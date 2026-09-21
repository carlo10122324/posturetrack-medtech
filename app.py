"""
PostureTrack MedTech - versione CLOUD v2 (streamlit-webrtc)
============================================================
Analisi in tempo reale della postura cervicale (Forward Head Posture).

Novita' della v2
----------------
1. CALIBRAZIONE personale: all'avvio (e con il pulsante "Ricalibra") l'app misura
   la TUA postura corretta per ~4 secondi. Da quel momento l'allarme scatta quando
   la testa si sposta in avanti di piu' della soglia RISPETTO A QUELLA BASE.
2. DUE MODALITA' di ripresa:
   - Frontale (webcam del portatile): angolo 3D testa-spalle ricavato dalle
     "world landmarks" di MediaPipe (profondita' stimata dal modello).
   - Laterale (profilo): angolo 2D orecchio-spalla rispetto alla verticale.
3. Stato ben leggibile: banner sul video, barra di deviazione con tacca della soglia
   e scheda colorata (verde / ambra / rossa).
4. ALLARME SONORO nel browser quando la postura scorretta e' confermata.

Avvio locale:  streamlit run app.py

NOTA: prototipo educativo. NON e' un dispositivo medico e non fornisce diagnosi.
"""

from __future__ import annotations

import io
import math
import os
import threading
import time
import wave
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import av
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

# ============================================================================
# 1. CONFIGURAZIONE GENERALE
# ============================================================================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
PL = mp_pose.PoseLandmark

MODEL_COMPLEXITY = 1        # 0 = leggero, 1 = bilanciato, 2 = accurato (piu' lento)
VISIBILITY_MIN = 0.5        # sotto questa "visibility" un landmark e' inaffidabile
EMA_ALPHA = 0.2             # smoothing esponenziale dell'angolo (piu' basso = piu' liscio)
SAMPLE_EVERY_S = 0.2        # ogni quanti secondi salvare un campione per il grafico
MAX_DT = 0.5                # tetto al dt per frame
UI_REFRESH_S = 0.2          # aggiornamento dell'interfaccia
CALIB_SECONDS = 4.0         # durata della calibrazione
CALIB_MIN_SAMPLES = 15      # campioni minimi per accettare la calibrazione
MIN_NECK_HEIGHT_M = 0.03    # altezza minima orecchio-spalla (m) per considerare valida la misura 3D

MODE_FRONTAL = "frontal"
MODE_LATERAL = "lateral"
MODE_LABELS = {
    "Frontale (webcam del portatile)": MODE_FRONTAL,
    "Laterale (di profilo)": MODE_LATERAL,
}

# Colori BGR (OpenCV)
COLOR_GOOD = (50, 160, 50)
COLOR_WARN = (0, 108, 239)
COLOR_BAD = (40, 40, 198)
COLOR_CALIB = (192, 101, 21)
COLOR_NONE = (110, 110, 110)
COLOR_REF = (255, 255, 255)

SIDES = {
    "sinistro": (PL.LEFT_EAR, PL.LEFT_SHOULDER, PL.LEFT_HIP),
    "destro": (PL.RIGHT_EAR, PL.RIGHT_SHOULDER, PL.RIGHT_HIP),
}

UPPER_IDS = set(range(0, 13)) | {PL.LEFT_HIP.value, PL.RIGHT_HIP.value}
UPPER_BODY_CONNECTIONS = [
    c for c in mp_pose.POSE_CONNECTIONS if c[0] in UPPER_IDS and c[1] in UPPER_IDS
]

Point = Tuple[float, float]


# ============================================================================
# 2. RETE: configurazione STUN / TURN per il deploy in cloud
# ============================================================================
def _secret(name: str) -> Optional[str]:
    """Legge un valore da st.secrets (Streamlit Cloud) o dalle variabili d'ambiente (HF Spaces)."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name)


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_open_relay(host: str, api_key: str) -> list:
    """Scarica le credenziali TURN dal servizio Open Relay (Metered). Ritorna [] se fallisce."""
    try:
        r = requests.get(f"https://{host}/api/v1/turn/credentials",
                         params={"apiKey": api_key}, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def get_rtc_configuration() -> dict:
    """
    STUN pubblico di Google (sempre) + TURN opzionale.
    Per attivare il TURN imposta nei Secrets / variabili d'ambiente UNA di queste opzioni:
      - TURN_URLS (separati da virgola), TURN_USERNAME, TURN_CREDENTIAL
      - OPEN_RELAY_API_HOST, OPEN_RELAY_API_KEY
    """
    ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]

    urls, user, cred = _secret("TURN_URLS"), _secret("TURN_USERNAME"), _secret("TURN_CREDENTIAL")
    if urls and user and cred:
        ice_servers.append({"urls": [u.strip() for u in urls.split(",")],
                            "username": user, "credential": cred})

    host, key = _secret("OPEN_RELAY_API_HOST"), _secret("OPEN_RELAY_API_KEY")
    if host and key:
        ice_servers.extend(_fetch_open_relay(host, key))

    return {"iceServers": ice_servers}


# ============================================================================
# 3. SUONO DI ALLARME
# ============================================================================
@st.cache_data(show_spinner=False)
def make_beep_wav() -> bytes:
    """Genera in memoria un breve allarme (due beep acuti + uno piu' grave) in formato WAV."""
    sr = 22050

    def tone(freq: float, dur: float, vol: float = 0.6) -> np.ndarray:
        t = np.arange(int(sr * dur)) / sr
        y = np.sin(2 * np.pi * freq * t)
        fade = int(sr * 0.01)                       # 10 ms di fade in/out contro i "click"
        env = np.ones_like(y)
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
        return vol * y * env

    gap = np.zeros(int(sr * 0.08))
    signal = np.concatenate([tone(880, 0.22), gap, tone(880, 0.22), gap, tone(660, 0.35)])
    pcm = (signal * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def play_beep(placeholder):
    """
    Riproduce il suono nel browser. Si svuota prima il segnaposto, cosi' Streamlit
    ricrea l'elemento <audio> e l'autoplay riparte anche per allarmi ravvicinati.
    """
    placeholder.empty()
    time.sleep(0.15)
    try:
        placeholder.audio(make_beep_wav(), format="audio/wav", autoplay=True)
    except TypeError:  # versioni vecchie di Streamlit senza il parametro autoplay
        placeholder.audio(make_beep_wav(), format="audio/wav")


# ============================================================================
# 4. LOGICA BIOMEDICA / MATEMATICA
# ============================================================================
def angle_from_vertical(p_top: Point, p_bottom: Point, facing: int) -> float:
    """
    MODALITA' LATERALE (2D). Angolo (gradi) del segmento p_bottom -> p_top
    rispetto alla verticale passante per p_bottom.

        theta = atan2( facing * (x_top - x_bottom),  y_bottom - y_top )

    - Coordinate in PIXEL (non normalizzate) per non deformare l'angolo.
    - L'asse y dell'immagine punta verso il basso.
    - `facing` (+1/-1): verso cui guarda la persona; theta > 0 = testa in avanti.
    """
    dx = facing * (p_top[0] - p_bottom[0])
    dy = p_bottom[1] - p_top[1]
    return math.degrees(math.atan2(dx, dy))


def frontal_head_angle(sh_y: float, sh_z: float, ear_y: float, ear_z: float) -> float:
    """
    MODALITA' FRONTALE (3D). Usa le coordinate "world" di MediaPipe (metri, origine
    al centro delle anche, y verso il basso, z minore = piu' vicino alla camera).

    Nel piano sagittale (asse verticale y, asse antero-posteriore z):
      - componente verticale   : sh_y - ear_y   (>0 se l'orecchio sta sopra la spalla)
      - componente in avanti   : sh_z - ear_z   (>0 se l'orecchio e' piu' vicino alla camera
                                                 della spalla = testa proiettata in avanti)

        theta = atan2( sh_z - ear_z,  sh_y - ear_y )

    theta > 0 = testa in avanti rispetto alle spalle (Forward Head Posture).
    """
    return math.degrees(math.atan2(sh_z - ear_z, sh_y - ear_y))


@dataclass
class PostureReading:
    """Risultato dell'analisi di un frame."""
    raw_angle: float                 # angolo grezzo (gradi) secondo la modalita' scelta
    ear_px: Point                    # punto "testa" per il disegno
    shoulder_px: Point               # punto "spalla" per il disegno
    extra_px: List[Point] = field(default_factory=list)


def _mean_point(points: List[Point]) -> Point:
    return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))


def _extract_frontal(results, lm, px) -> Optional[PostureReading]:
    world = results.pose_world_landmarks
    if world is None:
        return None
    wlm = world.landmark

    ls, rs = PL.LEFT_SHOULDER.value, PL.RIGHT_SHOULDER.value
    if min(lm[ls].visibility, lm[rs].visibility) < VISIBILITY_MIN:
        return None
    ears = [i for i in (PL.LEFT_EAR.value, PL.RIGHT_EAR.value)
            if lm[i].visibility >= VISIBILITY_MIN]
    if not ears:
        return None

    # media di entrambe le spalle e delle orecchie visibili -> misura stabile, niente "salti di lato"
    sh_y = (wlm[ls].y + wlm[rs].y) / 2
    sh_z = (wlm[ls].z + wlm[rs].z) / 2
    ear_y = sum(wlm[i].y for i in ears) / len(ears)
    ear_z = sum(wlm[i].z for i in ears) / len(ears)

    if sh_y - ear_y < MIN_NECK_HEIGHT_M:   # stima geometricamente implausibile
        return None

    raw = frontal_head_angle(sh_y, sh_z, ear_y, ear_z)
    ear_px = _mean_point([px(i) for i in ears])
    sh_px = _mean_point([px(ls), px(rs)])
    extra = [px(i) for i in ears] + [px(ls), px(rs)]
    return PostureReading(raw, ear_px, sh_px, extra)


def _extract_lateral(lm, px) -> Optional[PostureReading]:
    best_side, best_score = None, -1.0
    for side, (e, s, _) in SIDES.items():
        score = lm[e].visibility + lm[s].visibility
        if score > best_score:
            best_side, best_score = side, score

    ear_id, sh_id, _ = SIDES[best_side]
    if lm[ear_id].visibility < VISIBILITY_MIN or lm[sh_id].visibility < VISIBILITY_MIN:
        return None

    ear, shoulder, nose = px(ear_id), px(sh_id), px(PL.NOSE)
    facing = 1 if nose[0] > ear[0] else -1      # il naso sta davanti all'orecchio
    raw = angle_from_vertical(ear, shoulder, facing)
    return PostureReading(raw, ear, shoulder, [])


def extract_posture(results, w: int, h: int, mode: str) -> Optional[PostureReading]:
    if not results.pose_landmarks:
        return None
    lm = results.pose_landmarks.landmark

    def px(i) -> Point:
        return lm[i].x * w, lm[i].y * h

    if mode == MODE_FRONTAL:
        return _extract_frontal(results, lm, px)
    return _extract_lateral(lm, px)


# ============================================================================
# 5. STATO DELLA SESSIONE
# ============================================================================
@dataclass
class SessionStats:
    last_ts: float = 0.0
    t_start: float = 0.0
    good_s: float = 0.0
    bad_s: float = 0.0
    undetected_s: float = 0.0
    alerts: int = 0
    bad_since: Optional[float] = None
    is_bad: bool = False
    last_sample_ts: float = 0.0
    history: list = field(default_factory=list)   # (t, deviazione, soglia, scorretta)


def new_stats(now: float) -> SessionStats:
    return SessionStats(last_ts=now, t_start=now)


@dataclass
class LiveState:
    """Valori "live" letti dall'interfaccia."""
    state: str = "calibrating"      # calibrating | nodetect | good | pending | bad
    deviation: Optional[float] = None
    pending: float = 0.0
    calib_progress: float = 0.0


# ============================================================================
# 6. DISEGNO SUL FRAME
# ============================================================================
STATE_STYLE = {
    # stato: (colore BGR, testo del banner sul video)
    "calibrating": (COLOR_CALIB, "CALIBRAZIONE: siediti dritto e guarda lo schermo"),
    "nodetect": (COLOR_NONE, "NESSUNA PERSONA RILEVATA"),
    "good": (COLOR_GOOD, "POSTURA CORRETTA"),
    "pending": (COLOR_WARN, "ATTENZIONE: testa in avanti"),
    "bad": (COLOR_BAD, "POSTURA SCORRETTA - raddrizza la testa"),
}


def draw_overlay(img, results, reading: Optional[PostureReading], state: str,
                 deviation: Optional[float], threshold: float):
    h, w = img.shape[:2]
    color, banner_text = STATE_STYLE[state]

    # Scheletro MediaPipe (solo parte superiore del corpo)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            img, results.pose_landmarks, UPPER_BODY_CONNECTIONS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=3),
        )

    if reading is not None:
        ear = tuple(map(int, reading.ear_px))
        sh = tuple(map(int, reading.shoulder_px))
        cv2.line(img, sh, (sh[0], max(sh[1] - 180, 0)), COLOR_REF, 1, cv2.LINE_AA)  # verticale
        cv2.line(img, sh, ear, color, 5, cv2.LINE_AA)                                # collo
        for p in list(map(lambda q: tuple(map(int, q)), reading.extra_px)) + [ear, sh]:
            cv2.circle(img, p, 7, color, -1, cv2.LINE_AA)
            cv2.circle(img, p, 9, COLOR_REF, 1, cv2.LINE_AA)

    # Banner di stato in alto
    cv2.rectangle(img, (0, 0), (w, 44), color, -1)
    cv2.putText(img, banner_text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)

    # Barra di deviazione in basso: la tacca bianca al centro e' la soglia
    if deviation is not None and threshold > 0:
        bar_w, bar_h = int(w * 0.6), 16
        x0, y0 = (w - bar_w) // 2, h - 36
        cv2.rectangle(img, (x0, y0), (x0 + bar_w, y0 + bar_h), (60, 60, 60), -1)
        ratio = float(np.clip(deviation / (2 * threshold), 0.0, 1.0))
        cv2.rectangle(img, (x0, y0), (x0 + int(bar_w * ratio), y0 + bar_h), color, -1)
        mid = x0 + bar_w // 2
        cv2.line(img, (mid, y0 - 4), (mid, y0 + bar_h + 4), COLOR_REF, 2)
        cv2.putText(img, f"{deviation:+.0f} deg  (soglia {threshold:.0f})", (x0, y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ============================================================================
# 7. ELABORAZIONE VIDEO (thread separato, uno per ogni connessione)
# ============================================================================
class PostureProcessor(VideoProcessorBase):
    """
    Riceve ogni frame dal browser, calcola la postura, disegna l'overlay e restituisce
    il frame. Lo stato e' protetto da un lock e letto dal thread principale di Streamlit.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.threshold = 10.0       # aggiornati dagli slider ad ogni rerun
        self.tolerance = 3.0
        self.mode = MODE_FRONTAL
        self.stats = new_stats(time.time())
        self.live = LiveState()
        self.angle_ema: Optional[float] = None
        # calibrazione (parte automaticamente all'avvio)
        self.calibrating = True
        self.calib_start: Optional[float] = None
        self.calib_samples: List[float] = []
        self.baseline: Optional[float] = None
        self.pose = mp_pose.Pose(model_complexity=MODEL_COMPLEXITY,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

    # ---- comandi dal thread principale -------------------------------------
    def start_calibration(self):
        with self.lock:
            self.calibrating = True
            self.calib_start = None
            self.calib_samples = []
            self.baseline = None
            self.angle_ema = None
            self.stats.bad_since = None
            self.stats.is_bad = False
            self.live.state = "calibrating"
            self.live.calib_progress = 0.0
            self.live.deviation = None
            self.live.pending = 0.0

    def set_mode(self, mode: str):
        if mode != self.mode:
            self.mode = mode
            self.start_calibration()   # cambia il tipo di misura -> serve una nuova base

    def snapshot(self) -> dict:
        with self.lock:
            s, lv = self.stats, self.live
            return dict(state=lv.state, deviation=lv.deviation, pending=lv.pending,
                        calib_progress=lv.calib_progress, alerts=s.alerts,
                        good_s=s.good_s, bad_s=s.bad_s, is_bad=s.is_bad)

    def on_ended(self):
        try:
            self.pose.close()
        except Exception:
            pass

    # ---- elaborazione -------------------------------------------------------
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = self._analyze(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def _analyze(self, img):
        mode, thr, tol = self.mode, self.threshold, self.tolerance
        if mode == MODE_FRONTAL:
            img = cv2.flip(img, 1)          # anteprima "a specchio", piu' naturale davanti al PC
        h, w = img.shape[:2]

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)
        reading = extract_posture(results, w, h, mode)
        now = time.time()

        deviation: Optional[float] = None
        with self.lock:
            s, lv = self.stats, self.live
            dt = max(0.0, min(now - s.last_ts, MAX_DT))
            s.last_ts = now

            if reading is None:
                s.bad_since, s.is_bad = None, False
                lv.deviation, lv.pending = None, 0.0
                if self.calibrating:
                    # la calibrazione richiede una posa rilevata in modo continuo
                    self.calib_start, self.calib_samples = None, []
                    lv.calib_progress = 0.0
                    lv.state = "calibrating"
                else:
                    s.undetected_s += dt
                    lv.state = "nodetect"
            else:
                # smoothing esponenziale (EMA) contro il jitter dei landmarks
                raw = reading.raw_angle
                self.angle_ema = raw if self.angle_ema is None else (
                    EMA_ALPHA * raw + (1 - EMA_ALPHA) * self.angle_ema)
                smooth = self.angle_ema

                if self.calibrating:
                    if self.calib_start is None:
                        self.calib_start = now
                    self.calib_samples.append(smooth)
                    elapsed = now - self.calib_start
                    lv.calib_progress = min(elapsed / CALIB_SECONDS, 1.0)
                    lv.state, lv.deviation, lv.pending = "calibrating", None, 0.0
                    if elapsed >= CALIB_SECONDS and len(self.calib_samples) >= CALIB_MIN_SAMPLES:
                        # base personale = mediana dei campioni (robusta agli outlier)
                        self.baseline = float(np.median(self.calib_samples))
                        self.calibrating = False
                        self.stats = s = new_stats(now)    # le statistiche ripartono da qui

                if not self.calibrating and self.baseline is not None:
                    deviation = smooth - self.baseline      # > 0 = testa piu' avanti della tua base

                    # logica temporale: soglia superata per >= tolleranza secondi consecutivi
                    pending = 0.0
                    if deviation > thr:
                        if s.bad_since is None:
                            s.bad_since = now
                        pending = now - s.bad_since
                        confirmed_bad = pending >= tol
                    else:
                        s.bad_since = None
                        confirmed_bad = False

                    if confirmed_bad and not s.is_bad:
                        s.alerts += 1                       # fronte di salita -> nuovo alert
                    s.is_bad = confirmed_bad

                    if s.is_bad:                            # il periodo di tolleranza conta come corretto
                        s.bad_s += dt
                    else:
                        s.good_s += dt

                    if now - s.last_sample_ts >= SAMPLE_EVERY_S:
                        s.last_sample_ts = now
                        s.history.append((now - s.t_start, deviation, thr, s.is_bad))

                    lv.deviation, lv.pending = deviation, pending
                    lv.state = "bad" if s.is_bad else ("pending" if pending > 0 else "good")

            state = lv.state
            calib_left = max(0.0, CALIB_SECONDS * (1 - lv.calib_progress))

        draw_overlay(img, results, reading, state, deviation, thr)
        if state == "calibrating" and reading is not None:
            cv2.putText(img, f"{calib_left:.0f} s", (12, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 2, cv2.LINE_AA)
        return img


# ============================================================================
# 8. INTERFACCIA: SCHEDA DI STATO, METRICHE, DASHBOARD
# ============================================================================
def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _status_card(ph, bg: str, title: str, subtitle: str):
    ph.markdown(
        f"<div style='background:{bg};color:#fff;padding:20px 14px;border-radius:14px;"
        f"text-align:center;font-size:1.6rem;font-weight:700;line-height:1.25'>{title}"
        f"<div style='font-size:0.95rem;font-weight:400;margin-top:6px'>{subtitle}</div></div>",
        unsafe_allow_html=True)


def render_live(ui: dict, snap: Optional[dict], threshold: float, tolerance: float):
    if snap is None:
        _status_card(ui["status"], "#546e7a", "⏳ Connessione…", "Attendo il video dalla webcam")
        return

    state = snap["state"]
    if state == "calibrating":
        _status_card(ui["status"], "#1565c0", "🎯 Calibrazione in corso",
                     f"Siediti nella tua postura corretta e guarda lo schermo · "
                     f"{snap['calib_progress'] * 100:.0f}%")
    elif state == "nodetect":
        _status_card(ui["status"], "#546e7a", "👤 Nessuna persona rilevata",
                     "Inquadra spalle e orecchie, con buona luce")
    elif state == "good":
        _status_card(ui["status"], "#2e7d32", "✅ Postura Corretta", "Testa in linea con la tua base")
    elif state == "pending":
        _status_card(ui["status"], "#ef6c00", "⚠️ Attenzione",
                     f"Testa in avanti da {snap['pending']:.1f} s (allarme a {tolerance:.1f} s)")
    else:
        _status_card(ui["status"], "#c62828", "🚨 Postura Scorretta",
                     "Attenzione: Correggi la postura!")

    dev = snap["deviation"]
    ui["dev"].metric("Deviazione dalla tua postura corretta",
                     f"{dev:+.1f}°" if dev is not None else "--",
                     delta=f"soglia {threshold:.0f}°", delta_color="off")
    monitored = snap["good_s"] + snap["bad_s"]
    ui["pct"].metric("Postura corretta", f"{100 * snap['good_s'] / monitored:.0f}%" if monitored > 0 else "--")
    ui["alerts"].metric("Alert generati", snap["alerts"])


def render_dashboard(stats: SessionStats):
    monitored = stats.good_s + stats.bad_s
    if monitored < 1:
        return

    st.divider()
    st.header("📊 Dashboard di sessione")

    pct_good = 100 * stats.good_s / monitored
    pct_bad = 100 - pct_good
    df = pd.DataFrame(stats.history,
                      columns=["tempo_s", "deviazione_deg", "soglia_deg", "scorretta"])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Durata monitorata", fmt_time(monitored))
    c2.metric("Postura corretta", f"{pct_good:.1f}%")
    c3.metric("Postura scorretta", f"{pct_bad:.1f}%")
    c4.metric("Alert", stats.alerts)
    c5.metric("Deviazione media / max",
              f"{df.deviazione_deg.mean():.1f}° / {df.deviazione_deg.max():.1f}°" if not df.empty else "--")

    left, right = st.columns([1, 2])

    with left:
        fig1, ax1 = plt.subplots(figsize=(4, 4))
        ax1.pie([stats.good_s, stats.bad_s], labels=["Corretta", "Scorretta"],
                colors=["#2ecc71", "#e74c3c"], autopct="%1.1f%%", startangle=90,
                wedgeprops=dict(width=0.4, edgecolor="white"))
        ax1.set_title("Tempo in postura corretta vs scorretta")
        st.pyplot(fig1)

    with right:
        if not df.empty:
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(df.tempo_s, df.deviazione_deg, color="#3498db", lw=1.5,
                     label="Deviazione dalla postura corretta")
            ax2.step(df.tempo_s, df.soglia_deg, where="post", color="gray", ls="--", label="Soglia")
            bad = df[df.scorretta]
            ax2.scatter(bad.tempo_s, bad.deviazione_deg, s=10, color="#e74c3c",
                        label="Postura scorretta", zorder=3)
            ax2.set_xlabel("Tempo (s)")
            ax2.set_ylabel("Deviazione (°)")
            ax2.set_title("Andamento della deviazione cervicale durante la sessione")
            ax2.grid(alpha=0.3)
            ax2.legend()
            st.pyplot(fig2)

    st.download_button("⬇️ Scarica dati sessione (CSV)",
                       df.to_csv(index=False).encode("utf-8"),
                       file_name="posturetrack_sessione.csv", mime="text/csv")


def request_calibration():
    """Callback del pulsante 'Ricalibra'."""
    st.session_state["calib_req"] = True


# ============================================================================
# 9. APP STREAMLIT
# ============================================================================
def main():
    st.set_page_config(page_title="PostureTrack MedTech", page_icon="🦴", layout="wide")
    # il player audio dell'allarme resta nascosto (il suono parte comunque)
    st.markdown("<style>[data-testid='stAudio'], audio {display:none !important;}</style>",
                unsafe_allow_html=True)

    st.title("🦴 PostureTrack MedTech")
    st.caption("Monitoraggio in tempo reale della Forward Head Posture · prototipo educativo, "
               "non e' un dispositivo medico")

    with st.sidebar:
        st.header("⚙️ Parametri")
        mode_label = st.radio(
            "Posizione della webcam", list(MODE_LABELS.keys()),
            help="Frontale: la webcam del portatile davanti a te (stima 3D). "
                 "Laterale: webcam posta di profilo (misura 2D piu' precisa).")
        mode = MODE_LABELS[mode_label]
        threshold = st.slider(
            "Soglia di deviazione (gradi)", 3, 30, 10, 1,
            help="Di quanto la testa puo' andare in avanti rispetto alla TUA postura corretta "
                 "(calibrata all'avvio) prima che scatti l'allarme.")
        tolerance = st.slider(
            "Tempo di tolleranza (secondi)", 1.0, 10.0, 3.0, 0.5,
            help="Per quanti secondi consecutivi la deviazione deve restare sopra soglia "
                 "prima di segnalare 'Postura Scorretta'.")

        st.divider()
        st.subheader("🔔 Avviso sonoro")
        sound_on = st.toggle("Attiva avviso sonoro", value=True)
        repeat_s = st.slider("Ripeti l'avviso ogni (secondi)", 3, 30, 8, 1)
        test_sound = st.button("🔊 Prova suono", use_container_width=True)
        st.caption("Il browser puo' bloccare l'audio finche' non interagisci con la pagina: "
                   "premi START o 'Prova suono' almeno una volta e controlla il volume.")

        st.divider()
        st.button("🎯 Ricalibra postura corretta", on_click=request_calibration,
                  use_container_width=True)
        st.info("**Come usarla**\n\n"
                "1. Premi **START** e consenti la webcam\n"
                "2. Siediti **nella tua postura corretta**: per 4 secondi l'app la memorizza\n"
                "3. Se la testa va in avanti oltre la soglia, scatta l'allarme\n"
                "4. Premi **STOP** per vedere la dashboard")
        st.caption("🔒 Il video viene elaborato in tempo reale e non viene salvato.")

    video_col, metrics_col = st.columns([3, 2])

    with video_col:
        ctx = webrtc_streamer(
            key="posturetrack",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=get_rtc_configuration(),
            media_stream_constraints={
                "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                "audio": False,
            },
            video_processor_factory=PostureProcessor,
            async_processing=True,
        )
        st.caption("🟢 verde = corretta · 🟠 ambra = oltre soglia, sta per scattare l'allarme · "
                   "🔴 rosso = scorretta. La barra in basso mostra la deviazione: "
                   "la tacca bianca e' la soglia.")

    ui = {"status": metrics_col.empty(), "dev": metrics_col.empty()}
    sub_a, sub_b = metrics_col.columns(2)
    ui["pct"], ui["alerts"] = sub_a.empty(), sub_b.empty()
    sound_ph = metrics_col.empty()

    if test_sound:
        play_beep(sound_ph)

    # Passa parametri e comandi al processore (thread separato)
    proc = ctx.video_processor
    if proc is not None:
        proc.threshold = float(threshold)
        proc.tolerance = float(tolerance)
        proc.set_mode(mode)
        if st.session_state.pop("calib_req", False):
            proc.start_calibration()
        st.session_state["last_stats"] = proc.stats
    else:
        st.session_state.pop("calib_req", None)

    if ctx.state.playing:
        # Loop di aggiornamento dell'interfaccia. Quando l'utente preme STOP o cambia
        # uno slider, Streamlit interrompe questo script e lo rilancia da capo.
        while True:
            proc = ctx.video_processor
            if proc is None:
                render_live(ui, None, float(threshold), float(tolerance))
            else:
                snap = proc.snapshot()
                st.session_state["last_stats"] = proc.stats
                render_live(ui, snap, float(threshold), float(tolerance))

                # ---- allarme sonoro ----
                if snap["state"] == "bad":
                    now = time.time()
                    if sound_on and now - st.session_state.get("last_beep", 0.0) >= repeat_s:
                        st.session_state["last_beep"] = now
                        play_beep(sound_ph)
                else:
                    st.session_state["last_beep"] = 0.0   # al prossimo allarme suona subito
            time.sleep(UI_REFRESH_S)
    else:
        _status_card(ui["status"], "#546e7a", "Premi START", "per iniziare la sessione")
        if "last_stats" in st.session_state:
            render_dashboard(st.session_state["last_stats"])


if __name__ == "__main__":
    main()
