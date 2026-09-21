"""
PostureTrack MedTech - versione CLOUD (streamlit-webrtc)
=========================================================
Analisi in tempo reale della postura cervicale (Forward Head Posture).

A differenza della versione locale (cv2.VideoCapture), qui il video arriva dalla
webcam del BROWSER dell'utente tramite WebRTC: per questo funziona anche quando
l'app e' ospitata su un server (Streamlit Community Cloud, Hugging Face Spaces...).

Avvio locale:  streamlit run app.py

NOTA: prototipo educativo. NON e' un dispositivo medico e non fornisce diagnosi.
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import av
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import pandas as pd
import requests
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

# ============================================================================
# 1. CONFIGURAZIONE GENERALE
# ============================================================================
st.set_page_config(page_title="PostureTrack MedTech", page_icon="🦴", layout="wide")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
PL = mp_pose.PoseLandmark

MODEL_COMPLEXITY = 1      # 0 = leggero, 1 = bilanciato, 2 = accurato (piu' lento)
VISIBILITY_MIN = 0.5      # sotto questa "visibility" un landmark e' inaffidabile
EMA_ALPHA = 0.3           # smoothing esponenziale dell'angolo
SAMPLE_EVERY_S = 0.2      # ogni quanti secondi salvare un campione per il grafico
MAX_DT = 0.5              # tetto al dt per frame
UI_REFRESH_S = 0.2        # aggiornamento dei box metriche

# Colori BGR (OpenCV)
COLOR_GOOD = (0, 200, 0)      # verde  -> postura corretta
COLOR_WARN = (0, 165, 255)    # ambra  -> oltre soglia, tolleranza non esaurita
COLOR_BAD = (0, 0, 255)       # rosso  -> postura scorretta confermata
COLOR_REF = (255, 255, 255)   # bianco -> verticale di riferimento

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

    Su alcune piattaforme (Streamlit Community Cloud incluso) i pacchetti WebRTC
    possono essere bloccati e senza un server TURN il video non parte.
    Per attivare il TURN, imposta UNA di queste opzioni nei Secrets / variabili d'ambiente:
      - TURN_URLS (separati da virgola), TURN_USERNAME, TURN_CREDENTIAL
      - OPEN_RELAY_API_HOST, OPEN_RELAY_API_KEY  (credenziali Open Relay / Metered)
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
# 3. LOGICA BIOMEDICA / MATEMATICA
# ============================================================================
def angle_from_vertical(p_top: Point, p_bottom: Point, facing: int) -> float:
    """
    Angolo (gradi) del segmento p_bottom -> p_top rispetto alla verticale per p_bottom.

        theta = atan2( facing * (x_top - x_bottom),  y_bottom - y_top )

    - Coordinate in PIXEL (non normalizzate) per non deformare l'angolo con l'aspect ratio.
    - L'asse y dell'immagine punta verso il basso, quindi la componente verticale
      "verso l'alto" e' (y_bottom - y_top).
    - `facing` = +1/-1 (verso cui guarda la persona): angolo POSITIVO = testa in avanti.
    """
    dx = facing * (p_top[0] - p_bottom[0])
    dy = p_bottom[1] - p_top[1]
    return math.degrees(math.atan2(dx, dy))


@dataclass
class PostureReading:
    side: str
    ear: Point
    shoulder: Point
    hip: Optional[Point]
    neck_angle: float
    trunk_angle: Optional[float]


def extract_posture(pose_landmarks, w: int, h: int) -> Optional[PostureReading]:
    """Sceglie il lato piu' visibile e calcola angolo del collo e del busto."""
    lm = pose_landmarks.landmark

    best_side, best_score = None, -1.0
    for side, (e, s, _) in SIDES.items():
        score = lm[e].visibility + lm[s].visibility
        if score > best_score:
            best_side, best_score = side, score

    ear_id, sh_id, hip_id = SIDES[best_side]
    if lm[ear_id].visibility < VISIBILITY_MIN or lm[sh_id].visibility < VISIBILITY_MIN:
        return None

    def to_px(idx) -> Point:
        return lm[idx].x * w, lm[idx].y * h

    ear, shoulder, nose = to_px(ear_id), to_px(sh_id), to_px(PL.NOSE)
    hip = to_px(hip_id) if lm[hip_id].visibility >= VISIBILITY_MIN else None

    facing = 1 if nose[0] > ear[0] else -1   # il naso sta davanti all'orecchio

    neck = angle_from_vertical(ear, shoulder, facing)
    trunk = angle_from_vertical(shoulder, hip, facing) if hip else None
    return PostureReading(best_side, ear, shoulder, hip, neck, trunk)


# ============================================================================
# 4. STATO DELLA SESSIONE
# ============================================================================
@dataclass
class SessionStats:
    last_ts: float = field(default_factory=time.time)
    t_start: float = field(default_factory=time.time)
    good_s: float = 0.0
    bad_s: float = 0.0
    undetected_s: float = 0.0
    alerts: int = 0
    bad_since: Optional[float] = None
    is_bad: bool = False
    angle_ema: Optional[float] = None
    last_sample_ts: float = 0.0
    history: list = field(default_factory=list)   # (t, angolo, soglia, scorretta)
    # valori "live" letti dall'interfaccia
    detected: bool = False
    live_angle: Optional[float] = None
    live_trunk: Optional[float] = None
    live_pending: float = 0.0


# ============================================================================
# 5. ELABORAZIONE VIDEO (gira in un thread separato, uno per ogni connessione)
# ============================================================================
def draw_overlay(img, results, reading: Optional[PostureReading], color, angle: float):
    mp_drawing.draw_landmarks(
        img, results.pose_landmarks, UPPER_BODY_CONNECTIONS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=3),
    )
    if reading is None:
        return

    ear = tuple(map(int, reading.ear))
    sh = tuple(map(int, reading.shoulder))

    cv2.line(img, sh, (sh[0], max(sh[1] - 180, 0)), COLOR_REF, 1, cv2.LINE_AA)  # verticale
    cv2.line(img, sh, ear, color, 5, cv2.LINE_AA)                                # collo

    points = [ear, sh] + ([tuple(map(int, reading.hip))] if reading.hip else [])
    for p in points:
        cv2.circle(img, p, 7, color, -1, cv2.LINE_AA)
        cv2.circle(img, p, 9, COLOR_REF, 1, cv2.LINE_AA)

    cv2.putText(img, f"{angle:.1f} deg", (sh[0] + 12, sh[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


class PostureProcessor(VideoProcessorBase):
    """
    Riceve ogni frame dal browser, calcola la postura, disegna l'overlay e
    restituisce il frame elaborato. Le statistiche stanno in `self.stats`
    (protette da lock) e vengono lette dal thread principale di Streamlit.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.threshold = 18.0     # aggiornati dagli slider ad ogni rerun
        self.tolerance = 3.0
        self.stats = SessionStats()
        self.pose = mp_pose.Pose(model_complexity=MODEL_COMPLEXITY,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

    def on_ended(self):
        try:
            self.pose.close()
        except Exception:
            pass

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = self._analyze(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def _analyze(self, img):
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)
        reading = (extract_posture(results.pose_landmarks, w, h)
                   if results.pose_landmarks else None)

        now = time.time()
        thr, tol = self.threshold, self.tolerance
        angle = None
        color = COLOR_REF

        with self.lock:
            s = self.stats
            dt = min(now - s.last_ts, MAX_DT)
            s.last_ts = now

            if reading is None:
                # nessun rilevamento affidabile: azzera il timer, conta a parte
                s.bad_since, s.is_bad = None, False
                s.undetected_s += dt
                s.detected, s.live_angle, s.live_trunk, s.live_pending = False, None, None, 0.0
            else:
                # smoothing esponenziale (EMA) contro il jitter dei landmarks
                raw = reading.neck_angle
                s.angle_ema = raw if s.angle_ema is None else (
                    EMA_ALPHA * raw + (1 - EMA_ALPHA) * s.angle_ema)
                angle = s.angle_ema

                # logica temporale: soglia superata per >= tolleranza secondi consecutivi
                pending = 0.0
                if angle > thr:
                    if s.bad_since is None:
                        s.bad_since = now
                    pending = now - s.bad_since
                    confirmed_bad = pending >= tol
                else:
                    s.bad_since = None
                    confirmed_bad = False

                if confirmed_bad and not s.is_bad:
                    s.alerts += 1                      # fronte di salita -> nuovo alert
                s.is_bad = confirmed_bad

                # il periodo di tolleranza conta come "corretto"
                if s.is_bad:
                    s.bad_s += dt
                else:
                    s.good_s += dt

                if now - s.last_sample_ts >= SAMPLE_EVERY_S:
                    s.last_sample_ts = now
                    s.history.append((now - s.t_start, angle, thr, s.is_bad))

                s.detected, s.live_angle = True, angle
                s.live_trunk, s.live_pending = reading.trunk_angle, pending
                color = (COLOR_BAD if s.is_bad else COLOR_WARN if pending > 0 else COLOR_GOOD)

        if results.pose_landmarks:
            draw_overlay(img, results, reading, color, angle if angle is not None else 0.0)
        else:
            cv2.putText(img, "Nessuna persona rilevata", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_WARN, 2, cv2.LINE_AA)
        return img


# ============================================================================
# 6. METRICHE LIVE E DASHBOARD
# ============================================================================
def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def update_live_metrics(ui: dict, proc: PostureProcessor, threshold: float, tolerance: float):
    with proc.lock:  # copia veloce dei valori, poi rilascia il lock
        s = proc.stats
        detected, angle, trunk = s.detected, s.live_angle, s.live_trunk
        pending, is_bad, alerts = s.live_pending, s.is_bad, s.alerts
        good, bad = s.good_s, s.bad_s

    if not detected or angle is None:
        ui["angle"].metric("Angolo collo (da verticale)", "--")
        ui["status"].info("Profilo non rilevato: posizionati di lato, con orecchio e spalla visibili.")
    else:
        ui["angle"].metric("Angolo collo (da verticale)", f"{angle:.1f}°",
                           delta=f"soglia {threshold:.0f}°", delta_color="off")
        if is_bad:
            ui["status"].error("⚠️ Attenzione: Correggi la postura!")
        elif pending > 0:
            ui["status"].warning(f"Angolo oltre soglia… {pending:.1f} / {tolerance:.1f} s")
        else:
            ui["status"].success("✅ Postura Corretta")

    ui["trunk"].metric("Inclinazione busto (info)", f"{trunk:.1f}°" if trunk is not None else "n/d")
    monitored = good + bad
    ui["pct"].metric("Postura corretta (sessione)",
                     f"{100 * good / monitored:.0f}%" if monitored > 0 else "--")
    ui["alerts"].metric("Alert generati", alerts)


def render_dashboard(stats: SessionStats):
    monitored = stats.good_s + stats.bad_s
    if monitored < 1:
        return

    st.divider()
    st.header("📊 Dashboard di sessione")

    pct_good = 100 * stats.good_s / monitored
    pct_bad = 100 - pct_good
    df = pd.DataFrame(stats.history,
                      columns=["tempo_s", "angolo_deg", "soglia_deg", "scorretta"])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Durata monitorata", fmt_time(monitored))
    c2.metric("Postura corretta", f"{pct_good:.1f}%")
    c3.metric("Postura scorretta", f"{pct_bad:.1f}%")
    c4.metric("Alert", stats.alerts)
    c5.metric("Angolo medio / max",
              f"{df.angolo_deg.mean():.1f}° / {df.angolo_deg.max():.1f}°" if not df.empty else "--")

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
            ax2.plot(df.tempo_s, df.angolo_deg, color="#3498db", lw=1.5, label="Angolo collo")
            ax2.step(df.tempo_s, df.soglia_deg, where="post", color="gray", ls="--", label="Soglia")
            bad = df[df.scorretta]
            ax2.scatter(bad.tempo_s, bad.angolo_deg, s=10, color="#e74c3c",
                        label="Postura scorretta", zorder=3)
            ax2.set_xlabel("Tempo (s)")
            ax2.set_ylabel("Angolo da verticale (°)")
            ax2.set_title("Andamento dell'angolo cervicale durante la sessione")
            ax2.grid(alpha=0.3)
            ax2.legend()
            st.pyplot(fig2)

    st.download_button("⬇️ Scarica dati sessione (CSV)",
                       df.to_csv(index=False).encode("utf-8"),
                       file_name="posturetrack_sessione.csv", mime="text/csv")


# ============================================================================
# 7. INTERFACCIA STREAMLIT
# ============================================================================
st.title("🦴 PostureTrack MedTech")
st.caption("Monitoraggio in tempo reale della Forward Head Posture · prototipo educativo, "
           "non e' un dispositivo medico")

with st.sidebar:
    st.header("⚙️ Parametri")
    threshold = st.slider("Soglia angolo (gradi)", 5, 40, 18, 1,
                          help="Oltre questo angolo (testa in avanti rispetto alla verticale "
                               "della spalla) la postura e' considerata a rischio.")
    tolerance = st.slider("Tempo di tolleranza (secondi)", 1.0, 10.0, 3.0, 0.5,
                          help="Per quanti secondi consecutivi l'angolo deve restare sopra "
                               "soglia prima di segnalare 'Postura Scorretta'.")
    st.divider()
    st.info("**Come usarla**\n\n"
            "1. Premi **START** e consenti l'accesso alla webcam\n"
            "2. Posizionati **di lato** (vista sagittale), con orecchio e spalla visibili\n"
            "3. Premi **STOP** per vedere la dashboard di sessione")
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

ui = {
    "status": metrics_col.empty(),
    "angle": metrics_col.empty(),
    "trunk": metrics_col.empty(),
    "pct": metrics_col.empty(),
    "alerts": metrics_col.empty(),
}

# Passa i parametri degli slider al processore (thread separato)
if ctx.video_processor:
    ctx.video_processor.threshold = float(threshold)
    ctx.video_processor.tolerance = float(tolerance)
    # tiene un riferimento alle statistiche: resta disponibile anche dopo STOP
    st.session_state["last_stats"] = ctx.video_processor.stats

if ctx.state.playing:
    # Loop di aggiornamento delle metriche. Quando l'utente preme STOP o cambia
    # uno slider, Streamlit interrompe questo script e lo rilancia da capo.
    while True:
        proc = ctx.video_processor
        if proc is not None:
            st.session_state["last_stats"] = proc.stats
            update_live_metrics(ui, proc, float(threshold), float(tolerance))
        else:
            ui["status"].info("Connessione alla webcam in corso…")
        time.sleep(UI_REFRESH_S)
else:
    ui["status"].info("Premi **START** per iniziare la sessione.")
    if "last_stats" in st.session_state:
        render_dashboard(st.session_state["last_stats"])
