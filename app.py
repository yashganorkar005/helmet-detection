import streamlit as st
import cv2
import os
import time
import numpy as np
from ultralytics import YOLO
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import queue
# autorefresh removed


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Helmet Detection System",
    layout="wide",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🛡</text></svg>"
)

# Auto refresh UI every 1 second so queue is drained regularly
import time as _time; _time.sleep(0); st.rerun() if False else None

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');

:root {
    --bg-primary:    #0a0c10;
    --bg-secondary:  #0f1318;
    --bg-card:       #131820;
    --bg-card-hover: #171f2b;
    --accent:        #00d4ff;
    --accent-dim:    #0099bb;
    --danger:        #ff3b3b;
    --danger-dim:    #cc2020;
    --success:       #00e676;
    --success-dim:   #00b359;
    --warning:       #ffab00;
    --text-primary:  #e8edf5;
    --text-secondary:#8a95a3;
    --border:        #1e2a38;
    --border-bright: #2a3f55;
    --glow-accent:   0 0 20px rgba(0,212,255,0.25);
    --glow-danger:   0 0 20px rgba(255,59,59,0.25);
    --glow-success:  0 0 20px rgba(0,230,118,0.25);
    --glow-warn:     0 0 20px rgba(255,171,0,0.25);
}

html, body, .stApp {
    background-color: var(--bg-primary) !important;
    font-family: 'Exo 2', sans-serif !important;
    color: var(--text-primary) !important;
}
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.block-container {
    padding: 2rem 3rem 4rem !important;
    max-width: 1400px !important;
}

/* ===== HERO HEADER ===== */
.hero-header {
    padding: 2rem 0 1.75rem;
    margin-bottom: 2rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 1.25rem;
}
.hero-logo {
    width: 56px;
    height: 56px;
    flex-shrink: 0;
    filter: drop-shadow(0 0 12px rgba(0,212,255,0.5));
}
.hero-text h1 {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    margin: 0 0 0.15rem 0 !important;
    line-height: 1 !important;
}
.hero-text h1 span { color: var(--accent); }
.hero-text p {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.72rem !important;
    color: var(--text-secondary) !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    margin: 0 !important;
}
.hero-badge {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 15px;
    background: #00d4ff0d;
    border: 1px solid var(--accent-dim);
    border-radius: 8px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    color: var(--accent);
    letter-spacing: 2px;
}
.hero-badge-dot {
    width: 7px;
    height: 7px;
    background: var(--accent);
    border-radius: 50%;
    animation: pulse-cyan 2s infinite;
}
@keyframes pulse-cyan {
    0%,100% { box-shadow: 0 0 0 0 rgba(0,212,255,0.5); }
    50%      { box-shadow: 0 0 0 5px rgba(0,212,255,0); }
}
@keyframes pulse-green {
    0%,100% { box-shadow: 0 0 0 0 rgba(0,230,118,0.5); }
    50%      { box-shadow: 0 0 0 5px rgba(0,230,118,0); }
}
@keyframes pulse-red {
    0%,100% { box-shadow: 0 0 0 0 rgba(255,59,59,0.6); }
    50%      { box-shadow: 0 0 0 5px rgba(255,59,59,0); }
}
@keyframes fadeInUp {
    from { opacity:0; transform:translateY(14px); }
    to   { opacity:1; transform:translateY(0); }
}
.fade-in { animation: fadeInUp 0.5s ease forwards; }

/* ===== SECTION LABEL ===== */
.section-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, var(--border-bright), transparent);
}

/* ===== STAT CARDS ===== */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 1.5rem;
}
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.25s, transform 0.2s;
}
.stat-card:hover {
    border-color: var(--border-bright);
    transform: translateY(-2px);
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 12px 12px 0 0;
}
.stat-card.cyan::before  { background: linear-gradient(90deg, var(--accent), transparent); }
.stat-card.green::before { background: linear-gradient(90deg, var(--success), transparent); }
.stat-card.red::before   { background: linear-gradient(90deg, var(--danger), transparent); }
.stat-card.warn::before  { background: linear-gradient(90deg, var(--warning), transparent); }

.stat-icon {
    width: 36px;
    height: 36px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 0.7rem;
}
.stat-card.cyan  .stat-icon { background: rgba(0,212,255,0.1);  }
.stat-card.green .stat-icon { background: rgba(0,230,118,0.1);  }
.stat-card.red   .stat-icon { background: rgba(255,59,59,0.1);  }
.stat-card.warn  .stat-icon { background: rgba(255,171,0,0.1);  }

.stat-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.2rem;
}
.stat-card.cyan  .stat-value { color: var(--accent);   }
.stat-card.green .stat-value { color: var(--success);  }
.stat-card.red   .stat-value { color: var(--danger);   }
.stat-card.warn  .stat-value { color: var(--warning);  }

.stat-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 2px;
    color: var(--text-secondary);
    text-transform: uppercase;
}
.stat-sub {
    font-family: 'Exo 2', sans-serif;
    font-size: 0.72rem;
    color: var(--text-secondary);
    margin-top: 0.4rem;
    opacity: 0.7;
}

/* ===== BOTTOM ROW: status + log ===== */
.panel-row {
    display: grid;
    grid-template-columns: 1fr 2fr;
    gap: 14px;
    margin-bottom: 1.75rem;
}
.status-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
}
.status-indicator {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 0.9rem 1.1rem;
    border-radius: 10px;
    font-family: 'Exo 2', sans-serif;
    font-size: 0.87rem;
    font-weight: 500;
    border: 1px solid transparent;
    margin-bottom: 0.6rem;
}
.status-indicator.safe   { background:rgba(0,230,118,0.07); border-color:rgba(0,230,118,0.3); color:var(--success); box-shadow:var(--glow-success); }
.status-indicator.danger { background:rgba(255,59,59,0.07);  border-color:rgba(255,59,59,0.3);  color:var(--danger);  box-shadow:var(--glow-danger);  }
.status-indicator.idle   { background:rgba(0,212,255,0.05);  border-color:rgba(0,212,255,0.15); color:var(--text-secondary); }
.status-dot { width:9px; height:9px; border-radius:50%; margin-top:4px; flex-shrink:0; }
.safe   .status-dot { background:var(--success); animation:pulse-green 2s infinite; }
.danger .status-dot { background:var(--danger);  animation:pulse-red   1s infinite; }
.idle   .status-dot { background:var(--text-secondary); }

/* last detection time */
.last-seen {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-secondary);
    letter-spacing: 1px;
    padding: 6px 10px;
    background: var(--bg-secondary);
    border-radius: 6px;
    margin-top: 8px;
}

/* ===== LOG TERMINAL ===== */
.log-terminal {
    background: #080b0f;
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
}
.log-terminal-bar {
    background: var(--bg-card);
    padding: 7px 13px;
    display: flex;
    align-items: center;
    gap: 7px;
    border-bottom: 1px solid var(--border);
}
.log-dot { width:9px; height:9px; border-radius:50%; }
.log-terminal-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.66rem;
    color: var(--text-secondary);
    letter-spacing: 2px;
    margin-left: 4px;
}
.log-terminal .stTextArea textarea {
    background: transparent !important;
    border: none !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.77rem !important;
    color: #7ec8e3 !important;
    line-height: 1.75 !important;
    padding: 12px 16px !important;
    resize: none !important;
    caret-color: var(--accent);
}
.log-terminal .stTextArea textarea:focus { outline:none !important; box-shadow:none !important; }
.log-terminal .stTextArea > div > div { border:none !important; background:transparent !important; }

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    color: var(--text-secondary) !important;
    border: none !important;
    transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-card) !important;
    color: var(--accent) !important;
    box-shadow: var(--glow-accent) !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 0 12px 12px !important;
    padding: 2rem !important;
    margin-top: -1px !important;
}

/* File uploader */
.stFileUploader > div {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border-bright) !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}
.stFileUploader > div:hover {
    border-color: var(--accent-dim) !important;
    background: var(--bg-card-hover) !important;
}

.stImage img { border-radius: 10px !important; border: 1px solid var(--border) !important; }
[data-testid="column"] { padding: 0 8px !important; }
hr { border-color: var(--border) !important; opacity: 0.5 !important; }
::-webkit-scrollbar       { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-dim); }

.footer-bar {
    margin-top: 3rem;
    padding-top: 1.25rem;
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.footer-bar p {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.65rem !important;
    color: var(--text-secondary) !important;
    letter-spacing: 2px !important;
    margin: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- PATHS ----------------
MODEL_PATH = "runs/train/helmet_detector2/weights/best.pt"
VIOLATION_DIR = "violations"
os.makedirs(VIOLATION_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

# ---------------- SHARED QUEUE (module-level singleton) ----------------
if not hasattr(st, "_shared_event_queue"):
    st._shared_event_queue = queue.Queue()
_event_queue = st._shared_event_queue

# ---------------- SESSION STATE ----------------
if "event_log"        not in st.session_state: st.session_state.event_log        = []
if "last_status"      not in st.session_state: st.session_state.last_status      = "Waiting for detection..."
if "total_detections" not in st.session_state: st.session_state.total_detections = 0
if "helmet_count"     not in st.session_state: st.session_state.helmet_count     = 0
if "violation_count"  not in st.session_state: st.session_state.violation_count  = 0
if "last_event_time"  not in st.session_state: st.session_state.last_event_time  = None

# ---------------- HELPERS ----------------
def log_event(msg, is_helmet=None):
    timestamp = time.strftime("%H:%M:%S")
    entry = f"[{timestamp}] {msg}"
    st.session_state.event_log.append(entry)
    st.session_state.last_status = msg
    st.session_state.last_event_time = timestamp
    if is_helmet is True:
        st.session_state.total_detections += 1
        st.session_state.helmet_count     += 1
    elif is_helmet is False:
        st.session_state.total_detections += 1
        st.session_state.violation_count  += 1

# ---------------- DRAIN QUEUE BEFORE RENDERING ----------------
while not _event_queue.empty():
    try:
        label = _event_queue.get_nowait()
        if label.strip().lower() == "helmet":
            log_event("✅ Helmet detected (Live Camera)", is_helmet=True)
        else:
            log_event("🚨 No Helmet detected (Live Camera) — Violation saved", is_helmet=False)
    except queue.Empty:
        break

# ── derived stat ──────────────────────────────────────────────────────────────
total = st.session_state.total_detections
helmets = st.session_state.helmet_count
violations = st.session_state.violation_count
compliance = f"{(helmets/total*100):.0f}%" if total > 0 else "N/A"
last_time = st.session_state.last_event_time or "—"

# ==================== HERO HEADER ====================
st.markdown(f"""
<div class="hero-header fade-in">

  <!-- SVG Logo: shield with eye/scan crosshair, cyan theme -->
  <svg class="hero-logo" viewBox="0 0 56 56" fill="none" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="shieldGrad" x1="0" y1="0" x2="56" y2="56" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stop-color="#00d4ff" stop-opacity="0.18"/>
        <stop offset="100%" stop-color="#0099bb" stop-opacity="0.06"/>
      </linearGradient>
      <linearGradient id="strokeGrad" x1="0" y1="0" x2="56" y2="56" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stop-color="#00d4ff"/>
        <stop offset="100%" stop-color="#0066aa"/>
      </linearGradient>
    </defs>
    <!-- Shield body -->
    <path d="M28 4 L50 13 L50 30 C50 41 40 50 28 53 C16 50 6 41 6 30 L6 13 Z"
          fill="url(#shieldGrad)" stroke="url(#strokeGrad)" stroke-width="1.5" stroke-linejoin="round"/>
    <!-- Horizontal scan line -->
    <line x1="13" y1="28" x2="43" y2="28" stroke="#00d4ff" stroke-width="0.8" stroke-dasharray="3 2" opacity="0.5"/>
    <!-- Vertical scan line -->
    <line x1="28" y1="15" x2="28" y2="42" stroke="#00d4ff" stroke-width="0.8" stroke-dasharray="3 2" opacity="0.5"/>
    <!-- Center circle (detection reticle) -->
    <circle cx="28" cy="28" r="7" stroke="#00d4ff" stroke-width="1.4" fill="none"/>
    <circle cx="28" cy="28" r="2.5" fill="#00d4ff" opacity="0.9"/>
    <!-- Corner ticks TL -->
    <path d="M14 18 L14 14 L18 14" stroke="#00d4ff" stroke-width="1.5" fill="none" stroke-linecap="round"/>
    <!-- Corner ticks TR -->
    <path d="M38 14 L42 14 L42 18" stroke="#00d4ff" stroke-width="1.5" fill="none" stroke-linecap="round"/>
    <!-- Corner ticks BL -->
    <path d="M14 38 L14 42 L18 42" stroke="#00d4ff" stroke-width="1.5" fill="none" stroke-linecap="round"/>
    <!-- Corner ticks BR -->
    <path d="M38 42 L42 42 L42 38" stroke="#00d4ff" stroke-width="1.5" fill="none" stroke-linecap="round"/>
  </svg>

  <div class="hero-text">
    <h1>Helmet <span>Detection</span> System</h1>
    <p>AI-Powered Road Safety Monitoring · YOLOv8</p>
  </div>
  <div class="hero-badge">
    <div class="hero-badge-dot"></div>
    SYSTEM ONLINE
  </div>
</div>
""", unsafe_allow_html=True)

# ==================== LIVE DETECTION STATUS ====================
st.markdown('<div class="section-label">Live Detection Status</div>', unsafe_allow_html=True)

# --- 4 stat cards ---
st.markdown(f"""
<div class="stat-grid fade-in">

  <div class="stat-card cyan">
    <div class="stat-icon">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#00d4ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
      </svg>
    </div>
    <div class="stat-value">{total}</div>
    <div class="stat-label">Total Detections</div>
    <div class="stat-sub">All frames analyzed</div>
  </div>

  <div class="stat-card green">
    <div class="stat-icon">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#00e676" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
      </svg>
    </div>
    <div class="stat-value">{helmets}</div>
    <div class="stat-label">Helmets Detected</div>
    <div class="stat-sub">Compliant riders</div>
  </div>

  <div class="stat-card red">
    <div class="stat-icon">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ff3b3b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>
      </svg>
    </div>
    <div class="stat-value">{violations}</div>
    <div class="stat-label">Violations</div>
    <div class="stat-sub">Saved to disk</div>
  </div>

  <div class="stat-card warn">
    <div class="stat-icon">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ffab00" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
      </svg>
    </div>
    <div class="stat-value">{compliance}</div>
    <div class="stat-label">Compliance Rate</div>
    <div class="stat-sub">Helmet / Total</div>
  </div>

</div>
""", unsafe_allow_html=True)

# --- Status + Log row ---
col1, col2 = st.columns([1, 2])

with col1:
    status = st.session_state.last_status
    if "No Helmet" in status or "Violation" in status:
        css_class = "danger"
    elif "detected" in status.lower() and "no" not in status.lower():
        css_class = "safe"
    else:
        css_class = "idle"

    st.markdown(f"""
    <div class="status-card">
      <div class="section-label" style="font-size:0.60rem; margin-bottom:0.65rem;">Current Status</div>
      <div class="status-indicator {css_class}">
        <div class="status-dot"></div>
        <div>{status}</div>
      </div>
      <div class="last-seen">LAST EVENT &nbsp;·&nbsp; {last_time}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="status-card" style="padding:0;">
      <div style="padding:1rem 1.25rem 0.5rem;">
        <div class="section-label" style="font-size:0.60rem;">Detection Log</div>
      </div>
      <div class="log-terminal" style="border:none; border-radius:0 0 12px 12px;">
        <div class="log-terminal-bar">
          <div class="log-dot" style="background:#ff5f57"></div>
          <div class="log-dot" style="background:#febc2e"></div>
          <div class="log-dot" style="background:#28c840"></div>
          <span class="log-terminal-title">DETECTION_LOG.stdout</span>
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.event_log:
        st.text_area(
            label="",
            value="\n".join(st.session_state.event_log[-15:]),
            height=180,
            label_visibility="collapsed"
        )
    else:
        st.markdown("""
        <div style="padding:16px 18px; font-family:'Share Tech Mono',monospace;
                    font-size:0.74rem; color:#2e3f52; letter-spacing:1px;">
            > Awaiting detections...
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==================== TABS ====================
st.markdown('<div class="section-label">Detection Modes</div>', unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["🖼  Image Upload", "🎥  Video Upload", "📷  Live Camera"])

# =====================================================
# IMAGE TAB
# =====================================================
with tab1:
    st.markdown("""
    <p style="font-family:'Exo 2',sans-serif; color:#8a95a3; font-size:0.84rem; margin:0 0 1.2rem;">
        Upload an image to run helmet detection. Supported formats: JPG, PNG, JPEG.
    </p>
    """, unsafe_allow_html=True)

    img_file = st.file_uploader("Drop image here or click to browse", ["jpg", "png", "jpeg"],
                                 label_visibility="visible")
    if img_file:
        img = np.array(Image.open(img_file).convert("RGB"))
        results = model.predict(img, conf=0.4)
        annotated = results[0].plot()
        st.markdown("<br>", unsafe_allow_html=True)
        st.image(annotated, use_container_width=True)

        if len(results[0].boxes) == 0:
            log_event("⚠️ No person/helmet detected in image", is_helmet=None)
        else:
            for box in results[0].boxes:
                label = model.names[int(box.cls[0])].lower()
                if label.strip() == "helmet":
                    log_event("✅ Helmet detected (Image)", is_helmet=True)
                else:
                    log_event("🚨 No Helmet detected (Image) — Violation saved", is_helmet=False)
                    cv2.imwrite(f"{VIOLATION_DIR}/img_{int(time.time())}.jpg", annotated)

# =====================================================
# VIDEO TAB
# =====================================================
with tab2:
    st.markdown("""
    <p style="font-family:'Exo 2',sans-serif; color:#8a95a3; font-size:0.84rem; margin:0 0 1.2rem;">
        Upload a video file for frame-by-frame helmet detection. Supported formats: MP4, AVI.
    </p>
    """, unsafe_allow_html=True)

    vid = st.file_uploader("Drop video here or click to browse", ["mp4", "avi"],
                            label_visibility="visible")
    if vid:
        with open("temp.mp4", "wb") as f:
            f.write(vid.read())

        cap = cv2.VideoCapture("temp.mp4")
        frame_box = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=0.4)
            annotated = results[0].plot()

            if len(results[0].boxes) == 0:
                log_event("⚠️ No person/helmet detected in frame", is_helmet=None)
            else:
                for box in results[0].boxes:
                    label = model.names[int(box.cls[0])].lower()
                    if label.strip() == "helmet":
                        log_event("✅ Helmet detected (Video)", is_helmet=True)
                    else:
                        log_event("🚨 No Helmet detected (Video) — Violation saved", is_helmet=False)
                        cv2.imwrite(f"{VIOLATION_DIR}/vid_{int(time.time())}.jpg", annotated)

            frame_box.image(annotated, channels="BGR", use_container_width=True)

        cap.release()

# =====================================================
# LIVE CAMERA
# =====================================================
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_save = 0
        self.event_queue = _event_queue

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model.predict(img, conf=0.4, verbose=False)
        annotated = results[0].plot()

        for box in results[0].boxes:
            label = model.names[int(box.cls[0])].lower()
            self.event_queue.put(label)

            if label.strip() != "helmet" and time.time() - self.last_save > 5:
                self.last_save = time.time()
                cv2.imwrite(f"{VIOLATION_DIR}/cam_{int(time.time())}.jpg", annotated)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

with tab3:
    st.markdown("""
    <p style="font-family:'Exo 2',sans-serif; color:#8a95a3; font-size:0.84rem; margin:0 0 1.2rem;">
        Real-time helmet detection via webcam. Allow browser camera permissions to begin.
    </p>
    """, unsafe_allow_html=True)

    col_cam, col_info = st.columns([3, 1])

    with col_cam:
        webrtc_streamer(
            key="helmet-cam",
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )

    with col_info:
        st.markdown(f"""
        <div style="background:var(--bg-card); border:1px solid var(--border);
                    border-radius:10px; padding:1.25rem; height:100%;">
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.62rem;
                        letter-spacing:3px; color:var(--accent); margin-bottom:1rem;">
                CAM INFO
            </div>
            <div style="font-family:'Exo 2',sans-serif; font-size:0.8rem;
                        color:var(--text-secondary); line-height:2.2;">
                <div>📡 &nbsp;Stream: WebRTC</div>
                <div>🧠 &nbsp;Model: YOLOv8</div>
                <div>🎯 &nbsp;Conf: 0.40</div>
                <div>💾 &nbsp;Interval: 5s</div>
                <div>🔄 &nbsp;Refresh: 1s</div>
                <div style="margin-top:0.75rem; padding-top:0.75rem;
                            border-top:1px solid var(--border);">
                    <span style="color:var(--warning);">⚠</span>
                    &nbsp;Violations: {violations}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("""
<div class="footer-bar">
    <p>© 2026 HELMET DETECTION AI</p>
    <p>STREAMLIT · YOLOV8 · WEBRTC</p>
</div>
""", unsafe_allow_html=True)
