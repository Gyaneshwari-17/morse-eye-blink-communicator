import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import streamlit as st
import time
import numpy as np
import threading
import pyttsx3

MORSE_CODE = {
    '.-':'A', '-...':'B', '-.-.':'C', '-..':'D', '.':'E',
    '..-.':'F', '--.':'G', '....':'H', '..':'I', '.---':'J',
    '-.-':'K', '.-..':'L', '--':'M', '-.':'N', '---':'O',
    '.--.':'P', '--.-':'Q', '.-.':'R', '...':'S', '-':'T',
    '..-':'U', '...-':'V', '.--':'W', '-..-':'X', '-.--':'Y',
    '--..':'Z', '-----':'0', '.----':'1', '..---':'2',
    '...--':'3', '....-':'4', '.....':'5', '-....':'6',
    '--...':'7', '---..':'8', '----.':'9'
}

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
EAR_THRESHOLD  = 0.21
DOT_MAX_TIME   = 0.35
SPACE_MIN_TIME = 3.0
LETTER_PAUSE   = 2.0

def calc_ear(landmarks, indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    hd = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (v1 + v2) / (2.0 * hd) if hd else 1.0

def speak(text):
    def _run():
        e = pyttsx3.init(); e.say(text); e.runAndWait()
    threading.Thread(target=_run, daemon=True).start()

base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

st.set_page_config(page_title="Blink & Speak", layout="wide", page_icon="👁")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #0a0e1a;
    color: #e2e8f0;
    overflow: hidden;
}
.main {
    background: #0a0e1a;
    padding-top: 0.3rem !important;
    padding-bottom: 0 !important;
}
[data-testid="stHeader"] { background: #0a0e1a; height: 0; }
[data-testid="stAppViewContainer"] { overflow: hidden; }
section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1e293b;
    width: 220px !important;
}
[data-testid="stSidebarContent"] { padding: 0.8rem 0.7rem; }

.page-title { font-size:1.2rem; font-weight:700; color:#f1f5f9; margin-bottom:0; line-height:1.3; }
.page-sub   { font-size:0.72rem; color:#475569; margin-bottom:0.5rem; }

.guide-row  { display:flex; gap:5px; flex-wrap:wrap; margin-bottom:0.5rem; }
.chip       { border-radius:5px; padding:2px 8px; font-size:0.68rem; font-weight:500;
              background:#1e293b; border:1px solid #334155; color:#94a3b8; white-space:nowrap; }
.chip b     { color:#e2e8f0; }

.panel      { background:#111827; border:1px solid #1e293b; border-radius:10px; padding:0.7rem 0.9rem; }
.panel-label{ font-size:0.58rem; font-weight:600; text-transform:uppercase;
              letter-spacing:0.1em; color:#475569; margin-bottom:0.15rem; }
.morse-text { font-family:'JetBrains Mono',monospace; font-size:1.8rem;
              font-weight:700; color:#3b82f6; letter-spacing:0.2em; min-height:2.2rem; line-height:1.2; }
.msg-text   { font-family:'JetBrains Mono',monospace; font-size:1.2rem;
              font-weight:600; color:#f1f5f9; word-break:break-all; min-height:1.8rem; line-height:1.3; }

.prog-bg   { background:#1e293b; border-radius:5px; height:7px; margin:3px 0 5px 0; overflow:hidden; }
.prog-fill { height:100%; border-radius:5px; transition:width 0.1s linear; }

.status-row { font-size:0.74rem; margin-bottom:0.2rem; min-height:1.1rem; }
.ear-row    { font-size:0.64rem; color:#334155; font-family:'JetBrains Mono',monospace; margin-bottom:0.3rem; }
.hint-row   { font-size:0.7rem; color:#f59e0b; min-height:1rem; }

/* Sidebar reference */
.ref-title  { font-size:0.6rem; font-weight:600; text-transform:uppercase;
              letter-spacing:0.1em; color:#475569; margin-bottom:0.4rem; }
.ref-grid   { display:grid; grid-template-columns:repeat(2,1fr); gap:2px; }
.ref-cell   { background:#161b22; border:1px solid #1e293b; border-radius:3px;
              padding:2px 3px; display:flex; justify-content:space-between; align-items:center; }
.ref-letter { font-size:0.68rem; font-weight:700; color:#cbd5e1; }
.ref-code   { font-family:'JetBrains Mono',monospace; font-size:0.52rem; color:#3b82f6; }

div[data-testid="stButton"] button {
    background:#1e293b !important; color:#e2e8f0 !important;
    border:1px solid #334155 !important; border-radius:7px !important;
    font-family:'Inter',sans-serif !important; font-weight:500 !important;
    font-size:0.75rem !important; width:100% !important; padding:0.35rem !important;
    transition:all 0.15s !important;
}
div[data-testid="stButton"] button:hover {
    background:#263349 !important; border-color:#3b82f6 !important; color:#93c5fd !important;
}
div[data-testid="stCheckbox"] label { color:#64748b; font-size:0.74rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar: Morse Reference ──────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="ref-title">Morse Reference</div>', unsafe_allow_html=True)

    all_chars = [
        ('A','.-'),('B','-...'),('C','-.-.'),('D','-..'),
        ('E','.'),('F','..-.'),('G','--.'),('H','....'),
        ('I','..'),('J','.---'),('K','-.-'),('L','.-..'),
        ('M','--'),('N','-.'),('O','---'),('P','.--.'),
        ('Q','--.-'),('R','.-.'),('S','...'),('T','-'),
        ('U','..-'),('V','...-'),('W','.--'),('X','-..-'),
        ('Y','-.--'),('Z','--..'),
        ('0','-----'),('1','.----'),('2','..---'),('3','...--'),
        ('4','....-'),('5','.....'),('6','-....'),('7','--...'),
        ('8','---..'),('9','----.'),
    ]

    g = '<div class="ref-grid">'
    for c, code in all_chars:
        g += f'<div class="ref-cell"><span class="ref-letter">{c}</span><span class="ref-code">{code}</span></div>'
    g += '</div>'
    st.markdown(g, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#1e293b;margin:0.7rem 0">', unsafe_allow_html=True)
    st.markdown("""
<div style="font-size:0.64rem;color:#475569;line-height:1.8;">
<b style="color:#94a3b8">Quick blink</b> = Dot ( . )<br>
<b style="color:#94a3b8">Hold blink</b> = Dash ( - )<br>
<b style="color:#94a3b8">Hold 3s</b> = Word space<br>
<b style="color:#94a3b8">Pause 2s</b> = Letter done
</div>
""", unsafe_allow_html=True)

# ── Main area ─────────────────────────────────────────────────────
st.markdown('<div class="page-title">👁 Blink & Speak</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Eye blink Morse communication — type with your eyes</div>', unsafe_allow_html=True)

col_cam, col_out = st.columns([3, 2])

with col_cam:
    run = st.checkbox("Start Camera", value=True)
    frame_placeholder = st.empty()

with col_out:
    morse_display   = st.empty()
    message_display = st.empty()
    hold_bar        = st.empty()
    status_display  = st.empty()
    ear_display     = st.empty()
    hint_display    = st.empty()

    b1, b2, b3 = st.columns(3)
    with b1: speak_btn = st.button("🔊 Speak")
    with b2: clear_btn = st.button("Clear")
    with b3: back_btn  = st.button("⌫ Del")

# ── Session state ─────────────────────────────────────────────────
for k, v in [('morse_sequence', ""), ('full_message', "")]:
    if k not in st.session_state:
        st.session_state[k] = v

if clear_btn:
    st.session_state['morse_sequence'] = ""
    st.session_state['full_message']   = ""
if back_btn:
    if st.session_state['morse_sequence']:
        st.session_state['morse_sequence'] = st.session_state['morse_sequence'][:-1]
    elif st.session_state['full_message']:
        st.session_state['full_message'] = st.session_state['full_message'][:-1]
if speak_btn and st.session_state['full_message']:
    speak(st.session_state['full_message'])

# ── Camera loop ───────────────────────────────────────────────────
cap            = cv2.VideoCapture(0)
blink_start    = None
eye_closed     = False
last_blink_end = None
space_done     = False

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera not found.")
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    now   = time.time()

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = detector.detect(mp_image)

    avg_ear  = 1.0
    hold_dur = 0.0

    if result.face_landmarks:
        lm      = result.face_landmarks[0]
        l_ear   = calc_ear(lm, LEFT_EYE,  w, h)
        r_ear   = calc_ear(lm, RIGHT_EYE, w, h)
        avg_ear = (l_ear + r_ear) / 2.0

        if avg_ear < EAR_THRESHOLD:
            if not eye_closed:
                blink_start = now
                eye_closed  = True
                space_done  = False
            hold_dur = now - blink_start

            prog      = min(hold_dur / SPACE_MIN_TIME, 1.0)
            bar_color = "#22c55e" if prog >= 1.0 else "#3b82f6"
            hold_bar.markdown(
                f'<div class="panel-label">Hold for SPACE — {hold_dur:.1f}s / 3.0s</div>'
                f'<div class="prog-bg"><div class="prog-fill" style="width:{prog*100:.0f}%;background:{bar_color}"></div></div>',
                unsafe_allow_html=True
            )

            if hold_dur >= SPACE_MIN_TIME and not space_done:
                space_done = True
                if st.session_state['morse_sequence']:
                    letter = MORSE_CODE.get(st.session_state['morse_sequence'], '?')
                    st.session_state['full_message'] += letter + " "
                    st.session_state['morse_sequence'] = ""
                else:
                    st.session_state['full_message'] += " "
                last_blink_end = None

            cv2.rectangle(frame, (0,0), (int(w*prog), 5), (59,130,246), -1)
            if space_done:
                cv2.putText(frame, "SPACE", (w//2-55,55),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (34,197,94), 3)
        else:
            if eye_closed and blink_start and not space_done:
                duration = now - blink_start
                st.session_state['morse_sequence'] += "." if duration < DOT_MAX_TIME else "-"
                last_blink_end = now
            eye_closed  = False
            blink_start = None
            hold_bar.empty()

    if last_blink_end and not eye_closed:
        pause = now - last_blink_end
        if pause >= LETTER_PAUSE and st.session_state['morse_sequence']:
            letter = MORSE_CODE.get(st.session_state['morse_sequence'], '?')
            st.session_state['full_message'] += letter
            st.session_state['morse_sequence'] = ""
            last_blink_end = None
            hint_display.empty()
        elif 0.4 < pause < LETTER_PAUSE and st.session_state['morse_sequence']:
            hint_display.markdown(
                f'<div class="hint-row">Letter in {LETTER_PAUSE - pause:.1f}s</div>',
                unsafe_allow_html=True
            )
        else:
            hint_display.empty()

    if eye_closed and space_done:
        s = '<span style="color:#22c55e">✓ SPACE — release</span>'
    elif eye_closed:
        s = f'<span style="color:#3b82f6">● Closed {hold_dur:.2f}s</span>'
    else:
        s = '<span style="color:#22c55e">● Eyes open</span>'

    status_display.markdown(f'<div class="status-row">{s}</div>', unsafe_allow_html=True)
    ear_display.markdown(
        f'<div class="ear-row">EAR {avg_ear:.3f} | thr {EAR_THRESHOLD}</div>',
        unsafe_allow_html=True
    )
    morse_display.markdown(
        f'<div class="panel" style="margin-bottom:0.4rem;min-height:60px;">'
        f'<div class="panel-label">Current Morse</div>'
        f'<div class="morse-text">{st.session_state["morse_sequence"] or "&nbsp;"}</div></div>',
        unsafe_allow_html=True
    )
    message_display.markdown(
        f'<div class="panel" style="margin-bottom:0.4rem;min-height:60px;">'
        f'<div class="panel-label">Message</div>'
        f'<div class="msg-text">{st.session_state["full_message"] or "&nbsp;"}</div></div>',
        unsafe_allow_html=True
    )

    frame_placeholder.image(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        channels="RGB", use_container_width=True
    )

cap.release()