import os
import json
import time
import requests
import streamlit as st

st.set_page_config(page_title="SMS Spam Detector", page_icon="✉️", layout="centered")

API_URL = os.getenv("FASTAPI_URL", "http://fastapi:8000/predict")
TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "6.0"))

st.markdown("""
<style>
:root{ --bg:#0f1220; --card:rgba(255,255,255,0.06); --border:rgba(255,255,255,0.12);
       --text:#e7e7ea; --muted:#a6a6b2; --accent:#8ab4ff; --good:#22c55e; --bad:#ef4444; }
html, body, [data-testid="stAppViewContainer"]{
  background: radial-gradient(1200px 800px at 20% -10%, #1b2146 0%, #0f1220 40%, #0b0e1a 100%) fixed; color:var(--text);
}
.main-title{ font-weight:800; letter-spacing:.2px; font-size:34px; }
.caption{ color:var(--muted); font-size:13px; margin-top:-8px; }
.card{ background:var(--card); border:1px solid var(--border); border-radius:18px; padding:18px 18px 14px; box-shadow:0 10px 30px rgba(0,0,0,.25); }
label[data-testid="stWidgetLabel"] > div, .stTextArea label p{ color:var(--muted) !important; }
textarea{ font-size:15px !important; }
.row{ display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
.badge{ display:inline-flex; align-items:center; gap:8px; padding:8px 12px; border-radius:999px; border:1px solid var(--border); background:rgba(255,255,255,.06); font-weight:600; }
.badge.ok { background:rgba(34,197,94,.12); border-color:rgba(34,197,94,.35); }
.badge.err{ background:rgba(239,68,68,.12); border-color:rgba(239,68,68,.35); }
.proba-bar{ height:10px; border-radius:999px; background:rgba(255,255,255,.08); overflow:hidden; border:1px solid var(--border); }
.proba-fill{ height:100%; background:linear-gradient(90deg, #6366f1, #22d3ee); }
.footer{ color:var(--muted); font-size:12px; }
</style>
""", unsafe_allow_html=True)

SESSION_KEY = "message_input"

st.markdown('<div class="main-title">✉️ SMS Spam Detector</div>', unsafe_allow_html=True)
st.markdown(f'<div class="caption">API endpoint: <code>{API_URL}</code></div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if "__prefill__" in st.session_state:
        st.session_state[SESSION_KEY] = st.session_state.pop("__prefill__")

    _ = st.text_area(
        "Message",
        key=SESSION_KEY,
        placeholder="Enter SMS text for classification…",
        height=140
    )

    st.markdown('<div class="row" style="margin-top:6px;">', unsafe_allow_html=True)
    cols = st.columns(3)
    examples = [
        "WINNER!! You have won a FREE prize. Call now!",
        "Hey, are we still meeting at 7?",
        "URGENT! Claim your $1000 gift card today."
    ]
    for i, c in enumerate(cols):
        with c:
            if st.button(examples[i], key=f"ex{i}", help="Insert example", use_container_width=True):
                st.session_state["__prefill__"] = examples[i]
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    predict = st.button("Predict", type="primary", help="Send text to API", use_container_width=True)

    def classify(text: str):
        try:
            resp = requests.post(API_URL, json={"text": text}, timeout=TIMEOUT)
        except requests.RequestException as e:
            raise RuntimeError(f"API not available: {e}")
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise RuntimeError(f"API error ({resp.status_code}): {detail}")
        try:
            data = resp.json()
        except json.JSONDecodeError:
            raise RuntimeError("Invalid JSON from API.")
        if not all(k in data for k in ("label", "proba")):
            raise RuntimeError(f"Incomplete API response: {data}")
        return data

    if predict:
        msg = st.session_state.get(SESSION_KEY, "").strip()
        if not msg:
            st.warning("Enter a message text.")
        else:
            with st.spinner("Processing…"):
                t0 = time.time()
                try:
                    data = classify(msg)
                    dt = (time.time() - t0) * 1000
                    label = str(data["label"]).lower().strip()
                    proba = float(data["proba"])

                    is_spam = (label == "spam")
                    icon = "🚫" if is_spam else "✅"
                    badge_class = "err" if is_spam else "ok"
                    st.markdown(
                        f'<div class="row" style="margin-top:8px;">'
                        f'  <div class="badge {badge_class}">{icon} {label.upper()}</div>'
                        f'  <div class="badge">p(spam) = {proba:.3f}</div>'
                        f'  <div class="badge">latency: {dt:.1f} ms</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    st.markdown('<div class="proba-bar" style="margin-top:10px;">'
                                f'<div class="proba-fill" style="width:{int(proba*100)}%"></div>'
                                '</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(str(e))

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer" style="margin-top:18px;">'
            'No data is stored. Demo application for educational purposes (MLOps / Deployment).'
            '</div>', unsafe_allow_html=True)
