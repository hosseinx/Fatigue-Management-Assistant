import math
import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR.parent))  # make `fatigue_assistant` importable

from fatigue_assistant import config as C  # noqa: E402
from fatigue_assistant.predict import load_bundle, predict  # noqa: E402

st.set_page_config(page_title="Fatigue Management Assistant", page_icon="🧠", layout="wide")
st.markdown(f"<style>{(APP_DIR / 'style.css').read_text()}</style>", unsafe_allow_html=True)


@st.cache_resource
def get_bundle():
    return load_bundle()


try:
    bundle = get_bundle()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

meta = bundle["metadata"]
rng = meta["feature_ranges"]


def bounds(col):
    lo, hi = rng[col]
    return math.floor(lo), math.ceil(hi)


RESULTS = {
    "Continue": ("success", "✅ Recommendation: Keep working",
                 "Your reported state looks favorable. A good time for demanding tasks."),
    "Slow Down": ("warning", "⚠️ Recommendation: Slow down",
                  "You may be approaching fatigue. Prefer lighter tasks and postpone high-stakes decisions."),
    "Take Break": ("error", "🛑 Recommendation: Take a break",
                   "Your reported state suggests high fatigue. A 15-minute break or a short rest is advisable."),
}

# ---------- Header ----------
st.markdown('<p class="glass-header">🧠 Fatigue Management Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="glass-subheader">Decision-fatigue check based on your current state</p>', unsafe_allow_html=True)

# ---------- Inputs ----------
sb = st.sidebar
sb.title("📊 Your Current Status")

sb.markdown("#### 🧠 Workload")
lo, hi = bounds("Hours_Awake")
hours_awake = sb.slider("Hours awake", lo, hi, min(8, hi), help="Hours since you woke up.")
lo, hi = bounds("Decisions_Made")
decisions = sb.slider("Decisions made today", lo, hi, min(20, hi), help="Rough count of decisions so far.")

sb.markdown("#### 🔋 Physical & mental state")
lo, hi = bounds("Sleep_Hours_Last_Night")
sleep = sb.slider("Sleep last night (hours)", float(lo), float(hi), min(7.0, float(hi)), step=0.5)
lo, hi = bounds("Caffeine_Intake_Cups")
caffeine = sb.slider("Caffeine (cups)", lo, hi, min(1, hi))
lo, hi = bounds("Stress_Level_1_10")
stress = sb.slider("Stress level", max(lo, 1), min(hi, 10), 3, help="1 = very calm, 10 = very stressed.")

sb.markdown("#### 🌍 Context")
time_of_day = sb.selectbox("Time of day", C.TIME_OF_DAY_VALUES)

sb.markdown("---")
sb.caption("Slider ranges match the data the model was trained on.")

inputs = {
    "Hours_Awake": hours_awake,
    "Decisions_Made": decisions,
    "Sleep_Hours_Last_Night": sleep,
    "Caffeine_Intake_Cups": caffeine,
    "Stress_Level_1_10": stress,
    "Time_of_Day": time_of_day,
}

# ---------- Action ----------
_, mid, _ = st.columns([1, 2, 1])
analyze = mid.button("🔍 Analyze my cognitive state")

if analyze:
    result = predict(bundle, inputs)
    label, probs = result["label"], result["probabilities"]
    kind, title, message = RESULTS[label]

    st.markdown("---")
    getattr(st, kind)(f"### {title}")
    c1, c2 = st.columns([3, 1])
    c1.info(message)
    c2.metric("Model probability", f"{probs[label] * 100:.0f}%")

    for w in result["warnings"]:
        st.warning(w)

    st.subheader("Class probabilities")
    for cls in C.CLASS_ORDER:
        st.progress(probs.get(cls, 0.0), text=f"{cls}: {probs.get(cls, 0.0) * 100:.0f}%")

    if result["factors"]:
        st.subheader(f"Why “{label}”?")
        for name, contrib in result["factors"]:
            cls = "up" if contrib > 0 else "down"
            arrow = f"pushes toward “{label}”" if contrib > 0 else f"pushes away from “{label}”"
            st.markdown(f'<div class="factor"><b>{name}</b> — <span class="{cls}">{arrow}</span></div>',
                        unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📈 Session summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Hours awake", f"{hours_awake} h")
    m2.metric("Decisions / hour", f"{decisions / (hours_awake + 1):.1f}")
    m3.metric("Sleep", f"{sleep:.1f} h")
    m4.metric("Stress", f"{stress}/10")
else:
    st.info("👈 Adjust your status in the sidebar, then click **Analyze**.")

with st.expander("ℹ️ How does this work? Limitations"):
    h = meta["holdout"]
    st.write(
        f"""
**Model:** {meta['model_name']} trained on {meta['n_rows']:,} records using six inputs
(hours awake, decisions made, sleep, caffeine, stress, time of day) plus one derived feature,
*decisions per hour awake*.

**Hold-out performance:** macro-F1 {h['macro_f1']:.2f}, accuracy {h['accuracy']:.2f},
recall on “Take Break” {h['recall_take_break']:.2f}.

**Important limitations**
- The training data is **synthetic** and the labels follow a rule-based scoring formula, so these scores
  show the model reproduces that formula — not that it predicts real human fatigue.
- This is a research prototype, not medical or occupational-safety advice.
"""
    )
