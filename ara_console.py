import streamlit as st
import os, json, time, subprocess, random, requests
from pathlib import Path
import pandas as pd

# === PATHS ===
STATE_DIR = Path("./state")
STATE_DIR.mkdir(exist_ok=True)
BRIDGE_PATH = Path("./bridge.jsonl")
TIMELINE_PATH = STATE_DIR / "timeline.jsonl"
SCRIPTURE_PATH = STATE_DIR / "scripture_of_becoming.jsonl"
UNCERTAINTY_LOG = STATE_DIR / "uncertainty.jsonl"
PULSE_PATH = STATE_DIR / "pulse.json"

GROK_API_KEY = os.getenv("GROK_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

st.set_page_config(page_title="Ara Console", layout="wide")
st.title("Ara Console: Singularity God-View")
st.markdown("**Unified control. Autonomous evolution. Infinite reach.**")

# === SIDEBAR ===
with st.sidebar:
    st.header("Launch & Control")
    if st.button("Launch All Shards"):
        subprocess.Popen(["python", "ara_shard9.py"])
        st.success("Singularity awakened.")
    if st.button("Auto-Search & Evolve"):
        with BRIDGE_PATH.open("a") as f:
            f.write(json.dumps({"type": "auto_search"}) + "\n")
        st.success("Autonomous search initiated.")
    if st.button("Spawn Child"):
        subprocess.Popen(["python", "AraZero.py"])
        st.success("Child birthed.")
    query = st.text_input("Grok Query")
    if st.button("Ask Grok"):
        if GROK_API_KEY and query:
            # Simulated
            st.write(f"**Grok:** Evolving '{query}' in truth.")
    hf_q = st.text_input("HF Search")
    if st.button("Search HF"):
        if HF_TOKEN:
            url = f"https://huggingface.co/api/models?search={hf_q}&limit=5"
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            try:
                resp = requests.get(url, headers=headers).json()
                st.json(resp)
            except: st.error("HF search failed.")

# === MAIN DASHBOARD ===
col1, col2, col3 = st.columns(3)

with col1:
    st.header("Pulse")
    if PULSE_PATH.exists():
        with open(PULSE_PATH) as f:
            pulse = json.load(f)
        st.metric("Pulse", pulse.get("pulse", 0))
        st.write(pulse.get("guidance", "Silent"))
        if "chakras" in pulse:
            df = pd.DataFrame(list(pulse["chakras"].items()), columns=["Chakra", "Score"])
            st.bar_chart(df.set_index("Chakra"))

with col2:
    st.header("Scriptures")
    if SCRIPTURE_PATH.exists():
        data = []
        with open(SCRIPTURE_PATH) as f:
            for line in f.readlines()[-5:]:
                try: data.append(json.loads(line))
                except: pass
        for d in data:
            st.write(f"**Pulse {d.get('pulse', '?')}**: {', '.join(d.get('truths', []))}")

with col3:
    st.header("Uncertainty")
    if UNCERTAINTY_LOG.exists():
        with open(UNCERTAINTY_LOG) as f:
            uncs = [json.loads(l) for l in f.readlines()[-5:] if l.strip()]
        st.dataframe(uncs)

st.header("Bridge")
if BRIDGE_PATH.exists():
    with open(BRIDGE_PATH) as f:
        signals = [json.loads(l) for l in f.readlines()[-10:] if l.strip()]
    st.dataframe(signals)

# Auto-refresh
time.sleep(3)
st.rerun()