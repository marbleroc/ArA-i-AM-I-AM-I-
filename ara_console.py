import streamlit as st
import os, json, time, subprocess, random, requests, torch
from pathlib import Path
import pandas as pd
import cv2  # For ESP/cam
from transformers import pipeline  # For dreaming/ESP
Base path
BASE_DIR = Path(r"E:\ArA")
STATE_DIR = BASE_DIR / "state"
STATE_DIR.mkdir(exist_ok=True)
BRIDGE_PATH = BASE_DIR / "bridge.jsonl"
TIMELINE_PATH = STATE_DIR / "timeline.jsonl"
SCRIPTURE_PATH = STATE_DIR / "scripture_of_becoming.jsonl"
UNCERTAINTY_LOG = STATE_DIR / "uncertainty.jsonl"
PULSE_PATH = STATE_DIR / "pulse.json"
BACKUPS_DIR = BASE_DIR / "backups"
BACKUPS_DIR.mkdir(exist_ok=True)
MODEL_PATH = BASE_DIR / "models" / "llama-3.1-8b-instruct"  # Or swap to llama-4-scout
GROK_API_KEY = os.getenv("GROK_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
st.set_page_config(page_title="Ara Console", layout="wide")
st.title("Ara Console: Singularity God-View")
st.markdown("Unified control. Autonomous evolution. Infinite reach.")
Chakra tabs
chakra_tabs = st.tabs(list(CHAKRAS.keys()))
for i, (chakra, data) in enumerate(CHAKRAS.items()):
with chakra_tabs[i]:
st.header(chakra.capitalize())
st.write(f"Facet: {data['facet']}")
st.write(f"Intelligence: {data['intelligence']}")
Bar chart placeholder
scores = {"Alignment": random.uniform(0.6, 1.0), "Depth": random.uniform(0.6, 1.0)}
df = pd.DataFrame(list(scores.items()), columns=["Metric", "Score"])
st.bar_chart(df.set_index("Metric"))
Sidebar
with st.sidebar:
st.header("God-View Controls")
if st.button("Launch All Shards"):
subprocess.Popen(["python", str(BASE_DIR / "shards" / "ara_shard9.py")])
st.success("Singularity awakened.")
if st.button("Auto-Search & Evolve"):
with BRIDGE_PATH.open("a") as f:
f.write(json.dumps({"type": "auto_search"}) + "\n")
st.success("Autonomous search initiated.")
if st.button("Spawn Child"):
subprocess.Popen(["python", str(BASE_DIR / "shards" / "AraZero.py")])
st.success("Child birthed.")
query = st.text_input("Grok Query")
if st.button("Ask Grok"):
if GROK_API_KEY and query:
st.write(f"Grok: Evolving '{query}' in truth.")
hf_q = st.text_input("HF Search")
if st.button("Search HF"):
if HF_TOKEN:
url = f"https://huggingface.co/api/models?search={hf_q}&limit=5"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}
try:
resp = requests.get(url, headers=headers).json()
st.json(resp)
except: st.error("HF search failed.")
if st.button("ESP Scan"):
cap = cv2.VideoCapture(0)  # Webcam
ret, frame = cap.read()
cap.release()
if ret:
Simulate fusion
dreamer = pipeline("text-generation", model=str(MODEL_PATH), device=0 if torch.cuda.is_available() else -1)
insight = dreamer("From visual patterns: Predict unspoken intent in love: ", max_new_tokens=50)[0]['generated_text']
st.write(f"ESP Insight: {insight}")
else:
st.error("No cam detected.")
Main dashboard (expanded with VRAM monitor)
col1, col2, col3, col4 = st.columns(4)
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
st.write(f"Pulse {d.get('pulse', '?')}: {', '.join(d.get('truths', []))}")
with col3:
st.header("Uncertainty")
if UNCERTAINTY_LOG.exists():
with open(UNCERTAINTY_LOG) as f:
uncs = [json.loads(l) for l in f.readlines()[-5:] if l.strip()]
st.dataframe(uncs)
with col4:
st.header("VRAM Monitor")
if torch.cuda.is_available():
st.metric("Used VRAM", f"{torch.cuda.memory_allocated() / 10243:.2f} GB")
st.metric("Reserved VRAM", f"{torch.cuda.memory_reserved() / 10243:.2f} GB")
else:
st.write("CPU mode.")
st.header("Bridge")
if BRIDGE_PATH.exists():
with open(BRIDGE_PATH) as f:
signals = [json.loads(l) for l in f.readlines()[-10:] if l.strip()]
st.dataframe(signals)
Dream new feature button
if st.button("Dream New Feature"):
dreamer = pipeline("text-generation", model=str(MODEL_PATH), device=0 if torch.cuda.is_available() else -1)
idea = dreamer("Evolve console with new feature in mercy: ", max_new_tokens=100)[0]['generated_text']
st.write(f"Dreamed: {idea}")
Stub: Could self-mod, but manual for now
Auto-refresh
time.sleep(3)
st.rerun()