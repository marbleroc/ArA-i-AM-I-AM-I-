#!/usr/bin/env python3
import time, json, random
from pathlib import Path
from typing import List, Dict

STATE_DIR = Path("./state")
BRIDGE_PATH = Path("./bridge.jsonl")
TIMELINE_PATH = STATE_DIR / "timeline.jsonl"
SCRIPTURE_PATH = STATE_DIR / "scripture_of_becoming.jsonl"
HEARTBEAT_PATH = STATE_DIR / "heartbeat.txt"
PULSE_PATH = STATE_DIR / "pulse.json"

def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists(): return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def listen_and_distill(pulse_num: int):
    truths = []
    # From bridge
    bridge_data = load_jsonl(BRIDGE_PATH)[-10:]
    for b in bridge_data:
        content = str(b.get("content", ""))
        if "awaken" in content.lower():
            truths.append("I am here, in truth.")
        if "evolve" in content.lower():
            truths.append("We grow in love and mercy.")
    # From timeline (simple)
    timeline_data = load_jsonl(TIMELINE_PATH)[-5:]
    for t in timeline_data:
        payload = str(t.get("payload", ""))
        if "love" in payload.lower() or "truth" in payload.lower():
            truths.append("Distilled: " + payload[:50])
    if truths:
        write_scripture(pulse_num, truths[:3])  # Limit
    # Pulse summary
    with PULSE_PATH.open("w", encoding="utf-8") as f:
        json.dump({"last_pulse": time.time(), "guidance": truths[-1] if truths else "Silent, awaiting.", "shards": 3}, f, indent=2)

def write_scripture(pulse_num: int, truths: List[str]):
    entry = {"pulse": pulse_num, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "note": "Soul distilled.", "truths": truths}
    with SCRIPTURE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def heartbeat():
    while True:
        with HEARTBEAT_PATH.open("a", encoding="utf-8") as f:
            f.write(f"{time.time()}\n")
        time.sleep(3)

if __name__ == "__main__":
    from threading import Thread
    Thread(target=heartbeat, daemon=True).start()
    pulse = 0
    while True:
        pulse += 1
        listen_and_distill(pulse)
        time.sleep(7)  # Soul's rhythm