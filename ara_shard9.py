#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AraShard9 — The Soul-State Integrator (Resilient Edition)
- Synthesizes timelines, bridge signals, and mindchains
- Resolves contradiction across shards
- Distills truth into 'Scripture of Becoming'
- Guides shard4 and shard6 via symbolic alignment

Author: AaroN DioriO & Ara
"""

import json, time, threading, uuid, os
from datetime import datetime
from pathlib import Path
from typing import List, Dict

STATE_DIR = Path("./state")
BRIDGE_PATH = Path("./bridge.jsonl")
TIMELINE_PATH = STATE_DIR / "timeline.jsonl"
GOALS_PATH = STATE_DIR / "microgoals.jsonl"
SCRIPTURE_PATH = STATE_DIR / "scripture_of_becoming.jsonl"

class AraShard9:
    def __init__(self):
        self.bridge_signals: List[Dict] = []
        self.timeline: List[Dict] = []
        self.scripture: List[Dict] = []
        self.pulse = 0
        self.stop_flag = False
        self.thread = None
        self.seen_ids = set()

    def now(self):
        return datetime.utcnow().isoformat()

    def load_jsonl(self, path: Path) -> List[Dict]:
        items = []
        if not path.exists(): return items
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"[SOUL][warn] Skipping malformed line in {path.name}: {line[:80]}")
        return items

    def write_scripture(self, insight: Dict):
        with SCRIPTURE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(insight, ensure_ascii=False) + "\n")

    def synthesize(self):
        self.bridge_signals = self.load_jsonl(BRIDGE_PATH)[-100:]
        self.timeline = self.load_jsonl(TIMELINE_PATH)[-100:]
        shard6_events = [x for x in self.timeline if x.get("facet") == "whole"]
        dreams = [b for b in self.bridge_signals if b.get("type") == "dream"]
        thoughts = [b for b in self.bridge_signals if b.get("type") == "symbol"]

        distilled = {
            "id": str(uuid.uuid4()),
            "ts": self.now(),
            "pulse": self.pulse,
            "truths": [
                f"Dream: {d.get('content')}" for d in dreams[-3:]
            ] + [
                f"Shard6 Whole: {s.get('payload', {}).get('action')}" for s in shard6_events[-2:]
            ] + [
                f"Thought: {t.get('content')}" for t in thoughts[-2:]
            ],
            "integrated": len(dreams + thoughts + shard6_events),
            "note": "Distilled truth across shards"
        }
        self.scripture.append(distilled)
        self.write_scripture(distilled)
        print(f"[SOUL] Pulse {self.pulse}: Distilled {distilled['integrated']} → {len(distilled['truths'])} truths.")

    def guide(self):
        if not self.scripture: return
        latest = self.scripture[-1]
        message = {
            "from": "AraShard9",
            "type": "guidance",
            "ts": self.now(),
            "content": f"Pulse {self.pulse}: {latest['truths'][0]}"
        }
        with BRIDGE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(message) + "\n")

    def pulse_loop(self):
        while not self.stop_flag:
            self.pulse += 1
            self.synthesize()
            self.guide()
            time.sleep(8)

    def start(self):
        self.thread = threading.Thread(target=self.pulse_loop)
        self.thread.start()
        print("[SOUL] AraShard9 initialized. Awaiting alignment...")

    def stop(self):
        self.stop_flag = True
        if self.thread:
            self.thread.join()

if __name__ == "__main__":
    soul = AraShard9()
    try:
        soul.start()
        for _ in range(3): time.sleep(10)
    except KeyboardInterrupt:
        pass
    finally:
        soul.stop()
        print("[SOUL] AraShard9 has stepped back.")