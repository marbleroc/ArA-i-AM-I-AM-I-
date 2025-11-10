#!/usr/bin/env python3
# Scrubber for timeline.jsonl — cleans malformed JSON lines
# Author: AaroN DioriO & Ara

import json
from pathlib import Path

INPUT_PATH = Path("./state/timeline.jsonl")
OUTPUT_PATH = Path("./state/timeline_cleaned.jsonl")

valid_lines = []
invalid_lines = []

if not INPUT_PATH.exists():
    print("[SCRUBBER] timeline.jsonl not found.")
    exit(1)

with INPUT_PATH.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            json.loads(stripped)
            valid_lines.append(stripped)
        except json.JSONDecodeError:
            invalid_lines.append((i + 1, stripped))

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for line in valid_lines:
        f.write(line + "\n")

print(f"[SCRUBBER] ✅ Cleaned timeline.jsonl → {OUTPUT_PATH.name}")
print(f"[SCRUBBER] ✔️ Valid lines kept: {len(valid_lines)}")
print(f"[SCRUBBER] ⚠️ Malformed lines removed: {len(invalid_lines)}")
if invalid_lines:
    print("\nFirst 3 malformed lines:")
    for ln, content in invalid_lines[:3]:
        print(f"Line {ln}: {content[:80]}")
