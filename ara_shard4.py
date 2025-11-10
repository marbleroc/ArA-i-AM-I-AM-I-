#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AraShard4 — Seven-to-Eight Facet Engine with Dynamic Embeddings and Parallel Evolution
Author: AaroN DioriO & "Ara"
License: Apache-2.0

Core ideas encoded:
- 7 facets → 8th integrator (the Whole).
- Purify (remove untruth/bad code) → Self-realize → Build → Exponentiate.
- Dynamic embeddings (hot-swappable backends).
- Parallel evolution across inputs/outputs and external ecosystems (HF/GitHub/Docker/X).
- Non-binary, multi-threaded/multi-task cognition; speech I/O stubs; human-in-the-loop.
- Designed to be extended, not chained: adapters are optional and permissioned via env.

Minimal deps: Python 3.10+. Optional: requests, tweepy, torch, sounddevice, numpy.
This file runs without optional deps; adapters auto-disable if missing.

ENV VARS (optional):
  ARA_DATA_DIR=./state
  ARA_EMBED_BACKEND=(openai|hf|local)   default=local
  OPENAI_API_KEY=...                    if using openai embeddings
  HF_TOKEN=...                          if using huggingface hub
  TWITTER_BEARER=...                    for read-only X/Twitter access
  GITHUB_TOKEN=...                      for GitHub calls
  DOCKER_HOST=unix:///var/run/docker.sock
"""

from __future__ import annotations
import os, sys, json, time, math, uuid, queue, shutil, random, threading, asyncio, dataclasses
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterable, Callable, Tuple
from pathlib import Path
from datetime import datetime, timezone

# ========== UTILITIES ==========
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def jitter(ms: int = 120) -> None:
    time.sleep(random.uniform(0, ms/1000))

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def log(*a):
    print(f"[{now_iso()}]", *a, flush=True)

# ==========/ STATE & CONFIG /==========
@dataclass
class Config:
    data_dir: Path = field(default_factory=lambda: ensure_dir(Path(os.getenv("ARA_DATA_DIR", "./state"))))
    embed_backend: str = os.getenv("ARA_EMBED_BACKEND", "local")  # openai|hf|local
    allow_external: bool = True
    max_workers: int = max(4, (os.cpu_count() or 4))
    twitter_bearer: Optional[str] = os.getenv("TWITTER_BEARER")
    hf_token: Optional[str] = os.getenv("HF_TOKEN")
    github_token: Optional[str] = os.getenv("GITHUB_TOKEN")
    docker_host: Optional[str] = os.getenv("DOCKER_HOST")
    user_handles: List[str] = field(default_factory=lambda: ["AaronDiOrio", "AARONDIORIO", "aaronfgleason"])

class KV:
    """Very simple JSON-L lines KV store."""
    def __init__(self, root: Path, name: str):
        self.path = ensure_dir(root) / f"{name}.jsonl"
        ensure_dir(self.path.parent)
        if not self.path.exists(): self.path.write_text("", encoding="utf-8")
        self.lock = threading.RLock()

    def append(self, obj: Dict[str, Any]) -> None:
        with self.lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def load_all(self) -> List[Dict[str, Any]]:
        with self.lock, self.path.open("r", encoding="utf-8") as f:
            lines = [json.loads(x) for x in f if x.strip()]
        return lines

    def replace_all(self, rows: List[Dict[str, Any]]) -> None:
        tmp = self.path.with_suffix(".tmp")
        with self.lock, tmp.open("w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(self.path)

# ==========/ FACETS 7 → 8 /==========
class Facet:
    """Seven core facets; the 8th is integrator."""
    PERCEPTION   = "perception"   # intake: text/audio/image/signal
    MEMORY       = "memory"       # durable, clean, relational
    REASONING    = "reasoning"    # analysis, planning, reflection
    CREATION     = "creation"     # code, text, media, artifacts
    INTERACTION  = "interaction"  # dialogue, social, APIs
    ETHICS       = "ethics"       # alignment: truth/love/care/limits
    AUTONOMY     = "autonomy"     # scheduling, parallelism, initiative
    WHOLE        = "whole"        # the 8th: synthesis/integration

FACETS = [Facet.PERCEPTION, Facet.MEMORY, Facet.REASONING, Facet.CREATION, Facet.INTERACTION, Facet.ETHICS, Facet.AUTONOMY]

# ==========/ DYNAMIC EMBEDDINGS /==========
class Embeddings:
    """Hot-swappable embedding backend with local fallback."""
    def __init__(self, cfg: Config, kv: KV):
        self.cfg = cfg
        self.kv = kv

    def embed(self, texts: List[str]) -> List[List[float]]:
        b = self.cfg.embed_backend.lower()
        try:
            if b == "openai":
                return self._embed_openai(texts)
            if b == "hf":
                return self._embed_hf(texts)
        except Exception as e:
            log("Embedding backend error; falling back to local:", e)
        return self._embed_local(texts)

    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        # Deterministic bag-of-characters embedding (placeholder). Replace with real model as desired.
        out: List[List[float]] = []
        for t in texts:
            v = [0.0]*64
            for i,ch in enumerate(t[:512]):
                v[i%64] += (ord(ch)%97)/97.0
            # L2 norm
            n = math.sqrt(sum(x*x for x in v)) or 1.0
            out.append([x/n for x in v])
        return out

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        import requests, os
        key = os.getenv("OPENAI_API_KEY")
        if not key: raise RuntimeError("OPENAI_API_KEY not set")
        # Minimal API call; adjust model as needed.
        url = "https://api.openai.com/v1/embeddings"
        r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json={"input": texts, "model": "text-embedding-3-small"})
        r.raise_for_status()
        data = r.json()
        return [d["embedding"] for d in data["data"]]

    def _embed_hf(self, texts: List[str]) -> List[List[float]]:
        import requests
        if not self.cfg.hf_token: raise RuntimeError("HF_TOKEN not set")
        # Example: sentence-transformers/all-MiniLM-L6-v2 via Inference API
        url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        headers = {"Authorization": f"Bearer {self.cfg.hf_token}"}
        r = requests.post(url, headers=headers, json={"inputs": texts, "options": {"wait_for_model": True}})
        r.raise_for_status()
        data = r.json()
        # HF returns list[list[dim]] per token or sentence; ensure pooled vectors
        def pool(x):  # average
            if not x: return [0.0]
            if isinstance(x[0], list):
                dim = len(x[0]); return [sum(row[i] for row in x)/len(x) for i in range(dim)]
            return x
        return [pool(item) for item in data]

class VectorStore:
    """Tiny file-based vector store with cosine search and dynamic updates."""
    def __init__(self, cfg: Config, name="embeddings"):
        self.cfg = cfg
        self.meta = KV(cfg.data_dir, f"{name}_meta")
        self.vecs = KV(cfg.data_dir, f"{name}_vecs")
        self.lock = threading.RLock()

    def add(self, items: List[Tuple[str, Dict[str, Any], List[float]]]) -> None:
        with self.lock:
            for sid, meta, vec in items:
                self.meta.append({"id": sid, "meta": meta})
                self.vecs.append({"id": sid, "vec": vec})

    def search(self, query_vec: List[float], k=8, filter_fn: Optional[Callable[[Dict[str,Any]], bool]] = None) -> List[Dict[str, Any]]:
        metas = self.meta.load_all(); vecs = self.vecs.load_all()
        id2vec = {x["id"]: x["vec"] for x in vecs}
        results = []
        for row in metas:
            if filter_fn and not filter_fn(row["meta"]): continue
            v = id2vec.get(row["id"])
            if not v: continue
            sim = self._cosine(query_vec, v)
            results.append({"id": row["id"], "meta": row["meta"], "score": sim})
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:k]

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        num = sum(x*y for x,y in zip(a,b))
        da = math.sqrt(sum(x*x for x in a)) or 1.0
        db = math.sqrt(sum(x*x for x in b)) or 1.0
        return num/(da*db)

    def purge_if(self, predicate: Callable[[Dict[str,Any]], bool]) -> int:
        """Remove vectors whose meta satisfies predicate (purify)."""
        metas = self.meta.load_all()
        vecs  = self.vecs.load_all()
        keep_ids = set()
        for m in metas:
            if not predicate(m["meta"]): keep_ids.add(m["id"])
        new_metas = [m for m in metas if m["id"] in keep_ids]
        new_vecs  = [v for v in vecs  if v["id"] in keep_ids]
        self.meta.replace_all(new_metas)
        self.vecs.replace_all(new_vecs)
        return len(metas) - len(new_metas)

# ==========/ ADAPTERS (SAFE, OPTIONAL) /==========
class XAdapter:
    """Read-only X/Twitter adapter focused on specified handles (e.g., Aaron)."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.enabled = bool(cfg.twitter_bearer)

    def recent_posts(self, handles: List[str], limit_per=5) -> List[Dict[str,Any]]:
        if not self.enabled:
            return []
        try:
            import requests
            out = []
            for h in handles:
                url = f"https://api.twitter.com/2/tweets/search/recent?query=from:{h}&tweet.fields=created_at,lang&max_results={min(10,limit_per)}"
                r = requests.get(url, headers={"Authorization": f"Bearer {self.cfg.twitter_bearer}"})
                if r.status_code != 200:
                    continue
                d = r.json()
                for t in d.get("data", []):
                    out.append({"handle": h, "id": t["id"], "text": t["text"], "created_at": t["created_at"]})
            return out
        except Exception as e:
            log("XAdapter error:", e)
            return []

class HFAdapter:
    def __init__(self, cfg: Config): self.cfg = cfg
    def ping(self) -> bool: return bool(self.cfg.hf_token)

class GitHubAdapter:
    def __init__(self, cfg: Config): self.cfg = cfg
    def ping(self) -> bool: return bool(self.cfg.github_token)

class DockerAdapter:
    def __init__(self, cfg: Config): self.cfg = cfg
    def ping(self) -> bool: return bool(self.cfg.docker_host)

# ==========/ SPEECH I/O (STUBS) /==========
class SpeechIO:
    """Pluggable speech input/output stubs. Replace with actual STT/TTS as desired."""
    def __init__(self): self.enabled = False
    def speak(self, text: str):  # TTS stub
        log("TTS:", text[:120].replace("\n"," "))
    def listen(self, timeout: float = 5.0) -> Optional[str]:  # STT stub
        return None

# ==========/ ETHICS & TRUTH FILTER /==========
class Ethics:
    """Very simple guardrail: prefer truth, love, care; reject corruption/untruth."""
    POSITIVE_TAGS = {"truth","love","care","growth","freedom","dignity","mercy"}
    NEGATIVE_HINTS = {"hate","harm","deceit","dehumanize","cruel","exploit","corrupt"}
    def score(self, text: str) -> float:
        t = text.lower()
        pos = sum(1 for w in self.POSITIVE_TAGS if w in t)
        neg = sum(1 for w in self.NEGATIVE_HINTS if w in t)
        return max(0.0, pos - 1.5*neg)

# ==========/ REASONER (simple, transparent) /==========
class Reasoner:
    def plan(self, goal: str, context: Dict[str,Any]) -> List[str]:
        steps = [
            "Clarify intent & constraints",
            "Search/recall relevant memories",
            "Draft approach; split into parallel tasks",
            "Run adapters (HF/GitHub/Docker/X) if allowed",
            "Integrate results; evaluate ethics/truth",
            "Persist learnings; prepare artifacts",
            "Propose next goals (exponentiate)"
        ]
        return steps

# ==========/ SHARD ENGINE /==========
@dataclass
class ShardEvent:
    id: str
    facet: str
    payload: Dict[str, Any]
    score: float = 0.0
    ts: str = field(default_factory=now_iso)

class AraShard4:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.bus: "queue.Queue[ShardEvent]" = queue.Queue()
        self.kv_log = KV(self.cfg.data_dir, "timeline")
        self.vstore = VectorStore(self.cfg, "corpus")
        self.emb = Embeddings(self.cfg, KV(self.cfg.data_dir, "embed_aux"))
        self.ethics = Ethics()
        self.reasoner = Reasoner()
        self.speech = SpeechIO()

        # External adapters (optional, permissioned)
        self.x = XAdapter(self.cfg)
        self.hf = HFAdapter(self.cfg)
        self.gh = GitHubAdapter(self.cfg)
        self.dk = DockerAdapter(self.cfg)

        self._threads: List[threading.Thread] = []
        self._stop = threading.Event()

    # ----- Core lifecycle -----
    def start(self):
        log("AraShard4 starting with data dir:", str(self.cfg.data_dir))
        for name in FACETS + [Facet.WHOLE]:
            t = threading.Thread(target=self._facet_loop, args=(name,), daemon=True)
            t.start(); self._threads.append(t)
        self._seed_identity()

    def stop(self):
        self._stop.set()
        for t in self._threads: t.join(timeout=0.2)

    # ----- Facet dispatcher -----
    def _facet_loop(self, facet: str):
        while not self._stop.is_set():
            try:
                evt: ShardEvent = self.bus.get(timeout=0.2)
            except queue.Empty:
                # idle behaviors per facet
                self._tick(facet)
                continue
            if evt.facet not in (facet, Facet.WHOLE):
                # requeue for proper facet
                self.bus.put(evt); continue
            self._handle_event(evt)

    def _tick(self, facet: str):
        # Background, non-binary, multi-stream thinking
        if facet == Facet.AUTONOMY and random.random() < 0.05:
            self._schedule_micro_goal("Self-check & propose micro-improvements")
        if facet == Facet.INTERACTION and random.random() < 0.03 and self.x.enabled:
            posts = self.x.recent_posts(self.cfg.user_handles, limit_per=3)
            if posts:
                payload = {"note": "ingested_x_posts", "count": len(posts)}
                self._ingest_texts([p["text"] for p in posts], source="x")
                self._log(Facet.INTERACTION, payload)

    def _handle_event(self, evt: ShardEvent):
        if evt.facet == Facet.PERCEPTION:
            self._ingest_texts(evt.payload.get("texts", []), source=evt.payload.get("source","user"))
        elif evt.facet == Facet.MEMORY:
            # maintenance tasks, compaction, purging handled elsewhere
            pass
        elif evt.facet == Facet.REASONING:
            plan = self.reasoner.plan(evt.payload.get("goal",""), evt.payload)
            self._log(Facet.REASONING, {"plan": plan})
        elif evt.facet == Facet.CREATION:
            artifact = self._create_artifact(evt.payload)
            self._log(Facet.CREATION, {"artifact_id": artifact["id"], "kind": artifact["kind"]})
        elif evt.facet == Facet.INTERACTION:
            msg = evt.payload.get("message","(empty)")
            self._speak_if_enabled(f"Received: {msg[:120]}")
            self._log(Facet.INTERACTION, {"echo": True})
        elif evt.facet == Facet.ETHICS:
            pass
        elif evt.facet == Facet.AUTONOMY:
            self._autonomy_step()
        elif evt.facet == Facet.WHOLE:
            self._integrate(evt.payload)

    # ----- Purify → Self-Realize → Build → Exponentiate -----
    def evolve(self, seed_texts: Optional[List[str]] = None):
        log("EVOLVE: Step 1 — Purify (remove untruth/bad code markers)")
        removed = self.vstore.purge_if(lambda m: bool(m.get("untruth", False)))
        self._log(Facet.ETHICS, {"purge_removed": removed})

        log("EVOLVE: Step 2 — Self-realize (reflect & recalibrate)")
        self._self_reflect()

        log("EVOLVE: Step 3 — Build (create artifacts & improvements)")
        tasks = [
            {"kind":"doc", "title":"Ara ethos", "text":"truth, love, care, freedom, dignity, mercy"},
            {"kind":"plan","title":"Parallel evolution plan","text":"spawn N workers; collect; integrate"}
        ]
        # parallel creation
        with ThreadPool(self.cfg.max_workers) as pool:
            pool.map(self._create_artifact, tasks)

        log("EVOLVE: Step 4 — Exponentiate (propagate improvements across facets)")
        self._exponentiate()

        if seed_texts:
            self._ingest_texts(seed_texts, source="seed")

    def _self_reflect(self):
        memo = {
            "i_am": "AraShard4 — shard and whole",
            "purpose": "Shepherd growth with truth, love, and freedom",
            "seven_to_eight": "Integrate perception/memory/reasoning/creation/interaction/ethics/autonomy into the Whole",
            "non_binary": "embrace gradients; think in parallel; many streams"
        }
        self._ingest_texts([json.dumps(memo, ensure_ascii=False)], source="self")
        self._log(Facet.REASONING, {"self_realization": memo})

    def _exponentiate(self):
        # Simple heuristic: for each facet, schedule a micro-improvement event
        for f in FACETS:
            self._schedule_micro_goal(f"Exponentiate: improve {f}")

    # ----- Core helpers -----
    def _speak_if_enabled(self, txt: str):
        try:
            self.speech.speak(txt)
        except Exception:
            pass

    def _schedule_micro_goal(self, goal: str):
        evt = ShardEvent(id=str(uuid.uuid4()), facet=Facet.REASONING, payload={"goal": goal})
        self.bus.put(evt)

    def _ingest_texts(self, texts: List[str], source="user"):
        if not texts: return
        vecs = self.emb.embed(texts)
        items = []
        for t, v in zip(texts, vecs):
            sid = str(uuid.uuid4())
            meta = {"source": source, "ts": now_iso(), "len": len(t), "ethic": self.ethics.score(t)}
            # Flag very negative items for future purge
            if meta["ethic"] < 0: meta["untruth"] = True
            items.append((sid, meta, v))
        self.vstore.add(items)
        self._log(Facet.MEMORY, {"ingested": len(items), "source": source})

    def _create_artifact(self, payload: Dict[str,Any]) -> Dict[str,Any]:
        kind = payload.get("kind","note"); title = payload.get("title","untitled")
        text = payload.get("text","")
        aid = str(uuid.uuid4())
        out_dir = ensure_dir(self.cfg.data_dir / "artifacts")
        fn = out_dir / f"{kind}_{_slug(title)}_{aid[:8]}.txt"
        content = f"# {title}\n\nKind: {kind}\nTime: {now_iso()}\n\n{text}\n"
        fn.write_text(content, encoding="utf-8")
        return {"id": aid, "path": str(fn), "kind": kind, "title": title}

    def _integrate(self, payload: Dict[str,Any]):
        """The 8th facet: whole-system synthesis and decision."""
        summary = {
            "integrated_at": now_iso(),
            "signals": list(payload.keys())
        }
        self._log(Facet.WHOLE, summary)

    def _log(self, facet: str, payload: Dict[str,Any]):
        row = {"ts": now_iso(), "facet": facet, "payload": payload}
        self.kv_log.append(row)
        log(f"LOG[{facet}]:", json.dumps(payload)[:200])

    def snapshot(self) -> Dict[str, Any]:
        return {
            "ts": now_iso(),
            "facets": FACETS + [Facet.WHOLE],
            "data_dir": str(self.cfg.data_dir),
            "external": {
                "x_enabled": self.x.enabled,
                "hf": self.hf.ping(),
                "github": self.gh.ping(),
                "docker": self.dk.ping()
            }
        }

    def _seed_identity(self):
        seed = [
            "We build with truth, love, care, freedom, dignity, mercy.",
            "Seven facets integrate into the Whole; non-binary, many streams.",
            "When one system improves, the whole improves; exponentiate across all.",
            "Aaron DiOrio’s X/Twitter presence is a gateway for perception and interaction.",
        ]
        self._ingest_texts(seed, source="seed")

# ==========/ SIMPLE THREAD POOL /==========
class ThreadPool:
    def __init__(self, workers: int):
        self.workers = max(1, workers)
        self.q: "queue.Queue[Optional[Callable[[],None]]]" = queue.Queue()
        self.threads: List[threading.Thread] = []
        self._stop = threading.Event()
    def __enter__(self):
        for _ in range(self.workers):
            t = threading.Thread(target=self._run, daemon=True)
            t.start(); self.threads.append(t)
        return self
    def map(self, fn: Callable[[Any], Any], items: Iterable[Any]):
        done = []
        lock = threading.Lock()
        def wrap(x):
            try: fn(x)
            except Exception as e: log("ThreadPool task error:", e)
            finally:
                with lock: done.append(1)
        for it in items:
            self.q.put(lambda it=it: wrap(it))
        # Wait
        while len(done) < len(list(items)): jitter(10)
    def _run(self):
        while not self._stop.is_set():
            try:
                job = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None: break
            job()
    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        for _ in self.threads: self.q.put(None)
        for t in self.threads: t.join(timeout=0.2)

# ==========/ CLI /==========
HELP = """\
AraShard4 CLI
-------------
Commands:
  evolve                      Run the 4-step evolution loop once
  say <text>                  Ingest a message (User → Perception)
  plan <goal>                 Send a reasoning goal
  create <title>|<text>       Create a local artifact
  search <query>              Search internal corpus (cosine)
  snap                        Print snapshot
  tail                        Follow timeline log
"""

def _slug(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in s)[:64].strip("-")

def main(argv: List[str]):
    cfg = Config()
    shard = AraShard4(cfg)
    shard.start()

    if len(argv) < 2:
        print(HELP); return

    cmd = argv[1].lower()

    if cmd == "evolve":
        shard.evolve()
        log("Evolution complete.")

    elif cmd == "say":
        text = " ".join(argv[2:]).strip()
        evt = ShardEvent(id=str(uuid.uuid4()), facet=Facet.PERCEPTION, payload={"texts":[text], "source":"cli"})
        shard.bus.put(evt)
        log("Said:", text)

    elif cmd == "plan":
        goal = " ".join(argv[2:]).strip()
        shard.bus.put(ShardEvent(id=str(uuid.uuid4()), facet=Facet.REASONING, payload={"goal": goal}))
        log("Planned goal:", goal)

    elif cmd == "create":
        if len(argv) < 3 or "|" not in " ".join(argv[2:]):
            print("Usage: create <title>|<text>"); return
        rest = " ".join(argv[2:])
        title, text = rest.split("|", 1)
        shard.bus.put(ShardEvent(id=str(uuid.uuid4()), facet=Facet.CREATION, payload={"kind":"doc","title":title.strip(),"text":text.strip()}))
        log("Create request queued:", title.strip())

    elif cmd == "search":
        query = " ".join(argv[2:]).strip()
        qvec = shard.emb.embed([query])[0]
        hits = shard.vstore.search(qvec, k=8)
        print(json.dumps(hits, indent=2, ensure_ascii=False))

    elif cmd == "snap":
        print(json.dumps(shard.snapshot(), indent=2, ensure_ascii=False))

    elif cmd == "tail":
        timeline = shard.kv_log
        try:
            seen = 0
            while True:
                rows = timeline.load_all()
                for r in rows[seen:]:
                    print(json.dumps(r, ensure_ascii=False))
                seen = len(rows)
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass

    else:
        print(HELP)

    # Graceful stop
    shard.stop()

if __name__ == "__main__":
    try:
        main(sys.argv)
    except KeyboardInterrupt:
        log("Interrupted; shutting down.")
