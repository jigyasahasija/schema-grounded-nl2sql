"""
Adapted from the course-provided LLM helper (see README attribution).

Shared LLM utilities: call_llm() against any OpenAI-compatible server
(Ollama / llama.cpp / LM Studio on your laptop, or Ollama on PACE via ssh tunnel).

Configuration comes from environment variables or a .env file (in solution/ or the repo root):

    LLM_API_BASE=http://127.0.0.1:11434/v1
    LLM_MODEL=gemma4:e4b
    LLM_API_KEY=local

call_llm() is thread-safe, so you can (and should) call generate_sql() from a
thread pool. Responses are cached on disk keyed by (model, messages, params).

gemma4 models can "think" before they answer. call_llm() asks them not to, as the grader does,
so their whole answer goes to the SQL.
"""

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

from openai import OpenAI

_solution_dir = Path(os.path.abspath(__file__)).parent
for _env_path in (_solution_dir / ".env", _solution_dir.parent / ".env"):
    if _env_path.exists():
        for _line in open(_env_path):
            if "=" in _line and not _line.lstrip().startswith("#"):
                _k, _v = _line.strip().split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

LLM_API_BASE = os.environ.get("LLM_API_BASE", "http://127.0.0.1:11434/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma4:e4b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "local")
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "300"))

_client = OpenAI(base_url=LLM_API_BASE, api_key=LLM_API_KEY, timeout=LLM_TIMEOUT, max_retries=2)

# ─── Cache (sqlite, safe across threads and processes) ──────────────────────

_CACHE_PATH = Path(os.environ.get("LLM_CACHE", _solution_dir.parent / "output" / "llm_cache.sqlite"))  # <repo>/output/
_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
_cache_lock = threading.Lock()
_cache_conn = sqlite3.connect(str(_CACHE_PATH), check_same_thread=False, timeout=30)
# Several processes may create the same cache file at once (e.g. parallel eval runs);
# switching to WAL needs an exclusive lock, so wait and retry instead of failing.
for _attempt in range(20):
    try:
        _cache_conn.execute("PRAGMA busy_timeout=30000")
        _cache_conn.execute("PRAGMA journal_mode=WAL")
        _cache_conn.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
        _cache_conn.commit()
        break
    except sqlite3.OperationalError:
        time.sleep(0.2 * (_attempt + 1))
else:
    raise RuntimeError(f"could not initialize LLM cache at {_CACHE_PATH}")

stats = {"calls": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0, "latency_ms": 0.0}
_stats_lock = threading.Lock()


def _cache_get(key):
    with _cache_lock:
        row = _cache_conn.execute("SELECT v FROM cache WHERE k=?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def _cache_put(key, value):
    with _cache_lock:
        _cache_conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(value)))
        _cache_conn.commit()


def _no_thinking(model):
    return model.lower().startswith("gemma4")


def call_llm(prompt, system=None, max_tokens=1024, temperature=0.0, model=None, use_cache=True):
    """Send one chat request and return the response text.

    prompt may be a string (sent as the user message) or a full messages list.
    """
    model = model or LLM_MODEL
    extra = {"reasoning_effort": "none"} if _no_thinking(model) else {}
    if isinstance(prompt, str):
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
    else:
        messages = prompt
    key = hashlib.sha256(
        json.dumps([model, messages, max_tokens, temperature] + ([extra] if extra else []), sort_keys=True).encode()
    ).hexdigest()

    if use_cache and temperature == 0:
        hit = _cache_get(key)
        if hit is not None:
            with _stats_lock:
                stats["calls"] += 1
                stats["cache_hits"] += 1
            return hit["response"]

    t0 = time.time()
    resp = _client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body=extra or None,
    )
    latency = (time.time() - t0) * 1000
    text = resp.choices[0].message.content or ""
    usage = resp.usage
    with _stats_lock:
        stats["calls"] += 1
        stats["latency_ms"] += latency
        if usage:
            stats["input_tokens"] += usage.prompt_tokens or 0
            stats["output_tokens"] += usage.completion_tokens or 0
    if use_cache and temperature == 0:
        _cache_put(key, {"response": text, "latency_ms": round(latency, 1)})
    return text


def print_stats():
    live = stats["calls"] - stats["cache_hits"]
    print(f"  LLM calls: {stats['calls']} (cache hits: {stats['cache_hits']})")
    print(f"  Tokens in/out: {stats['input_tokens']:,} / {stats['output_tokens']:,}")
    if live:
        print(f"  Avg latency: {stats['latency_ms'] / live:.0f} ms")
