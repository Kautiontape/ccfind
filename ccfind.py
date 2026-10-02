#!/usr/bin/env python3
"""ccfind: read-only search and viewer for Claude Code session history.

Indexes ~/.claude/projects/**/*.jsonl into a SQLite FTS5 database under
~/.cache/ccfind. Session files are only ever opened for reading: nothing is
written under ~/.claude, and nothing is ever sent to Claude. The only "action"
is copying a `cd <dir> && claude --resume <id>` command to the clipboard.
"""
from __future__ import annotations

import html
import json
import math
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

HOME = Path.home()
CLAUDE_DIR = Path(os.environ.get("CLAUDE_CONFIG_DIR") or HOME / ".claude")
PROJECTS = CLAUDE_DIR / "projects"
CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME") or HOME / ".cache") / "ccfind"
DB_PATH = Path(os.environ.get("CCFIND_DB") or CACHE_DIR / "index.db")
SCRIPT = Path(__file__).resolve()
WEB_HTML = SCRIPT.with_name("ccfind_web.html")
SCHEMA_VERSION = "10"  # bump whenever parsing changes so old rows get rebuilt

TOOL_INDEX_CAP = 8_000    # chars of each tool call / tool output kept in the index
TEXT_INDEX_CAP = 50_000   # chars of each prompt / reply / thought kept in the index
VIEW_CAP = 30_000         # chars of each tool input/output sent to the viewers

# Pasted text in a prompt: <pasted_content id="b035">…</pasted_content id="b035"> (the id is optional,
# and a paste cut off at the end of the message has no closing tag).
PASTE_RE = re.compile(r'<pasted_content(?:\s+id="[^"]*")?\s*>\n?(.*?)(?:\n?</pasted_content(?:\s+id="[^"]*")?\s*>|\Z)', re.S)
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
# A published Artifact: claude.ai/code/artifact/<uuid> or claude.ai/artifact/<22-character id>.
# Truncated ids ("…/artifact/ce3c3c65-f38c-") and placeholders ("…/artifact/{uuid}") don't match.
ARTIFACT_RE = re.compile(rf"(?:https?://)?claude\.ai/(?:code/)?artifact/({_UUID}|[A-Za-z0-9]{{22}})(?![\w-])")
# An artifact's uuid in its own Artifact tool output (a Docs artifact's document id is its uuid).
ARTIFACT_ALIAS_RE = re.compile(rf'(?:^\[Artifact |own id, "|Claude Docs document \(project ")({_UUID})', re.M)
HTML_TITLE_RE = re.compile(r"<title[^>]*>\s*(.*?)\s*</title>", re.I | re.S)
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

# How much a hit in each kind of message counts toward a session's relevance.
KIND_WEIGHT_SQL = (
    "CASE m.kind WHEN 'title' THEN 4.0 WHEN 'user' THEN 2.0 WHEN 'assistant' THEN 1.4 "
    "WHEN 'summary' THEN 1.0 WHEN 'tool_use' THEN 1.0 WHEN 'thinking' THEN 0.8 "
    "WHEN 'tool_result' THEN 0.6 WHEN 'output' THEN 0.6 ELSE 0.4 END"
)
KIND_GROUPS = {
    "user": ["user"], "you": ["user"], "me": ["user"],
    "claude": ["assistant"], "assistant": ["assistant"],
    "thinking": ["thinking"], "think": ["thinking"],
    "tools": ["tool_use", "tool_result"], "tool": ["tool_use", "tool_result"],
    "output": ["output"], "system": ["system"], "summary": ["summary"], "title": ["title"],
}
KIND_LABEL = {
    "user": "You", "assistant": "Claude", "thinking": "Thinking", "tool_use": "Tool call",
    "tool_result": "Tool output", "output": "Command output", "system": "System",
    "summary": "Compaction summary", "title": "Title",
}


# ---------------------------------------------------------------- parsing

def _tag(s: str, name: str) -> str | None:
    m = re.search(rf"<{name}>(.*?)</{name}>", s, re.S)
    return m.group(1).strip() if m else None


def classify_user_text(text: str):
    """Return (kind, text) for a user-side text block, or None to drop it."""
    s = text.lstrip()
    if not s or s.startswith("Caveat: The messages below were generated"):
        return None
    head = s[:300]
    if "<command-name>" in head or s.startswith("<command-message>"):
        name = _tag(s, "command-name") or "/" + (_tag(s, "command-message") or "")
        return "user", f"{name} {_tag(s, 'command-args') or ''}".strip()
    if s.startswith("<bash-input>"):
        return "user", "! " + (_tag(s, "bash-input") or "")
    if s.startswith(("<local-command-stdout>", "<local-command-stderr>", "<bash-stdout>", "<bash-stderr>")):
        parts = [_tag(s, t) for t in ("local-command-stdout", "local-command-stderr", "bash-stdout", "bash-stderr")]
        return "output", ANSI_RE.sub("", "\n".join(p for p in parts if p))
    if s.startswith(("<task-notification>", "<system-reminder>", "<agent-message", "[Request interrupted")):
        return "system", s
    return "user", text


def split_pastes(text: str):
    """A prompt split into plain-text and pasted parts, or None if nothing was pasted."""
    if "<pasted_content" not in text:
        return None
    parts, pos = [], 0
    for m in PASTE_RE.finditer(text):
        before = text[pos:m.start()].strip("\n")
        if before.strip():
            parts.append({"paste": False, "text": before})
        body = m.group(1).strip("\n")
        parts.append({"paste": True, "text": body, "lines": body.count("\n") + 1 if body else 0})
        pos = m.end()
    after = text[pos:].strip("\n")
    if after.strip():
        parts.append({"paste": False, "text": after})
    return parts


def _paste_label(n: int) -> str:
    return f"[pasted {n} line{'' if n == 1 else 's'}]"


def collapse_pastes(text: str) -> str:
    """The prompt with each paste shrunk to '[pasted N lines]' (for titles and previews)."""
    parts = split_pastes(text)
    if parts is None:
        return text
    return "\n".join(_paste_label(p["lines"]) if p["paste"] else p["text"] for p in parts)


def result_text(c) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for b in c:
            if isinstance(b, str):
                out.append(b)
            elif isinstance(b, dict):
                if b.get("type") == "text":
                    out.append(b.get("text") or "")
                elif b.get("type") == "image":
                    out.append("[image]")
                elif b.get("type") == "tool_reference":
                    out.append(f"[tool: {b.get('tool_name', '')}]")
        return "\n".join(out)
    return "" if c is None else json.dumps(c, ensure_ascii=False)


def tool_summary(inp) -> str:
    """One-line description of a tool call (its most telling argument)."""
    if not isinstance(inp, dict):
        return ""
    for k in ("command", "file_path", "notebook_path", "pattern", "url", "query",
              "description", "skill", "prompt", "path", "subject"):
        v = inp.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip().splitlines()[0][:240]
    return ""


def tool_text(inp) -> str:
    """Searchable text of a tool call: every argument value."""
    if not isinstance(inp, dict):
        return "" if inp is None else str(inp)
    return "\n".join(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                     for v in inp.values() if isinstance(v, (str, list, dict)))


def parse_file(path, full: bool = False):
    """Parse one transcript into (info, items). Opens the file read-only."""
    info: dict = dict(cwds=[], custom_title=None, agent_name=None, ai_title=None, summary=None,
                first_prompt=None, started=None, ended=None, branch=None, model=None,
                session_id=None, n_prompts=0, n_replies=0, n_tools=0,
                tokens=0, peak_ctx=0, out_tokens=0)
    items = []
    tool_names = {}
    calls = {}   # API message id -> (context size, output tokens); a reply is logged once per content block
    with open(path, "rb") as fh:
        for lineno, raw in enumerate(fh, 1):
            try:
                r = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(r, dict):
                continue
            t = r.get("type")
            if t == "custom-title":
                info["custom_title"] = r.get("customTitle") or info["custom_title"]
                continue
            if t == "agent-name":
                info["agent_name"] = r.get("agentName") or info["agent_name"]
                continue
            if t == "ai-title":
                info["ai_title"] = r.get("aiTitle") or info["ai_title"]
                continue
            if t == "summary":
                info["summary"] = r.get("summary") or info["summary"]
                continue
            if t not in ("user", "assistant", "attachment"):
                continue
            if not info["session_id"] and r.get("sessionId"):
                info["session_id"] = r["sessionId"]
            cwd = r.get("cwd")
            if cwd and cwd not in info["cwds"]:
                info["cwds"].append(cwd)
            br = r.get("gitBranch")
            if br and br != "HEAD" and not info["branch"]:
                info["branch"] = br
            ts = r.get("timestamp")
            if ts:
                if not info["started"] or ts < info["started"]:
                    info["started"] = ts
                if not info["ended"] or ts > info["ended"]:
                    info["ended"] = ts
            base = {"ts": ts, "uuid": r.get("uuid"), "line": lineno}

            if t == "attachment":
                a = r.get("attachment") or {}
                if a.get("type") == "queued_command" and a.get("prompt"):
                    c = classify_user_text(result_text(a["prompt"]))
                    if c:
                        items.append({**base, "kind": c[0], "text": c[1]})
                continue

            msg = r.get("message") or {}
            content = msg.get("content")
            if t == "user":
                if r.get("isMeta"):
                    continue
                if r.get("isCompactSummary"):
                    items.append({**base, "kind": "summary", "text": result_text(content)})
                    continue
                blocks = [{"type": "text", "text": content}] if isinstance(content, str) else (content or [])
                for b in blocks:
                    if not isinstance(b, dict):
                        continue
                    bt = b.get("type")
                    if bt == "text":
                        c = classify_user_text(b.get("text") or "")
                        if c:
                            items.append({**base, "kind": c[0], "text": c[1]})
                    elif bt == "tool_result":
                        tid = b.get("tool_use_id")
                        items.append({**base, "kind": "tool_result", "text": result_text(b.get("content")),
                                      "tool": tool_names.get(tid), "tool_id": tid,
                                      "is_error": bool(b.get("is_error"))})
                    elif bt == "image" and full:
                        items.append({**base, "kind": "image", "text": "[image]"})
            else:
                if msg.get("model") and msg["model"] != "<synthetic>":
                    info["model"] = msg["model"]
                u = msg.get("usage")
                if isinstance(u, dict) and msg.get("id"):
                    ctx, out = _call_size(u)
                    prev = calls.get(msg["id"], (0, 0))
                    calls[msg["id"]] = (max(prev[0], ctx), max(prev[1], out))
                blocks = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
                for b in blocks:
                    if not isinstance(b, dict):
                        continue
                    bt = b.get("type")
                    if bt == "text":
                        txt = b.get("text") or ""
                        if txt.strip():
                            kind = "system" if r.get("isApiErrorMessage") else "assistant"
                            items.append({**base, "kind": kind, "text": txt})
                    elif bt == "thinking":
                        txt = b.get("thinking") or ""
                        if txt.strip():
                            items.append({**base, "kind": "thinking", "text": txt})
                    elif bt == "tool_use":
                        name = b.get("name") or "?"
                        tool_names[b.get("id")] = name
                        inp = b.get("input")
                        it = {**base, "kind": "tool_use", "text": tool_text(inp), "tool": name,
                              "tool_id": b.get("id"), "summary": tool_summary(inp)}
                        if full:
                            it["input"] = inp
                        items.append(it)

    for it in items:
        if it["kind"] == "user":
            # Shell-mode "! cmd" lines and bare slash commands ("/clear") aren't messages to Claude;
            # slash commands with a request attached are.
            bare = it["text"].startswith("! ") or (it["text"].startswith("/") and len(it["text"].split(None, 1)) < 2)
            if not bare:
                info["n_prompts"] += 1
                if not info["first_prompt"]:
                    info["first_prompt"] = it["text"]
        elif it["kind"] == "assistant":
            info["n_replies"] += 1
        elif it["kind"] == "tool_use":
            info["n_tools"] += 1
    _token_totals(info, [c for c in calls.values() if c[0]])
    if not info["first_prompt"]:
        info["first_prompt"] = next((it["text"] for it in items if it["kind"] == "user"), None)
    if info["first_prompt"]:
        info["first_prompt"] = collapse_pastes(info["first_prompt"])
    return info, items


def _call_size(u: dict) -> tuple[int, int]:
    """(context size, output tokens) of one API call. When a call ran several
    iterations (e.g. the advisor tool), the top-level usage adds them all up,
    so measure only the main model's own iterations."""
    ctx_of = lambda d: sum(d.get(k) or 0 for k in ("input_tokens", "cache_read_input_tokens",
                                                    "cache_creation_input_tokens"))
    its = [i for i in u.get("iterations") or [] if isinstance(i, dict) and i.get("type", "message") == "message"]
    if its:
        return max(ctx_of(i) for i in its), sum(i.get("output_tokens") or 0 for i in its)
    return ctx_of(u), u.get("output_tokens") or 0


def _token_totals(info, calls):
    """How much went through the conversation, in tokens.

    Summing raw input tokens would count the whole context again on every turn.
    Instead, add up how much the context grew between consecutive API calls
    (Claude's previous reply, tool output, your next prompt), plus the first
    prompt and the last reply. That leaves out the fixed system prompt/tools/
    CLAUDE.md overhead, ignores cache hits and misses, and keeps counting
    across compactions (a compaction shrinks the context; growth after it counts).
    """
    if not calls:
        return
    info["peak_ctx"] = max(c[0] for c in calls)
    info["out_tokens"] = sum(c[1] for c in calls)
    growth = sum(max(0, b[0] - a[0]) for a, b in zip(calls, calls[1:]))
    first_prompt = len(info["first_prompt"] or "") // 4
    info["tokens"] = growth + first_prompt + calls[-1][1]


def fmt_tok(n: int | None) -> str:
    n = n or 0
    if n < 1000:
        return str(n)
    if n < 10_000:
        return f"{n / 1000:.1f}k"
    if n < 1_000_000:
        return f"{n // 1000}k"
    return f"{n / 1e6:.1f}M"


def one_line(s: str | None, n: int = 120) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def session_title(info) -> str:
    for k in ("custom_title", "agent_name", "ai_title", "summary", "first_prompt"):
        if info.get(k):
            return one_line(info[k])
    return "(empty session)"


def _sanitize(p: str) -> str:
    """Claude Code's project-folder name for a path: every non-alphanumeric
    UTF-16 code unit becomes '-' (so an emoji becomes two dashes)."""
    return "".join(c if c.isascii() and c.isalnum() else "-" * (2 if ord(c) > 0xFFFF else 1) for c in p)


def decode_projdir(projdir: str) -> str | None:
    """Best-effort inverse of _sanitize: find an existing directory that encodes to projdir."""
    def walk(base: str, rest: str) -> str | None:
        if not rest:
            return base
        try:
            names = os.listdir(base)
        except OSError:
            return None
        for n in sorted(names, key=len, reverse=True):
            enc = "-" + _sanitize(n)
            if rest.startswith(enc) and os.path.isdir(os.path.join(base, n)):
                found = walk(os.path.join(base, n), rest[len(enc):])
                if found:
                    return found
        return None
    return walk("/", projdir)


def choose_cwd(projdir: str, cwds: list[str]) -> str | None:
    """The directory `claude --resume` must be run from: the cwd whose encoding
    matches the project folder the transcript lives in."""
    for c in cwds:
        if _sanitize(c) == projdir:
            return c
    return cwds[0] if cwds else decode_projdir(projdir)


# ---------------------------------------------------------------- index

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS files(
    id INTEGER PRIMARY KEY, path TEXT UNIQUE, mtime INTEGER, size INTEGER,
    sid TEXT, agent TEXT, agent_type TEXT, agent_desc TEXT, projdir TEXT, tokens INTEGER);
CREATE INDEX IF NOT EXISTS files_sid ON files(sid);
CREATE TABLE IF NOT EXISTS sessions(
    sid TEXT PRIMARY KEY, path TEXT, projdir TEXT, cwd TEXT, title TEXT, first_prompt TEXT,
    started TEXT, ended TEXT, branch TEXT, model TEXT, n_prompts INTEGER, n_replies INTEGER,
    size INTEGER, n_tools INTEGER, tokens INTEGER, peak_ctx INTEGER, out_tokens INTEGER, n_artifacts INTEGER);
CREATE TABLE IF NOT EXISTS msgs(
    id INTEGER PRIMARY KEY, file_id INTEGER, sid TEXT, agent TEXT, kind TEXT, tool TEXT,
    ts TEXT, uuid TEXT, line INTEGER, body TEXT);
CREATE INDEX IF NOT EXISTS msgs_file ON msgs(file_id);
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
    body, sid, content='msgs', content_rowid='id', tokenize='unicode61 remove_diacritics 2');
"""


def _private_cache_dir() -> Path:
    """The index holds the full text of every session, so keep it as private as the transcripts."""
    d = DB_PATH.parent
    d.mkdir(parents=True, exist_ok=True)
    try:
        if d.stat().st_mode & 0o077:
            os.chmod(d, 0o700)
        for f in d.iterdir():
            if f.stat().st_mode & 0o077:
                os.chmod(f, 0o600)
    except OSError:
        pass
    return d


def connect() -> sqlite3.Connection:
    _private_cache_dir()
    db = sqlite3.connect(DB_PATH, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.executescript(SCHEMA)
    row = db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
    if row is None:
        db.execute("INSERT OR REPLACE INTO meta VALUES('schema', ?)", (SCHEMA_VERSION,))
        db.commit()
    elif int(row[0]) < int(SCHEMA_VERSION):
        # Index built by an older parser: start over (it only takes ~30s).
        db.close()
        for suffix in ("", "-wal", "-shm"):
            Path(str(DB_PATH) + suffix).unlink(missing_ok=True)
        return connect()
    elif int(row[0]) > int(SCHEMA_VERSION):
        sys.exit("ccfind: the index was built by a newer ccfind; restart this process")
    return db


def scan_files() -> dict[str, tuple]:
    """Every transcript on disk: path -> (mtime_ns, size, projdir, sid_from_path, agent)."""
    found = {}
    if not PROJECTS.is_dir():
        return found
    for proj in os.scandir(PROJECTS):
        if not proj.is_dir():
            continue
        for root, _dirs, names in os.walk(proj.path):
            rel = Path(root).relative_to(proj.path).parts
            for n in names:
                if not n.endswith(".jsonl"):
                    continue
                p = os.path.join(root, n)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                stem = n[:-6]
                if not rel:      # <projdir>/<sid>.jsonl, the main transcript
                    sid, agent = (stem, "") if UUID_RE.match(stem) else (None, stem)
                else:            # <projdir>/<sid>/subagents/agent-*.jsonl and friends
                    sid, agent = (rel[0] if UUID_RE.match(rel[0]) else None), stem
                found[p] = (st.st_mtime_ns, st.st_size, proj.name, sid, agent)
    return found


def _index_worker(path: str):
    try:
        info, items = parse_file(path)
    except OSError:
        return path, None, None
    rows = []
    for it in items:
        cap = TOOL_INDEX_CAP if it["kind"] in ("tool_use", "tool_result", "output") else TEXT_INDEX_CAP
        body = it["text"]
        if it["kind"] == "user" and (parts := split_pastes(body)):
            # Collapsed prompt first (what previews show), then the pasted text so it stays searchable.
            body = collapse_pastes(body) + "\n\n" + "\n\n".join(p["text"] for p in parts if p["paste"])
        body = body[:cap]
        if it["kind"] == "tool_use":
            body = f"{it['tool']}\n{body}"
        if body.strip():
            rows.append((it["kind"], it.get("tool"), it["ts"], it["uuid"], it["line"], body))
    # Artifacts Claude did something with. Ones you only pasted don't count: hook and
    # classifier runs get whole transcripts pasted in, links and all.
    # Few transcripts link to claude.ai at all; only those need the full parse.
    info["n_artifacts"] = 0
    if any("claude.ai/" in it["text"] for it in items):
        try:
            info["n_artifacts"] = sum(a["how"] != "linked" for a in collect_artifacts(build_transcript(path)[1]))
        except OSError:
            pass
    return path, info, rows


def _agent_meta(path: str):
    try:
        with open(path[:-6] + ".meta.json", "rb") as fh:
            m = json.load(fh)
        return m.get("agentType"), m.get("description")
    except (OSError, ValueError, AttributeError):
        return None, None


def _drop_file(db, fid: int):
    db.execute("INSERT INTO fts(fts, rowid, body, sid) SELECT 'delete', id, body, sid FROM msgs WHERE file_id=?", (fid,))
    db.execute("DELETE FROM msgs WHERE file_id=?", (fid,))


def _sync_orphans(db):
    """Give subagent transcripts whose main transcript is gone a stub session row,
    so their messages still show up in search."""
    db.execute("DELETE FROM sessions WHERE sid NOT IN (SELECT sid FROM files)")
    for r in db.execute(
            "SELECT sid, MIN(projdir), GROUP_CONCAT(agent_desc, ' · '), SUM(size) FROM files "
            "WHERE agent != '' AND sid NOT IN (SELECT sid FROM sessions) GROUP BY sid").fetchall():
        sid, projdir, descs, size = r[0], r[1], r[2], r[3]
        started, ended = db.execute("SELECT MIN(ts), MAX(ts) FROM msgs WHERE sid = ?", (sid,)).fetchone()
        first = db.execute("SELECT body FROM msgs WHERE sid = ? AND kind = 'user' ORDER BY ts LIMIT 1",
                           (sid,)).fetchone()
        first = first[0] if first else None
        _put_session(db, sid=sid, path=str(PROJECTS / projdir / f"{sid}.jsonl"), projdir=projdir,
                     cwd=decode_projdir(projdir), title="[main transcript deleted] " + one_line(descs or first or sid, 100),
                     first_prompt=one_line(first, 400), started=started, ended=ended, size=size,
                     n_prompts=0, n_replies=0, n_tools=0, tokens=0, peak_ctx=0, out_tokens=0)


def _put_session(db, **cols):
    db.execute(f"INSERT OR REPLACE INTO sessions({', '.join(cols)}) VALUES({', '.join('?' * len(cols))})",
               list(cols.values()))


_index_lock = threading.Lock()


def update_index(progress: bool = False, wait: bool = True) -> int:
    """Bring the index up to date with what's on disk. Returns files (re)indexed.
    With wait=False, returns 0 at once if another process is already indexing."""
    import fcntl
    with _index_lock, open(_private_cache_dir() / "index.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError:
            return 0
        db = connect()
        try:
            return _update_index(db, progress)
        finally:
            db.close()


def _update_index(db, progress: bool) -> int:
    found = scan_files()
    known = {r["path"]: (r["id"], r["mtime"], r["size"], r["sid"], r["agent"])
             for r in db.execute("SELECT id, path, mtime, size, sid, agent FROM files")}
    stale = [p for p, v in found.items() if p not in known or known[p][1:3] != v[:2]]
    gone = [p for p in known if p not in found]

    if gone:
        for p in gone:
            fid, _, _, sid, agent = known[p]
            _drop_file(db, fid)
            db.execute("DELETE FROM files WHERE id=?", (fid,))
            if not agent:
                db.execute("DELETE FROM sessions WHERE sid=?", (sid,))
        _sync_orphans(db)
        db.commit()
    if not stale:
        db.execute("INSERT OR REPLACE INTO meta VALUES('checked', ?)", (str(time.time()),))
        db.commit()
        return 0

    # Big files first so the pool stays busy; small ones fill in at the end.
    stale.sort(key=lambda p: -found[p][1])
    total, done, t0 = len(stale), 0, time.time()
    workers = min(len(stale), os.cpu_count() or 4)
    pool = ProcessPoolExecutor(max_workers=workers) if total > 8 else None
    results = pool.map(_index_worker, stale, chunksize=2) if pool else map(_index_worker, stale)
    try:
        for path, info, rows in results:
            done += 1
            mtime, size, projdir, sid, agent = found[path]
            if info is None:
                continue
            if sid is None:
                sid = info.get("session_id") or Path(path).stem
            if path in known:
                _drop_file(db, known[path][0])
            atype, adesc = _agent_meta(path) if agent else (None, None)
            fid = db.execute(
                "INSERT INTO files(path, mtime, size, sid, agent, agent_type, agent_desc, projdir, tokens) "
                "VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET mtime=excluded.mtime, "
                "size=excluded.size, sid=excluded.sid, agent=excluded.agent, agent_type=excluded.agent_type, "
                "agent_desc=excluded.agent_desc, projdir=excluded.projdir, tokens=excluded.tokens RETURNING id",
                (path, mtime, size, sid, agent, atype, adesc, projdir, info["tokens"])).fetchone()[0]
            if not agent:
                titles = [info[k] for k in ("custom_title", "agent_name", "ai_title", "summary") if info.get(k)]
                if titles:
                    rows.append(("title", None, info["ended"], None, 0, "\n".join(titles)))
                _put_session(db, sid=sid, path=path, projdir=projdir, cwd=choose_cwd(projdir, info["cwds"]),
                             title=session_title(info), first_prompt=one_line(info["first_prompt"], 400),
                             started=info["started"], ended=info["ended"], branch=info["branch"],
                             model=info["model"], size=size,
                             **{k: info[k] for k in ("n_prompts", "n_replies", "n_tools", "tokens",
                                                     "peak_ctx", "out_tokens", "n_artifacts")})
            db.executemany(
                "INSERT INTO msgs(file_id, sid, agent, kind, tool, ts, uuid, line, body) VALUES(?,?,?,?,?,?,?,?,?)",
                [(fid, sid, agent, *r) for r in rows])
            db.execute("INSERT INTO fts(rowid, body, sid) SELECT id, body, sid FROM msgs WHERE file_id=?", (fid,))
            if done % 100 == 0:
                db.commit()
            if progress:
                print(f"\r  indexing {done}/{total} files ({time.time() - t0:.0f}s)", end="", file=sys.stderr, flush=True)
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
        db.commit()
    _sync_orphans(db)
    if total > 200:
        db.execute("INSERT INTO fts(fts) VALUES('optimize')")
    db.execute("INSERT OR REPLACE INTO meta VALUES('checked', ?)", (str(time.time()),))
    db.commit()
    if progress:
        print(f"\r  indexed {total} files in {time.time() - t0:.1f}s{' ' * 20}", file=sys.stderr)
    return total


# ---------------------------------------------------------------- queries

FILTER_KEYS = {"p": "project", "project": "project", "in": "in", "kind": "in",
               "since": "since", "after": "since", "before": "before", "until": "before",
               "id": "id", "sort": "sort", "agents": "agents", "sub": "agents",
               "prompts": "prompts", "msgs": "prompts", "messages": "prompts",
               "tokens": "tokens", "tok": "tokens", "size": "tokens", "has": "has"}
RANGE_COLS = {"prompts": "n_prompts", "tokens": "tokens"}
CMP_RE = re.compile(r"^(>=|<=|>|<|=)?(\d+(?:\.\d+)?)([km]?)$", re.I)
SORT_ALIASES = {"new": "new", "newest": "new", "recent": "new", "date": "new", "old": "old", "oldest": "old",
                "big": "big", "biggest": "big", "largest": "big", "size": "big", "tokens": "big", "long": "big"}
TOKEN_RE = re.compile(r'(-?)(?:([A-Za-z]+):)?("[^"]*"?|\S+)')
DUR_RE = re.compile(r"(\d+)\s*(mo|y|w|d|h|m|s)", re.I)


def parse_when(v: str) -> str | None:
    """'2w', '3d6h', '6mo', '2026-07-20', '2026-07-20T14:30' -> UTC ISO string."""
    v = v.strip()
    if re.fullmatch(r"(\d+\s*(mo|y|w|d|h|m|s)\s*)+", v, re.I):
        now = datetime.now(timezone.utc)
        days = secs = 0
        for n, u in DUR_RE.findall(v):
            n, u = int(n), u.lower()
            days += {"y": 365 * n, "mo": 30 * n, "w": 7 * n, "d": n}.get(u, 0)
            secs += {"h": 3600 * n, "m": 60 * n, "s": n}.get(u, 0)
        return (now - timedelta(days=days, seconds=secs)).strftime("%Y-%m-%dT%H:%M:%S")
    try:
        d = datetime.fromisoformat(v.replace(" ", "T"))
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.astimezone()
    return d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def parse_cmp(v: str):
    """'5' (at least 5), '>=5', '<3', '50k', '1.5m' -> (op, number), or None."""
    m = CMP_RE.match(v.strip())
    if not m:
        return None
    n = float(m.group(2)) * {"": 1, "k": 1_000, "m": 1_000_000}[m.group(3).lower()]
    return m.group(1) or ">=", int(n)


def _fts_quote(s: str, prefix: bool) -> str:
    return '"' + s.replace('"', '""') + '"' + ("*" if prefix else "")


class Query:
    """A search string parsed into an FTS5 expression plus filters.

    Words match as prefixes ("deploy" finds "deployment"); "quoted text" is an
    exact phrase; -word excludes; OR between terms. Filters: p:<project>,
    in:you|claude|thinking|tools|output, since:2w, before:2026-05-01, id:<sid>,
    prompts:5 (at least 5 prompts; also >5 <5 <=5), tokens:50k (at least 50k
    tokens; also <20k), has:artifacts (Claude published, opened or linked an
    Artifact; -has:artifacts for none), sort:new|old|big, agents:no.
    """

    def __init__(self, q: str = "", **overrides):
        self.raw = q or ""
        self.projects: list[str] = []
        self.projdirs: list[str] = []
        self.kinds: set[str] = set()
        self.since = self.before = self.sid = None
        self.sort = None
        self.agents = True
        self.artifacts = None   # True: only sessions with artifacts, False: only without
        self.ranges: list[tuple[str, str, int]] = []   # (sessions column, operator, value)
        pos, neg, self.terms = [], [], []
        for m in TOKEN_RE.finditer(self.raw):
            negate, key, val = m.group(1), (m.group(2) or "").lower(), m.group(3)
            if key in FILTER_KEYS and val and not val.startswith('"'):
                self._filter(FILTER_KEYS[key], val, negate)
                continue
            if m.group(2):
                val = m.group(2) + ":" + val
            if val == "OR" and not negate:
                if pos and pos[-1] != "OR":
                    pos.append("OR")
                continue
            phrase = val.startswith('"')
            text = val.strip('"')
            if not re.search(r"\w", text):
                continue
            (neg if negate else pos).append(_fts_quote(text, prefix=not phrase))
            if not negate:
                self.terms.append((text, phrase))
        while pos and pos[-1] == "OR":
            pos.pop()
        if UUID_RE.match(self.raw.strip()):          # a pasted session id
            self.sid, pos, neg, self.terms = self.raw.strip(), [], [], []
        expr = " ".join(pos)
        if expr and neg:
            expr = f"({expr}) NOT " + " NOT ".join(neg)
        self.expr = expr or None
        for k, v in overrides.items():
            if v not in (None, "", [], set()):
                if k in ("projects", "projdirs"):
                    getattr(self, k).extend(p for p in v if p)
                elif k == "kinds":
                    self.kinds |= {x for g in v for x in KIND_GROUPS.get(g, [g])}
                elif k in ("min_prompts", "min_tokens"):
                    self.ranges.append((RANGE_COLS[k[4:]], ">=", int(v)))
                else:
                    setattr(self, k, v)

    def _filter(self, key, val, negate):
        if key == "project":
            self.projects.append(val)
        elif key == "in":
            for g in val.lower().split(","):
                self.kinds.update(KIND_GROUPS.get(g, []))
        elif key in ("since", "before"):
            setattr(self, key, parse_when(val))
        elif key == "id":
            self.sid = val
        elif key == "sort":
            self.sort = SORT_ALIASES.get(val.lower())
        elif key in RANGE_COLS:
            c = parse_cmp(val)
            if c:
                self.ranges.append((RANGE_COLS[key], *c))
        elif key == "agents":
            self.agents = val.lower() not in ("no", "off", "false", "0")
        elif key == "has" and val.lower().startswith("art"):
            self.artifacts = not negate

    @property
    def fts(self) -> str | None:
        return f"{{body}} : ({self.expr})" if self.expr else None

    def term_patterns(self) -> list[str]:
        """Regex sources (Python- and JS-compatible) that highlight the matched words."""
        pats = []
        for text, phrase in self.terms:
            words = re.findall(r"\w+", text)
            if not words:
                continue
            p = r"[\W_]+".join(re.escape(w) for w in words)
            pats.append(r"\b" + p + (r"\b" if phrase else r"\w*"))
        return pats

    def session_where(self, alias="s"):
        where, params = [], []
        for p in self.projects:
            where.append(f"({alias}.cwd LIKE ? OR {alias}.projdir LIKE ?)")
            params += [f"%{p}%", f"%{_sanitize(p)}%"]
        if self.projdirs:
            where.append(f"{alias}.projdir IN ({','.join('?' * len(self.projdirs))})")
            params += self.projdirs
        if self.since:
            where.append(f"{alias}.ended >= ?")
            params.append(self.since)
        if self.before:
            where.append(f"{alias}.started <= ?")
            params.append(self.before)
        if self.sid:
            where.append(f"{alias}.sid LIKE ?")
            params.append(self.sid + "%")
        for col, op, n in self.ranges:      # col and op come from fixed tables, never from input
            where.append(f"COALESCE({alias}.{col}, 0) {op} ?")
            params.append(n)
        if self.artifacts is not None:
            where.append(f"COALESCE({alias}.n_artifacts, 0) {'>' if self.artifacts else '='} 0")
        return where, params


def _session_dict(r, extra=None) -> dict:
    cwd = r["cwd"]
    d = {
        "sid": r["sid"], "title": r["title"] or "(untitled)", "project": project_label(cwd, r["projdir"]),
        "cwd": cwd, "projdir": r["projdir"], "started": r["started"], "ended": r["ended"],
        "branch": r["branch"], "model": r["model"], "path": r["path"],
        **{k: r[k] or 0 for k in ("n_prompts", "n_replies", "n_tools", "tokens", "peak_ctx", "out_tokens", "n_artifacts")},
        "first_prompt": r["first_prompt"], "resume_cmd": resume_cmd(cwd, r["sid"]),
        "cwd_exists": bool(cwd and os.path.isdir(cwd)), "resumable": os.path.exists(r["path"] or ""),
    }
    if extra:
        d.update(extra)
    return d


def search(db, q: Query, limit=50, offset=0, n_snippets=3):
    """Sessions matching q, best first, each with its top snippets."""
    t0 = time.time()
    where, params = q.session_where()
    if not q.fts:
        sql = "SELECT * FROM sessions s" + (" WHERE " + " AND ".join(where) if where else "")
        total = db.execute(sql.replace("*", "COUNT(*)", 1), params).fetchone()[0]
        order = {"old": "s.started ASC", "big": "s.tokens DESC"}.get(q.sort, "s.ended DESC")
        rows = db.execute(sql + f" ORDER BY {order} LIMIT ? OFFSET ?", params + [limit, offset]).fetchall()
        out = [_session_dict(r, {"hits": 0, "snippets": []}) for r in rows]
        _add_agent_counts(db, out)
        return {"total": total, "sessions": out, "ms": int((time.time() - t0) * 1000)}

    mwhere, mparams = _msg_where(q)
    sql = (f"SELECT m.sid AS sid, MIN(f.rank * {KIND_WEIGHT_SQL}) AS best, COUNT(*) AS n "
           "FROM (SELECT rowid, rank FROM fts WHERE fts MATCH ?) f JOIN msgs m ON m.id = f.rowid "
           "JOIN sessions s ON s.sid = m.sid")
    allw = where + mwhere
    if allw:
        sql += " WHERE " + " AND ".join(allw)
    sql += " GROUP BY m.sid"
    try:
        groups = db.execute(sql, [q.fts] + params + mparams).fetchall()
    except sqlite3.OperationalError as e:
        return {"total": 0, "sessions": [], "error": str(e), "ms": int((time.time() - t0) * 1000)}

    meta = {}
    if groups:
        sids = [g["sid"] for g in groups]
        for chunk in range(0, len(sids), 900):
            part = sids[chunk:chunk + 900]
            for r in db.execute(f"SELECT * FROM sessions WHERE sid IN ({','.join('?' * len(part))})", part):
                meta[r["sid"]] = r
    now = time.time()

    def score(g):
        r = meta.get(g["sid"])
        age_days = (now - _epoch(r["ended"] if r else None)) / 86400
        return -g["best"] + 0.35 * math.log1p(g["n"]) + 0.3 * math.exp(-age_days / 60)

    if q.sort == "new":
        groups.sort(key=lambda g: (meta[g["sid"]]["ended"] or "") if g["sid"] in meta else "", reverse=True)
    elif q.sort == "old":
        groups.sort(key=lambda g: (meta[g["sid"]]["started"] or "") if g["sid"] in meta else "")
    elif q.sort == "big":
        groups.sort(key=lambda g: (meta[g["sid"]]["tokens"] or 0) if g["sid"] in meta else 0, reverse=True)
    else:
        groups.sort(key=score, reverse=True)
    page = [g for g in groups[offset:offset + limit] if g["sid"] in meta]
    snips = snippets_for(db, q, [g["sid"] for g in page], per_session=n_snippets) if page else {}
    out = [_session_dict(meta[g["sid"]], {"hits": g["n"], "snippets": snips.get(g["sid"], [])}) for g in page]
    _add_agent_counts(db, out)
    return {"total": len(groups), "sessions": out, "ms": int((time.time() - t0) * 1000)}


def _msg_where(q: Query):
    where, params = [], []
    if q.kinds:
        where.append(f"m.kind IN ({','.join('?' * len(q.kinds))})")
        params += sorted(q.kinds)
    if not q.agents:
        where.append("m.agent = ''")
    return where, params


def _add_agent_counts(db, sessions):
    if not sessions:
        return
    sids = [s["sid"] for s in sessions]
    counts = {r[0]: (r[1], r[2]) for r in db.execute(
        f"SELECT sid, COUNT(*), SUM(tokens) FROM files WHERE agent != '' AND sid IN ({','.join('?' * len(sids))}) "
        "GROUP BY sid", sids)}
    for s in sessions:
        s["n_agents"], s["agent_tokens"] = counts.get(s["sid"], (0, 0))
        s["agent_tokens"] = s["agent_tokens"] or 0


def snippets_for(db, q: Query, sids, per_session=3, limit=None):
    """Best-ranked snippets for each session in sids."""
    mwhere, mparams = _msg_where(q)
    out: dict[str, list] = {}
    for sid in sids:
        match = f"({q.fts}) AND sid : {_fts_quote(sid, False)}"
        sql = (f"SELECT m.agent, m.kind, m.tool, m.ts, m.line, m.uuid, "
               f"snippet(fts, 0, char(2), char(3), '…', 16) AS snip, fts.rank * {KIND_WEIGHT_SQL} AS r "
               "FROM fts JOIN msgs m ON m.id = fts.rowid WHERE fts MATCH ?")
        if mwhere:
            sql += " AND " + " AND ".join(mwhere)
        sql += " ORDER BY r LIMIT ?"
        rows = db.execute(sql, [match] + mparams + [limit or per_session]).fetchall()
        out[sid] = [{"agent": r["agent"], "kind": r["kind"], "tool": r["tool"], "ts": r["ts"],
                     "line": r["line"], "uuid": r["uuid"], "snippet": _clean_snippet(r)} for r in rows]
    return out


def _clean_snippet(r) -> str:
    snip = one_line(r["snip"], 400)
    # Tool-call bodies start with the tool name (so it's searchable); the label already says it.
    if r["kind"] == "tool_use" and r["tool"] and snip.startswith(r["tool"] + " "):
        snip = snip[len(r["tool"]) + 1:]
    return snip


def hit_lines(db, q: Query, sid: str) -> list[dict]:
    """Every (agent, line) in a session that matches q, in file order."""
    if not q.fts:
        return []
    mwhere, mparams = _msg_where(q)
    sql = "SELECT m.agent, m.line, m.kind FROM fts JOIN msgs m ON m.id = fts.rowid WHERE fts MATCH ?"
    if mwhere:
        sql += " AND " + " AND ".join(mwhere)
    sql += " ORDER BY m.agent, m.line LIMIT 5000"
    rows = db.execute(sql, [f"({q.fts}) AND sid : {_fts_quote(sid, False)}"] + mparams).fetchall()
    return [dict(r) for r in rows]


def get_session(db, sid: str):
    r = db.execute("SELECT * FROM sessions WHERE sid = ?", (sid,)).fetchone()
    if r is None and len(sid) >= 6:
        rs = db.execute("SELECT * FROM sessions WHERE sid LIKE ? LIMIT 2", (sid + "%",)).fetchall()
        r = rs[0] if len(rs) == 1 else None
    return r


def session_files(db, sid: str):
    return db.execute("SELECT path, agent, agent_type, agent_desc, size, tokens FROM files WHERE sid = ? ORDER BY agent",
                      (sid,)).fetchall()


def build_transcript(path: str):
    """Entries for display: tool calls paired with their outputs."""
    info, items = parse_file(path, full=True)
    entries, by_tool_id = [], {}
    for it in items:
        k = it["kind"]
        if k == "tool_use":
            e = {"kind": "tool", "tool": it["tool"], "summary": it.get("summary", ""), "ts": it["ts"],
                 "line": it["line"], "uuid": it["uuid"], "input": _cap_input(it.get("input")),
                 "result": None, "result_line": None, "is_error": False}
            by_tool_id[it.get("tool_id")] = e
            entries.append(e)
        elif k == "tool_result":
            e = by_tool_id.get(it.get("tool_id"))
            text, cut = _cap(it["text"])
            if e is not None and e["result"] is None:
                e.update(result=text, result_cut=cut, result_line=it["line"], is_error=it.get("is_error", False))
            else:
                entries.append({"kind": "tool", "tool": it.get("tool") or "?", "summary": "", "ts": it["ts"],
                                "line": it["line"], "uuid": it["uuid"], "input": None, "result": text,
                                "result_cut": cut, "result_line": it["line"], "is_error": it.get("is_error", False)})
        else:
            e = {"kind": k, "text": it["text"], "ts": it["ts"], "line": it["line"], "uuid": it["uuid"]}
            if k == "user" and (parts := split_pastes(it["text"])):
                e["parts"] = parts
                e["text"] = "\n\n".join(p["text"] for p in parts)   # what "copy message" copies
            entries.append(e)
    return info, entries


def _cap(s: str, n: int = VIEW_CAP):
    if s and len(s) > n:
        return s[:n], len(s) - n
    return s, 0


def _cap_input(inp):
    if isinstance(inp, dict):
        return {k: (_cap(v)[0] if isinstance(v, str) else v) for k, v in inp.items()}
    return inp


# ---------------------------------------------------------------- artifacts

# How a transcript touched an artifact. The strongest one is shown.
ARTIFACT_HOW = {"linked": 0, "mentioned": 1, "opened": 2, "updated": 3, "published": 4, "deleted": 5}


def artifact_url(aid: str) -> str:
    return f"https://claude.ai/code/artifact/{aid}" if UUID_RE.match(aid) else f"https://claude.ai/artifact/{aid}"


def _clean_title(s: str) -> str:
    return " ".join(html.unescape(s).replace("**", "").replace("`", "").split())[:120]


def _linked_title(text: str, start: int) -> str | None:
    """The name an artifact link is given in Claude's text: [Name](url) or **Name**: url."""
    before = text[text.rfind("\n", 0, start) + 1:start]
    m = re.search(r"\[([^\]\n]+)\]\(\s*<?$", before) or re.search(r"\*\*([^*\n]{2,120})\*\*\s*[:—–-]?\s*<?$", before)
    # "**Report:** url" is a label, not a name.
    if m and not ARTIFACT_RE.search(m[1]) and not m[1].lstrip().startswith("http") and not m[1].rstrip().endswith(":"):
        return _clean_title(m[1])
    return None


def _file_title(path: str) -> str | None:
    """<title> of a published page that's still on disk."""
    try:
        with open(path, "rb") as fh:
            m = HTML_TITLE_RE.search(fh.read(65536).decode("utf-8", "replace"))
    except OSError:
        return None
    return _clean_title(m[1]) if m else None


def collect_artifacts(entries) -> list[dict]:
    """Artifacts a transcript published, opened or linked to, in the order they first come up.

    Links that only appear in other tools' output don't count: an Artifact "list" or a fetched
    note names artifacts from other sessions. Such output can still name an artifact found here."""
    arts: dict[str, dict] = {}
    names: dict[str, tuple[int, str]] = {}   # id -> (priority, title)
    written: dict[str, str] = {}             # file name -> <title> it was last written with
    aliases: dict[str, str] = {}             # uuid -> short id of the same artifact

    def name(aid, title, prio):
        if title and prio >= names.get(aid, (-1, ""))[0]:
            names[aid] = (prio, title)

    def add(aid, e, how, version=None, desc=None):
        a = arts.setdefault(aid, {"id": aid, "first": e["line"], "line": e["line"], "how": how, "version": None, "desc": None})
        if ARTIFACT_HOW[how] > ARTIFACT_HOW[a["how"]]:
            a["how"], a["line"] = how, e["line"]
        if version:
            a["version"] = max(a["version"] or 0, version)
        if desc:
            a["desc"] = desc

    for e in entries:
        k = e["kind"]
        if k in ("assistant", "summary", "user"):
            for m in ARTIFACT_RE.finditer(e["text"]):
                add(m[1], e, "linked" if k == "user" else "mentioned")
                if k != "user":
                    name(m[1], _linked_title(e["text"], m.start()), 2)
            continue
        if k != "tool":
            continue
        tool, res = e.get("tool") or "", e.get("result") or ""
        inp = e["input"] if isinstance(e.get("input"), dict) else {}
        url = ARTIFACT_RE.search(inp.get("url") or "") if isinstance(inp.get("url"), str) else None
        # A page that gets published later, written with Write, Edit or a shell heredoc.
        for key in ("content", "new_string"):
            if isinstance(inp.get(key), str) and isinstance(inp.get("file_path"), str) and (m := HTML_TITLE_RE.search(inp[key])):
                written[os.path.basename(inp["file_path"])] = _clean_title(m[1])
        if isinstance(inp.get("command"), str) and (m := HTML_TITLE_RE.search(inp["command"])):
            for f in re.findall(r"([\w.-]+\.html?)\b", inp["command"][:m.start()]):
                written[f] = _clean_title(m[1])
        if tool == "Artifact":
            action = inp.get("action") or "publish"
            target = url[1] if url else None
            if action == "list" and not url:
                for line in res.splitlines():
                    if line.startswith("- ") and (m := ARTIFACT_RE.search(line)):
                        name(m[1], _clean_title(line[2:m.start()].rstrip(" —-")), 3)
                continue
            if action == "publish":
                if not target and not e.get("is_error"):
                    type_url = ARTIFACT_RE.search(inp.get("type_url") or "")
                    target = next((m[1] for m in ARTIFACT_RE.finditer(res) if not type_url or m[1] != type_url[1]), None)
                if not target:
                    continue
                v = re.search(r"\(Version (\d+)\)", res)
                add(target, e, "updated" if url else "published", version=v and int(v[1]), desc=inp.get("description"))
                path = inp.get("file_path")
                if isinstance(inp.get("title"), str):
                    name(target, _clean_title(inp["title"]), 5)
                if isinstance(path, str) and not inp.get("asset"):
                    name(target, written.get(os.path.basename(path)) or _file_title(path), 4)
                    name(target, os.path.basename(path), 1)
            elif target:
                how = {"write_db": "updated", "delete": "updated" if inp.get("path") else "deleted"}.get(action, "opened")
                add(target, e, how)
                if action == "read" and (m := HTML_TITLE_RE.search(res)):
                    name(target, _clean_title(m[1]), 4)
            if target and not UUID_RE.match(target):
                for m in ARTIFACT_ALIAS_RE.finditer(res):
                    aliases[m[1]] = target
        elif tool.startswith("Artifact"):   # ArtifactData, ArtifactComments
            if url:
                add(url[1], e, "opened" if inp.get("action") in (None, "get", "list", "query", "read") else "updated")
        elif "Claude_Docs" in tool:
            create = (inp.get("container") or {}).get("create") if isinstance(inp.get("container"), dict) else None
            for m in ARTIFACT_RE.finditer(res):
                add(m[1], e, "published" if create else "updated")
                if isinstance(create, dict) and isinstance(create.get("name"), str):
                    name(m[1], _clean_title(create["name"]), 5)
        elif inp:
            # Links Claude wrote somewhere else: a file, a ticket, a note.
            for m in ARTIFACT_RE.finditer(json.dumps(inp, ensure_ascii=False)):
                add(m[1], e, "mentioned")

    # The same artifact can show up by its uuid and by its short id.
    for uid, short in aliases.items():
        if uid not in arts:
            continue
        a = arts.pop(uid)
        b = arts.setdefault(short, a | {"id": short})
        if b is not a:
            if ARTIFACT_HOW[a["how"]] > ARTIFACT_HOW[b["how"]]:
                b["how"], b["line"] = a["how"], a["line"]
            b["first"] = min(a["first"], b["first"])
            b["version"] = b["version"] or a["version"]
            b["desc"] = b["desc"] or a["desc"]
        if names.get(uid, (-1,))[0] > names.get(short, (-1,))[0]:
            names[short] = names[uid]
    out = sorted(arts.values(), key=lambda a: a["first"])
    for a in out:
        a["url"] = artifact_url(a["id"])
        a["title"] = names.get(a["id"], (0, None))[1]
    return out


# ---------------------------------------------------------------- project tree

# Where worktree tools put checkouts, and how many path parts below that name one:
# herdr uses ~/.herdr/worktrees/<repo name>/<worktree>, Claude Code <repo>/.claude/worktrees/<worktree>.
WORKTREE_DIRS = {"/.herdr/worktrees/": 2, "/.claude/worktrees/": 1}
HERDR_LOG = Path(os.environ.get("XDG_CONFIG_HOME") or HOME / ".config") / "herdr" / "herdr-server.log"


def worktree_root(cwd: str) -> str | None:
    """The checkout directory, if cwd is inside a linked git worktree."""
    for marker, depth in WORKTREE_DIRS.items():
        head, sep, tail = cwd.partition(marker)
        parts = tail.split("/")
        if sep and len(parts) >= depth and all(parts[:depth]):
            return head + marker + "/".join(parts[:depth])
    return cwd if _linked_gitdir(cwd) else None


def _linked_gitdir(checkout: str) -> str | None:
    """Where a linked worktree's .git file points: <repo>/.git/worktrees/<name>."""
    try:
        with open(os.path.join(checkout, ".git")) as fh:
            line = fh.readline().strip()
    except OSError:      # no .git, or a .git directory (a main checkout)
        return None
    if not line.startswith("gitdir:"):
        return None
    gitdir = os.path.normpath(os.path.join(checkout, line[7:].strip()))
    _, sep, name = gitdir.rpartition("/worktrees/")
    return gitdir if sep and "/" not in name else None   # a submodule's points at .../modules/<name>


def _main_checkout(gitdir: str) -> str | None:
    """The repo a linked worktree belongs to, from its git dir. Works after the repo is gone."""
    try:
        with open(os.path.join(gitdir, "commondir")) as fh:
            common = os.path.normpath(os.path.join(gitdir, fh.read().strip()))
    except OSError:
        common = gitdir.rpartition("/worktrees/")[0]
    try:
        with open(os.path.join(common, "config")) as fh:
            m = re.search(r"^\s*worktree\s*=\s*(.+?)\s*$", fh.read(), re.M)
    except OSError:
        m = None
    if m:                # a submodule: its git dir is the superproject's .git/modules/<name>
        return os.path.normpath(os.path.join(common, m[1]))
    if os.path.basename(common) == ".git":
        return os.path.dirname(common)
    sup, sep, name = common.partition("/.git/modules/")
    return os.path.join(sup, name) if sep else None


def _herdr_checkouts() -> dict[str, str]:
    """checkout -> repo for the worktrees herdr has logged creating."""
    out = {}
    try:
        with open(HERDR_LOG, errors="replace") as fh:
            for line in fh:
                if "checkout_path=" in line and (m := re.search(r'repo_root=(.+?) branch=".*" checkout_path=(.+)', line)):
                    out[m[2].rstrip()] = m[1]
    except OSError:
        pass
    return out


def _has_branch(repo: str, branch: str) -> bool:
    try:
        r = subprocess.run(["git", "-C", repo, "for-each-ref", "--count=1", "--format=x",
                            f"refs/heads/{branch}", f"refs/remotes/*/{branch}"],
                           capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.stdout.strip() == "x"


def worktree_parents(worktrees: dict[str, set[str]], projects: set[str]) -> dict[str, str]:
    """The repo each worktree was made from, worked out again from what's on disk each time.

    worktrees maps each checkout to the branches its sessions were on; projects are the
    other directories sessions ran in. A worktree that can't be placed is left out."""
    parents, herdr = {}, None
    for wt in worktrees:
        if gitdir := _linked_gitdir(wt):
            parent = _main_checkout(gitdir)
        elif "/.claude/worktrees/" in wt:
            parent = wt.partition("/.claude/worktrees/")[0]
        else:
            herdr = _herdr_checkouts() if herdr is None else herdr
            parent = herdr.get(wt)
        if parent:
            parents[wt] = parent
    # A deleted herdr worktree's folder still names its repo. When more than one repo has
    # that name, keep the ones still on disk, then the ones that still have its branch.
    known = projects | set(parents.values())
    for wt, branches in worktrees.items():
        if wt in parents or "/.herdr/worktrees/" not in wt:
            continue
        name = wt.partition("/.herdr/worktrees/")[2].split("/")[0]
        cands = [p for p in known if os.path.basename(p) == name]
        if len(cands) > 1:
            cands = [p for p in cands if os.path.isdir(p)]
        if len(cands) > 1:
            cands = [p for p in cands if any(_has_branch(p, b) for b in branches)]
        if len(cands) == 1:
            parents[wt] = cands[0]
    return parents


def project_tree(db) -> dict[str, dict]:
    """Every directory sessions ran in, keyed by path, with the folders above them.
    Worktrees hang off the repo they were made from. Roots are HOME and "/"."""
    rows = db.execute("SELECT projdir, cwd, COUNT(*) n, MAX(ended) last, json_group_array(DISTINCT branch) branches "
                      "FROM sessions GROUP BY projdir").fetchall()
    home = str(HOME)
    tree: dict[str, dict] = {}

    def node(path, parent=None):
        """The node for path, created along with the folders above it (up to HOME or "/")."""
        if path not in tree:
            tree[path] = {"name": os.path.basename(path) or path, "projdirs": [], "n": 0, "last": "",
                          "kids": [], "wt": parent is not None}
            if path not in (home, "/"):
                parent = parent or (os.path.dirname(path) if path.startswith("/") else "/")
                node(parent)["kids"].append(path)
        return tree[path]

    keys, worktrees = {}, {}
    for r in rows:
        cwd = (r["cwd"] or r["projdir"]).rstrip("/") or "/"
        wt = worktree_root(cwd) if cwd.startswith("/") else None
        keys[r["projdir"]] = wt or cwd
        if wt:
            # A detached checkout is on "HEAD", which every clone with a remote has.
            worktrees.setdefault(wt, set()).update(b for b in json.loads(r["branches"]) if b and b != "HEAD")
    parents = worktree_parents(worktrees, {k for k in keys.values() if k not in worktrees})

    node(home)
    node("/")
    for r in rows:
        key = keys[r["projdir"]]
        parent = parents.get(key)
        if parent and (parent + "/").startswith(key + "/"):    # never under itself
            parent = None
        n = node(key, parent)
        n["projdirs"].append(r["projdir"])
        n["n"] += r["n"]
        n["last"] = max(n["last"], r["last"] or "")

    def total(path):
        t = tree[path]
        t["total"], t["recent"] = t["n"], t["last"]
        for k in t["kids"]:
            total(k)
            t["total"] += tree[k]["total"]
            if tree[k]["wt"]:
                t["recent"] = max(t["recent"], tree[k]["recent"])
        t["kids"].sort(key=lambda k: (tree[k]["wt"], tree[k]["name"].lstrip(".").casefold()))
    total(home)
    total("/")
    return tree


def project_options(tree: dict[str, dict], n_recent: int = 8) -> dict:
    """The project picker: the most recently used projects, then every folder as an indented tree.
    A folder with nothing of its own and one subfolder shares a row with it ("obsidian/Main")."""
    home = str(HOME)
    rows = []

    def walk(path, depth, label):
        t = tree[path]
        while not t["projdirs"] and len(t["kids"]) == 1 and not tree[t["kids"][0]]["wt"]:
            path = t["kids"][0]
            t = tree[path]
            label += "/" + t["name"]
        rows.append({"key": path, "label": label, "depth": depth, "n": t["total"]})
        for k in t["kids"]:
            walk(k, depth + 1, ("⎇ " if tree[k]["wt"] else "") + tree[k]["name"])

    # HOME and "/" themselves list only their own sessions: everything under them is "All projects".
    for root, label in ((home, "~"), ("/", "/")):
        if tree[root]["projdirs"]:
            rows.append({"key": root, "label": label, "depth": 0, "n": tree[root]["n"]})
    for k in tree[home]["kids"]:
        walk(k, 0, tree[k]["name"])
    for k in tree["/"]["kids"]:
        walk(k, 0, "/" + tree[k]["name"])

    used = [p for p, t in tree.items() if not t["wt"] and (t["projdirs"] or any(tree[k]["wt"] for k in t["kids"]))]
    used.sort(key=lambda p: tree[p]["recent"], reverse=True)
    recent = [{"key": p, "label": project_label(p), "n": tree[p]["n"] if p in (home, "/") else tree[p]["total"]}
              for p in used[:n_recent]]
    return {"recent": recent, "tree": rows}


def project_projdirs(tree: dict[str, dict], key: str) -> list[str]:
    """The project folders of every session at key or under it (HOME and "/": only their own)."""
    if key not in tree:
        return []
    if key in (str(HOME), "/"):
        return list(tree[key]["projdirs"])
    out, stack = [], [key]
    while stack:
        t = tree[stack.pop()]
        out += t["projdirs"]
        stack += t["kids"]
    return out


# ---------------------------------------------------------------- helpers

def resume_cmd(cwd: str | None, sid: str) -> str:
    return f"cd {shlex.quote(cwd)} && claude --resume {sid}" if cwd else f"claude --resume {sid}"


def project_label(cwd: str | None, projdir: str | None = None) -> str:
    if not cwd:
        return projdir or "?"
    try:
        parts = Path(cwd).relative_to(HOME).parts
    except ValueError:
        return cwd
    if not parts:
        return "~"
    return "/".join(parts[-2:]) if len(parts) > 2 else "~/" + "/".join(parts)


def _epoch(ts: str | None) -> float:
    if not ts:
        return 0.0
    try:
        return datetime.fromisoformat(ts).timestamp()
    except ValueError:
        return 0.0


def fmt_ts(ts: str | None, fmt="%Y-%m-%d %H:%M") -> str:
    if not ts:
        return ""
    try:
        return datetime.fromisoformat(ts).astimezone().strftime(fmt)
    except ValueError:
        return ts


def ago(ts: str | None) -> str:
    if not ts:
        return ""
    s = time.time() - _epoch(ts)
    for unit, n in (("y", 31536000), ("mo", 2592000), ("w", 604800), ("d", 86400), ("h", 3600), ("m", 60)):
        if s >= n:
            return f"{int(s // n)}{unit} ago"
    return "just now"


def copy_to_clipboard(text: str) -> bool:
    if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-copy"):
        cmd = ["wl-copy"]
    elif os.environ.get("DISPLAY") and shutil.which("xclip"):
        cmd = ["xclip", "-selection", "clipboard"]
    elif os.environ.get("DISPLAY") and shutil.which("xsel"):
        cmd = ["xsel", "-b", "-i"]
    elif shutil.which("pbcopy"):
        cmd = ["pbcopy"]
    else:
        return False
    return subprocess.run(cmd, input=text.encode(), check=False).returncode == 0


def ensure_index(max_age: float = 0, quiet: bool = False):
    """Update the index unless it was checked within max_age seconds."""
    if max_age and DB_PATH.exists():
        db = connect()
        row = db.execute("SELECT value FROM meta WHERE key='checked'").fetchone()
        db.close()
        if row and time.time() - float(row[0]) < max_age:
            return
    first = not DB_PATH.exists()
    if first and not quiet:
        print("Building the search index (first run only; later runs only re-read changed files)…", file=sys.stderr)
    update_index(progress=not quiet)


# ---------------------------------------------------------------- terminal output

C = {"b": "\x1b[1m", "d": "\x1b[2m", "i": "\x1b[3m", "r": "\x1b[0m", "cy": "\x1b[36m", "ma": "\x1b[35m",
     "ye": "\x1b[33m", "gr": "\x1b[32m", "re": "\x1b[31m", "bl": "\x1b[34m", "hl": "\x1b[30;43m"}
MATCH_MARK = "◆"


def ansi_snippet(s: str) -> str:
    return s.replace("\x02", C["hl"]).replace("\x03", C["r"])


def highlight(text: str, pats: list[str], reset: str = "") -> str:
    if not pats:
        return text
    rx = re.compile("|".join(f"(?:{p})" for p in pats), re.I)
    return rx.sub(lambda m: C["hl"] + m.group(0) + C["r"] + reset, text)


def kind_label(kind, tool=None, agent=None) -> str:
    if agent and kind in ("user", "assistant"):
        return "Subagent prompt" if kind == "user" else "Subagent"
    lab = KIND_LABEL.get(kind, kind)
    if kind in ("tool_use", "tool_result") and tool:
        lab = f"{tool} {'call' if kind == 'tool_use' else 'output'}"
    return lab + (" · subagent" if agent else "")


def plural(n, word: str) -> str:
    n = n or 0
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def size_line(r, agents=()) -> str:
    """'12 prompts · 34 replies · 230 tool calls · 106k tokens (peak context 98k) · 3 subagents (85k tokens)'"""
    parts = [plural(r["n_prompts"], "prompt"), f"{r['n_replies'] or 0} repl{'y' if r['n_replies'] == 1 else 'ies'}",
             plural(r["n_tools"], "tool call"),
             f"{fmt_tok(r['tokens'])} tokens" + (f" (peak context {fmt_tok(r['peak_ctx'])})" if r["peak_ctx"] else "")]
    if agents:
        parts.append(f"{plural(len(agents), 'subagent')} ({fmt_tok(sum(a['tokens'] or 0 for a in agents))} tokens)")
    return " · ".join(parts)


def print_header(r, agents=(), out=sys.stdout):
    cwd = r["cwd"]
    w = out.write
    w(f"{C['b']}{r['title']}{C['r']}\n")
    w(f"{C['cy']}{cwd or r['projdir']}{C['r']}" + (f"  {C['d']}⎇ {r['branch']}{C['r']}" if r["branch"] else "") + "\n")
    w(f"{C['d']}{fmt_ts(r['started'])} → {fmt_ts(r['ended'])} ({ago(r['ended'])})"
      + (f" · {r['model']}" if r["model"] else "") + f"{C['r']}\n")
    w(f"{size_line(r, agents)}\n")
    w(f"{C['ye']}{resume_cmd(cwd, r['sid'])}{C['r']}\n")
    if not os.path.exists(r["path"]):
        w(f"{C['re']}⚠ the main transcript was deleted, so this session can't be resumed "
          f"(its subagent transcripts are still readable){C['r']}\n")
    elif cwd and not os.path.isdir(cwd):
        w(f"{C['re']}⚠ that directory no longer exists{C['r']}\n")
    w(f"{C['d']}{r['path']}{C['r']}\n")


def render_transcript(entries, pats, hit_set, show_tools="auto", show_thinking=False, out=sys.stdout):
    """Write a readable transcript. Messages containing a match get MATCH_MARK in
    their header so the pager can jump between them."""
    rx = re.compile("|".join(f"(?:{p})" for p in pats), re.I) if pats else None
    w = out.write
    for e in entries:
        k = e["kind"]
        hit = e["line"] in hit_set or (e.get("result_line") in hit_set)
        mark = f" {C['ye']}{MATCH_MARK} match{C['r']}" if hit else ""
        when = f"{C['d']}{fmt_ts(e['ts'], '%Y-%m-%d %H:%M')}{C['r']}"
        if k == "tool":
            if show_tools == "none" and not hit:
                continue
            color = C["re"] if e.get("is_error") else C["d"]
            w(f"  {color}⏺ {e['tool']}{C['r']} {highlight(one_line(e['summary'], 150), pats)}{mark}\n")
            body = e.get("result") or ""
            if show_tools == "full" or hit:
                lines = body.splitlines()
                if not (show_tools == "full"):
                    # Show a window around the first match instead of the whole output.
                    first = next((i for i, ln in enumerate(lines) if rx and rx.search(ln)), 0)
                    lo = max(0, first - 8)
                    lines = (["…"] if lo else []) + lines[lo:lo + 30] + (["…"] if lo + 30 < len(lines) else [])
                inp = e.get("input")
                if hit and isinstance(inp, dict) and rx:
                    for kk, vv in inp.items():
                        if isinstance(vv, str) and rx.search(vv) and vv.strip() != e["summary"]:
                            for ln in vv.splitlines()[:30]:
                                w(f"    {C['bl']}│{C['r']} {highlight(ln, pats)}\n")
                for ln in lines:
                    w(f"    {C['d']}│{C['r']} {highlight(ln, pats)}\n")
            continue
        if k == "thinking" and not (show_thinking or hit):
            continue
        if k == "image":
            w(f"  {C['d']}[image]{C['r']}\n")
            continue
        label = {"user": f"{C['b']}{C['cy']}▌ You", "assistant": f"{C['b']}{C['ma']}▌ Claude",
                 "thinking": f"{C['d']}{C['i']}▌ Thinking", "output": f"{C['gr']}▌ Output",
                 "summary": f"{C['bl']}▌ Compaction summary"}.get(k, f"{C['d']}▌ {k}")
        w(f"\n{label}{C['r']} {when}{mark}\n")
        if e.get("parts"):
            for part in e["parts"]:
                if not part["paste"]:
                    for ln in part["text"].rstrip().splitlines():
                        w(f"  {highlight(ln, pats)}\n")
                    continue
                # Pastes stay collapsed to one line unless they contain a match.
                lines = part["text"].splitlines()
                first = next((ln.strip() for ln in lines if ln.strip()), "")
                expand = bool(rx and rx.search(part["text"]))
                preview = "" if expand else " " + highlight(one_line(first, 100), pats)
                w(f"  {C['d']}┃ {_paste_label(part['lines'])}{C['r']}{preview}\n")
                if expand:
                    i0 = next(i for i, ln in enumerate(lines) if rx.search(ln))
                    lo = max(0, i0 - 8)
                    window = (["…"] if lo else []) + lines[lo:lo + 30] + (["…"] if lo + 30 < len(lines) else [])
                    for ln in window:
                        w(f"  {C['d']}┃{C['r']}   {highlight(ln, pats)}\n")
            continue
        style = C["d"] if k in ("thinking", "system", "output") else ""
        for ln in e["text"].rstrip().splitlines():
            w(f"  {style}{highlight(ln, pats, style)}{C['r'] if style else ''}\n")


def cmd_show(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="ccfind show", description="Print a session transcript.")
    ap.add_argument("sid")
    ap.add_argument("-q", "--query", default="", help="highlight and jump to matches of this search")
    ap.add_argument("-a", "--agent", default=None, help="show this subagent's transcript")
    ap.add_argument("-t", "--tools", choices=["none", "auto", "full"], default="auto",
                    help="tool calls: none, auto (one line, matches expanded), full")
    ap.add_argument("--thinking", action="store_true", help="include thinking blocks")
    ap.add_argument("--pager", action="store_true", help="open in less, jumping to the first match")
    ap.add_argument("--no-color", action="store_true")
    a = ap.parse_args(argv)
    ensure_index(max_age=60, quiet=True)
    db = connect()
    r = get_session(db, a.sid)
    if r is None:
        sys.exit(f"ccfind: no session {a.sid}")
    q = Query(a.query)
    pats = q.term_patterns()
    hits = hit_lines(db, q, r["sid"])
    files = session_files(db, r["sid"])
    agents = [f for f in files if f["agent"]]

    import io
    buf = io.StringIO()
    print_header(r, agents, out=buf)
    hit_agents = []
    if a.agent:
        f = next((f for f in agents if f["agent"].endswith(a.agent)), None)
        if f is None:
            sys.exit(f"ccfind: no subagent {a.agent}")
        targets = [f]
    else:
        targets = [{"path": r["path"], "agent": "", "agent_desc": None, "agent_type": None}]
        hit_agents = sorted({h["agent"] for h in hits if h["agent"]})
        if not os.path.exists(r["path"]):
            hit_agents = [f["agent"] for f in agents]
        targets += [f for f in agents if f["agent"] in hit_agents]
    if hits:
        buf.write(f"{C['ye']}{len(hits)} matching messages — in less, press n / N to jump between {MATCH_MARK} marks"
                  f"{C['r']}\n")
    for f in targets:
        if f["agent"]:
            buf.write(f"\n{C['b']}{C['bl']}{'═' * 20} Subagent: {f['agent_desc'] or f['agent']}"
                      f" ({f['agent_type'] or 'agent'}) {'═' * 20}{C['r']}\n")
        if not os.path.exists(f["path"]):
            buf.write(f"{C['re']}(transcript file is gone: {f['path']}){C['r']}\n")
            continue
        _, entries = build_transcript(f["path"])
        hit_set = {h["line"] for h in hits if h["agent"] == f["agent"]}
        render_transcript(entries, pats, hit_set, a.tools, a.thinking, out=buf)
    text = buf.getvalue()
    if a.no_color or (not a.pager and not sys.stdout.isatty()):
        text = ANSI_RE.sub("", text)
    if a.pager:
        cmd = ["less", "-R", "-j4"] + ([f"+/{MATCH_MARK}"] if hits else [])
        try:
            subprocess.run(cmd, input=text.encode(), check=False)
        except FileNotFoundError:
            sys.stdout.write(text)
    else:
        try:
            sys.stdout.write(text)
        except BrokenPipeError:
            pass


def cmd_search(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="ccfind search", usage="ccfind search [-n N] [--json] query…",
                                 description="Search sessions and print results.",
                                 epilog=Query.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--limit", type=int, default=15)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    # Anything argparse doesn't know is query text, so "-word" exclusions pass through.
    a, words = ap.parse_known_args(argv)
    # A shell-quoted argument with spaces in it was meant as a phrase, unless it
    # holds query syntax (filters like prompts:5, -exclusions, quotes, OR).
    def phrase(w):
        return (re.search(r"\s", w) and '"' not in w and " OR " not in w
                and not re.search(r"(^|\s)(-\w|[A-Za-z]+:\S)", w))
    words = [f'"{w}"' if phrase(w) else w for w in words]
    ensure_index(max_age=30, quiet=a.json)
    db = connect()
    q = Query(" ".join(words))
    res = search(db, q, limit=a.limit)
    if a.json:
        for s in res["sessions"]:
            for sn in s["snippets"]:
                sn["snippet"] = sn["snippet"].replace("\x02", "").replace("\x03", "")
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return
    if res.get("error"):
        sys.exit(f"ccfind: bad query: {res['error']}")
    color = sys.stdout.isatty()
    lines = [f"{C['d']}{res['total']} sessions · {res['ms']} ms{C['r']}"]
    for s in res["sessions"]:
        lines.append("")
        lines.append(f"{C['b']}{s['title']}{C['r']}  {C['d']}{s['hits']} hits{C['r']}" if s["hits"]
                     else f"{C['b']}{s['title']}{C['r']}")
        lines.append(f"  {C['cy']}{s['project']}{C['r']}  {C['d']}{fmt_ts(s['ended'])} ({ago(s['ended'])}) · "
                     f"{plural(s['n_prompts'], 'prompt')} · {fmt_tok(s['tokens'])} tokens{C['r']}")
        for sn in s["snippets"]:
            lines.append(f"  {C['d']}[{kind_label(sn['kind'], sn['tool'], sn['agent'])}]{C['r']} {ansi_snippet(sn['snippet'])}")
        lines.append(f"  {C['ye']}{s['resume_cmd']}{C['r']}")
    text = "\n".join(lines) + "\n"
    sys.stdout.write(text if color else ANSI_RE.sub("", text).replace("\x02", "").replace("\x03", ""))


# ---------------------------------------------------------------- fzf TUI

def _fzf_list(argv):
    q = Query(" ".join(argv))
    db = connect()
    res = search(db, q, limit=400, n_snippets=0)
    if res.get("error"):
        print(f"\t{C['re']}query error: {res['error']}{C['r']}")
        return
    out = []
    for s in res["sessions"]:
        hits = f"  {C['ye']}{s['hits']}×{C['r']}" if s["hits"] else ""
        agents = f" {C['d']}+{s['n_agents']} subagents{C['r']}" if s.get("n_agents") else ""
        size = f"{s['n_prompts']:>4} msg {fmt_tok(s['tokens']):>5} tok"
        size = f"{C['d']}{size}{C['r']}" if s["n_prompts"] < 3 else size
        out.append(f"{s['sid']}\t{C['d']}{fmt_ts(s['ended'], '%Y-%m-%d %H:%M')}{C['r']}  {size}  "
                   f"{C['cy']}{one_line(s['project'], 30):<30}{C['r']}  {s['title']}{agents}{hits}")
    try:
        sys.stdout.write("\n".join(out) + ("\n" if out else ""))
    except BrokenPipeError:
        pass


def _fzf_preview(argv):
    sid, query = argv[0], " ".join(argv[1:])
    if not sid:
        return
    db = connect()
    r = get_session(db, sid)
    if r is None:
        return
    q = Query(query)
    print_header(r, [f for f in session_files(db, r["sid"]) if f["agent"]])
    print(f"{C['d']}{'─' * 60}{C['r']}")
    if q.fts:
        snips = snippets_for(db, q, [r["sid"]], limit=40)[r["sid"]]
        total = len(hit_lines(db, q, r["sid"]))
        print(f"{C['b']}{total} matching messages{C['r']}" + (" (top 40)" if total > 40 else ""))
        for sn in snips:
            print(f"{C['d']}{fmt_ts(sn['ts'], '%m-%d %H:%M')} [{kind_label(sn['kind'], sn['tool'], sn['agent'])}]"
                  f"{C['r']} {ansi_snippet(sn['snippet'])}")
    else:
        prompts = db.execute("SELECT ts, body FROM msgs WHERE sid = ? AND kind = 'user' AND agent = '' "
                             "ORDER BY line LIMIT 30", (r["sid"],)).fetchall()
        print(f"{C['b']}Your prompts{C['r']}")
        for p in prompts:
            print(f"{C['d']}{fmt_ts(p['ts'], '%m-%d %H:%M')}{C['r']} {one_line(p['body'], 300)}")


FZF_HELP = """\
enter  read transcript (n/N in less jumps between matches)   ctrl-y  copy resume command
alt-i  copy session id   alt-p  copy file path   alt-w  open in web viewer   ctrl-/  toggle preview
filters: p:<project> in:you|claude|thinking|tools since:2w before:2026-05-01 prompts:5 tokens:50k sort:new|big
         -word  "exact phrase"  a OR b          columns: last active · your prompts · tokens · project · title"""


def cmd_tui(argv):
    if not shutil.which("fzf"):
        sys.exit("ccfind: fzf not found; use `ccfind search <query>` or `ccfind web`")
    ensure_index()
    me = f"{shlex.quote(sys.executable)} {shlex.quote(str(SCRIPT))}"
    header = FZF_HELP
    copied = lambda what: f"+change-header(✓ copied {what}\n{header})"
    args = [
        "fzf", "--ansi", "--disabled", "--no-sort", "--layout=reverse", "--info=inline-right",
        "--delimiter=\t", "--with-nth=2..", "--prompt=ccfind> ", "--header", header, "--header-first",
        "--query", " ".join(argv),
        "--bind", f"start:reload({me} _list {{q}})",
        "--bind", f"change:reload(sleep 0.12; {me} _list {{q}})+change-header({header})",
        "--preview", f"{me} _preview {{1}} {{q}}",
        "--preview-window", "right,55%,wrap,<110(down,55%,wrap)",
        "--bind", f"enter:execute({me} show {{1}} -q {{q}} --pager)",
        "--bind", f"ctrl-y:execute-silent({me} copy cmd {{1}}){copied('resume command')}",
        "--bind", f"alt-i:execute-silent({me} copy id {{1}}){copied('session id')}",
        "--bind", f"alt-p:execute-silent({me} copy path {{1}}){copied('file path')}",
        "--bind", f"alt-w:execute-silent({me} web --background --open {{1}} --query {{q}}){copied('nothing — opened browser')}",
        "--bind", "ctrl-/:change-preview-window(hidden|)",
    ]
    subprocess.run(args, check=False)


def cmd_copy(argv):
    what, sid = argv[0], argv[1]
    db = connect()
    r = get_session(db, sid)
    if r is None:
        sys.exit(1)
    text = {"cmd": resume_cmd(r["cwd"], r["sid"]), "id": r["sid"], "path": r["path"]}[what]
    if not copy_to_clipboard(text):
        print(text)


def cmd_cmd(argv):
    db = connect()
    for sid in argv:
        r = get_session(db, sid)
        print(resume_cmd(r["cwd"], r["sid"]) if r else f"# no session {sid}")


# ---------------------------------------------------------------- web UI

def cmd_web(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="ccfind web", description="Read-only web viewer on localhost.")
    ap.add_argument("--port", type=int, default=int(os.environ.get("CCFIND_PORT", 8977)))
    ap.add_argument("--no-open", action="store_true", help="don't open a browser")
    ap.add_argument("--open", metavar="SID", default=None, help="open this session")
    ap.add_argument("--query", default="")
    ap.add_argument("--background", action="store_true",
                    help="detach; exit after 30 idle minutes (used by the TUI's alt-w)")
    a = ap.parse_args(argv)

    from urllib.parse import urlencode
    params = {k: v for k, v in (("q", a.query), ("sid", a.open)) if v}
    url = f"http://127.0.0.1:{a.port}/" + ("?" + urlencode(params) if params else "")

    if _server_alive(a.port):
        if not a.no_open:
            _open_browser(url)
        else:
            print(f"ccfind web is already running at {url}")
        return
    if a.background:
        cmd = [sys.executable, str(SCRIPT), "web", "--port", str(a.port), "--idle-exit", "1800"]
        if a.no_open:
            cmd.append("--no-open")
        else:
            cmd += ["--open-url", url]
        subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
        return
    serve(a.port, open_url=None if a.no_open else url)


def _server_alive(port: int) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ping", timeout=0.5) as r:
            return r.read() == b"ccfind"
    except OSError:
        return False


def _open_browser(url: str):
    import webbrowser
    webbrowser.open(url)


def serve(port: int, open_url: str | None = None, idle_exit: float = 0):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlparse

    ensure_index(max_age=10)
    state = {"last_request": time.time(), "last_refresh": time.time()}
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    def maybe_refresh():
        if time.time() - state["last_refresh"] > 20 and not _index_lock.locked():
            state["last_refresh"] = time.time()
            threading.Thread(target=update_index, kwargs={"wait": False}, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send(self, code, body: bytes, ctype="application/json; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode())

        def do_GET(self):
            state["last_request"] = time.time()
            # Only answer to our own origin: blocks DNS-rebinding pages from reading transcripts.
            if self.headers.get("Host") not in allowed_hosts:
                return self._send(403, b"forbidden", "text/plain")
            u = urlparse(self.path)
            qs = {k: v[-1] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/":
                    return self._send(200, WEB_HTML.read_bytes(), "text/html; charset=utf-8")
                if u.path == "/api/ping":
                    return self._send(200, b"ccfind", "text/plain")
                maybe_refresh()
                db = connect()
                try:
                    if u.path == "/api/search":
                        return self._json(self._search(db, qs))
                    if u.path == "/api/session":
                        return self._json(self._session(db, qs))
                    if u.path == "/api/projects":
                        return self._json(project_options(project_tree(db)))
                    if u.path == "/api/stats":
                        return self._json(stats(db))
                finally:
                    db.close()
                return self._send(404, b"not found", "text/plain")
            except Exception as e:  # keep serving; show the error in the page
                return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def _search(self, db, qs):
            q = Query(qs.get("q", ""),
                      kinds=[k for k in qs.get("kinds", "").split(",") if k],
                      since=parse_when(qs["since"]) if qs.get("since") else None,
                      before=parse_when(qs["before"]) if qs.get("before") else None,
                      sort=SORT_ALIASES.get(qs.get("sort") or ""),
                      min_prompts=int(qs.get("min_prompts") or 0) or None,
                      min_tokens=int(qs.get("min_tokens") or 0) or None,
                      artifacts=True if qs.get("artifacts") == "1" else None)
            if qs.get("agents") == "0":
                q.agents = False
            if qs.get("project"):
                # A project that's no longer indexed matches nothing, not everything.
                q.projdirs = project_projdirs(project_tree(db), qs["project"]) or [""]
            res = search(db, q, limit=int(qs.get("limit", 40)), offset=int(qs.get("offset", 0)))
            res["patterns"] = q.term_patterns()
            for s in res["sessions"]:
                s["ago"] = ago(s["ended"])
            return res

        def _session(self, db, qs):
            r = get_session(db, qs.get("sid", ""))
            if r is None:
                return {"error": "no such session"}
            q = Query(qs.get("q", ""), kinds=[k for k in qs.get("kinds", "").split(",") if k])
            agent = qs.get("agent", "")
            files = session_files(db, r["sid"])
            path = r["path"]
            if agent:
                f = next((f for f in files if f["agent"] == agent), None)
                if f is None:
                    return {"error": "no such subagent"}
                path = f["path"]
            hits = hit_lines(db, q, r["sid"])
            entries = []
            if os.path.exists(path):
                _, entries = build_transcript(path)
            hit_counts = {}
            for h in hits:
                hit_counts[h["agent"]] = hit_counts.get(h["agent"], 0) + 1
            return {
                "session": _session_dict(r, {"ago": ago(r["ended"])}),
                "agent": agent, "path": path, "exists": os.path.exists(path),
                "agents": [{"agent": f["agent"], "type": f["agent_type"], "desc": f["agent_desc"],
                            "tokens": f["tokens"] or 0, "hits": hit_counts.get(f["agent"], 0)}
                           for f in files if f["agent"]],
                "hit_lines": sorted({h["line"] for h in hits if h["agent"] == agent}),
                "main_hits": hit_counts.get("", 0),
                "patterns": q.term_patterns(),
                "artifacts": collect_artifacts(entries),
                "entries": entries,
            }

        def do_POST(self):
            self._send(405, b"read-only", "text/plain")

        do_PUT = do_DELETE = do_PATCH = do_POST

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    httpd.daemon_threads = True
    print(f"ccfind web: http://127.0.0.1:{port}/  (read-only; Ctrl-C to stop)", file=sys.stderr)
    if open_url:
        threading.Timer(0.3, _open_browser, args=(open_url,)).start()
    if idle_exit:
        def watchdog():
            while True:
                time.sleep(30)
                if time.time() - state["last_request"] > idle_exit:
                    httpd.shutdown()
                    return
        threading.Thread(target=watchdog, daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


def stats(db) -> dict:
    g = lambda sql: db.execute(sql).fetchone()[0]
    checked = db.execute("SELECT value FROM meta WHERE key='checked'").fetchone()
    return {
        "sessions": g("SELECT COUNT(*) FROM sessions"),
        "files": g("SELECT COUNT(*) FROM files"),
        "subagent_files": g("SELECT COUNT(*) FROM files WHERE agent != ''"),
        "messages": g("SELECT COUNT(*) FROM msgs"),
        "oldest": g("SELECT MIN(started) FROM sessions"),
        "newest": g("SELECT MAX(ended) FROM sessions"),
        "db_bytes": DB_PATH.stat().st_size + (os.path.getsize(str(DB_PATH) + "-wal")
                                              if os.path.exists(str(DB_PATH) + "-wal") else 0),
        "db_path": str(DB_PATH),
        "checked": float(checked[0]) if checked else None,
    }


# ---------------------------------------------------------------- main

USAGE = f"""\
ccfind: read-only search for Claude Code sessions in {PROJECTS}

  ccfind [query…]            interactive search (fzf)
  ccfind web [--port N]      browser UI on http://127.0.0.1:8977
  ccfind search <query…>     print matching sessions (--json for scripts)
  ccfind show <sid> [-q Q]   print a transcript (--pager, --tools full, --thinking)
  ccfind cmd <sid>           print the resume command
  ccfind index [--rebuild]   update the index (runs automatically)
  ccfind stats               index size and coverage

{Query.__doc__}"""


def main():
    os.umask(0o077)
    argv = sys.argv[1:]
    if argv and argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return
    sub = argv[0] if argv else ""
    rest = argv[1:]
    if sub == "search":
        cmd_search(rest)
    elif sub == "show":
        cmd_show(rest)
    elif sub == "web":
        if "--idle-exit" in rest:   # internal: detached server started by --background
            i = rest.index("--idle-exit")
            idle = float(rest[i + 1])
            port = int(rest[rest.index("--port") + 1]) if "--port" in rest else 8977
            url = rest[rest.index("--open-url") + 1] if "--open-url" in rest else None
            serve(port, open_url=url, idle_exit=idle)
        else:
            cmd_web(rest)
    elif sub == "cmd":
        cmd_cmd(rest)
    elif sub == "copy":
        cmd_copy(rest)
    elif sub == "index":
        if "--rebuild" in rest and DB_PATH.exists():
            for suffix in ("", "-wal", "-shm"):
                p = Path(str(DB_PATH) + suffix)
                if p.exists():
                    p.unlink()
        update_index(progress=True)
    elif sub == "stats":
        ensure_index(max_age=60, quiet=True)
        s = stats(connect())
        for k, v in s.items():
            if k == "db_bytes":
                v = f"{v / 1e6:.0f} MB"
            elif k == "checked" and v:
                v = datetime.fromtimestamp(v).strftime("%Y-%m-%d %H:%M:%S")
            elif k in ("oldest", "newest"):
                v = fmt_ts(v)
            print(f"{k:>15}: {v}")
    elif sub == "_list":
        _fzf_list(rest)
    elif sub == "_preview":
        _fzf_preview(rest)
    else:
        cmd_tui(argv)


if __name__ == "__main__":
    main()
