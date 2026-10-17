"""Read public progress and results from this repository's local Codex sessions."""

from __future__ import annotations

from collections import deque
from contextlib import closing
import datetime as dt
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import threading
import time


PREFIX = "session:"
PRIVATE = {".git", ".codex", ".venv", ".python", ".runtime", "node_modules", "__pycache__"}
RESULT_ROOTS = ("train/experiments", "experiments", "results", "outputs", "artifacts")


def local_path(value) -> Path:
    value = str(value)
    if value.startswith("\\\\?\\"):
        value = value[4:]
    return Path(value).resolve()


def redact(text: str) -> str:
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}", "[REDACTED]", text)
    text = re.sub(r"(?i)(authorization\s*[:=]\s*bearer\s+)\S+", r"\1[REDACTED]", text)
    return re.sub(r"(?i)((?:api[_-]?key|access[_-]?token|password|secret)\s*[=:]\s*)[^\s,;]+", r"\1[REDACTED]", text)


def public_text(payload: dict) -> str:
    return "\n".join(item.get("text", "") for item in payload.get("content", [])
                     if isinstance(item, dict) and item.get("type") in {"output_text", "text"})


class CodexSessions:
    def __init__(self, root: Path, home: Path | None = None):
        self.root = local_path(root)
        self.home = local_path(home or os.environ.get("CODEX_HOME", Path.home() / ".codex"))
        self._roots = [self.root]
        self._roots_at = 0.0
        self._cache = {}
        self._lock = threading.RLock()

    def roots(self) -> list[Path]:
        with self._lock:
            if time.monotonic() - self._roots_at > 15:
                roots = [self.root]
                try:
                    result = subprocess.run(["git", "-C", str(self.root), "worktree", "list", "--porcelain"],
                                            capture_output=True, text=True, encoding="utf-8", errors="replace",
                                            timeout=5, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                    if result.returncode == 0:
                        for line in result.stdout.splitlines():
                            if line.startswith("worktree "):
                                path = local_path(line[9:])
                                if path.is_dir() and path not in roots:
                                    roots.append(path)
                except (OSError, subprocess.TimeoutExpired):
                    pass
                self._roots, self._roots_at = roots, time.monotonic()
            return list(self._roots)

    def index(self) -> list[dict]:
        databases = [p for p in self.home.glob("state_*.sqlite") if p.stem[6:].isdigit()]
        if not databases:
            raise RuntimeError("Local Codex session index is unavailable.")
        database = max(databases, key=lambda p: int(p.stem[6:]))
        roots = self.roots()
        try:
            with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=3)) as db:
                db.row_factory = sqlite3.Row
                columns = {row[1] for row in db.execute("PRAGMA table_info(threads)")}
                if not {"id", "cwd", "rollout_path", "title"} <= columns:
                    raise RuntimeError("This Codex session index format is unsupported.")
                fields = [name for name in ("id", "cwd", "rollout_path", "title", "name", "source", "archived", "updated_at") if name in columns]
                rows = db.execute("SELECT " + ",".join(fields) + " FROM threads" +
                                  (" WHERE archived=0" if "archived" in columns else "")).fetchall()
        except sqlite3.Error:
            raise RuntimeError("Local Codex session index could not be read; retry shortly.") from None
        sessions = []
        for value in rows:
            row = dict(value)
            if not row.get("cwd") or not row.get("rollout_path") or "subagent" in (row.get("source") or ""):
                continue
            cwd, rollout = local_path(row["cwd"]), local_path(row["rollout_path"])
            root = next((p for p in roots if cwd.is_relative_to(p)), None)
            if root is None or not any(rollout.is_relative_to(self.home / p) for p in ("sessions", "archived_sessions")):
                continue
            if not rollout.is_file():
                continue
            row.update(root=root, rollout=rollout, cwd=str(cwd), title=row.get("name") or row["title"])
            sessions.append(row)
        return sorted(sessions, key=lambda row: row.get("updated_at", 0), reverse=True)

    def row(self, task_id: str) -> dict:
        if not task_id.startswith(PREFIX):
            raise ValueError("Use a session task_id returned by list_tasks.")
        session_id = task_id[len(PREFIX):]
        row = next((row for row in self.index() if row["id"] == session_id), None)
        if row is None:
            raise ValueError("This local session is not an available ARBoids session.")
        return row

    def _references(self, state: dict, text: str):
        # Only validated local repository paths are retained; transcript files stay private.
        candidates = re.findall(r"`([^`\n]{1,500})`|\]\(([^)\n]{1,500})\)", text)
        for parts in candidates:
            candidate = next(part for part in parts if part).strip("<>")
            candidate = re.sub(r":\d+(?::\d+)?$", "", candidate)
            if candidate.startswith("/") and re.match(r"/[A-Za-z]:[/\\]", candidate):
                candidate = candidate[1:]
            try:
                path = local_path(state["root"] / candidate)
                relative = path.relative_to(state["root"])
                if not any(part.lower() in PRIVATE or part.lower().startswith(".env") for part in relative.parts) and path.is_file():
                    state["references"].add(relative.as_posix())
            except (OSError, ValueError):
                continue

    def _event(self, state: dict, seq: int, at: str, kind: str, text: str):
        text = redact(text)
        self._references(state, text)
        state["events"].append({"seq": seq, "at": at, "kind": kind, "detail": text[:5000]})

    def _consume(self, state: dict, record: dict, seq: int):
        kind, payload = record.get("type"), record.get("payload", {})
        if not isinstance(payload, dict):
            return
        at, event_type = record.get("timestamp", ""), payload.get("type")
        if kind == "event_msg":
            if event_type == "task_started":
                state.update(status="running", started_at=at, finished_at=None, final_message=None,
                             current_action="Codex turn started", progress={})
                self._event(state, seq, at, "started", "Codex turn started")
            elif event_type in {"task_complete", "turn_aborted", "task_failed"}:
                status = {"task_complete": "completed", "turn_aborted": "interrupted", "task_failed": "failed"}[event_type]
                state.update(status=status, finished_at=at, current_action=f"Latest Codex turn {status}")
                message = payload.get("last_agent_message")
                if isinstance(message, str) and message:
                    state["final_message"] = redact(message)[:24000]
                    self._references(state, message)
                self._event(state, seq, at, status, state["current_action"])
            elif event_type == "agent_message" and payload.get("phase") in {None, "commentary", "final_answer"}:
                self._message(state, seq, at, payload.get("phase"), payload.get("message", ""))
            if event_type in {"task_started", "task_complete", "turn_aborted", "task_failed", "token_count", "agent_message"}:
                state["last_activity_at"] = at
        elif kind == "response_item":
            # Reasoning, instructions, user messages, and raw tool outputs are never exposed.
            if event_type == "message" and payload.get("role") == "assistant" and payload.get("phase") in {None, "commentary", "final_answer"}:
                self._message(state, seq, at, payload.get("phase"), public_text(payload))
            elif event_type in {"function_call", "custom_tool_call"}:
                name = payload.get("name", "tool")
                if name != "exec":
                    state["current_action"] = f"Using {name}"
                self._event(state, seq, at, "tool", f"Using {name}")
            elif event_type in {"function_call_output", "custom_tool_call_output"}:
                output = payload.get("output", "")
                if isinstance(output, str):
                    # Export only explicitly labelled experiment progress, never raw command output.
                    lines = [line for line in output.splitlines() if re.match(r"^\[(?:PROGRESS|EVAL)\]\s", line) and len(line) < 1000]
                    if lines:
                        state["progress"]["latest_output"] = redact(lines[-1])
                        self._event(state, seq, at, "experiment_progress", "\n".join(lines[-5:]))
            if event_type not in {"reasoning", "message"} or payload.get("role") == "assistant":
                state["last_activity_at"] = at

    def _message(self, state, seq, at, phase, message):
        if not isinstance(message, str) or not message:
            return
        if phase == "final_answer":
            state["final_message"] = redact(message)[:24000]
            state.update(status="completed", finished_at=at)
        else:
            state["current_action"] = redact(message)[:2000]
        self._event(state, seq, at, "final_message" if phase == "final_answer" else "agent_message", message)

    def _read(self, row: dict) -> dict:
        with self._lock:
            path = row["rollout"]
            stat = path.stat()
            state = self._cache.get(row["id"])
            if state is None or state["position"] > stat.st_size or state["file_identity"] != (str(path), stat.st_ino):
                state = {"root": row["root"], "position": 0, "file_identity": (str(path), stat.st_ino),
                         "events": deque(maxlen=200), "references": set(), "status": "unknown", "current_action": "No public activity observed",
                         "final_message": None, "progress": {}, "last_activity_at": None, "started_at": None, "finished_at": None}
                self._cache[row["id"]] = state
                start = max(0, stat.st_size - 2 * 1024 * 1024)
            else:
                start = state["position"]
            with path.open("rb") as stream:
                stream.seek(start)
                if start and not state["position"]:
                    stream.readline()
                    state["position"] = stream.tell()
                # Capture this snapshot only; an actively growing transcript cannot extend the read indefinitely.
                while stream.tell() < stat.st_size:
                    line = stream.readline(10 * 1024 * 1024)
                    if not line.endswith(b"\n"):
                        break
                    state["position"] = stream.tell()
                    try:
                        record = json.loads(line)
                    except (ValueError, UnicodeError):
                        continue
                    self._consume(state, record, state["position"])
            return {**state, "events": list(state["events"]), "references": set(state["references"]), "progress": dict(state["progress"])}

    def snapshot(self, task_id: str, row: dict | None = None, compact=False) -> dict:
        row = row or self.row(task_id)
        state = self._read(row)
        status = state["status"]
        activity_age = None
        if state["last_activity_at"]:
            try:
                activity_age = max(0, (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(state["last_activity_at"].replace("Z", "+00:00"))).total_seconds())
            except ValueError:
                pass
        if status == "running" and (activity_age is None or activity_age > 300):
            status = "unknown"
        references = sorted(state["references"])
        artifacts = []
        for relative in references[-30:]:
            path = row["root"] / relative
            try:
                artifacts.append({"path": relative, "bytes": path.stat().st_size})
            except OSError:
                continue
        return {"task_id": PREFIX + row["id"], "thread_id": row["id"], "title": redact(row["title"]),
                "kind": "codex_session", "source": "local_codex_session", "read_only": True,
                "project": str(row["root"]), "cwd": row["cwd"], "status": status,
                "status_scope": "latest_codex_turn", "status_source": "local_transcript", "live_process_verified": False,
                "last_observed_status": state["status"], "last_activity_at": state["last_activity_at"],
                "activity_age_seconds": round(activity_age, 1) if activity_age is not None else None,
                "started_at": state["started_at"], "finished_at": state["finished_at"],
                "current_action": state["current_action"], "progress": dict(state["progress"]),
                "final_message": (state["final_message"] or "")[:1200 if compact else 24000],
                "artifacts": artifacts[:10] if compact else artifacts,
                "result_directories": [p for p in RESULT_ROOTS if (row["root"] / p).is_dir()],
                "latest_event_sequence": state["position"], "poll_after_seconds": 15,
                "note": "Latest-turn completion does not prove background or remote experiments finished. Running is observed transcript activity; stale activity is unknown. Read actual result files to verify experiments."}

    def list(self, limit=10, exclude=()) -> dict:
        rows = [row for row in self.index() if row["id"] not in exclude]
        results = [self.snapshot(PREFIX + row["id"], row, compact=True) for row in rows[:max(1, min(limit, 50))]]
        return {"sessions": results, "total_sessions": len(rows), "workspaces": [str(p) for p in self.roots()]}

    def events(self, task_id: str, after_sequence=0, limit=20) -> dict:
        state = self._read(self.row(task_id))
        events = [event for event in state["events"] if event["seq"] > after_sequence]
        events = events[-max(1, min(limit, 50)):] if after_sequence == 0 else events[:max(1, min(limit, 50))]
        return {"task_id": task_id, "events": events,
                "next_sequence": events[-1]["seq"] if events else state["position"],
                "history_limited": True, "note": "Recent public messages, tool names, and labelled experiment progress only."}
