"""Local ARBoids task bridge for ChatGPT over the official Secure MCP Tunnel."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import datetime as dt
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from typing import Any

from sessions import CodexSessions, PREFIX as SESSION_PREFIX

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_DB = HERE / ".runtime" / "tasks.sqlite3"
TERMINAL = {"completed", "failed", "cancelled", "timed_out", "interrupted"}
HIDDEN = {".git", ".venv", ".python", ".runtime", ".codex", "node_modules", "__pycache__"}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        import pywintypes
        import win32api
        import win32con
        import win32process

        try:
            handle = win32api.OpenProcess(win32con.PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            try:
                return win32process.GetExitCodeProcess(handle) == 259
            finally:
                handle.Close()
        except (OSError, pywintypes.error):
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def database_retry(function):
    """Replay rolled-back operations on a fresh connection after transient I/O."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        for attempt in range(5):
            try:
                return function(*args, **kwargs)
            except sqlite3.OperationalError as error:
                code = (getattr(error, "sqlite_errorcode", 0) or 0) & 255
                transient = code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED, sqlite3.SQLITE_IOERR}
                transient |= str(error).lower() in {"disk i/o error", "database is locked", "database table is locked"}
                if not transient or attempt == 4:
                    raise
                time.sleep(.1 * 2 ** attempt)
    return wrapped


class Store:
    def __init__(self, path: Path = DEFAULT_DB, root: Path = ROOT):
        self.path, self.root = Path(path).resolve(), Path(root).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Keep shared WAL/SHM files open across short Windows transactions.
        self._anchor = sqlite3.connect(self.path, timeout=15, check_same_thread=False)
        try:
            self._initialize()
            self._anchor.execute("SELECT count(*) FROM sqlite_master").fetchone()
        except Exception:
            self.close()
            raise

    @database_retry
    def _initialize(self):
        with self.connect() as db:
            if db.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal":
                db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY, request_id TEXT UNIQUE NOT NULL,
                    fingerprint TEXT NOT NULL, payload TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued', created_at TEXT NOT NULL,
                    started_at TEXT, finished_at TEXT, heartbeat_at TEXT,
                    worker_pid INTEGER, codex_pid INTEGER, thread_id TEXT,
                    current_action TEXT NOT NULL DEFAULT 'Waiting for local Codex',
                    final_message TEXT, error TEXT, exit_code INTEGER,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    progress TEXT NOT NULL DEFAULT '{}',
                    experiment_started INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL, at TEXT NOT NULL, kind TEXT NOT NULL,
                    detail TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS events_task ON events(task_id, seq);
                CREATE TABLE IF NOT EXISTS service_state (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                );
            """)

    def close(self):
        anchor = getattr(self, "_anchor", None)
        if anchor is not None:
            self._anchor = None
            anchor.close()

    @database_retry
    def set_context(self, key: str, value: dict):
        with self.connect() as db:
            db.execute("INSERT INTO service_state(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                       (key, json.dumps(value, ensure_ascii=False)))

    @database_retry
    def context(self) -> dict:
        with self.connect() as db:
            return {row["key"]: json.loads(row["value"]) for row in db.execute("SELECT key,value FROM service_state")}

    @contextlib.contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @database_retry
    def row(self, task_id: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown task_id. Use list_tasks to find a valid ID.")
        value = dict(row)
        value["payload"] = json.loads(value["payload"])
        value["progress"] = json.loads(value["progress"])
        return value

    @database_retry
    def update(self, task_id: str, **values):
        allowed = {"status", "started_at", "finished_at", "heartbeat_at", "worker_pid",
                   "codex_pid", "thread_id", "current_action", "final_message", "error",
                   "exit_code", "cancel_requested", "progress", "experiment_started"}
        if not values or not values.keys() <= allowed:
            raise ValueError("Invalid task update")
        if "progress" in values:
            values["progress"] = json.dumps(values["progress"], ensure_ascii=False)
        with self.connect() as db:
            db.execute("UPDATE tasks SET " + ",".join(f"{key}=?" for key in values) + " WHERE id=?",
                       (*values.values(), task_id))

    @database_retry
    def event(self, task_id: str, kind: str, detail: str):
        with self.connect() as db:
            db.execute("INSERT INTO events(task_id,at,kind,detail) VALUES(?,?,?,?)",
                       (task_id, now(), kind, detail[-16000:]))

    @database_retry
    def submit(self, payload: dict, request_id: str) -> tuple[str, bool]:
        if not request_id.strip() or len(request_id) > 160:
            raise ValueError("request_id must contain 1 to 160 characters; reuse it when retrying.")
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute("SELECT id,fingerprint FROM tasks WHERE request_id=?", (request_id,)).fetchone()
            if prior:
                if prior["fingerprint"] != fingerprint:
                    raise ValueError("This request_id already belongs to a different task.")
                return prior["id"], False
            count = db.execute("SELECT count(*) FROM tasks WHERE status IN ('queued','running')").fetchone()[0]
            if count >= 20:
                raise ValueError("The local queue is full (20 tasks). Wait for existing work to finish.")
            task_id = uuid.uuid4().hex
            db.execute("INSERT INTO tasks(id,request_id,fingerprint,payload,created_at) VALUES(?,?,?,?,?)",
                       (task_id, request_id, fingerprint, json.dumps(payload, ensure_ascii=False), now()))
            db.execute("INSERT INTO events(task_id,at,kind,detail) VALUES(?,?,?,?)",
                       (task_id, now(), "queued", "Task accepted; calls may disconnect while local work continues."))
        return task_id, True

    @database_retry
    def retry_failed(self, task_id: str, request_id: str) -> tuple[str, bool]:
        if not request_id.strip() or len(request_id) > 160:
            raise ValueError("request_id must contain 1 to 160 characters.")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
            if prior is None:
                raise ValueError("Unknown task_id.")
            progress = json.loads(prior["progress"])
            attempts = progress.setdefault("attempt_history", [])
            if any(attempt["request_id"] == request_id for attempt in attempts):
                return task_id, False
            if (prior["status"] != "failed" or prior["started_at"] or prior["thread_id"]
                    or prior["experiment_started"] or prior["cancel_requested"] or alive(prior["worker_pid"])):
                raise ValueError("Only a failed task that never started execution can be retried automatically.")
            attempt = dict(request_id=request_id, status=prior["status"], error=prior["error"],
                           finished_at=prior["finished_at"], worker_pid=prior["worker_pid"], retried_at=now())
            attempts.append(attempt)
            # Retain the original task ID and queue slot, so a dependent queued
            # continuation cannot jump ahead of its previously failed integrator.
            db.execute("UPDATE tasks SET status='queued',finished_at=NULL,error=NULL,worker_pid=NULL,exit_code=NULL,"
                       "current_action='Waiting to retry before execution',progress=? WHERE id=?",
                       (json.dumps(progress, ensure_ascii=False), task_id))
            db.execute("INSERT INTO events(task_id,at,kind,detail) VALUES(?,?,?,?)",
                       (task_id, now(), "retry_queued", json.dumps(attempt, ensure_ascii=False)))
        return task_id, True

    @database_retry
    def claim(self, task_id: str) -> bool:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            active = db.execute("SELECT id,worker_pid,codex_pid FROM tasks WHERE status='running'").fetchall()
            for row in active:
                if row['id'] == task_id and row['worker_pid'] == os.getpid() and row['codex_pid'] is None:
                    return True
                if alive(row["worker_pid"]):
                    return False
                db.execute("UPDATE tasks SET status='interrupted',finished_at=?,error=? WHERE id=?",
                           (now(), "Local worker exited before reporting completion.", row["id"]))
            first = db.execute("SELECT id FROM tasks WHERE status='queued' ORDER BY created_at,rowid LIMIT 1").fetchone()
            if first is None or first["id"] != task_id:
                return False
            db.execute("UPDATE tasks SET status='running',started_at=?,heartbeat_at=?,worker_pid=? WHERE id=?",
                       (now(), now(), os.getpid(), task_id))
        return True

    def snapshot(self, task_id: str) -> dict:
        task = self.row(task_id)
        if task["status"] == "running" and not alive(task["worker_pid"]):
            self.update(task_id, status="interrupted", finished_at=now(), error="Local worker is no longer running.")
            task = self.row(task_id)
        keys = ("id", "status", "created_at", "started_at", "finished_at", "heartbeat_at", "thread_id",
                "current_action", "final_message", "error", "exit_code", "cancel_requested", "progress")
        result = {key: task[key] for key in keys}
        result["task_id"] = result.pop("id")
        result["title"] = task["payload"]["title"]
        result["kind"] = task["payload"]["kind"]
        result["project"] = str(self.root)
        if task["payload"].get("retry_of"):
            result["retry_of"] = task["payload"]["retry_of"]
        context = self.context()
        policy = context.get("execution_policy", {"mode": "read_only", "remote_execution_enabled": False})
        read_only = policy.get("remote_execution_enabled") is not True
        result["execution_policy"] = policy
        repair = task["progress"].get("execution_repair") or context.get("execution_repair")
        if repair:
            result["execution_repair"] = {
                "status": repair.get("status"), "updated_at": repair.get("updated_at"),
                "context_tool": "get_project_context",
                "execution_order": repair.get("execution_order", []),
                "next_action": ("User stopped this execution chain. Pro provides plans and reads results only; local Codex execution is started manually by the user. Do not retry or resubmit cancelled tasks."
                                if read_only else "Read get_project_context.execution_repair once per updated_at before continuing or submitting more work. Existing running jobs must finish naturally; do not duplicate the restored integration task.")
            }
        if task["progress"].get("attempt_history"):
            result["attempt_history"] = task["progress"]["attempt_history"]
        result["poll_after_seconds"] = 15 if task["status"] not in TERMINAL else None
        with self.connect() as db:
            result["latest_event_sequence"] = db.execute("SELECT coalesce(max(seq),0) FROM events WHERE task_id=?", (task_id,)).fetchone()[0]
        if task["payload"]["kind"] == "experiment":
            folder = self.root / "train" / "experiments" / f"mcp_{task_id}"
            result["run_directory"] = folder.relative_to(self.root).as_posix()
            result["artifacts"] = [
                {"path": p.relative_to(self.root).as_posix(), "bytes": p.stat().st_size}
                for p in sorted(folder.glob("*")) if p.is_file()
            ]
            metric_files = sorted(folder.glob("metrics*.csv"), key=lambda p: p.stat().st_mtime)
            if metric_files:
                try:
                    with metric_files[-1].open(encoding="utf-8", newline="") as stream:
                        rows = list(csv.DictReader(stream))
                    result["latest_evaluation"] = rows[-1] if rows else None
                    result["metrics_file"] = metric_files[-1].relative_to(self.root).as_posix()
                except (OSError, csv.Error):
                    result["latest_evaluation"] = None
        return result


def clean_environment() -> dict:
    env = os.environ.copy()
    for name in ("CONTROL_PLANE_API_KEY", "OPENAI_API_KEY", "CODEX_THREAD_ID", "CODEX_SESSION_ID"):
        env.pop(name, None)
    env.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg")
    return env


def codex_command() -> list[str]:
    binary = shutil.which("codex.exe")
    if binary:
        return [binary]
    launcher = shutil.which("codex")
    if not launcher:
        raise RuntimeError("Codex CLI is not on PATH. Install it and sign in with ChatGPT first.")
    if os.name == "nt":
        entry = Path(launcher).parent / "node_modules" / "@openai" / "codex" / "bin" / "codex.js"
        node = shutil.which("node")
        if not node or not entry.is_file():
            raise RuntimeError("Cannot locate the installed Codex Node entry point.")
        return [node, str(entry)]
    return [launcher]


class ProcessTree:
    """A Windows job object ensures cancellation also stops training descendants."""

    def __init__(self, process: subprocess.Popen):
        self.process, self.handle = process, None
        if os.name == "nt":
            import win32job
            self.handle = win32job.CreateJobObject(None, "")
            info = win32job.QueryInformationJobObject(self.handle, win32job.JobObjectExtendedLimitInformation)
            info["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            win32job.SetInformationJobObject(self.handle, win32job.JobObjectExtendedLimitInformation, info)
            try:
                win32job.AssignProcessToJobObject(self.handle, int(process._handle))
            except Exception:
                self.handle.Close()
                self.handle = None
                process.kill()
                process.wait()
                raise

    def close(self):
        if self.handle is not None:
            self.handle.Close()
            self.handle = None
        elif os.name != "nt":
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGKILL)


def spawn_options() -> dict:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def launch_worker(store: Store, task_id: str):
    process = subprocess.Popen([sys.executable, "-X", "utf8", str(Path(__file__).resolve()),
                      "--root", str(store.root), "--db", str(store.path), "worker", task_id],
                     cwd=store.root, env=clean_environment(), stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **spawn_options())
    with store.connect() as db:
        db.execute("UPDATE tasks SET worker_pid=? WHERE id=? AND status='queued'", (process.pid, task_id))


def task_prompt(store: Store, task_id: str) -> str:
    task = store.row(task_id)
    data = task["payload"]
    repair = task["progress"].get("execution_repair") or store.context().get("execution_repair")
    prefix = (f"Work only on the authorized ARBoids project at {store.root}. "
              "Follow its AGENTS.md instructions. Use Windows PowerShell 7 with login=false; Python uses -X utf8. "
              "Complete this task and report actual checks and results. Do not make unsolicited documentation, "
              "backups, commits, or changes outside the project. Do not start other agents. "
              "The MCP bridge observes your command events and experiment progress. "
              "Use Chinese for progress updates and the final response.\n\n")
    if repair:
        prefix += "Execution repair context (verify the current files and use their actual output paths):\n" + json.dumps(repair, ensure_ascii=False) + "\n\n"
    if data["kind"] == "experiment":
        command = f'& \'{sys.executable}\' -X utf8 \'{Path(__file__).resolve()}\' --root \'{store.root}\' --db \'{store.path}\' experiment {task_id}'
        return prefix + ("Run this already configured ARBoids experiment exactly once using the following command. "
                         "Do not edit source or configurations and do not reinstall dependencies. "
                         "Use an exec yield interval, poll the process until completion, and inspect the actual metrics. "
                         "Return the step count, evaluation success rate/reward, checkpoint path, and any failure. "
                         "This is an integration smoke check, not evidence of convergence.\n\n" + command)
    return prefix + data["task"]


def handle_codex_event(store: Store, task_id: str, event: dict):
    event_type = event.get("type", "unknown")
    if event_type == "thread.started":
        store.update(task_id, thread_id=event.get("thread_id"))
    item = event.get("item", {})
    item_type = item.get("type", "")
    if item_type in {"command_execution", "mcp_tool_call", "web_search", "file_change"}:
        detail = item.get("command") or item.get("tool") or item.get("query") or item_type
        store.update(task_id, current_action=str(detail)[:4000])
        store.event(task_id, event_type, json.dumps(item, ensure_ascii=False))
    elif item_type == "agent_message" and item.get("text"):
        store.update(task_id, final_message=item["text"][-60000:])
        store.event(task_id, "agent_message", item["text"])
    elif event_type in {"error", "turn.failed"}:
        error = event.get("error")
        message = event.get("message") or (error.get("message") if isinstance(error, dict) else error) or json.dumps(event)
        store.event(task_id, event_type, str(message))
        store.update(task_id, error=str(message)[-12000:])


def run_worker(store: Store, task_id: str) -> int:
    process = tree = None
    try:
        while True:
            task = store.row(task_id)
            if task["status"] in TERMINAL:
                return 0
            if task["cancel_requested"]:
                store.update(task_id, status="cancelled", finished_at=now(), current_action="Cancelled before execution")
                return 0
            if store.claim(task_id):
                break
            time.sleep(1)
        store.event(task_id, "running", "Starting signed-in Codex CLI in the ARBoids workspace.")
        command = codex_command() + ["-a", "never", "-s", "workspace-write", "-C", str(store.root),
                                     "exec", "--json", "--color", "never", "-"]
        process = subprocess.Popen(command, cwd=store.root, env=clean_environment(),
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, encoding="utf-8", errors="replace", **spawn_options())
        tree = ProcessTree(process)
        store.update(task_id, codex_pid=process.pid, current_action="Codex is preparing the task")
        process.stdin.write(task_prompt(store, task_id))
        process.stdin.close()
        lines: queue.Queue = queue.Queue(maxsize=1000)

        def read_stream(stream, source):
            try:
                for line in stream:
                    lines.put((source, line))
            finally:
                lines.put((source, None))

        for source, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
            threading.Thread(target=read_stream, args=(stream, source), daemon=True).start()
        deadline = time.monotonic() + store.row(task_id)["payload"]["timeout_seconds"]
        ended, last_heartbeat, forced_status = set(), 0.0, None
        while len(ended) < 2 or not lines.empty():
            if time.monotonic() - last_heartbeat >= 2:
                task = store.row(task_id)
                store.update(task_id, heartbeat_at=now())
                last_heartbeat = time.monotonic()
                if task["cancel_requested"] or time.monotonic() >= deadline:
                    forced_status = "cancelled" if task["cancel_requested"] else "timed_out"
                    tree.close()
                    break
            try:
                source, line = lines.get(timeout=0.5)
            except queue.Empty:
                continue
            if line is None:
                ended.add(source)
            elif source == "stdout":
                try:
                    handle_codex_event(store, task_id, json.loads(line))
                except json.JSONDecodeError:
                    store.event(task_id, "output", line.strip())
            elif line.strip():
                store.event(task_id, "diagnostic", line.strip())
        exit_code = process.wait(timeout=15)
        task = store.row(task_id)
        status = forced_status or ("completed" if exit_code == 0 else "failed")
        if data := task["progress"].get("experiment_exit_code"):
            if status == "completed" and data != 0:
                status = "failed"
        if task["payload"]["kind"] == "experiment" and status == "completed" and "experiment_exit_code" not in task["progress"]:
            status = "failed"
            store.update(task_id, error="Codex returned without completing the requested experiment.")
        store.update(task_id, status=status, exit_code=exit_code, finished_at=now(), heartbeat_at=now(),
                     current_action=f"Task {status}")
        store.event(task_id, status, f"Codex exit code: {exit_code}")
        return exit_code
    except Exception as error:
        phase = "execution" if process else "queue_before_execution"
        diagnostic = f"{type(error).__name__}: {error}"
        if isinstance(error, sqlite3.Error):
            diagnostic += f" [{getattr(error, 'sqlite_errorname', 'SQLITE_ERROR')}; phase={phase}]"
        store.update(task_id, status="failed", error=diagnostic, finished_at=now())
        store.event(task_id, "failed", diagnostic)
        return 1
    finally:
        if tree:
            tree.close()
        if process and process.poll() is None:
            process.kill()
            process.wait()


def parse_progress(progress: dict, line: str) -> dict:
    progress = dict(progress, latest_output=line.strip()[-2000:], updated_at=now())
    match = re.search(r"\[PROGRESS\] step=(\d+)/(\d+) updates=(\d+) elapsed=([\d.]+)s", line)
    if match:
        step, total, updates, seconds = match.groups()
        progress.update(stage="training", step=int(step), total_steps=int(total),
                        updates=int(updates), elapsed_seconds=float(seconds),
                        percent=round(100 * int(step) / int(total), 1))
    match = re.search(r"\[EVAL\] step=(\d+) success_rate=([\d.]+) reward=([\d.eE+-]+)", line)
    if match:
        step, success, reward = match.groups()
        progress["evaluation"] = {"step": int(step), "success_rate": float(success), "reward": float(reward)}
    return progress


def run_experiment(store: Store, task_id: str) -> int:
    task = store.row(task_id)
    data = task["payload"]
    if data["kind"] != "experiment" or task["status"] != "running":
        raise ValueError("This task is not an active experiment.")
    with store.connect() as db:
        changed = db.execute("UPDATE tasks SET experiment_started=1 WHERE id=? AND experiment_started=0", (task_id,)).rowcount
    if not changed:
        raise ValueError("The experiment has already been started; inspect its existing result.")
    python = store.root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    script = "train/run_adversarial_loop.py" if data["mode"] == "adversarial" else "train/train.py"
    command = [str(python), "-X", "utf8", "-u", script, "--config", data["config"],
               "--device", data["device"], "--seed", str(data["seed"]), "--run-id", f"mcp_{task_id}"]
    if data["mode"] == "adversarial":
        command += ["--rounds", str(data["rounds"])]
    progress = {"stage": "starting", "config": data["config"], "device": data["device"], "seed": data["seed"]}
    store.update(task_id, progress=progress)
    store.event(task_id, "experiment_started", subprocess.list2cmdline(command))
    process = subprocess.Popen(command, cwd=store.root, env=clean_environment(), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    for line in process.stdout:
        print(line, end="", flush=True)
        progress = parse_progress(progress, line)
        if "[EVAL]" in line:
            store.event(task_id, "evaluation", line.strip())
        elif "[PROGRESS]" in line:
            store.event(task_id, "training_progress", line.strip())
        store.update(task_id, progress=progress)
    code = process.wait()
    progress.update(stage="finished" if code == 0 else "failed", experiment_exit_code=code, updated_at=now())
    store.update(task_id, progress=progress)
    store.event(task_id, "experiment_finished", f"Experiment exit code: {code}")
    return code


def safe_file(root: Path, relative_path: str) -> Path:
    requested = Path(relative_path)
    if requested.is_absolute() or any(part.lower() in HIDDEN or part.lower().startswith(".env") for part in requested.parts):
        raise ValueError("Only non-secret files inside the ARBoids project are readable.")
    target = (root / requested).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        raise ValueError("File must exist inside the ARBoids project.")
    if any(part.lower() in HIDDEN or part.lower().startswith(".env") for part in target.relative_to(root.resolve()).parts):
        raise ValueError("Private runtime and credential files are not exposed.")
    return target


def create_server(store: Store, port: int = 8787, sessions: CodexSessions | None = None):
    from mcp.server.fastmcp import FastMCP, Image
    from mcp.types import ToolAnnotations
    from starlette.responses import JSONResponse

    read = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)
    write = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=True)
    sessions = sessions or CodexSessions(store.root)
    policy = store.context().get("execution_policy", {"mode": "read_only", "remote_execution_enabled": False})
    read_only = policy.get("remote_execution_enabled") is not True

    def require_remote_execution():
        if store.context().get("execution_policy", {}).get("remote_execution_enabled") is not True:
            raise ValueError("Remote execution is disabled by the user. Pro provides plans and reads results only; start execution manually in local Codex.")

    def execution_tool(**options):
        return (lambda function: function) if read_only else server.tool(**options)

    def snapshot(task_id):
        return sessions.snapshot(task_id) if task_id.startswith(SESSION_PREFIX) else store.snapshot(task_id)

    def project_root(task_id):
        return sessions.row(task_id)["root"] if task_id.startswith(SESSION_PREFIX) else store.root

    mode_instructions = ("The user stopped Pro-managed execution. This MCP is read-only. Pro proposes plans and may inspect progress and results; the user starts execution manually in local Codex. Do not retry or recreate the cancelled task chain. "
                         if read_only else "Direct and supervise local Codex work on ARBoids. Submit new work with a unique request_id and reuse it on retries. Work continues after calls return. ")
    server = FastMCP("ARBoids Local Codex", host="127.0.0.1", port=port, stateless_http=True, json_response=True,
                     instructions=mode_instructions + "list_tasks includes existing local Codex sessions in this repository and its Git worktrees. Session task_ids start with session: and are read-only observations; their status refers to the latest Codex turn, not proof of experiment completion. Read actual result files using the session task_id to select its workspace. Read get_task_status, then use wait_for_task_update with its latest_event_sequence to follow progress. Inspect real metrics and artifacts before reporting success. Never claim completion while queued/running. An idle ChatGPT conversation does not receive automatic pushes; supervision requires tool calls.")

    @server.custom_route("/health", methods=["GET"])
    async def health(request):
        return JSONResponse({"service": "arboids-local-codex", "version": "1.3.0", "project": str(store.root), "pid": os.getpid(),
                             "execution_mode": "read_only" if read_only else "execution"})

    @server.tool(annotations=read)
    def get_project_context() -> dict[str, Any]:
        """Use this first to identify the ARBoids project, execution boundaries, and available experiment entry points."""
        return {"project": str(store.root), "executor": "manually started local Codex" if read_only else "signed-in Codex CLI", "sandbox": "read-only MCP" if read_only else "workspace-write",
                "concurrency": 0 if read_only else 1, "experiments": ["train/configs/smoke.yaml", "train/configs/paper-parameters-smoke.yaml", "train/configs/adversarial-smoke.yaml"],
                "progress": "Call get_task_status to see Codex activity, heartbeat, training steps and evaluation metrics.",
                "existing_sessions": "list_tasks also discovers non-archived local Codex sessions in this repository and registered Git worktrees. Use their session: task_id with status, event, file and artifact tools. No messages or control commands are sent to those sessions.",
                "workspaces": [str(path) for path in sessions.roots()],
                "execution_repair": store.context().get("execution_repair"),
                "execution_policy": store.context().get("execution_policy", policy),
                "instructions": ("Pro provides plans and may read progress and results. It cannot submit, retry, run or cancel tasks. Actual execution is initiated manually in local Codex."
                                 if read_only else "Tasks can edit ARBoids files and run experiments. Model calls use the existing Codex sign-in. This server has no public listener.")}

    def submit(payload: dict, request_id: str) -> dict:
        require_remote_execution()
        task_id, created = store.submit(payload, request_id)
        if created:
            try:
                launch_worker(store, task_id)
            except Exception as error:
                store.update(task_id, status="failed", error=str(error), finished_at=now())
                raise
        return store.snapshot(task_id)

    @execution_tool(annotations=write)
    def submit_codex_task(task: str, request_id: str, title: str = "ARBoids task", timeout_seconds: int = 3600) -> dict[str, Any]:
        """Use this when the user wants local Codex to implement, debug, or evaluate ARBoids. This writes files or runs commands. Returns a task_id immediately; poll its status. Provide a concrete task and unique request_id."""
        if not task.strip() or len(task) > 60000:
            raise ValueError("task must contain 1 to 60000 characters.")
        if not 30 <= timeout_seconds <= 86400:
            raise ValueError("timeout_seconds must be between 30 and 86400.")
        return submit({"kind": "codex", "task": task, "title": title[:200], "timeout_seconds": timeout_seconds}, request_id)

    @execution_tool(annotations=write)
    def retry_failed_task(task_id: str, request_id: str) -> dict[str, Any]:
        """Retry only a failed task whose execution never started. Preserves its task ID, original queue position and failure history; active work continues. Idempotent for the same request_id."""
        require_remote_execution()
        retry_id, created = store.retry_failed(task_id, request_id)
        if created:
            try:
                launch_worker(store, retry_id)
            except Exception as error:
                store.update(retry_id, status="failed", error=str(error), finished_at=now())
                raise
        return store.snapshot(retry_id)

    @execution_tool(annotations=write)
    def run_arboids_experiment(request_id: str, config: str = "train/configs/smoke.yaml", device: str = "cuda:0", seed: int = 42, mode: str = "train", rounds: int = 2, timeout_seconds: int = 3600) -> dict[str, Any]:
        """Use this to ask local Codex to run an ARBoids training experiment and return live progress plus real metrics/checkpoints. Defaults to the existing 2000-step smoke check. Runs in the background; poll get_task_status. Never interpret a smoke check as convergence."""
        target = safe_file(store.root, config)
        if not target.is_relative_to(store.root / "train" / "configs") or target.suffix not in {".yaml", ".yml"}:
            raise ValueError("config must be a YAML file under train/configs.")
        if mode not in {"train", "adversarial"} or not re.fullmatch(r"cpu|cuda(?::\d+)?", device):
            raise ValueError("mode must be train/adversarial and device must be cpu/cuda/cuda:N.")
        if not 1 <= rounds <= 20 or not 0 <= seed < 2**32 or not 30 <= timeout_seconds <= 86400:
            raise ValueError("Invalid rounds, seed or timeout_seconds.")
        return submit({"kind": "experiment", "title": f"ARBoids {mode}: {target.name}",
                       "config": target.relative_to(store.root).as_posix(), "device": device, "seed": seed,
                       "mode": mode, "rounds": rounds, "timeout_seconds": timeout_seconds}, request_id)

    @server.tool(annotations=read)
    def list_tasks(limit: int = 10) -> dict[str, Any]:
        """Find managed jobs AND existing local ARBoids Codex sessions, including Git worktrees. session: IDs are read-only; get_task_status and result tools accept these IDs too."""
        limit = max(1, min(limit, 50))
        with store.connect() as db:
            ids = db.execute("SELECT id FROM tasks ORDER BY created_at DESC,rowid DESC LIMIT ?", (limit,)).fetchall()
            managed_threads = {row[0] for row in db.execute("SELECT thread_id FROM tasks WHERE thread_id IS NOT NULL")}
        tasks = [store.snapshot(row["id"]) for row in ids]
        monitoring = {"available": True}
        try:
            discovered = sessions.list(limit, exclude=managed_threads)
            tasks.extend(discovered.pop("sessions"))
            monitoring.update(discovered)
        except (RuntimeError, OSError) as error:
            monitoring = {"available": False, "error": str(error)}
        tasks.sort(key=lambda task: task.get("last_activity_at") or task.get("started_at") or task.get("created_at") or "", reverse=True)
        return {"tasks": tasks[:limit], "session_monitoring": monitoring}

    @server.tool(annotations=read)
    def get_task_status(task_id: str) -> dict[str, Any]:
        """Use this repeatedly to supervise Codex. Returns lifecycle state, heartbeat, current command, training step/percent, latest evaluation, result summary and artifact paths. Poll every 15-30 seconds while queued/running."""
        return snapshot(task_id)

    @server.tool(annotations=read)
    async def wait_for_task_update(task_id: str, after_sequence: int = 0, timeout_seconds: int = 25) -> dict[str, Any]:
        """Use this to keep supervising a running task without busy polling. Returns on a new command/progress/evaluation event, task completion, or after at most 25 seconds. Pass latest_event_sequence from the previous status as after_sequence."""
        deadline = time.monotonic() + max(0, min(timeout_seconds, 25))
        while True:
            status = await asyncio.to_thread(snapshot, task_id)
            if status["status"] in TERMINAL or status["latest_event_sequence"] > after_sequence:
                return {"changed": True, "task": status}
            if time.monotonic() >= deadline:
                return {"changed": False, "task": status}
            await asyncio.sleep(.5)

    @server.tool(annotations=read)
    def get_task_events(task_id: str, after_sequence: int = 0, limit: int = 20) -> dict[str, Any]:
        """Use this to inspect Codex command results, diagnostics or failures. Pass next_sequence on the next call to fetch only new events."""
        if task_id.startswith(SESSION_PREFIX):
            return sessions.events(task_id, after_sequence, limit)
        store.row(task_id)
        with store.connect() as db:
            rows = db.execute("SELECT seq,at,kind,detail FROM events WHERE task_id=? AND seq>? ORDER BY seq LIMIT ?",
                              (task_id, max(0, after_sequence), max(1, min(limit, 50)))).fetchall()
        events = [dict(row) for row in rows]
        return {"task_id": task_id, "events": events, "next_sequence": events[-1]["seq"] if events else after_sequence}

    @execution_tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False))
    def cancel_task(task_id: str) -> dict[str, Any]:
        """Use this only when the user wants to stop a task. Stops Codex and its experiment subprocesses; partial artifacts remain available."""
        require_remote_execution()
        if task_id.startswith(SESSION_PREFIX):
            raise ValueError("Existing Codex sessions are observed read-only; stop them in their original Codex chat.")
        task = store.row(task_id)
        if task["status"] not in TERMINAL:
            store.update(task_id, cancel_requested=1)
            if task["status"] == "queued":
                store.update(task_id, status="cancelled", finished_at=now(), current_action="Cancelled before execution")
        return store.snapshot(task_id)

    @server.tool(annotations=read)
    def read_project_file(path: str, offset: int = 0, max_characters: int = 16000, task_id: str = "") -> str:
        """Read ARBoids code/configuration, metrics or logs. Paths are project-relative; pass a session: task_id to read that session's Git worktree. Private files are excluded. Use read_artifact for plots."""
        target = safe_file(project_root(task_id), path)
        if target.suffix.lower() not in {".py", ".yaml", ".yml", ".json", ".csv", ".txt", ".md", ".log", ".toml", ".sh", ".ps1"}:
            raise ValueError("This file type is not exposed as text.")
        if target.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("File is too large; use the task's structured status/metrics.")
        content = target.read_text(encoding="utf-8", errors="replace")
        begin = max(0, offset)
        return content[begin:begin + max(1, min(max_characters, 60000))]

    @server.tool(annotations=read)
    def list_project_files(path: str = "", task_id: str = "", limit: int = 50) -> dict[str, Any]:
        """Find result files in a project directory. Pass the session: task_id to browse its workspace and use result_directories from get_task_status as starting points."""
        root = project_root(task_id)
        target = (root / path).resolve()
        if Path(path).is_absolute() or not target.is_relative_to(root) or not target.is_dir():
            raise ValueError("Choose a directory inside the selected ARBoids workspace.")
        if any(part.lower() in HIDDEN or part.lower().startswith(".env") for part in target.relative_to(root).parts):
            raise ValueError("Private directories are not exposed.")
        files = []
        for child in target.iterdir():
            if child.name.lower() in HIDDEN or child.name.lower().startswith(".env") or child.is_symlink():
                continue
            stat = child.stat()
            files.append({"path": child.relative_to(root).as_posix(), "directory": child.is_dir(),
                          "bytes": stat.st_size if child.is_file() else None, "modified_at": dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).isoformat()})
        files.sort(key=lambda item: item["modified_at"], reverse=True)
        return {"project": str(root), "path": path, "files": files[:max(1, min(limit, 100))], "total_entries": len(files)}

    @server.tool(annotations=read, structured_output=False)
    def read_artifact(task_id: str, path: str):
        """Use this to view a plot or read a result file produced by an experiment. Use paths from get_task_status.artifacts. Model checkpoints return metadata rather than their binary content."""
        if task_id.startswith(SESSION_PREFIX):
            target = safe_file(project_root(task_id), path)
        else:
            task = store.row(task_id)
            if task["payload"]["kind"] != "experiment":
                raise ValueError("This task has no managed experiment directory.")
            target = safe_file(store.root, path)
            folder = (store.root / "train" / "experiments" / f"mcp_{task_id}").resolve()
            if not target.is_relative_to(folder):
                raise ValueError("Artifact does not belong to this task.")
        if target.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} and target.stat().st_size <= 5 * 1024 * 1024:
            return Image(path=str(target))
        if target.suffix.lower() in {".csv", ".json", ".yaml", ".yml", ".txt", ".md", ".log"} and target.stat().st_size <= 200000:
            return target.read_text(encoding="utf-8", errors="replace")
        return json.dumps({"path": path, "bytes": target.stat().st_size, "note": "Binary/large artifact is stored locally."})

    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--port", type=int, default=8787)
    serve.add_argument("--transport", choices=["streamable-http", "stdio"], default="streamable-http")
    for name in ("worker", "experiment", "status"):
        commands.add_parser(name).add_argument("task_id")
    args = parser.parse_args()
    store = Store(args.db, args.root)
    if args.command == "worker":
        return run_worker(store, args.task_id)
    if args.command == "experiment":
        return run_experiment(store, args.task_id)
    if args.command == "status":
        print(json.dumps(store.snapshot(args.task_id), ensure_ascii=False, indent=2))
        return 0
    with store.connect() as db:
        queued = db.execute("SELECT id,worker_pid FROM tasks WHERE status='queued'").fetchall()
    for row in queued:
        if not alive(row["worker_pid"]):
            launch_worker(store, row["id"])
    create_server(store, args.port).run(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
