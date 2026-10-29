"""Durable Pro plans and delivery through the installed Codex desktop MCP."""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

from sessions import PREFIX, redact


def timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class DesktopUnavailable(RuntimeError):
    """The desktop cannot accept a call; nothing has been sent."""


class DeliveryUncertain(RuntimeError):
    """A call may have reached the desktop; never retry automatically."""


def desktop_parameters():
    from mcp.client.stdio import StdioServerParameters

    if os.name != "nt":
        raise DesktopUnavailable("This desktop delivery adapter requires Windows.")
    pipes = [name for name in os.listdir("\\\\.\\pipe\\") if name.startswith("codex-app-tools-")]
    if len(pipes) != 1:
        raise DesktopUnavailable("Open one Codex desktop instance; the plan remains in the inbox.")
    base = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    scripts = list((base / "plugins/cache/openai-bundled/codex-app-tools").glob("*/server.mjs"))
    node = shutil.which("node")
    if not scripts or not node:
        raise DesktopUnavailable("The installed Codex app-tools MCP or Node runtime is unavailable.")
    script = max(scripts, key=lambda path: path.stat().st_mtime)
    environment = os.environ.copy()
    for key in ("CONTROL_PLANE_API_KEY", "OPENAI_API_KEY", "CODEX_THREAD_ID", "CODEX_SESSION_ID"):
        environment.pop(key, None)
    environment["CODEX_APP_TOOLS_PIPE_PATH"] = "\\\\.\\pipe\\" + pipes[0]
    return StdioServerParameters(command=node, args=[str(script)], env=environment, cwd=str(script.parent))


async def desktop_call(caller_thread_id, tool, arguments):
    """Use the official bundled MCP, with this bridge chat as the actual caller."""
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    attempted = False
    try:
        parameters = desktop_parameters()
        async with asyncio.timeout(45):
            with open(os.devnull, "w") as errors:
                async with stdio_client(parameters, errlog=errors) as (reader, writer):
                    async with ClientSession(reader, writer) as client:
                        await client.initialize()
                        catalog = await client.list_tools()
                        if tool not in {item.name for item in catalog.tools}:
                            raise DesktopUnavailable("The desktop does not expose the required tool.")
                        attempted = True
                        result = await client.call_tool(tool, arguments, meta={"openai/threadId": caller_thread_id})
                        content = [item.text for item in result.content if item.type == "text"]
                        if result.isError:
                            raise DeliveryUncertain(redact("\n".join(content))[:1500])
                        return {"accepted_by_desktop": True, "content": content}
    except Exception as error:
        if attempted:
            raise DeliveryUncertain("Desktop acknowledgement unavailable. Inspect the target chat before retrying.") from error
        raise DesktopUnavailable("Desktop connection unavailable; the message has not been sent.") from error


class Inbox:
    def __init__(self, store, sessions, sender=None):
        self.store, self.sessions = store, sessions
        self.sender = sender or desktop_call
        with store.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS inbox (
                id TEXT PRIMARY KEY, request_id TEXT UNIQUE NOT NULL, fingerprint TEXT NOT NULL,
                title TEXT NOT NULL, body TEXT NOT NULL, target_session_id TEXT NOT NULL,
                mode TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, receipt TEXT, error TEXT
            )""")

    def policy(self):
        return self.store.context().get("messaging_policy", {})

    def save(self, body, request_id, title="Pro plan", session_id="", mode="store_only"):
        if self.policy().get("inbox_enabled") is not True:
            raise ValueError("The local inbox is disabled.")
        if mode not in {"store_only", "deliver"}:
            raise ValueError("Invalid delivery mode.")
        if mode == "deliver" and self.policy().get("session_delivery_enabled") is not True:
            raise ValueError("Session delivery is disabled; use save_plan instead.")
        if not body.strip() or len(body) > 60000 or "\x00" in body:
            raise ValueError("Message must contain 1 to 60000 characters without NUL bytes.")
        if not request_id.strip() or len(request_id) > 160:
            raise ValueError("request_id must contain 1 to 160 characters; reuse it on retries.")
        if not title.strip() or len(title) > 200:
            raise ValueError("title must contain 1 to 200 characters.")
        # Fingerprint the caller's request so changing the default cannot replay it.
        payload = [body, title, session_id, mode]
        fingerprint = hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()
        policy = self.policy()
        default_target = policy.get("default_receiver_session_id")
        if default_target and session_id in ("", PREFIX + policy.get("relay_thread_id", "")):
            session_id = default_target
        if session_id:
            self.sessions.row(session_id)
        elif mode == "deliver":
            raise ValueError("Choose an existing session: ID from list_tasks.")
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute("SELECT id,fingerprint FROM inbox WHERE request_id=?", (request_id,)).fetchone()
            if prior:
                if prior["fingerprint"] != fingerprint:
                    raise ValueError("This request_id already belongs to a different message.")
                message_id = prior["id"]
            else:
                message_id = uuid.uuid4().hex
                at = timestamp()
                db.execute("INSERT INTO inbox(id,request_id,fingerprint,title,body,target_session_id,mode,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,'stored',?,?)",
                           (message_id, request_id, fingerprint, title, body, session_id, mode, at, at))
        return self.get(message_id)

    def get(self, message_id):
        with self.store.connect() as db:
            row = db.execute("SELECT * FROM inbox WHERE id=?", (message_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown inbox message_id.")
        value = dict(row)
        value.pop("fingerprint")
        value["receipt"] = json.loads(value["receipt"]) if value["receipt"] else None
        value["execution_status"] = "not_started_by_inbox" if value["status"] == "stored" else "inspect_target_session"
        value["note"] = "Saving does not execute. A delivery acknowledgement is not task completion."
        return value

    def list(self, limit=20):
        with self.store.connect() as db:
            rows = db.execute("SELECT id,request_id,title,target_session_id,mode,status,created_at,updated_at,error FROM inbox ORDER BY created_at DESC,rowid DESC LIMIT ?",
                              (max(1, min(limit, 100)),)).fetchall()
        return {"messages": [dict(row) for row in rows], "storage": str(self.store.path)}

    async def deliver(self, message_id, session_id=""):
        policy = self.policy()
        if policy.get("session_delivery_enabled") is not True:
            raise ValueError("Session delivery is disabled.")
        caller = policy.get("relay_thread_id")
        if not caller:
            raise ValueError("The local bridge chat is not configured.")
        message = self.get(message_id)
        target = session_id or message["target_session_id"]
        self.sessions.row(target)
        if message["target_session_id"] and message["target_session_id"] != target:
            raise ValueError("This message already targets a different session.")
        # Commit the claim before sending. On timeout/crash, do not resend and risk a duplicate turn.
        with self.store.connect() as db:
            claimed = db.execute("UPDATE inbox SET status='dispatching',target_session_id=?,updated_at=? WHERE id=? AND status='stored'",
                                 (target, timestamp(), message_id)).rowcount
        if not claimed:
            return self.get(message_id)
        prompt = (f"Pro 发来的 ARBoids 方案（收件箱编号 {message_id}，标题：{message['title']}）。\n"
                  "用户已选择将这条方案发到本会话开始处理。按本会话的授权范围和 AGENTS.md 执行，"
                  "先核对当前代码与证据；保护已有结果，协调正在运行的任务。\n\n" + message["body"])
        receipt, error = None, None
        try:
            receipt = await self.sender(caller, "send_message_to_thread", {"threadId": target[len(PREFIX):], "hostId": "local", "prompt": prompt})
            status = "delivered"
        except DesktopUnavailable as problem:
            status, error = "stored", str(problem)
        except Exception as problem:
            status, error = "delivery_uncertain", str(problem)
        with self.store.connect() as db:
            db.execute("UPDATE inbox SET status=?,receipt=?,error=?,updated_at=? WHERE id=?",
                       (status, json.dumps(receipt, ensure_ascii=False) if receipt else None, redact(error) if error else None, timestamp(), message_id))
        return self.get(message_id)
