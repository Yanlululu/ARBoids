import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sqlite3
import sys
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch
import control
from inbox import Inbox, DesktopUnavailable, DeliveryUncertain

from server import ProcessTree, Store, TERMINAL, alive, create_server, parse_progress, safe_file, spawn_options
from sessions import CodexSessions


class BridgeTests(unittest.TestCase):
    def test_health_connection_reset_during_service_restart_is_not_a_fatal_error(self):
        opener = Mock()
        opener.open.side_effect = ConnectionResetError('service is stopping')
        with patch('control.urllib.request.build_opener', return_value=opener):
            self.assertIsNone(control.health(8787))

    @unittest.skipUnless(os.name == "nt", "Windows live proxy detection")
    def test_live_proxy_wins_over_stale_shell_and_falls_back_to_direct(self):
        with patch.dict(os.environ, {"CONTROL_PLANE_HTTP_PROXY": "http://127.0.0.1:10808"}, clear=True), \
             patch("control.urllib.request.getproxies_registry", return_value={"https": "http://127.0.0.1:10793"}), \
             patch("control.urllib.request.getproxies", return_value={}):
            self.assertEqual(control.proxy_candidates(), ["http://127.0.0.1:10793", "http://127.0.0.1:10808", None])

    def test_auto_route_falls_back_only_for_transport_failures(self):
        with patch("control.proxy_candidates", return_value=["http://127.0.0.1:1", None]), \
             patch("control.verify_tunnel_access", side_effect=[control.TunnelTransportError("offline"), {"cloud_access": "verified"}]):
            selected, result = control.select_route("tunnel_test", {"CONTROL_PLANE_API_KEY": "test-key", "HTTPS_PROXY": "http://stale:1"})
            self.assertNotIn("CONTROL_PLANE_HTTP_PROXY", selected)
            self.assertNotIn("HTTPS_PROXY", selected)
            self.assertIn("127.0.0.1", selected["NO_PROXY"])
            self.assertEqual(result["cloud_access"], "verified")
        with patch("control.proxy_candidates", return_value=["http://127.0.0.1:1", None]), \
             patch("control.verify_tunnel_access", side_effect=RuntimeError("HTTP 401")) as check:
            with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
                control.select_route("tunnel_test", {})
            self.assertEqual(check.call_count, 1)

    def test_proxy_watcher_switches_route_only_after_cloud_verification(self):
        state = dict(token="owner", proxy="http://old:1", preferred_proxy="http://old:1")
        runtime = dict(process_running=True, tunnel_id="tunnel_test", process={"target_value": "http://127.0.0.1:8787/mcp"})
        from contextlib import nullcontext
        with patch("control.runtime_lock", side_effect=nullcontext), patch("control.read_watch", return_value=state), \
             patch("control.runtime_status", return_value=runtime), patch("control.polling_healthy", return_value=True), \
             patch("control.proxy_candidates", return_value=["http://new:2", None]), \
             patch("control.select_route", return_value=({"CONTROL_PLANE_HTTP_PROXY": "http://new:2"}, {})), \
             patch("control.attach_runtime") as reconnect, patch("control.write_watch") as write, \
             patch("control.time.sleep", side_effect=StopIteration):
            with self.assertRaises(StopIteration):
                control.watch_tunnel(8787, "tunnel_test", "arboids", "owner")
            self.assertEqual(reconnect.call_count, 1)
            self.assertTrue(reconnect.call_args.kwargs["restart"])
            self.assertEqual(write.call_args.args[0]["proxy"], "http://new:2")

    def test_failed_new_proxy_leaves_existing_tunnel_running(self):
        state = dict(token="owner", proxy="http://old:1", preferred_proxy="http://old:1")
        runtime = dict(process_running=True, tunnel_id="tunnel_test", process={"target_value": "http://127.0.0.1:8787/mcp"})
        from contextlib import nullcontext
        with patch("control.runtime_lock", side_effect=nullcontext), patch("control.read_watch", return_value=state), \
             patch("control.runtime_status", return_value=runtime), patch("control.polling_healthy", return_value=True), \
             patch("control.proxy_candidates", return_value=["http://new:2", None]), \
             patch("control.select_route", side_effect=control.TunnelTransportError("offline")), \
             patch("control.attach_runtime") as reconnect, patch("control.write_watch"), \
             patch("control.time.sleep", side_effect=StopIteration):
            with self.assertRaises(StopIteration):
                control.watch_tunnel(8787, "tunnel_test", "arboids", "owner")
            reconnect.assert_not_called()

    def test_watch_state_never_persists_runtime_key(self):
        with patch("control.WATCH_STATE", self.root / "watch.json"), \
             patch("control.proxy_candidates", return_value=[None]), patch("control.subprocess.Popen", return_value=Mock(pid=123)):
            control.start_watcher(8787, "tunnel_test", "arboids", {"CONTROL_PLANE_API_KEY": "secret-test-value"})
            self.assertNotIn("secret-test-value", (self.root / "watch.json").read_text())

    def test_runtime_identity_mismatch_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "another tunnel"):
            control.runtime_matches({"tunnel_id": "tunnel_other", "process": {"target_value": "http://127.0.0.1:8787/mcp"}}, "tunnel_test", 8787)
        with self.assertRaisesRegex(RuntimeError, "MCP endpoint"):
            control.runtime_matches({"tunnel_id": "tunnel_test", "process": {"target_value": "http://127.0.0.1:9999/mcp"}}, "tunnel_test", 8787)

    def test_runtime_status_does_not_trigger_authenticated_remote_metadata_lookup(self):
        environment = {"CONTROL_PLANE_API_KEY": "runtime-test-key", "OPENAI_API_KEY": "other-test-key", "OPENAI_ADMIN_KEY": "admin-test-key"}
        with patch("control.subprocess.run", return_value=Mock(returncode=0, stdout='{"process_running":true}')) as run:
            self.assertTrue(control.runtime_status("arboids", environment)["process_running"])
            self.assertFalse({"CONTROL_PLANE_API_KEY", "OPENAI_API_KEY", "OPENAI_ADMIN_KEY"} & run.call_args.kwargs["env"].keys())
            self.assertIn("CONTROL_PLANE_API_KEY", environment)

    def test_runtime_management_uses_current_proxy_and_bypasses_local_mcp(self):
        environment = control.management_environment({"CONTROL_PLANE_HTTP_PROXY": "http://127.0.0.1:10793", "HTTPS_PROXY": "http://stale:1"})
        self.assertEqual(environment["HTTPS_PROXY"], "http://127.0.0.1:10793")
        self.assertEqual(environment["HTTP_PROXY"], "http://127.0.0.1:10793")
        self.assertEqual(environment["CONTROL_PLANE_HTTP_PROXY"], "http://127.0.0.1:10793")
        self.assertIn("127.0.0.1", environment["NO_PROXY"])
        self.assertNotIn("MCP_HTTP_PROXY", environment)

    def test_watcher_does_not_recreate_a_stopped_or_removed_runtime(self):
        from contextlib import nullcontext
        for runtime in ({}, {"process_running": False, "tunnel_id": "tunnel_test", "process": {"target_value": "http://127.0.0.1:8787/mcp"}}):
            state = dict(token="owner")
            with patch("control.runtime_lock", side_effect=nullcontext), patch("control.read_watch", return_value=state), \
                 patch("control.runtime_status", return_value=runtime), patch("control.attach_runtime") as reconnect, \
                 patch("control.write_watch") as write:
                control.watch_tunnel(8787, "tunnel_test", "arboids", "owner")
                reconnect.assert_not_called()
                self.assertEqual(write.call_args.args[0]["status"], "stopped")

    def test_system_proxy_is_scoped_to_the_control_plane(self):
        with patch.dict(os.environ, {}, clear=True), patch("control.urllib.request.getproxies", return_value={"https": "http://127.0.0.1:10808"}):
            environment = control.tunnel_environment()
        self.assertEqual(environment["CONTROL_PLANE_HTTP_PROXY"], "http://127.0.0.1:10808")
        self.assertNotIn("HTTP_PROXY", environment)
        self.assertNotIn("HTTPS_PROXY", environment)
        self.assertNotIn("TUNNEL_CLIENT_HTTP_PROXY", environment)
        self.assertNotIn("MCP_HTTP_PROXY", environment)

    def test_explicit_control_plane_proxy_is_preserved(self):
        with patch.dict(os.environ, {"CONTROL_PLANE_HTTP_PROXY": "http://127.0.0.1:9000"}, clear=True), patch("control.urllib.request.getproxies", return_value={"https": "http://127.0.0.1:10808"}):
            environment = control.tunnel_environment()
        self.assertEqual(environment["CONTROL_PLANE_HTTP_PROXY"], "http://127.0.0.1:9000")

    @unittest.skipUnless(os.name == "nt", "Windows system proxy fallback")
    def test_no_proxy_alone_does_not_hide_windows_system_proxy(self):
        with patch.dict(os.environ, {"NO_PROXY": "localhost,127.0.0.1"}, clear=True), patch("control.urllib.request.getproxies", return_value={"no": "localhost,127.0.0.1"}), patch("control.urllib.request.getproxies_registry", return_value={"https": "http://127.0.0.1:10808"}):
            environment = control.tunnel_environment()
        self.assertEqual(environment["CONTROL_PLANE_HTTP_PROXY"], "http://127.0.0.1:10808")

    def test_command_line_proxy_overrides_environment_without_proxying_local_mcp(self):
        with patch.dict(os.environ, {"CONTROL_PLANE_HTTP_PROXY": "http://127.0.0.1:9000"}, clear=True):
            environment = control.tunnel_environment("http://127.0.0.1:10808")
        self.assertEqual(environment["CONTROL_PLANE_HTTP_PROXY"], "http://127.0.0.1:10808")
        self.assertNotIn("MCP_HTTP_PROXY", environment)
        with self.assertRaises(ValueError):
            control.tunnel_environment("http://user:password@127.0.0.1:10808")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.store = Store(self.root / ".runtime" / "tasks.sqlite3", self.root)
        self.payload = {"kind": "codex", "task": "Read the configuration", "title": "check", "timeout_seconds": 60}

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def mailbox(self, sender=None):
        self.store.set_context("messaging_policy", {"inbox_enabled": True, "session_delivery_enabled": True, "relay_thread_id": "bridge-chat"})
        sessions = Mock()
        def row(session_id):
            if session_id != "session:target":
                raise ValueError("Not an available ARBoids session")
            return {"id": "target", "root": self.root}
        sessions.row.side_effect = row
        return Inbox(self.store, sessions, sender=sender)

    def test_saved_plan_does_not_execute_and_duplicate_requests_preserve_one_record(self):
        sender = AsyncMock()
        inbox = self.mailbox(sender)
        first = inbox.save("先审查现有证据，再给方案", "save-one")
        self.assertEqual(first["status"], "stored")
        self.assertEqual(inbox.save("先审查现有证据，再给方案", "save-one")["id"], first["id"])
        self.assertEqual(len(inbox.list()["messages"]), 1)
        sender.assert_not_called()
        with self.assertRaisesRegex(ValueError, "different message"):
            inbox.save("changed plan", "save-one")

    def test_duplicate_parallel_delivery_sends_only_once_to_the_exact_chat(self):
        sender = AsyncMock(return_value={"accepted_by_desktop": True})
        inbox = self.mailbox(sender)
        saved = inbox.save("执行方案", "delivery-one", session_id="session:target", mode="deliver")
        async def deliver_twice():
            await asyncio.gather(inbox.deliver(saved["id"]), inbox.deliver(saved["id"]))
        asyncio.run(deliver_twice())
        self.assertEqual(sender.await_count, 1)
        self.assertEqual(sender.call_args.args[2]["threadId"], "target")
        self.assertEqual(inbox.get(saved["id"])["status"], "delivered")
        self.assertEqual(inbox.get(saved["id"])["execution_status"], "inspect_target_session")

    def test_uncertain_delivery_never_auto_retries_but_offline_keeps_the_plan(self):
        sender = AsyncMock(side_effect=DeliveryUncertain("connection lost after sending"))
        inbox = self.mailbox(sender)
        saved = inbox.save("plan", "uncertain", session_id="session:target", mode="deliver")
        asyncio.run(inbox.deliver(saved["id"]))
        asyncio.run(inbox.deliver(saved["id"]))
        self.assertEqual(sender.await_count, 1)
        self.assertEqual(inbox.get(saved["id"])["status"], "delivery_uncertain")
        sender.side_effect = DesktopUnavailable("desktop closed before send")
        saved = inbox.save("plan", "offline", session_id="session:target", mode="deliver")
        self.assertEqual(asyncio.run(inbox.deliver(saved["id"]))["status"], "stored")

    def test_inbox_target_scope_and_live_policy_are_enforced(self):
        inbox = self.mailbox(AsyncMock())
        with self.assertRaisesRegex(ValueError, "ARBoids session"):
            inbox.save("plan", "bad-target", session_id="session:unrelated")
        saved = inbox.save("plan", "valid", session_id="session:target")
        self.store.set_context("messaging_policy", {"inbox_enabled": True, "session_delivery_enabled": False})
        with self.assertRaisesRegex(ValueError, "disabled"):
            asyncio.run(inbox.deliver(saved["id"]))

    def test_default_receiver_routes_bridge_and_empty_targets_without_replaying_old_requests(self):
        sender = AsyncMock(return_value={"accepted_by_desktop": True})
        inbox = self.mailbox(sender)
        inbox.sessions.row.side_effect = lambda target: {"id": target.removeprefix("session:"), "root": self.root}
        old = inbox.save("old plan", "old-request", session_id="session:bridge-chat", mode="deliver")
        asyncio.run(inbox.deliver(old["id"]))
        self.store.set_context("messaging_policy", dict(inbox.policy(), default_receiver_session_id="session:target"))
        retried = inbox.save("old plan", "old-request", session_id="session:bridge-chat", mode="deliver")
        asyncio.run(inbox.deliver(retried["id"]))
        self.assertEqual(retried["target_session_id"], "session:bridge-chat")
        self.assertEqual(sender.await_count, 1)
        for requested in ("", "session:bridge-chat"):
            saved = inbox.save("new plan", "new-" + requested, session_id=requested, mode="deliver")
            self.assertEqual(saved["target_session_id"], "session:target")
            asyncio.run(inbox.deliver(saved["id"]))
            self.assertEqual(sender.call_args.args[2]["threadId"], "target")
        explicit = inbox.save("other plan", "explicit", session_id="session:other", mode="deliver")
        self.assertEqual(explicit["target_session_id"], "session:other")

    def test_invalid_default_receiver_is_rejected_without_falling_back_to_bridge(self):
        inbox = self.mailbox(AsyncMock())
        self.store.set_context("messaging_policy", dict(inbox.policy(), default_receiver_session_id="session:unrelated"))
        with self.assertRaisesRegex(ValueError, "ARBoids session"):
            inbox.save("plan", "invalid-default", session_id="session:bridge-chat", mode="deliver")
        self.assertEqual(inbox.list()["messages"], [])

    def test_inbox_mcp_tools_do_not_reenable_cancelled_job_execution(self):
        self.mailbox()
        async def inspect():
            mcp = create_server(self.store)
            tools = {tool.name: tool for tool in await mcp.list_tools()}
            self.assertFalse(tools["save_plan"].annotations.readOnlyHint)
            self.assertFalse(tools["send_message_to_codex"].annotations.readOnlyHint)
            self.assertTrue(tools["list_inbox"].annotations.readOnlyHint)
            self.assertNotIn("submit_codex_task", tools)
            _, saved = await mcp.call_tool("save_plan", {"plan": "存储测试", "request_id": "via-mcp"})
            self.assertEqual(saved["status"], "stored")
            with self.store.connect() as db:
                self.assertEqual(db.execute("SELECT count(*) FROM tasks").fetchone()[0], 0)
        asyncio.run(inspect())

    def test_submission_retry_is_idempotent_and_conflicts_are_rejected(self):
        first, created = self.store.submit(self.payload, "one")
        retry, created_again = self.store.submit(self.payload, "one")
        self.assertEqual(first, retry)
        self.assertTrue(created)
        self.assertFalse(created_again)
        with self.assertRaisesRegex(ValueError, "different task"):
            self.store.submit(dict(self.payload, task="A different request"), "one")

    def test_transient_io_failure_during_queue_claim_is_retried(self):
        task_id, _ = self.store.submit(self.payload, "one")
        original = sqlite3.connect
        calls = 0
        def flaky(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls < 3:
                raise sqlite3.OperationalError("disk I/O error")
            return original(*args, **kwargs)
        with patch("server.sqlite3.connect", side_effect=flaky), patch("server.time.sleep"):
            self.assertTrue(self.store.claim(task_id))
        self.assertEqual(self.store.row(task_id)["status"], "running")
        self.assertEqual(calls, 3)

    def test_claim_retry_after_uncertain_commit_does_not_wait_on_itself(self):
        task_id, _ = self.store.submit(self.payload, 'one')
        original = self.store.connect
        first = True
        @contextmanager
        def uncertain():
            nonlocal first
            with original() as db:
                yield db
            if first:
                first = False
                raise sqlite3.OperationalError('disk I/O error')
        with patch.object(self.store,'connect',uncertain),patch('server.time.sleep'):
            self.assertTrue(self.store.claim(task_id))
        self.assertEqual(self.store.row(task_id)['worker_pid'],os.getpid())

    def test_recovery_preserves_failed_task_and_waits_for_active_task(self):
        active, _ = self.store.submit(self.payload, "active")
        self.assertTrue(self.store.claim(active))
        failed, _ = self.store.submit(self.payload, "failed")
        continuation, _ = self.store.submit(self.payload, "continuation")
        self.store.update(failed, status="failed", error="OperationalError: disk I/O error")
        retry, created = self.store.retry_failed(failed, "retry")
        again, created_again = self.store.retry_failed(failed, "retry")
        self.assertEqual((retry, created, again, created_again), (retry, True, retry, False))
        self.assertEqual(retry, failed)
        self.assertEqual(self.store.row(failed)["progress"]["attempt_history"][0]["error"], "OperationalError: disk I/O error")
        self.assertIsNone(self.store.row(retry)["error"])
        self.assertFalse(self.store.claim(retry))
        self.store.update(active, status="completed")
        self.assertFalse(self.store.claim(continuation))
        self.assertTrue(self.store.claim(retry))
        self.store.update(retry, status="failed")
        with self.assertRaisesRegex(ValueError, "never started"):
            self.store.retry_failed(retry, "unsafe")

    def test_repair_context_is_visible_to_pro_and_the_recovered_executor(self):
        self.store.set_context('execution_repair',dict(status='verified',policy='frozen-code paired evaluation'))
        task_id, _ = self.store.submit(self.payload,'one')
        self.assertEqual(self.store.snapshot(task_id)['execution_repair']['status'],'verified')
        from server import task_prompt
        self.assertIn('frozen-code paired evaluation',task_prompt(self.store,task_id))

    def test_concurrent_stores_keep_wal_and_records_consistent(self):
        def write(index):
            store = Store(self.store.path, self.root)
            try:
                task_id, _ = store.submit(self.payload, f"parallel-{index}")
                store.event(task_id, "checked", str(index))
                return task_id
            finally:
                store.close()
        with ThreadPoolExecutor(max_workers=4) as pool:
            ids = list(pool.map(write, range(12)))
        self.assertEqual(len(set(ids)), 12)
        with self.store.connect() as db:
            self.assertEqual(db.execute("PRAGMA quick_check").fetchone()[0], "ok")
            self.assertEqual(db.execute("SELECT count(*) FROM events").fetchone()[0], 24)
        self.assertTrue(Path(str(self.store.path) + "-wal").exists())

    def test_queue_runs_only_one_task_in_submission_order(self):
        first, _ = self.store.submit(self.payload, "one")
        second, _ = self.store.submit(self.payload, "two")
        self.assertFalse(self.store.claim(second))
        self.assertTrue(self.store.claim(first))
        self.assertFalse(self.store.claim(second))
        self.store.update(first, status="completed")
        self.assertTrue(self.store.claim(second))

    def test_worker_disappearance_is_reported_instead_of_running_forever(self):
        task_id, _ = self.store.submit(self.payload, "one")
        self.store.update(task_id, status="running", worker_pid=123456789)
        self.assertEqual(self.store.snapshot(task_id)["status"], "interrupted")

    def test_path_boundaries_exclude_credentials_and_runtime(self):
        visible = self.root / "visible.py"
        visible.write_text("print('ok')", encoding="utf-8")
        self.assertEqual(safe_file(self.root, "visible.py"), visible)
        for path in ("../outside.py", ".env", ".ENV.json", ".runtime/tasks.sqlite3", ".RUNTIME/state.json", ".venv/Scripts/python.exe", str(visible)):
            with self.subTest(path=path), self.assertRaises(ValueError):
                safe_file(self.root, path)

    def test_progress_keeps_training_and_evaluation_evidence(self):
        state = parse_progress({}, "[PROGRESS] step=250/2000 updates=0 elapsed=3.5s")
        self.assertEqual((state["step"], state["percent"]), (250, 12.5))
        state = parse_progress(state, "[EVAL] step=1000 success_rate=0.400 reward=-1.5e+1")
        self.assertEqual(state["evaluation"], {"step": 1000, "success_rate": .4, "reward": -15.})
        state = parse_progress(state, "[INFO] saved model -> model.pth")
        self.assertEqual(state["percent"], 12.5)
        self.assertEqual(state["evaluation"]["step"], 1000)

    def test_event_cursor_does_not_repeat_old_events(self):
        task_id, _ = self.store.submit(self.payload, "one")
        self.store.event(task_id, "command", "python train/train.py")
        with self.store.connect() as db:
            rows = db.execute("SELECT seq,kind FROM events WHERE task_id=? ORDER BY seq", (task_id,)).fetchall()
            later = db.execute("SELECT kind FROM events WHERE task_id=? AND seq>?", (task_id, rows[0]["seq"])).fetchall()
        self.assertEqual([row["kind"] for row in later], ["command"])

    def test_mcp_execution_tools_are_declared_as_writes(self):
        self.store.set_context('execution_policy',dict(mode='execution',remote_execution_enabled=True))
        async def inspect_tools():
            tools = await create_server(self.store).list_tools()
            by_name = {tool.name: tool for tool in tools}
            self.assertFalse(by_name["submit_codex_task"].annotations.readOnlyHint)
            self.assertFalse(by_name["run_arboids_experiment"].annotations.readOnlyHint)
            self.assertFalse(by_name["cancel_task"].annotations.readOnlyHint)
            self.assertFalse(by_name["retry_failed_task"].annotations.readOnlyHint)
            self.assertTrue(by_name["get_task_status"].annotations.readOnlyHint)
            self.assertTrue(by_name["read_project_file"].annotations.readOnlyHint)
            self.assertTrue(by_name["wait_for_task_update"].annotations.readOnlyHint)
        asyncio.run(inspect_tools())

    def test_default_mcp_is_read_only_and_does_not_expose_execution_tools(self):
        async def inspect():
            mcp = create_server(self.store)
            catalog = await mcp.list_tools()
            self.assertTrue(all(tool.annotations.readOnlyHint for tool in catalog))
            self.assertTrue({'get_task_status','read_project_file','get_project_context'} <= {tool.name for tool in catalog})
            for name in ('submit_codex_task','retry_failed_task','run_arboids_experiment','cancel_task'):
                self.assertNotIn(name,{tool.name for tool in catalog})
            _, context = await mcp.call_tool('get_project_context',{})
            self.assertEqual(context['concurrency'],0)
            self.assertFalse(context['execution_policy']['remote_execution_enabled'])
        asyncio.run(inspect())

    def test_cached_write_tool_is_blocked_when_user_disables_remote_execution(self):
        self.store.set_context('execution_policy',dict(mode='execution',remote_execution_enabled=True))
        async def inspect():
            mcp = create_server(self.store)
            self.store.set_context('execution_policy',dict(mode='read_only',remote_execution_enabled=False))
            for name,arguments in (
                ('submit_codex_task',dict(task='must not execute',request_id='disabled')),
                ('retry_failed_task',dict(task_id='unused',request_id='disabled')),
                ('cancel_task',dict(task_id='unused')),
            ):
                with self.assertRaisesRegex(Exception,'Remote execution is disabled'):
                    await mcp.call_tool(name,arguments)
            with self.store.connect() as db:
                self.assertEqual(db.execute('SELECT count(*) FROM tasks').fetchone()[0],0)
        asyncio.run(inspect())

    def test_repeat_numbered_metrics_are_returned_as_actual_evidence(self):
        task_id, _ = self.store.submit(dict(self.payload, kind="experiment"), "one")
        folder = self.root / "train" / "experiments" / f"mcp_{task_id}"
        folder.mkdir(parents=True)
        (folder / "metrics1.csv").write_text("step,def_sr,reward\n2000,0.4,-2.5\n", encoding="utf-8")
        result = self.store.snapshot(task_id)
        self.assertEqual(result["latest_evaluation"]["step"], "2000")
        self.assertTrue(result["metrics_file"].endswith("metrics1.csv"))

    def test_cancelled_queued_task_cannot_be_claimed(self):
        task_id, _ = self.store.submit(self.payload, "one")
        self.store.update(task_id, status="cancelled", cancel_requested=1)
        self.assertFalse(self.store.claim(task_id))
        self.assertIn(self.store.snapshot(task_id)["status"], TERMINAL)

    def test_process_tree_cancellation_stops_descendants(self):
        pid_file = self.root / "child.pid"
        code = ("import subprocess,sys,time; from pathlib import Path; "
                "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); "
                f"Path({str(pid_file)!r}).write_text(str(child.pid)); time.sleep(60)")
        process = subprocess.Popen([sys.executable, "-c", code], **spawn_options())
        tree = None
        try:
            tree = ProcessTree(process)
            deadline = time.monotonic() + 10
            while not pid_file.exists() and time.monotonic() < deadline:
                time.sleep(.05)
            child_pid = int(pid_file.read_text())
            self.assertTrue(alive(child_pid))
            if tree:
                tree.close()
            process.wait(timeout=10)
            deadline = time.monotonic() + 5
            while alive(child_pid) and time.monotonic() < deadline:
                time.sleep(.05)
            self.assertFalse(alive(child_pid))
        finally:
            tree.close()
            if process.poll() is None:
                process.kill()
                process.wait()


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        self.root, self.worktree, self.home = [self.base / name for name in ("project", "worktree", "codex")]
        for path in (self.root, self.worktree, self.home / "sessions"):
            path.mkdir(parents=True)
        with closing(sqlite3.connect(self.home / "state_5.sqlite")) as db, db:
            db.execute("CREATE TABLE threads(id TEXT,cwd TEXT,rollout_path TEXT,title TEXT,name TEXT,source TEXT,archived INTEGER,updated_at INTEGER)")
        self.sessions = CodexSessions(self.root, self.home)
        self.roots_patch = patch.object(self.sessions, "roots", return_value=[self.root, self.worktree])
        self.roots_patch.start()
        self.add_session("one", self.worktree)

    def tearDown(self):
        self.roots_patch.stop()
        self.temporary.cleanup()

    def add_session(self, session_id, cwd, archived=0, rollout=None):
        path = rollout or self.home / "sessions" / (session_id + ".jsonl")
        path.touch()
        with closing(sqlite3.connect(self.home / "state_5.sqlite")) as db, db:
            db.execute("INSERT INTO threads VALUES(?,?,?,?,?,?,?,?)", (session_id, str(cwd), str(path), "Initial prompt", "Named chat " + session_id, "vscode", archived, time.time()))
        return path

    def append(self, kind, payload, at=None, newline=True):
        record = {"type": kind, "timestamp": at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "payload": payload}
        with (self.home / "sessions" / "one.jsonl").open("ab") as stream:
            stream.write(json.dumps(record).encode() + (b"\n" if newline else b""))

    def test_only_registered_project_sessions_and_safe_transcripts_are_exposed(self):
        unrelated = self.base / "unrelated" / "ARBoids"
        unrelated.mkdir(parents=True)
        self.add_session("other", unrelated)
        self.add_session("archived", self.root, archived=1)
        self.add_session("outside-transcript", self.root, rollout=self.base / "private.jsonl")
        self.assertEqual([row["id"] for row in self.sessions.index()], ["one"])
        with self.assertRaises(ValueError):
            self.sessions.snapshot("session:other")

    def test_incremental_running_completion_and_partial_record(self):
        self.append("event_msg", {"type": "task_started"})
        first = self.sessions.snapshot("session:one")
        self.assertEqual(first["status"], "running")
        self.assertEqual(first["title"], "Named chat one")
        self.append("event_msg", {"type": "task_complete", "last_agent_message": "Verified results"}, newline=False)
        self.assertEqual(self.sessions.snapshot("session:one")["status"], "running")
        with (self.home / "sessions" / "one.jsonl").open("ab") as stream:
            stream.write(b"\n")
        final = self.sessions.snapshot("session:one")
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["final_message"], "Verified results")
        events = self.sessions.events("session:one", first["latest_event_sequence"])
        self.assertEqual([event["kind"] for event in events["events"]], ["completed"])

    def test_stale_running_is_unknown_and_aborted_is_not_completed(self):
        self.append("event_msg", {"type": "task_started"}, at="2000-01-01T00:00:00Z")
        self.assertEqual(self.sessions.snapshot("session:one")["status"], "unknown")
        self.append("event_msg", {"type": "turn_aborted"})
        self.assertEqual(self.sessions.snapshot("session:one")["status"], "interrupted")

    def test_reasoning_instructions_and_raw_tool_secrets_are_excluded(self):
        self.append("response_item", {"type": "reasoning", "summary": [{"text": "private reasoning"}]})
        self.append("response_item", {"type": "message", "role": "developer", "content": [{"type": "text", "text": "private instructions"}]})
        self.append("response_item", {"type": "function_call_output", "output": "raw-private-output\n[PROGRESS] step=10/100"})
        self.append("response_item", {"type": "message", "role": "assistant", "phase": "commentary", "content": [{"type": "output_text", "text": "Checking sk-example123456789 and password=hiddenvalue"}]})
        output = json.dumps({"status": self.sessions.snapshot("session:one"), "events": self.sessions.events("session:one")})
        for secret in ("private reasoning", "private instructions", "raw-private-output", "sk-example123456789", "hiddenvalue"):
            self.assertNotIn(secret, output)
        self.assertIn("step=10/100", output)

    def test_mcp_existing_tools_read_the_correct_worktree_and_reject_escape(self):
        for root, value in ((self.root, "wrong workspace"), (self.worktree, "real metrics")):
            (root / "results").mkdir()
            (root / "results" / "metrics.csv").write_text(value, encoding="utf-8")
        self.append("event_msg", {"type": "task_complete", "last_agent_message": "Result: `results/metrics.csv`"})
        store = Store(self.root / ".runtime" / "tasks.sqlite3", self.root)
        async def verify():
            mcp = create_server(store, sessions=self.sessions)
            _, listed = await mcp.call_tool("list_tasks", {"limit": 20})
            self.assertEqual(listed["tasks"][0]["task_id"], "session:one")
            _, status = await mcp.call_tool("get_task_status", {"task_id": "session:one"})
            self.assertEqual(status["artifacts"][0]["path"], "results/metrics.csv")
            text = await mcp.call_tool("read_project_file", {"task_id": "session:one", "path": "results/metrics.csv"})
            self.assertIn("real metrics", str(text))
            self.assertNotIn("wrong workspace", str(text))
            artifact = await mcp.call_tool("read_artifact", {"task_id": "session:one", "path": "results/metrics.csv"})
            self.assertIn("real metrics", str(artifact))
            for path in ("../private.txt", ".runtime/tasks.sqlite3"):
                with self.assertRaises(Exception):
                    await mcp.call_tool("read_project_file", {"task_id": "session:one", "path": path})
            with self.assertRaises(Exception):
                await mcp.call_tool("cancel_task", {"task_id": "session:one"})
        try:
            asyncio.run(verify())
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
