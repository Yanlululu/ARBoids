"""Start the local bridge or attach an existing official Secure MCP Tunnel."""

import argparse
import contextlib
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STATE = HERE / ".runtime" / "service.json"
SERVICE = "arboids-local-codex"
WATCH_STATE = HERE / ".runtime" / "tunnel-watch.json"
TUNNEL_CLIENT = Path.home() / ".codex" / "bin" / "tunnel-client" / "tunnel-client.exe"


def health(port):
    try:
        direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with direct.open(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            result = json.load(response)
        if result.get("service") != SERVICE or Path(result["project"]).resolve() != ROOT:
            raise RuntimeError("This port belongs to another service.")
        return result
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return None


def start(port):
    current = health(port)
    if current:
        return current
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
    process = subprocess.Popen([sys.executable, "-X", "utf8", str(HERE / "server.py"), "serve", "--port", str(port)],
                               cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, **options)
    for _ in range(40):
        time.sleep(.25)
        current = health(port)
        if current:
            STATE.parent.mkdir(exist_ok=True)
            STATE.write_text(json.dumps({"pid": current["pid"], "launcher_pid": process.pid, "port": port}), encoding="utf-8")
            return current
        if process.poll() is not None:
            raise RuntimeError(f"MCP process exited with code {process.returncode}.")
    process.terminate()
    raise RuntimeError("Local MCP did not become ready.")


def stop(port):
    current = health(port)
    if not current:
        return {"running": False}
    if STATE.exists():
        recorded = json.loads(STATE.read_text(encoding="utf-8-sig"))
        if recorded["pid"] != current["pid"]:
            raise RuntimeError("Service identity changed; inspect the listener before stopping it.")
    os.kill(current["pid"], signal.SIGTERM)
    for _ in range(20):
        if health(port) is None:
            return {"running": False, "workers": "Submitted jobs continue. Use cancel_task to stop a job."}
        time.sleep(.25)
    raise RuntimeError("The service did not stop.")


def tunnel_environment(proxy=None):
    environment = os.environ.copy()
    if proxy is not None:
        parsed = urllib.parse.urlsplit(proxy)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
            raise ValueError("Use an HTTP proxy URL without embedded credentials.")
        environment["CONTROL_PLANE_HTTP_PROXY"] = proxy
    if not environment.get("CONTROL_PLANE_HTTP_PROXY"):
        proxies = urllib.request.getproxies()
        proxy = proxies.get("https") or proxies.get("http")
        if not proxy and os.name == "nt":
            # NO_PROXY alone makes urllib skip the Windows system proxy.
            proxies = urllib.request.getproxies_registry()
            proxy = proxies.get("https") or proxies.get("http")
        if proxy:
            # Only the OpenAI control plane uses this route. Local MCP stays direct.
            environment["CONTROL_PLANE_HTTP_PROXY"] = proxy
    return environment


class TunnelTransportError(RuntimeError):
    pass


def proxy_candidates(proxy=None):
    """Read the live Windows setting first; a stale shell must not pin an old port."""
    if proxy is not None:
        return [tunnel_environment(proxy)["CONTROL_PLANE_HTTP_PROXY"]]
    candidates = []
    if os.name == "nt":
        current = urllib.request.getproxies_registry()
        candidates.extend([current.get("https"), current.get("http")])
    current = urllib.request.getproxies()
    candidates.extend([os.environ.get("CONTROL_PLANE_HTTP_PROXY"), current.get("https"), current.get("http")])
    result = []
    for candidate in candidates:
        if candidate and candidate not in result:
            try:
                tunnel_environment(candidate)  # Reject credentials and unsupported proxy schemes.
                result.append(candidate)
            except ValueError:
                continue
    return result + [None]


def route_environment(environment, proxy):
    result = environment.copy()
    # The local MCP endpoint must remain direct even when a shell has proxy variables.
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
                 "TUNNEL_CLIENT_HTTP_PROXY", "MCP_HTTP_PROXY", "CONTROL_PLANE_HTTP_PROXY"):
        result.pop(name, None)
    result["NO_PROXY"] = "localhost,127.0.0.1,::1"
    if proxy:
        result["CONTROL_PLANE_HTTP_PROXY"] = proxy
    return result


def select_route(tunnel_id, environment, proxy=None):
    for candidate in proxy_candidates(proxy):
        selected = route_environment(environment, candidate)
        try:
            return selected, verify_tunnel_access(tunnel_id, selected)
        except TunnelTransportError:
            continue
    raise TunnelTransportError("No current system proxy, environment proxy, or direct route could reach OpenAI. The existing runtime was left running.")


def verify_tunnel_access(tunnel_id, environment):
    proxy = environment.get("CONTROL_PLANE_HTTP_PROXY")
    try:
        with httpx.Client(proxy=proxy, trust_env=False, timeout=8) as client:
            response = client.get(f"https://api.openai.com/v1/tunnels/{tunnel_id}",
                                  headers={"Authorization": "Bearer " + environment["CONTROL_PLANE_API_KEY"]})
            response.raise_for_status()
            data = response.json()
        return {"tunnel_id": data.get("id", tunnel_id), "cloud_access": "verified"}
    except httpx.HTTPStatusError as error:
        raise RuntimeError(f"Tunnel access check returned HTTP {error.response.status_code}; the existing runtime was left running.") from None
    except httpx.RequestError:
        raise TunnelTransportError("Tunnel access check could not reach OpenAI; the existing runtime was left running.") from None


@contextlib.contextmanager
def runtime_lock():
    """Serialize explicit reconnects and the watcher without locking the message database."""
    if os.name == "nt":
        import win32event
        name = "Local\\ARBoidsTunnel-" + hashlib.sha256(str(ROOT).encode()).hexdigest()[:20]
        handle = win32event.CreateMutex(None, False, name)
        acquired = False
        try:
            acquired = win32event.WaitForSingleObject(handle, 60000) in (0, 128)
            if not acquired:
                raise RuntimeError("Another tunnel reconnect is in progress; retry shortly.")
            yield
        finally:
            if acquired:
                win32event.ReleaseMutex(handle)
            handle.Close()
    else:
        import fcntl
        STATE.parent.mkdir(exist_ok=True)
        with (STATE.parent / "tunnel.lock").open("a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def read_watch():
    try:
        return json.loads(WATCH_STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_watch(value):
    WATCH_STATE.parent.mkdir(exist_ok=True)
    temp = WATCH_STATE.with_suffix(f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temp.replace(WATCH_STATE)


def runtime_status(alias, environment=None):
    # With credentials, this CLI also performs a remote metadata lookup. Keep status
    # local and check actual cloud polling through the runtime's health endpoint.
    local_environment = (environment if environment is not None else os.environ).copy()
    for key in ("CONTROL_PLANE_API_KEY", "OPENAI_API_KEY", "OPENAI_ADMIN_KEY"):
        local_environment.pop(key, None)
    result = subprocess.run([str(TUNNEL_CLIENT), "runtimes", "status", alias, "--json"], env=local_environment,
                            capture_output=True, text=True, encoding="utf-8", timeout=15,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    return json.loads(result.stdout) if result.returncode == 0 else {}


def management_environment(environment):
    # The official runtimes management commands use standard proxy variables;
    # their child poller additionally uses CONTROL_PLANE_HTTP_PROXY.
    proxy = environment.get("CONTROL_PLANE_HTTP_PROXY")
    result = route_environment(environment, proxy)
    if proxy:
        result.update(HTTP_PROXY=proxy, HTTPS_PROXY=proxy)
    return result


def runtime_matches(state, tunnel_id, port):
    if state and (state.get("tunnel_id") != tunnel_id or
                  state.get("target_value", state.get("process", {}).get("target_value")) != f"http://127.0.0.1:{port}/mcp"):
        raise RuntimeError("This runtime alias belongs to another tunnel or MCP endpoint; it will not be changed.")


def polling_healthy(state):
    url = state.get("health_details_url")
    if not state.get("process_running") or not url:
        return False
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return False
    try:
        with httpx.Client(trust_env=False, timeout=3) as client:
            details = client.get(url, params={"details": "true"}).json()
        control = details.get("components", {}).get("control-plane", {})
        return control.get("status") == "ok" and control.get("consecutive_failures", 0) == 0
    except (httpx.HTTPError, ValueError):
        return False


def attach_runtime(port, tunnel_id, alias, environment, restart=False):
    existing = runtime_status(alias, environment)
    runtime_matches(existing, tunnel_id, port)
    options = dict(env=management_environment(environment), check=True, capture_output=True, text=True, encoding="utf-8",
                   timeout=90, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if restart and existing.get("process_running"):
        subprocess.run([str(TUNNEL_CLIENT), "runtimes", "stop", alias, "--json"], **options)
    command = [str(TUNNEL_CLIENT), "runtimes", "connect", "--alias", alias, "--profile", alias,
               "--tunnel-id", tunnel_id, "--mcp-server-url", f"http://127.0.0.1:{port}/mcp",
               "--runtime-api-key", "env:CONTROL_PLANE_API_KEY", "--json"]
    subprocess.run(command, **options)
    return runtime_status(alias, environment)


def watch_tunnel(port, tunnel_id, alias, token, proxy=None):
    """Keep the runtime key only in this process while adapting the outbound route."""
    environment = os.environ.copy()
    failures = 0
    while True:
        delay = 20
        try:
            with runtime_lock():
                state = read_watch()
                if state.get("token") != token:
                    return
                runtime = runtime_status(alias, environment)
                runtime_matches(runtime, tunnel_id, port)
                # Respect a user explicitly stopping a runtime; do not resurrect it.
                if not runtime or not runtime.get("process_running"):
                    state.update(status="stopped", pid=None, note="Runtime stopped. Reconnect when needed.")
                    write_watch(state)
                    return
                healthy = polling_healthy(runtime)
                failures = 0 if healthy else failures + 1
                desired = proxy_candidates(proxy)[0]
                changed = desired != state.get("preferred_proxy")
                if changed or failures >= 3:
                    selected, _ = select_route(tunnel_id, environment, proxy)
                    route = selected.get("CONTROL_PLANE_HTTP_PROXY")
                    if route != state.get("proxy") or failures >= 3:
                        attach_runtime(port, tunnel_id, alias, selected, restart=True)
                        environment = selected
                        state["proxy"] = route
                        state["reconnections"] = state.get("reconnections", 0) + 1
                        failures = 0
                    state["preferred_proxy"] = desired
                state.update(pid=os.getpid(), status="watching", checked_at=time.time(), consecutive_failures=failures, error=None)
                write_watch(state)
        except Exception as error:
            # State is non-secret; never write command output, headers, or the environment.
            with runtime_lock():
                state = read_watch()
                if state.get("token") != token:
                    return
                detail = str(error) if isinstance(error, RuntimeError) else type(error).__name__
                state.update(status="waiting_for_network", checked_at=time.time(), error=detail)
                write_watch(state)
            delay = 60
        time.sleep(delay)


def start_watcher(port, tunnel_id, alias, environment, proxy=None):
    token = uuid.uuid4().hex
    command = [sys.executable, "-X", "utf8", str(HERE / "control.py"), "watch", "--port", str(port),
               "--tunnel-id", tunnel_id, "--alias", alias, "--watch-token", token]
    if proxy:
        command.extend(["--control-plane-proxy", proxy])
    state = dict(token=token, pid=None, status="starting", tunnel_id=tunnel_id, alias=alias, port=port,
                 proxy=environment.get("CONTROL_PLANE_HTTP_PROXY"), preferred_proxy=proxy_candidates(proxy)[0],
                 key_storage="process memory only", reconnections=0)
    write_watch(state)
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **options)
    state["pid"] = process.pid
    write_watch(state)
    return state


def connect(port, tunnel_id, alias, restart_tunnel=False, proxy=None):
    if not tunnel_id or not re.fullmatch(r"tunnel_[A-Za-z0-9_-]+", tunnel_id):
        raise ValueError("Supply the tunnel_id shown in official OpenAI Platform settings.")
    if not TUNNEL_CLIENT.is_file():
        raise RuntimeError("The official tunnel-client executable is not installed.")
    print(json.dumps(start(port), ensure_ascii=False), flush=True)
    environment = os.environ.copy()
    if not environment.get("CONTROL_PLANE_API_KEY"):
        environment["CONTROL_PLANE_API_KEY"] = getpass.getpass("Restricted tunnel runtime API key (hidden; not saved): ")
    if not environment["CONTROL_PLANE_API_KEY"]:
        raise ValueError("A restricted tunnel runtime key is required.")
    try:
        environment, verified = select_route(tunnel_id, environment, proxy)
        print(json.dumps(verified), flush=True)
        print(json.dumps({"control_plane_proxy": environment.get("CONTROL_PLANE_HTTP_PROXY", "direct"), "proxy_mode": "fixed" if proxy else "automatic"}), flush=True)
        with runtime_lock():
            prior = read_watch()
            needs_restart = restart_tunnel or not prior or prior.get("proxy") != environment.get("CONTROL_PLANE_HTTP_PROXY")
            state = attach_runtime(port, tunnel_id, alias, environment, restart=needs_restart)
            watcher = start_watcher(port, tunnel_id, alias, environment, proxy)
        summary = {name: state.get(name) for name in ("alias", "tunnel_id", "process_running", "healthy", "ready", "health_details_url")}
        summary["control_plane_proxy_configured"] = bool(environment.get("CONTROL_PLANE_HTTP_PROXY"))
        summary["automatic_reconnection"] = True
        summary["watcher_pid"] = watcher["pid"]
        summary["note"] = "Local readiness is separate from cloud polling; verify control-plane status before testing ChatGPT."
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        environment.pop("CONTROL_PLANE_API_KEY", None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "stop", "restart", "status", "connect", "watch"])
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--tunnel-id")
    parser.add_argument("--alias", default="arboids")
    parser.add_argument("--control-plane-proxy", help="Explicit HTTP proxy for OpenAI tunnel communication only.")
    parser.add_argument("--restart-tunnel", action="store_true", help="Restart this tunnel runtime after secure key entry; local Codex jobs continue.")
    parser.add_argument("--watch-token", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.action == "watch":
        if not args.watch_token or not os.environ.get("CONTROL_PLANE_API_KEY"):
            raise ValueError("The proxy watcher must be started by connect.")
        watch_tunnel(args.port, args.tunnel_id, args.alias, args.watch_token, args.control_plane_proxy)
        return
    if args.action == "connect":
        connect(args.port, args.tunnel_id, args.alias, args.restart_tunnel, args.control_plane_proxy)
        return
    if args.action in {"stop", "restart"}:
        result = stop(args.port)
    if args.action in {"start", "restart"}:
        result = start(args.port)
    if args.action == "status":
        result = health(args.port) or {"running": False}
        watch = read_watch()
        if watch:
            from server import alive
            result["tunnel_watcher"] = {key: value for key, value in watch.items() if key != "token"}
            result["tunnel_watcher"]["process_running"] = alive(watch.get("pid"))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
