"""Start the local bridge or attach an existing official Secure MCP Tunnel."""

import argparse
import getpass
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

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STATE = HERE / ".runtime" / "service.json"
SERVICE = "arboids-local-codex"


def health(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
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


def verify_tunnel_access(tunnel_id, environment):
    proxy = environment.get("CONTROL_PLANE_HTTP_PROXY")
    try:
        with httpx.Client(proxy=proxy, trust_env=False, timeout=15) as client:
            response = client.get(f"https://api.openai.com/v1/tunnels/{tunnel_id}",
                                  headers={"Authorization": "Bearer " + environment["CONTROL_PLANE_API_KEY"]})
            response.raise_for_status()
            data = response.json()
        return {"tunnel_id": data.get("id", tunnel_id), "cloud_access": "verified"}
    except httpx.HTTPStatusError as error:
        raise RuntimeError(f"Tunnel access check returned HTTP {error.response.status_code}; the existing runtime was left running.") from None
    except httpx.RequestError:
        raise RuntimeError("Tunnel access check could not reach OpenAI; the existing runtime was left running.") from None


def connect(port, tunnel_id, alias, restart_tunnel=False, proxy=None):
    if not tunnel_id or not re.fullmatch(r"tunnel_[A-Za-z0-9_-]+", tunnel_id):
        raise ValueError("Supply the tunnel_id shown in official OpenAI Platform settings.")
    executable = Path.home() / ".codex" / "bin" / "tunnel-client" / "tunnel-client.exe"
    if not executable.is_file():
        raise RuntimeError("The official tunnel-client executable is not installed.")
    print(json.dumps(start(port), ensure_ascii=False), flush=True)
    environment = tunnel_environment(proxy)
    print(json.dumps({"control_plane_proxy": environment.get("CONTROL_PLANE_HTTP_PROXY", "direct")}), flush=True)
    if not environment.get("CONTROL_PLANE_API_KEY"):
        environment["CONTROL_PLANE_API_KEY"] = getpass.getpass("Restricted tunnel runtime API key (hidden; not saved): ")
    if not environment["CONTROL_PLANE_API_KEY"]:
        raise ValueError("A restricted tunnel runtime key is required.")
    command = [str(executable), "runtimes", "connect", "--alias", alias, "--profile", alias,
               "--tunnel-id", tunnel_id, "--mcp-server-url", f"http://127.0.0.1:{port}/mcp",
               "--runtime-api-key", "env:CONTROL_PLANE_API_KEY", "--json"]
    try:
        print(json.dumps(verify_tunnel_access(tunnel_id, environment)), flush=True)
        if restart_tunnel:
            existing = subprocess.run([str(executable), "runtimes", "status", alias, "--json"],
                                      env=environment, capture_output=True, text=True, encoding="utf-8")
            if existing.returncode == 0:
                state = json.loads(existing.stdout)
                if state.get("tunnel_id") != tunnel_id:
                    raise RuntimeError("This alias belongs to another tunnel; it will not be stopped.")
                if state.get("process_running"):
                    subprocess.run([str(executable), "runtimes", "stop", alias, "--json"],
                                   env=environment, check=True)
        subprocess.run(command, env=environment, check=True)
        result = subprocess.run([str(executable), "runtimes", "status", alias, "--json"],
                                env=environment, check=True, capture_output=True, text=True, encoding="utf-8")
        state = json.loads(result.stdout)
        summary = {name: state.get(name) for name in ("alias", "tunnel_id", "process_running", "healthy", "ready", "health_details_url")}
        summary["control_plane_proxy_configured"] = bool(environment.get("CONTROL_PLANE_HTTP_PROXY"))
        summary["note"] = "Local readiness is separate from cloud polling; verify control-plane status before testing ChatGPT."
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        environment.pop("CONTROL_PLANE_API_KEY", None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "stop", "restart", "status", "connect"])
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--tunnel-id")
    parser.add_argument("--alias", default="arboids")
    parser.add_argument("--control-plane-proxy", help="Explicit HTTP proxy for OpenAI tunnel communication only.")
    parser.add_argument("--restart-tunnel", action="store_true", help="Restart this tunnel runtime after secure key entry; local Codex jobs continue.")
    args = parser.parse_args()
    if args.action == "connect":
        connect(args.port, args.tunnel_id, args.alias, args.restart_tunnel, args.control_plane_proxy)
        return
    if args.action in {"stop", "restart"}:
        result = stop(args.port)
    if args.action in {"start", "restart"}:
        result = start(args.port)
    if args.action == "status":
        result = health(args.port) or {"running": False}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
