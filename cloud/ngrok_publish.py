"""Publish the current ngrok public URL into Supabase runtime config."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request

from cloud.runtime_config import upsert_runtime_config

RUNTIME_KEY = "primary_backend"
DEFAULT_NGROK_API = "http://127.0.0.1:4040/api/tunnels"


def _fetch_ngrok_payload(api_url: str) -> dict:
    req = urllib.request.Request(api_url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _https_tunnel_url(payload: dict) -> str | None:
    tunnels = payload.get("tunnels") or []
    for tunnel in tunnels:
        public_url = str(tunnel.get("public_url") or "")
        proto = str(tunnel.get("proto") or "")
        if public_url.startswith("https://") and proto in {"https", "http"}:
            return public_url.rstrip("/")
    return None


def current_ngrok_value(api_url: str = DEFAULT_NGROK_API) -> dict:
    try:
        payload = _fetch_ngrok_payload(api_url)
        public_url = _https_tunnel_url(payload)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        public_url = None

    return {
        "url": public_url,
        "online": bool(public_url),
        "backend_label": "Home strong backend",
        "backend_quality": "strong",
        "source": "ngrok",
    }


def publish_current_url(api_url: str = DEFAULT_NGROK_API) -> dict:
    value = current_ngrok_value(api_url)
    upsert_runtime_config(RUNTIME_KEY, value)
    return value


def _status(value: dict) -> str:
    return value["url"] if value["online"] else "offline"


def _publish_or_report(value: dict) -> bool:
    try:
        upsert_runtime_config(RUNTIME_KEY, value)
    except Exception as exc:
        status = _status(value)
        print(
            "[ngrok] Supabase publish failed; use this URL manually: "
            f"{status}. Error: {exc}",
            flush=True,
        )
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish ngrok URL to Supabase runtime config")
    parser.add_argument("--api-url", default=DEFAULT_NGROK_API, help="ngrok local API URL")
    parser.add_argument("--print-only", action="store_true", help="Print the current ngrok HTTPS URL without writing to Supabase")
    parser.add_argument("--watch", action="store_true", help="Continuously watch and publish changes")
    parser.add_argument("--interval", type=float, default=10.0, help="Watch interval in seconds")
    args = parser.parse_args()

    if not args.watch:
        value = current_ngrok_value(args.api_url)
        status = _status(value)
        if args.print_only:
            print(f"Current ngrok backend: {status}")
            return
        if _publish_or_report(value):
            print(f"Published primary backend: {status}")
        return

    last_url: str | None | object = object()
    last_publish_ok: bool | None = None
    while True:
        value = current_ngrok_value(args.api_url)
        current_url = value["url"]
        publish_ok = True if args.print_only else _publish_or_report(value)
        if current_url != last_url or publish_ok != last_publish_ok:
            status = _status(value)
            if args.print_only:
                print(f"[ngrok] current backend: {status}", flush=True)
            elif publish_ok:
                print(f"[ngrok] primary backend updated: {status}", flush=True)
            last_url = current_url
            last_publish_ok = publish_ok
        time.sleep(max(args.interval, 2.0))


if __name__ == "__main__":
    main()
