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


def publish_current_url(api_url: str = DEFAULT_NGROK_API) -> dict:
    try:
        payload = _fetch_ngrok_payload(api_url)
        public_url = _https_tunnel_url(payload)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        public_url = None

    value = {
        "url": public_url,
        "online": bool(public_url),
        "backend_label": "Home strong backend",
        "backend_quality": "strong",
        "source": "ngrok",
    }
    upsert_runtime_config(RUNTIME_KEY, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish ngrok URL to Supabase runtime config")
    parser.add_argument("--api-url", default=DEFAULT_NGROK_API, help="ngrok local API URL")
    parser.add_argument("--watch", action="store_true", help="Continuously watch and publish changes")
    parser.add_argument("--interval", type=float, default=10.0, help="Watch interval in seconds")
    args = parser.parse_args()

    if not args.watch:
        value = publish_current_url(args.api_url)
        status = value["url"] if value["online"] else "offline"
        print(f"Published primary backend: {status}")
        return

    last_url: str | None | object = object()
    while True:
        value = publish_current_url(args.api_url)
        current_url = value["url"]
        if current_url != last_url:
            status = current_url if value["online"] else "offline"
            print(f"[ngrok] primary backend updated: {status}", flush=True)
            last_url = current_url
        time.sleep(max(args.interval, 2.0))


if __name__ == "__main__":
    main()
