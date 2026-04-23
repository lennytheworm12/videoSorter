"""Shared YouTube network policy for transcript and metadata fetches."""

from __future__ import annotations

import os
import pathlib
import re
import time
from dataclasses import dataclass, field
from typing import Callable, TypeVar

from dotenv import load_dotenv

load_dotenv()

T = TypeVar("T")


def _parse_proxy(raw: str) -> str:
    """Convert host:port:user:pass or standard URL to http://user:pass@host:port."""
    raw = raw.strip()
    if raw.startswith("http"):
        return raw
    parts = raw.split(":")
    if len(parts) == 4:
        host, port, user, password = parts
        return f"http://{user}:{password}@{host}:{port}"
    raise ValueError(f"Unrecognised proxy format: {raw}")


def _parse_delay_list(raw: str, default: list[int]) -> list[int]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    parsed: list[int] = []
    for value in values:
        try:
            parsed.append(max(1, int(value)))
        except ValueError:
            continue
    return parsed or list(default)


def load_proxy_list() -> list[str]:
    proxy_file = pathlib.Path("proxies.txt")
    if proxy_file.exists():
        raw_entries = [
            line.strip()
            for line in proxy_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    else:
        raw_entries = [
            part.strip()
            for part in os.environ.get("PROXY_LIST", "").split(",")
            if part.strip()
        ]
    return [_parse_proxy(proxy) for proxy in raw_entries]


def is_rate_limited(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "too many requests" in msg or ("ip" in msg and "block" in msg)


def is_proxy_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "502",
            "bad gateway",
            "proxyerror",
            "tunnel connection failed",
            "unable to connect to proxy",
        )
    )


def retry_delay_from_exception(exc: Exception, default: int) -> int:
    match = re.search(r"retry[^\d]*(\d+(?:\.\d+)?)\s*s", str(exc), re.IGNORECASE)
    if match:
        return max(default, int(float(match.group(1))) + 2)
    return default


@dataclass
class YouTubeNetworkPolicy:
    proxy_urls: list[str] = field(default_factory=load_proxy_list)
    direct_retry_delays: list[int] = field(
        default_factory=lambda: _parse_delay_list(
            os.environ.get("YOUTUBE_DIRECT_RETRY_DELAYS", "5,15,45"),
            [5, 15, 45],
        )
    )
    proxy_retry_delays: list[int] = field(
        default_factory=lambda: _parse_delay_list(
            os.environ.get("YOUTUBE_PROXY_RETRY_DELAYS", "20,40"),
            [20, 40],
        )
    )
    local_blocked_until: float = 0.0
    _proxy_index: int = 0

    def has_proxies(self) -> bool:
        return bool(self.proxy_urls)

    def direct_available(self) -> bool:
        return time.time() >= self.local_blocked_until

    def next_proxy_url(self) -> str | None:
        if not self.proxy_urls:
            return None
        url = self.proxy_urls[self._proxy_index % len(self.proxy_urls)]
        self._proxy_index += 1
        return url

    def block_direct_for(self, seconds: int) -> None:
        self.local_blocked_until = max(self.local_blocked_until, time.time() + max(1, seconds))

    def seconds_until_direct_retry(self) -> int:
        remaining = int(round(self.local_blocked_until - time.time()))
        return max(0, remaining)

    def describe(self) -> str:
        if self.has_proxies():
            return f"[transcribe] {len(self.proxy_urls)} proxy(ies) loaded — local IP first, proxies only after local restriction"
        return "[transcribe] No proxy configured — local IP only"

    def run(
        self,
        label: str,
        operation: Callable[[str | None], T],
    ) -> T:
        last_exc: Exception | None = None

        if self.direct_available():
            try:
                return operation(None)
            except Exception as exc:
                last_exc = exc
                if not is_rate_limited(exc):
                    raise

                for wait in self.direct_retry_delays:
                    print(f"    [{label}] local rate limit — waiting {wait}s before retrying direct…", flush=True)
                    time.sleep(wait)
                    try:
                        return operation(None)
                    except Exception as retry_exc:
                        last_exc = retry_exc
                        if not is_rate_limited(retry_exc):
                            raise

                cooldown = retry_delay_from_exception(last_exc, self.direct_retry_delays[-1] if self.direct_retry_delays else 30)
                self.block_direct_for(cooldown)
                if self.has_proxies():
                    print(
                        f"    [{label}] local IP appears restricted — switching to proxies "
                        f"for {self.seconds_until_direct_retry()}s",
                        flush=True,
                    )
                else:
                    raise last_exc
        elif self.has_proxies():
            print(
                f"    [{label}] local IP cooling down ({self.seconds_until_direct_retry()}s left) — using proxies",
                flush=True,
            )

        if not self.has_proxies():
            if last_exc:
                raise last_exc
            return operation(None)

        proxy_waits = [0] + self.proxy_retry_delays
        for wait in proxy_waits:
            proxy_url = self.next_proxy_url()
            try:
                return operation(proxy_url)
            except Exception as exc:
                last_exc = exc
                if is_proxy_error(exc):
                    print(f"    [{label}] proxy error — cycling to next proxy…", flush=True)
                    continue
                if is_rate_limited(exc):
                    delay = retry_delay_from_exception(exc, wait or (self.proxy_retry_delays[0] if self.proxy_retry_delays else 30))
                    print(f"    [{label}] proxy rate limit — waiting {delay}s…", flush=True)
                    time.sleep(delay)
                    continue
                raise

        if last_exc:
            raise last_exc
        raise RuntimeError(f"{label} failed without a captured exception")
