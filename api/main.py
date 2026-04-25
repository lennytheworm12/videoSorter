"""FastAPI query service for the web frontend."""

from __future__ import annotations

from contextlib import asynccontextmanager
from contextlib import contextmanager
from datetime import datetime, timezone
import logging
import os
import re
from threading import Condition, Lock
import time
import urllib.error
import urllib.request

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.env import load_project_env
from core.game_registry import DEFAULT_GAME, normalize_game
from retrieval.questions import normalize
from retrieval.query import answer as rag_answer, current_retrieval_mode

load_project_env()


class QueryRequest(BaseModel):
    question: str = Field(min_length=2)
    game: str = DEFAULT_GAME
    show_sources: bool = True
    split_detail: bool = False
    top_k: int = Field(default=35, ge=1, le=80)


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    normalized_question: str
    metadata: dict


_QUERY_LIMIT_LOCK = Lock()
_QUERY_COUNT_BY_DAY_AND_IP: dict[tuple[str, str], int] = {}
_QUERY_SLOT_CONDITION = Condition(Lock())
_ACTIVE_QUERY_COUNT = 0
_WAITING_QUERY_COUNT = 0


class _HealthAccessLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, tuple) or len(args) < 3:
            return True
        path = str(args[2])
        return not path.startswith("/health")


def _cors_origins() -> list[str]:
    raw = os.environ.get(
        "CORS_ORIGINS",
        ",".join(
            [
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                "https://lennytheworm12.github.io",
            ]
        ),
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _cors_origin_regex() -> str | None:
    raw = os.environ.get("CORS_ORIGIN_REGEX", "").strip()
    return raw or None


def _auth_required() -> bool:
    return os.environ.get("REQUIRE_AUTH", "false").strip().lower() in {"1", "true", "yes"}


def _vector_backend() -> str:
    return os.environ.get("VECTOR_BACKEND", "sqlite").strip().lower()


def _embedding_backend() -> str:
    return os.environ.get("EMBEDDING_BACKEND", "local").strip().lower() or "local"


def _daily_query_limit() -> int:
    raw = os.environ.get("DAILY_QUERY_LIMIT", "100").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 100


def _backend_label() -> str:
    return os.environ.get("BACKEND_LABEL", "Query backend").strip() or "Query backend"


def _backend_quality() -> str:
    return os.environ.get("BACKEND_QUALITY", "standard").strip().lower() or "standard"


def _retrieval_mode() -> str:
    raw = os.environ.get("RETRIEVAL_MODE", "").strip().lower()
    if raw:
        return raw
    embedding_backend = _embedding_backend()
    if embedding_backend == "hf_remote":
        return "semantic-hf-remote"
    if embedding_backend == "bm25_only":
        return "bm25-fallback"
    if _vector_backend() == "sqlite":
        return "semantic-local"
    return "semantic-local"


def _max_active_queries() -> int:
    raw = os.environ.get("MAX_ACTIVE_QUERIES", "8").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 8


def _max_queued_queries() -> int:
    raw = os.environ.get("MAX_QUEUED_QUERIES", "8").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 8


def _queue_wait_timeout_seconds() -> float:
    raw = os.environ.get("QUEUE_WAIT_TIMEOUT_SECONDS", "20").strip()
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 20.0


def _validate_runtime_config() -> None:
    if _auth_required():
        missing = [
            name
            for name in ("SUPABASE_URL",)
            if not os.environ.get(name)
        ]
        has_public_key = bool(
            os.environ.get("SUPABASE_PUBLISHABLE_KEY")
            or os.environ.get("SUPABASE_ANON_KEY")
            or os.environ.get("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY")
            or os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY")
        )
        if not has_public_key:
            missing.append("SUPABASE_PUBLISHABLE_KEY or SUPABASE_ANON_KEY")
        if missing:
            raise RuntimeError(
                "REQUIRE_AUTH=true but missing auth env vars: " + ", ".join(missing)
            )

    if _vector_backend() == "supabase" and not os.environ.get("SUPABASE_DATABASE_URL"):
        raise RuntimeError(
            "VECTOR_BACKEND=supabase requires SUPABASE_DATABASE_URL"
        )

    if _embedding_backend() == "hf_remote" and not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "EMBEDDING_BACKEND=hf_remote requires HF_TOKEN"
        )

    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is required for the hosted query API"
        )


def _configure_access_log_filters() -> None:
    logger = logging.getLogger("uvicorn.access")
    if getattr(logger, "_videosorter_health_filter_installed", False):
        return
    logger.addFilter(_HealthAccessLogFilter())
    logger._videosorter_health_filter_installed = True  # type: ignore[attr-defined]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _configure_access_log_filters()
    _validate_runtime_config()
    yield


app = FastAPI(title="videoSorter Query API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_origin_regex=_cors_origin_regex(),
    allow_credentials=_auth_required(),
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


def _validate_supabase_token(authorization: str | None = Header(default=None)) -> dict:
    if not _auth_required():
        return {"id": "local-dev"}
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    supabase_url = os.environ.get("SUPABASE_URL")
    anon_key = (
        os.environ.get("SUPABASE_PUBLISHABLE_KEY")
        or os.environ.get("SUPABASE_ANON_KEY")
        or os.environ.get("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY")
        or os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY")
    )
    if not supabase_url or not anon_key:
        raise HTTPException(status_code=500, detail="Supabase auth env vars are not configured")

    req = urllib.request.Request(
        supabase_url.rstrip("/") + "/auth/v1/user",
        headers={
            "Authorization": authorization,
            "apikey": anon_key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=401, detail="Invalid bearer token")
    except urllib.error.HTTPError as exc:
        raise HTTPException(status_code=401, detail="Invalid bearer token") from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=503, detail="Could not validate bearer token") from exc
    return {"id": "supabase-user"}


def _split_answer_sources(text: str) -> tuple[str, list[str]]:
    match = re.search(r"\n+\s*---\s*\n\s*(Sources(?:[^\n]*)?)\n", text, flags=re.IGNORECASE)
    if not match:
        return text, []
    answer_text = text[:match.start()]
    source_text = text[match.start():]
    source_lines = [
        line.rstrip()
        for line in re.sub(r"^\s*---\s*\n", "", source_text, count=1).splitlines()
        if line.strip()
    ]
    return answer_text.rstrip(), source_lines


def _client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "").strip()
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _enforce_daily_query_limit(request: Request) -> None:
    limit = _daily_query_limit()
    if limit <= 0:
        return

    today = datetime.now(timezone.utc).date().isoformat()
    client_ip = _client_ip(request)
    current_key = (today, client_ip)

    with _QUERY_LIMIT_LOCK:
        stale_keys = [key for key in _QUERY_COUNT_BY_DAY_AND_IP if key[0] != today]
        for key in stale_keys:
            _QUERY_COUNT_BY_DAY_AND_IP.pop(key, None)

        next_count = _QUERY_COUNT_BY_DAY_AND_IP.get(current_key, 0) + 1
        if next_count > limit:
            raise HTTPException(
                status_code=429,
                detail=f"Daily query limit reached for this IP ({limit} per day).",
            )
        _QUERY_COUNT_BY_DAY_AND_IP[current_key] = next_count


def _queue_snapshot() -> dict[str, int]:
    with _QUERY_SLOT_CONDITION:
        return {
            "active_queries": _ACTIVE_QUERY_COUNT,
            "queued_queries": _WAITING_QUERY_COUNT,
            "max_active_queries": _max_active_queries(),
            "max_queued_queries": _max_queued_queries(),
        }


@contextmanager
def _acquire_query_slot() -> None:
    max_active = _max_active_queries()
    if max_active <= 0:
        yield
        return

    global _ACTIVE_QUERY_COUNT, _WAITING_QUERY_COUNT
    wait_timeout = _queue_wait_timeout_seconds()
    start = time.monotonic()

    with _QUERY_SLOT_CONDITION:
        if _ACTIVE_QUERY_COUNT >= max_active:
            if _WAITING_QUERY_COUNT >= _max_queued_queries():
                raise HTTPException(
                    status_code=503,
                    detail="Query backend is saturated: active slots and queue are full.",
                )
            _WAITING_QUERY_COUNT += 1
            try:
                while _ACTIVE_QUERY_COUNT >= max_active:
                    remaining = wait_timeout - (time.monotonic() - start)
                    if remaining <= 0:
                        raise HTTPException(
                            status_code=503,
                            detail="Query backend queue wait timeout exceeded.",
                        )
                    _QUERY_SLOT_CONDITION.wait(timeout=remaining)
            finally:
                _WAITING_QUERY_COUNT -= 1

        _ACTIVE_QUERY_COUNT += 1

    try:
        yield
    finally:
        with _QUERY_SLOT_CONDITION:
            _ACTIVE_QUERY_COUNT -= 1
            _QUERY_SLOT_CONDITION.notify()


@app.get("/health")
def health() -> dict:
    payload = {
        "ok": True,
        "backend_label": _backend_label(),
        "backend_quality": _backend_quality(),
        "retrieval_mode": _retrieval_mode(),
        "semantic_enabled": _retrieval_mode().startswith("semantic"),
        "vector_backend": _vector_backend(),
        "embedding_backend": _embedding_backend(),
        "auth_required": _auth_required(),
        "daily_query_limit": _daily_query_limit(),
    }
    payload.update(_queue_snapshot())
    return payload


@app.post("/api/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    request: Request,
    _user: dict = Depends(_validate_supabase_token),
) -> QueryResponse:
    _enforce_daily_query_limit(request)
    with _acquire_query_slot():
        game = normalize_game(req.game)
        parsed = normalize(req.question, game=game)
        subject = parsed.get("subject") if game == "aoe2" else None
        generated = rag_answer(
            question=parsed["normalized"],
            role=parsed.get("role"),
            subject=subject,
            game=game,
            top_k=req.top_k,
            show_sources=req.show_sources,
            aoe2_split_detail=req.split_detail,
        )
    answer_text, sources = _split_answer_sources(generated)
    effective_retrieval_mode = current_retrieval_mode() or _retrieval_mode()
    return QueryResponse(
        answer=answer_text,
        sources=sources,
        normalized_question=parsed["normalized"],
        metadata={
            "game": game,
            "subject": subject,
            "role": parsed.get("role"),
            "reasoning": parsed.get("reasoning"),
            "backend_label": _backend_label(),
            "backend_quality": _backend_quality(),
            "retrieval_mode": effective_retrieval_mode,
            "semantic_enabled": effective_retrieval_mode.startswith("semantic"),
        },
    )
