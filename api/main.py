"""FastAPI query service for the web frontend."""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
import urllib.error
import urllib.request

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.env import load_project_env
from core.game_registry import DEFAULT_GAME, normalize_game
from retrieval.questions import normalize
from retrieval.query import answer as rag_answer

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


def _cors_origins() -> list[str]:
    raw = os.environ.get("CORS_ORIGINS", "http://localhost:3000")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _auth_required() -> bool:
    return os.environ.get("REQUIRE_AUTH", "false").strip().lower() in {"1", "true", "yes"}


def _vector_backend() -> str:
    return os.environ.get("VECTOR_BACKEND", "sqlite").strip().lower()


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

    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is required for the hosted query API"
        )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _validate_runtime_config()
    yield


app = FastAPI(title="videoSorter Query API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
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
    marker = "\n\n---\nSources"
    if marker not in text:
        return text, []
    answer_text, source_text = text.split(marker, 1)
    source_lines = [
        line.rstrip()
        for line in ("Sources" + source_text).splitlines()
        if line.strip()
    ]
    return answer_text.rstrip(), source_lines


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "vector_backend": _vector_backend(),
        "auth_required": _auth_required(),
    }


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest, _user: dict = Depends(_validate_supabase_token)) -> QueryResponse:
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
    return QueryResponse(
        answer=answer_text,
        sources=sources,
        normalized_question=parsed["normalized"],
        metadata={
            "game": game,
            "subject": subject,
            "role": parsed.get("role"),
            "reasoning": parsed.get("reasoning"),
        },
    )
