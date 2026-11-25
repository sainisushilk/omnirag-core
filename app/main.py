"""FastAPI application: admin API, public chat API and static assets."""

from __future__ import annotations

import hmac
import logging
import os
import re
import threading
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app import __version__, llm_router
from app.database import Database, load_or_create_secret
from app.rag_engine import KnowledgeBaseError, RagEngine, extract_pdf_text, fetch_url

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("omnirag")

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
ADMIN_PREFIX = "/api/admin"
MAX_PUBLIC_BODY_BYTES = 64 * 1024
MULTIPART_ALLOWANCE_BYTES = 1024 * 1024
MAX_TEXT_CHARS = 1_000_000
UNAVAILABLE_MESSAGE = "The assistant is temporarily unavailable. Please try again later."
UNAUTHORIZED_MESSAGE = "Invalid or missing admin password."

admin_header = APIKeyHeader(name="x-admin-pass", auto_error=False)


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from None
    if value < 0:
        raise RuntimeError(f"{name} must not be negative")
    return value


def password_matches(supplied: str | None, expected: str) -> bool:
    # HTTP header bytes are decoded as latin-1, so encoding back recovers the
    # original bytes; clients send non-ASCII passwords as UTF-8.
    if supplied is None:
        return False
    try:
        supplied_bytes = supplied.encode("latin-1")
    except UnicodeEncodeError:
        return False
    return hmac.compare_digest(supplied_bytes, expected.encode("utf-8"))


class RequestGuardMiddleware:
    """Rejects unauthenticated admin requests and oversized bodies before parsing them."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        state = scope["app"].state
        headers = {key: value for key, value in scope["headers"]}
        is_admin = scope["path"].startswith(ADMIN_PREFIX)
        if is_admin:
            raw = headers.get(b"x-admin-pass")
            supplied = raw.decode("latin-1") if raw is not None else None
            if not password_matches(supplied, state.admin_password):
                await JSONResponse({"detail": UNAUTHORIZED_MESSAGE}, 401)(scope, receive, send)
                return
        limit = state.max_admin_body if is_admin else MAX_PUBLIC_BODY_BYTES
        declared = headers.get(b"content-length", b"0")
        if not declared.isdigit() or int(declared) > limit:
            await JSONResponse({"detail": "Request body is too large."}, 413)(scope, receive, send)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise HTTPException(413, "Request body is too large.")
            return message

        await self.app(scope, limited_receive, send)


class RateLimiter:
    """Sliding one-minute window per client, held in memory."""

    WINDOW_SECONDS = 60.0

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}
        self._last_sweep = time.monotonic()

    def allow(self, client: str) -> bool:
        if self.limit == 0:
            return True
        now = time.monotonic()
        cutoff = now - self.WINDOW_SECONDS
        with self._lock:
            if now - self._last_sweep > self.WINDOW_SECONDS:
                self._hits = {k: v for k, v in self._hits.items() if v and v[-1] > cutoff}
                self._last_sweep = now
            hits = self._hits.setdefault(client, deque())
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True


# --------------------------------------------------------------------- schemas

Trimmed = Annotated[str, StringConstraints(strip_whitespace=True, max_length=500)]
NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class SettingsUpdate(BaseModel):
    """Partial settings; omitted fields are left unchanged."""

    model_config = ConfigDict(extra="forbid")

    chat_provider: Literal["openai", "groq", "gemini", "anthropic", "ollama", "custom"] | None = None
    chat_model: NonEmpty | None = None
    embed_provider: Literal["openai", "gemini", "ollama", "custom"] | None = None
    embed_model: NonEmpty | None = None
    openai_api_key: Trimmed | None = None
    groq_api_key: Trimmed | None = None
    gemini_api_key: Trimmed | None = None
    anthropic_api_key: Trimmed | None = None
    custom_llm_api_key: Trimmed | None = None
    custom_embed_api_key: Trimmed | None = None
    ollama_base_url: Trimmed | None = None
    custom_llm_base_url: Trimmed | None = None
    custom_embed_base_url: Trimmed | None = None
    bot_title: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=60)
    ] | None = None
    brand_color: str | None = None

    @field_validator("ollama_base_url", "custom_llm_base_url", "custom_embed_base_url")
    @classmethod
    def check_base_url(cls, value: str | None) -> str | None:
        if not value:
            return value
        parts = urlsplit(value)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("must be a valid http:// or https:// URL")
        return value.rstrip("/")

    @field_validator("brand_color")
    @classmethod
    def check_brand_color(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
            raise ValueError("must be a hex color such as #4f46e5")
        return value


class UrlTraining(BaseModel):
    url: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2048)]


class TextTraining(BaseModel):
    title: NonEmpty
    text: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_TEXT_CHARS)
    ]


class ChatRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


class DocumentRecord(BaseModel):
    id: str
    title: str
    source: str
    source_type: str
    chunk_count: int
    char_count: int
    created_at: str


class TestResult(BaseModel):
    success: bool
    message: str
    provider: str
    model: str


class SettingsSaved(BaseModel):
    settings: dict[str, str]
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- dependencies


def require_admin(
    request: Request, supplied: Annotated[str | None, Depends(admin_header)]
) -> None:
    if not password_matches(supplied, request.app.state.admin_password):
        raise HTTPException(401, UNAUTHORIZED_MESSAGE)


def get_db(request: Request) -> Database:
    return request.app.state.db


def get_engine(request: Request) -> RagEngine:
    return request.app.state.engine


Db = Annotated[Database, Depends(get_db)]
Engine = Annotated[RagEngine, Depends(get_engine)]

# ---------------------------------------------------------------------- routes

public = APIRouter()
admin = APIRouter(prefix=ADMIN_PREFIX, dependencies=[Depends(require_admin)], tags=["admin"])


@public.get("/health", tags=["public"])
def health(db: Db) -> dict[str, str]:
    db.ping()
    return {"status": "ok", "version": __version__}


@public.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("admin")


@public.get("/admin", include_in_schema=False)
def admin_page() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "admin.html",
        headers={
            "Cache-Control": "no-cache",
            "Content-Security-Policy": "frame-ancestors 'none'",
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        },
    )


@public.get("/api/public/widget-config", tags=["public"])
def widget_config(db: Db) -> dict[str, str]:
    settings = db.get_masked_settings()
    return {"bot_title": settings["bot_title"], "brand_color": settings["brand_color"]}


@public.post("/api/public/chat", tags=["public"])
def public_chat(payload: ChatRequest, request: Request, engine: Engine) -> ChatResponse:
    client = request.client.host if request.client else "unknown"
    if not request.app.state.rate_limiter.allow(client):
        raise HTTPException(
            429, "Too many questions. Please wait a minute.", headers={"Retry-After": "60"}
        )
    # Visitors get a generic message; provider details stay in the server log.
    try:
        return ChatResponse(**engine.answer(payload.question))
    except (llm_router.LLMConfigError, KnowledgeBaseError) as exc:
        logger.error("Chat is not available: %s", exc)
        raise HTTPException(503, UNAVAILABLE_MESSAGE) from exc
    except llm_router.LLMProviderError as exc:
        logger.error("Provider error during chat: %s", exc)
        raise HTTPException(502, UNAVAILABLE_MESSAGE) from exc


@admin.get("/settings")
def read_settings(db: Db) -> dict[str, str]:
    return db.get_masked_settings()


@admin.post("/settings")
def save_settings(payload: SettingsUpdate, db: Db, engine: Engine) -> SettingsSaved:
    db.save_settings(payload.model_dump(exclude_none=True))
    warnings: list[str] = []
    try:
        engine.check_embedding_compatibility()
    except KnowledgeBaseError as exc:
        warnings.append(str(exc))
    return SettingsSaved(settings=db.get_masked_settings(), warnings=warnings)


@admin.post("/test-llm")
def test_llm(db: Db, payload: SettingsUpdate | None = None) -> TestResult:
    submitted = payload.model_dump(exclude_none=True) if payload else {}
    settings = db.merge_settings(submitted)
    provider, model = settings["chat_provider"], settings["chat_model"]
    try:
        reply = llm_router.test_connection(settings)
    except (llm_router.LLMConfigError, llm_router.LLMProviderError) as exc:
        return TestResult(success=False, message=str(exc), provider=provider, model=model)
    return TestResult(
        success=True, message=reply or "Connected (empty reply)", provider=provider, model=model
    )


@admin.get("/documents")
def list_documents(db: Db, engine: Engine) -> dict[str, Any]:
    return {"documents": db.list_documents(), "total_chunks": engine.count_chunks()}


@admin.delete("/documents/{document_id}")
def delete_document(document_id: str, engine: Engine) -> dict[str, str]:
    if not engine.delete_document(document_id):
        raise HTTPException(404, "Document not found.")
    return {"status": "deleted"}


@admin.post("/train/pdf", status_code=201)
def train_pdf(
    request: Request, engine: Engine, file: Annotated[UploadFile, File()]
) -> DocumentRecord:
    limit = request.app.state.max_upload_bytes
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"The PDF is larger than {limit // (1024 * 1024)} MB.")
    raw_name = (file.filename or "").replace("\\", "/")
    name = PurePosixPath(raw_name).name.strip()[:200] or "document.pdf"
    return DocumentRecord(**engine.index_document(name, name, "pdf", extract_pdf_text(data)))


@admin.post("/train/url", status_code=201)
def train_url(payload: UrlTraining, request: Request, engine: Engine) -> DocumentRecord:
    url, title, text = fetch_url(payload.url, request.app.state.allow_private_urls)
    return DocumentRecord(**engine.index_document(title[:200], url, "url", text))


@admin.post("/train/text", status_code=201)
def train_text(payload: TextTraining, engine: Engine) -> DocumentRecord:
    return DocumentRecord(
        **engine.index_document(payload.title, payload.title, "text", payload.text)
    )


@admin.delete("/reset")
def reset_knowledge_base(engine: Engine) -> dict[str, Any]:
    return {"status": "reset", "documents_removed": engine.reset()}


# ----------------------------------------------------------------- application


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    data_dir = Path(os.environ.get("DATA_DIR", "").strip() or BASE_DIR / "data")
    data_dir.mkdir(parents=True, exist_ok=True)
    max_upload = env_int("MAX_UPLOAD_MB", 25) * 1024 * 1024
    state = app.state
    state.admin_password = load_or_create_secret(
        "ADMIN_PASSWORD", data_dir / ".admin_password", 18
    )
    master_secret = load_or_create_secret("MASTER_SECRET_KEY", data_dir / ".master_secret_key", 48)
    state.max_upload_bytes = max_upload
    # JSON may escape each character into several bytes, hence the wider text allowance.
    state.max_admin_body = max(max_upload, MAX_TEXT_CHARS * 6) + MULTIPART_ALLOWANCE_BYTES
    state.allow_private_urls = (
        os.environ.get("ALLOW_PRIVATE_URLS", "").strip().lower() in ("1", "true", "yes")
    )
    state.rate_limiter = RateLimiter(env_int("CHAT_RATE_LIMIT_PER_MINUTE", 30))
    state.db = Database(data_dir / "omnirag.db", master_secret)
    state.engine = RagEngine(state.db, data_dir / "qdrant")
    logger.info("OmniRAG Core %s is ready", __version__)
    try:
        yield
    finally:
        state.engine.close()


def knowledge_base_error(_: Request, exc: Exception) -> JSONResponse:
    status = exc.status_code if isinstance(exc, KnowledgeBaseError) else 500
    return JSONResponse({"detail": str(exc)}, status)


def config_error(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, 400)


def provider_error(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, 502)


def create_app() -> FastAPI:
    application = FastAPI(
        title="OmniRAG Core",
        version=__version__,
        description="Self-hosted RAG appliance with a universal website chat widget.",
        lifespan=lifespan,
    )
    application.add_middleware(RequestGuardMiddleware)
    # Added last so it is outermost: the widget runs on arbitrary origins, and
    # authentication uses a header rather than cookies.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["content-type", "x-admin-pass"],
    )
    application.add_exception_handler(KnowledgeBaseError, knowledge_base_error)
    application.add_exception_handler(llm_router.LLMConfigError, config_error)
    application.add_exception_handler(llm_router.LLMProviderError, provider_error)
    application.include_router(public)
    application.include_router(admin)
    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return application


app = create_app()
