"""Automated test suite. LiteLLM and HTTP fetching are mocked, so no network or keys are needed."""

from __future__ import annotations

import hashlib
import io
import json
import re
import sqlite3
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import requests
from fastapi.testclient import TestClient

from app import llm_router, rag_engine
from app.database import (
    DEFAULT_SETTINGS,
    SECRET_KEYS,
    Database,
    SecretDecryptionError,
    load_or_create_secret,
    mask_secret,
)
from app.main import RateLimiter, create_app
from app.rag_engine import (
    EMPTY_KB_MESSAGE,
    NOT_FOUND_MESSAGE,
    KnowledgeBaseError,
    chunk_text,
    extract_html_text,
    extract_pdf_text,
    fetch_url,
    validate_public_url,
)

PASSWORD = "test-admin-password"
MASTER = "test-master-secret"
AUTH = {"x-admin-pass": PASSWORD}
STATIC = Path(__file__).resolve().parent.parent / "static"


class FakeLLM:
    """Deterministic stand-in for litellm.completion and litellm.embedding."""

    def __init__(self) -> None:
        self.dimension = 256
        self.completions: list[dict[str, Any]] = []
        self.embeddings: list[dict[str, Any]] = []
        self.reply: str | None = "Our refund window is 30 days."
        self.completion_error: Exception | None = None
        self.embedding_error: Exception | None = None

    def vector(self, text: str) -> list[float]:
        values = [0.0] * self.dimension
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            values[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension] += 1.0
        return values if any(values) else [1.0] * self.dimension

    def completion(self, **kwargs: Any) -> Any:
        self.completions.append(kwargs)
        if self.completion_error:
            raise self.completion_error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))])

    def embedding(self, **kwargs: Any) -> Any:
        self.embeddings.append(kwargs)
        if self.embedding_error:
            raise self.embedding_error
        data = [{"index": i, "embedding": self.vector(t)} for i, t in enumerate(kwargs["input"])]
        return SimpleNamespace(data=list(reversed(data)))


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    fake = FakeLLM()
    monkeypatch.setattr(llm_router.litellm, "completion", fake.completion)
    monkeypatch.setattr(llm_router.litellm, "embedding", fake.embedding)
    return fake


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, llm: FakeLLM):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ADMIN_PASSWORD", PASSWORD)
    monkeypatch.setenv("MASTER_SECRET_KEY", MASTER)
    monkeypatch.setenv("CHAT_RATE_LIMIT_PER_MINUTE", "0")
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    monkeypatch.delenv("ALLOW_PRIVATE_URLS", raising=False)
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def ready(client: TestClient) -> TestClient:
    """A client with an OpenAI key saved."""
    save(client, openai_api_key="sk-openai-secret-ab12")
    return client


def save(client: TestClient, **values: str) -> dict[str, Any]:
    response = client.post("/api/admin/settings", headers=AUTH, json=values)
    assert response.status_code == 200, response.text
    return response.json()


def train(client: TestClient, title: str, text: str):
    return client.post("/api/admin/train/text", headers=AUTH, json={"title": title, "text": text})


def ask(client: TestClient, question: str):
    return client.post("/api/public/chat", json={"question": question})


def raw_setting(tmp_path: Path, key: str) -> str:
    conn = sqlite3.connect(tmp_path / "omnirag.db")
    try:
        return conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()[0]
    finally:
        conn.close()


def make_pdf(*pages: str) -> bytes:
    """Build a minimal valid PDF with one line of text per page."""
    objects: list[bytes] = []
    count = len(pages)
    kids = " ".join(f"{4 + i * 2} 0 R" for i in range(count))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {count} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, text in enumerate(pages):
        stream = zlib.compress(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode())
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {5 + i * 2} 0 R "
            "/Resources << /Font << /F1 3 0 R >> >> >>".encode()
        )
        objects.append(
            f"<< /Length {len(stream)} /Filter /FlateDecode >>\nstream\n".encode()
            + stream
            + b"\nendstream"
        )
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(out)


class FakeWeb:
    """Routes requests.Session.get and DNS lookups to in-memory pages."""

    def __init__(self) -> None:
        self.pages: dict[str, tuple[int, dict[str, str], bytes]] = {}
        self.addresses: dict[str, str] = {}
        self.fetched: list[str] = []

    def get(self, _session: Any, url: str, **kwargs: Any) -> requests.Response:
        assert kwargs["allow_redirects"] is False
        self.fetched.append(url)
        status, headers, body = self.pages[url]
        response = requests.Response()
        response.status_code = status
        response.url = url
        response.headers.update(headers)
        response.encoding = requests.utils.get_encoding_from_headers(response.headers)
        response.raw = io.BytesIO(body)
        return response

    def getaddrinfo(self, host: str, port: int, **_: Any) -> list[Any]:
        address = self.addresses.get(host)
        if address is None:
            raise OSError("unknown host")
        return [(2, 1, 6, "", (address, port))]


@pytest.fixture
def web(monkeypatch: pytest.MonkeyPatch) -> FakeWeb:
    fake = FakeWeb()
    monkeypatch.setattr(requests.Session, "get", lambda self, url, **kw: fake.get(self, url, **kw))
    monkeypatch.setattr(rag_engine.socket, "getaddrinfo", fake.getaddrinfo)
    return fake


# ------------------------------------------------- health, public config, static


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_widget_config_exposes_only_public_fields(client):
    save(client, bot_title="Acme Bot", brand_color="#ff0000")
    response = client.get("/api/public/widget-config")
    assert response.status_code == 200
    assert response.json() == {"bot_title": "Acme Bot", "brand_color": "#ff0000"}


def test_admin_page_and_root_redirect(client):
    assert client.get("/", follow_redirects=False).status_code == 307
    page = client.get("/admin")
    assert page.status_code == 200
    assert "Test LLM Connection" in page.text
    assert "frame-ancestors 'none'" in page.headers["content-security-policy"]


def test_widget_is_served_with_cors(client):
    response = client.get("/static/widget.js", headers={"Origin": "https://shop.example"})
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert response.headers["access-control-allow-origin"] == "*"


def test_widget_detects_origin_and_isolates_css():
    source = (STATIC / "widget.js").read_text()
    assert "document.currentScript" in source
    assert "innerHTML" not in source
    # "all:unset" on SVG paths would reset the CSS "d" property and hide the icons.
    assert "#omnirag-root *:not(path){all:unset" in source
    css = source[source.index("function css(") : source.index("function readableTextColor")]
    rules = re.findall(r'^\s+"(.*)$', css, flags=re.M)
    assert rules
    for rule in rules:
        selectors = re.sub(r"^@media \([^)]*\)\{", "", rule).split("{")[0]
        for selector in selectors.split(","):
            assert selector.startswith("#omnirag-root"), selector


@pytest.mark.parametrize("name", ["admin.html", "widget.js"])
def test_static_files_load_no_external_dependencies(name):
    source = (STATIC / name).read_text()
    assert not re.search(r"<link[^>]+href=[\"']https?:|@import|import\(|cdn\.", source)
    assert not re.search(r"<script[^>]+src=[\"']https?://(?!your-server)", source)


def test_cors_preflight_for_chat(client):
    response = client.options(
        "/api/public/chat",
        headers={
            "Origin": "https://shop.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"


# ------------------------------------------------------------------ authentication

ADMIN_ROUTES = [
    ("GET", "/api/admin/settings"),
    ("POST", "/api/admin/settings"),
    ("POST", "/api/admin/test-llm"),
    ("GET", "/api/admin/documents"),
    ("DELETE", "/api/admin/documents/some-id"),
    ("POST", "/api/admin/train/pdf"),
    ("POST", "/api/admin/train/url"),
    ("POST", "/api/admin/train/text"),
    ("DELETE", "/api/admin/reset"),
]


@pytest.mark.parametrize(("method", "path"), ADMIN_ROUTES)
@pytest.mark.parametrize("headers", [{}, {"x-admin-pass": "wrong"}, {"x-admin-pass": ""}])
def test_admin_routes_require_password(client, method, path, headers):
    assert client.request(method, path, headers=headers).status_code == 401


def test_every_admin_route_is_covered_by_the_auth_test(client):
    paths = client.get("/openapi.json").json()["paths"]
    declared = {
        (method.upper(), path.replace("{document_id}", "some-id"))
        for path, methods in paths.items()
        if path.startswith("/api/admin")
        for method in methods
    }
    assert declared == set(ADMIN_ROUTES)


def test_correct_password_is_accepted(client):
    assert client.get("/api/admin/settings", headers=AUTH).status_code == 200


def test_non_ascii_password(tmp_path, monkeypatch, llm):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ADMIN_PASSWORD", "pässwörd✓")
    monkeypatch.setenv("MASTER_SECRET_KEY", MASTER)
    with TestClient(create_app()) as c:
        good = c.get("/api/admin/settings", headers={b"x-admin-pass": "pässwörd✓".encode()})
        bad = c.get("/api/admin/settings", headers={b"x-admin-pass": "pässwörd".encode()})
    assert (good.status_code, bad.status_code) == (200, 401)


def test_missing_credentials_are_generated_and_persisted(tmp_path, monkeypatch, llm):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("MASTER_SECRET_KEY", raising=False)
    for _ in range(2):
        with TestClient(create_app()) as c:
            password_file = tmp_path / ".admin_password"
            assert password_file.stat().st_mode & 0o077 == 0
            assert (tmp_path / ".master_secret_key").stat().st_mode & 0o077 == 0
            headers = {"x-admin-pass": password_file.read_text()}
            assert c.get("/api/admin/settings", headers=headers).status_code == 200
            assert c.get("/api/admin/settings", headers={"x-admin-pass": ""}).status_code == 401


def test_empty_secret_file_is_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    (tmp_path / "secret").write_text("")
    with pytest.raises(RuntimeError, match="empty"):
        load_or_create_secret("ADMIN_PASSWORD", tmp_path / "secret", 18)


# -------------------------------------------------- secret encryption and masking


@pytest.mark.parametrize("key", SECRET_KEYS)
def test_secret_is_encrypted_masked_and_preserved(client, tmp_path, key):
    secret = f"sk-{key}-value-ab12"
    save(client, **{key: secret})

    ciphertext = raw_setting(tmp_path, key)
    assert ciphertext.startswith("gAAAA")
    assert secret not in ciphertext
    assert secret.encode() not in (tmp_path / "omnirag.db").read_bytes()

    masked = client.get("/api/admin/settings", headers=AUTH).json()[key]
    assert masked == "********ab12"

    save(client, **{key: masked, "bot_title": "Renamed"})
    assert raw_setting(tmp_path, key) == ciphertext
    assert client.app.state.db.get_settings()[key] == secret


def test_secret_can_be_replaced_and_cleared(client):
    save(client, openai_api_key="sk-first-secret-1111")
    assert save(client, openai_api_key=" sk-second-secret-2222 ")["settings"][
        "openai_api_key"
    ] == "********2222"
    assert client.app.state.db.get_settings()["openai_api_key"] == "sk-second-secret-2222"
    assert save(client, openai_api_key="")["settings"]["openai_api_key"] == ""
    assert client.app.state.db.get_settings()["openai_api_key"] == ""


def test_omitted_fields_are_unchanged(ready):
    save(ready, chat_model="gpt-4o")
    settings = ready.app.state.db.get_settings()
    assert settings["openai_api_key"] == "sk-openai-secret-ab12"
    assert settings["chat_model"] == "gpt-4o"


def test_short_secrets_are_fully_masked():
    assert mask_secret("") == ""
    assert mask_secret("abcd1234") == "********"
    assert mask_secret("long-enough-ab12") == "********ab12"


def test_changed_master_key_never_destroys_ciphertext(tmp_path):
    Database(tmp_path / "a.db", "one").save_settings({"openai_api_key": "sk-secret-abcdef"})
    other = Database(tmp_path / "a.db", "two")
    assert isinstance(other.get_settings()["openai_api_key"], SecretDecryptionError)
    masked = other.get_masked_settings()["openai_api_key"]
    other.save_settings({"openai_api_key": masked})
    with pytest.raises(llm_router.LLMConfigError, match="MASTER_SECRET_KEY"):
        llm_router.build_chat_params(other.get_settings())
    restored = Database(tmp_path / "a.db", "one")
    assert restored.get_settings()["openai_api_key"] == "sk-secret-abcdef"


@pytest.mark.parametrize(
    "payload",
    [
        {"chat_provider": "unknown"},
        {"embed_provider": "anthropic"},
        {"chat_model": "  "},
        {"brand_color": "red;}"},
        {"ollama_base_url": "ftp://host"},
        {"custom_llm_base_url": "not a url"},
        {"unexpected": "field"},
    ],
)
def test_invalid_settings_are_rejected(client, payload):
    assert client.post("/api/admin/settings", headers=AUTH, json=payload).status_code == 422


# ---------------------------------------------------------------- provider routing


def settings_with(**values: Any) -> dict[str, Any]:
    return {**DEFAULT_SETTINGS, **values}


@pytest.mark.parametrize(
    ("provider", "model", "extra", "expected"),
    [
        ("openai", "gpt-4o", {"openai_api_key": "k"}, {"model": "openai/gpt-4o", "api_key": "k"}),
        (
            "groq",
            "openai/gpt-oss-20b",
            {"groq_api_key": "k"},
            {"model": "groq/openai/gpt-oss-20b", "api_key": "k"},
        ),
        (
            "gemini",
            "gemini/gemini-2.0-flash",
            {"gemini_api_key": "k"},
            {"model": "gemini/gemini-2.0-flash", "api_key": "k"},
        ),
        (
            "anthropic",
            "claude-sonnet-5-5",
            {"anthropic_api_key": "k"},
            {"model": "anthropic/claude-sonnet-5-5", "api_key": "k"},
        ),
        (
            "ollama",
            "llama3.2",
            {"ollama_base_url": "http://ollama:11434/"},
            {"model": "ollama_chat/llama3.2", "api_base": "http://ollama:11434"},
        ),
        (
            "custom",
            "openai/gpt-4o",
            {"custom_llm_base_url": "https://openrouter.ai/api/v1", "custom_llm_api_key": "k"},
            {
                "model": "openai/openai/gpt-4o",
                "api_base": "https://openrouter.ai/api/v1",
                "api_key": "k",
            },
        ),
        (
            "custom",
            "local",
            {"custom_llm_base_url": "http://lmstudio:1234/v1"},
            {
                "model": "openai/local",
                "api_base": "http://lmstudio:1234/v1",
                "api_key": llm_router.KEYLESS_PLACEHOLDER,
            },
        ),
    ],
)
def test_chat_routing(provider, model, extra, expected):
    settings = settings_with(chat_provider=provider, chat_model=model, **extra)
    assert llm_router.build_chat_params(settings) == expected


@pytest.mark.parametrize(
    ("provider", "extra", "expected"),
    [
        (
            "openai",
            {"openai_api_key": "k"},
            {"model": "openai/m", "api_key": "k", "encoding_format": "float"},
        ),
        ("gemini", {"gemini_api_key": "k"}, {"model": "gemini/m", "api_key": "k"}),
        (
            "ollama",
            {"ollama_base_url": "http://ollama:11434"},
            {"model": "ollama/m", "api_base": "http://ollama:11434"},
        ),
        (
            "custom",
            {"custom_embed_base_url": "http://tei/v1", "custom_embed_api_key": "k"},
            {
                "model": "openai/m",
                "api_base": "http://tei/v1",
                "api_key": "k",
                "encoding_format": "float",
            },
        ),
    ],
)
def test_embedding_routing(provider, extra, expected):
    settings = settings_with(embed_provider=provider, embed_model="m", **extra)
    assert llm_router.build_embed_params(settings) == expected


@pytest.mark.parametrize("provider", ["openai", "groq", "gemini", "anthropic", "custom"])
def test_missing_chat_configuration(provider):
    with pytest.raises(llm_router.LLMConfigError, match="not configured"):
        llm_router.build_chat_params(settings_with(chat_provider=provider, chat_model="m"))


@pytest.mark.parametrize("provider", ["openai", "gemini", "custom"])
def test_missing_embedding_configuration(provider):
    with pytest.raises(llm_router.LLMConfigError, match="not configured"):
        llm_router.build_embed_params(settings_with(embed_provider=provider, embed_model="m"))


def test_embeddings_are_batched_and_kept_in_order(llm):
    texts = [f"word{i}" for i in range(150)]
    vectors = llm_router.embed(settings_with(openai_api_key="k"), texts)
    assert [len(call["input"]) for call in llm.embeddings] == [64, 64, 22]
    assert vectors == [llm.vector(text) for text in texts]


def test_incomplete_embedding_response(monkeypatch):
    monkeypatch.setattr(
        llm_router.litellm, "embedding", lambda **_: SimpleNamespace(data=[{"embedding": [1.0]}])
    )
    with pytest.raises(llm_router.LLMProviderError, match="1 vectors for 2 inputs"):
        llm_router.embed(settings_with(openai_api_key="k"), ["a", "b"])


def test_chat_retries_without_unsupported_temperature(monkeypatch, llm):
    def completion(**kwargs):
        if "temperature" in kwargs:
            raise ValueError("Unsupported value: temperature does not support 0.1")
        return llm.completion(**kwargs)

    monkeypatch.setattr(llm_router.litellm, "completion", completion)
    reply = llm_router.chat(settings_with(openai_api_key="k"), [{"role": "user", "content": "x"}])
    assert reply == llm.reply


# ----------------------------------------------------------------- test connection


def test_connection_with_saved_settings(ready, llm):
    llm.reply = "OK"
    response = ready.post("/api/admin/test-llm", headers=AUTH)
    assert response.json() == {
        "success": True,
        "message": "OK",
        "provider": "openai",
        "model": "gpt-4o-mini",
    }
    call = llm.completions[0]
    assert call["messages"] == [{"role": "user", "content": "Reply with OK"}]
    assert call["api_key"] == "sk-openai-secret-ab12"


def test_connection_with_submitted_settings_does_not_save(ready, llm):
    llm.reply = "OK"
    submitted = {
        "chat_provider": "custom",
        "chat_model": "deepseek-chat",
        "custom_llm_base_url": "https://api.deepseek.com/v1",
        "custom_llm_api_key": "sk-deepseek",
    }
    assert ready.post("/api/admin/test-llm", headers=AUTH, json=submitted).json()["success"]
    assert llm.completions[0]["model"] == "openai/deepseek-chat"
    assert llm.completions[0]["api_base"] == "https://api.deepseek.com/v1"
    assert ready.get("/api/admin/settings", headers=AUTH).json()["chat_provider"] == "openai"


def test_connection_with_masked_key_uses_the_saved_key(ready, llm):
    ready.post("/api/admin/test-llm", headers=AUTH, json={"openai_api_key": "********ab12"})
    assert llm.completions[0]["api_key"] == "sk-openai-secret-ab12"


def test_connection_returns_exact_provider_error_without_the_key(ready, llm):
    llm.completion_error = RuntimeError("AuthenticationError: bad key sk-openai-secret-ab12")
    body = ready.post("/api/admin/test-llm", headers=AUTH).json()
    assert body["success"] is False
    assert body["message"] == "AuthenticationError: bad key [redacted]"


def test_connection_without_key(client, llm):
    body = client.post("/api/admin/test-llm", headers=AUTH).json()
    assert body["success"] is False
    assert "API key is not configured" in body["message"]
    assert llm.completions == []


# ----------------------------------------------------------------- chunking, parsing


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text)
    assert len(chunks) > 5
    assert all(len(chunk) <= 800 for chunk in chunks)
    for left, right in zip(chunks, chunks[1:]):
        assert right[:110] in left
    assert chunks[0].startswith("word0 ")
    assert chunks[-1].endswith("word999")


def test_unbroken_text_uses_exact_window():
    text = "".join(chr(97 + i % 26) for i in range(2000))
    chunks = chunk_text(text)
    assert [len(c) for c in chunks] == [800, 800, 640]
    assert chunks[0][-120:] == chunks[1][:120]


def test_chunk_edge_cases():
    assert chunk_text("  \n ") == []
    assert chunk_text("short") == ["short"]
    with pytest.raises(ValueError):
        chunk_text("x", size=10, overlap=10)


def test_pdf_text_covers_all_pages():
    text = extract_pdf_text(make_pdf("First page facts", "Second page facts"))
    assert "First page facts" in text and "Second page facts" in text


@pytest.mark.parametrize("data", [b"not a pdf", b"%PDF-1.4 broken", make_pdf("")])
def test_bad_pdfs(data):
    with pytest.raises(KnowledgeBaseError):
        extract_pdf_text(data)


def test_html_boilerplate_is_removed():
    title, text = extract_html_text(
        "<html><head><title>FAQ</title><style>.x{}</style></head><body><nav>Menu</nav>"
        "<script>alert(1)</script><main>We ship worldwide</main><footer>Legal</footer></body></html>"
    )
    assert title == "FAQ"
    assert text == "We ship worldwide"


# ---------------------------------------------------------------------- training


def test_train_pdf(ready):
    response = ready.post(
        "/api/admin/train/pdf",
        headers=AUTH,
        files={"file": ("../dir/manual.pdf", make_pdf("Battery lasts ten hours"), "application/pdf")},
    )
    assert response.status_code == 201, response.text
    assert response.json()["source"] == "manual.pdf"
    assert response.json()["source_type"] == "pdf"
    assert ask(ready, "How long does the battery last?").json()["sources"] == ["manual.pdf"]


def test_train_pdf_rejects_other_files(ready):
    files = {"file": ("x.pdf", b"hello", "application/pdf")}
    assert ready.post("/api/admin/train/pdf", headers=AUTH, files=files).status_code == 422


def test_train_pdf_size_limit(ready):
    files = {"file": ("x.pdf", b"%PDF-" + b"0" * (1024 * 1024), "application/pdf")}
    response = ready.post("/api/admin/train/pdf", headers=AUTH, files=files)
    assert response.status_code == 413


def test_train_url(ready, web):
    web.addresses["example.com"] = "93.184.215.14"
    web.pages["https://example.com/faq"] = (
        200,
        {"Content-Type": "text/html; charset=utf-8"},
        "<title>FAQ</title><nav>Menu</nav><p>Café delivery takes two days</p>".encode(),
    )
    response = ready.post(
        "/api/admin/train/url", headers=AUTH, json={"url": "https://example.com/faq"}
    )
    assert response.status_code == 201, response.text
    assert response.json()["title"] == "FAQ"
    assert ask(ready, "How long does delivery take?").json()["sources"] == [
        "https://example.com/faq"
    ]


def test_url_redirects_are_followed_and_rechecked(web):
    web.addresses.update({"example.com": "93.184.215.14", "internal": "10.0.0.5"})
    web.pages["http://example.com/a"] = (302, {"Location": "/b"}, b"")
    web.pages["http://example.com/b"] = (200, {"Content-Type": "text/plain"}, b"hello")
    web.pages["http://example.com/evil"] = (302, {"Location": "http://internal/"}, b"")
    assert fetch_url("http://example.com/a") == ("http://example.com/b",) * 2 + ("hello",)
    with pytest.raises(KnowledgeBaseError, match="private"):
        fetch_url("http://example.com/evil")
    assert "http://internal/" not in web.fetched


def test_url_redirect_loop(web):
    web.addresses["example.com"] = "93.184.215.14"
    web.pages["http://example.com/"] = (302, {"Location": "/"}, b"")
    with pytest.raises(KnowledgeBaseError, match="too many"):
        fetch_url("http://example.com/")


@pytest.mark.parametrize(
    "address", ["127.0.0.1", "10.1.2.3", "169.254.169.254", "192.168.1.1", "::1", "::ffff:10.0.0.1"]
)
def test_private_addresses_are_blocked(web, address):
    web.addresses["host.test"] = address
    with pytest.raises(KnowledgeBaseError, match="private"):
        validate_public_url("http://host.test/", allow_private=False)
    validate_public_url("http://host.test/", allow_private=True)


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "example.com", "http://u:p@example.com/", "http://nowhere.test"]
)
def test_invalid_urls(ready, web, url):
    response = ready.post("/api/admin/train/url", headers=AUTH, json={"url": url})
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("status", "content_type", "body", "expected"),
    [
        (404, "text/html", b"missing", 502),
        (200, "image/png", b"png", 422),
        (200, "text/html", b"<script>x</script>", 422),
    ],
)
def test_unusable_pages(ready, web, status, content_type, body, expected):
    web.addresses["example.com"] = "93.184.215.14"
    web.pages["http://example.com/"] = (status, {"Content-Type": content_type}, body)
    response = ready.post("/api/admin/train/url", headers=AUTH, json={"url": "http://example.com/"})
    assert response.status_code == expected


def test_train_reports_provider_and_config_errors(client, llm):
    assert train(client, "T", "text").status_code == 400
    save(client, openai_api_key="sk-openai-secret-ab12")
    llm.embedding_error = RuntimeError("quota exceeded")
    response = train(client, "T", "text")
    assert response.status_code == 502
    assert response.json()["detail"] == "quota exceeded"
    assert client.get("/api/admin/documents", headers=AUTH).json() == {
        "documents": [],
        "total_chunks": 0,
    }


def test_failed_metadata_write_rolls_back_vectors(ready, monkeypatch):
    def fail(*_args, **_kwargs):
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(ready.app.state.db, "add_document", fail)
    with pytest.raises(sqlite3.OperationalError):
        train(ready, "T", "some text")
    assert ready.app.state.engine.count_chunks() == 0


def test_documents_list_and_delete(ready):
    kept = train(ready, "Kept", "alpha beta").json()
    removed = train(ready, "Removed", "gamma delta").json()
    assert ready.delete(f"/api/admin/documents/{removed['id']}", headers=AUTH).status_code == 200
    body = ready.get("/api/admin/documents", headers=AUTH).json()
    assert [doc["id"] for doc in body["documents"]] == [kept["id"]]
    assert body["total_chunks"] == 1
    assert ready.delete(f"/api/admin/documents/{removed['id']}", headers=AUTH).status_code == 404


# -------------------------------------------------------------------- RAG and chat


def test_grounded_answer_cites_source(ready, llm):
    train(ready, "Refund Policy", "Customers may request a refund within 30 days of purchase.")
    response = ask(ready, "What is the refund window?")
    assert response.status_code == 200
    assert response.json() == {
        "answer": "Our refund window is 30 days.",
        "sources": ["Refund Policy"],
    }
    system, user = llm.completions[0]["messages"]
    assert "ONLY" in system["content"] and NOT_FOUND_MESSAGE in system["content"]
    assert "refund within 30 days" in user["content"]
    assert "What is the refund window?" in user["content"]


def test_retrieval_uses_the_top_four_chunks(ready, llm):
    for name in ("one", "two", "three", "four"):
        train(ready, f"Match {name}", f"shipping cost details {name}")
    for name in ("five", "six"):
        train(ready, f"Other {name}", f"unrelated gardening notes {name}")
    sources = ask(ready, "shipping cost details").json()["sources"]
    assert sorted(sources) == sorted(f"Match {n}" for n in ("one", "two", "three", "four"))
    assert llm.completions[0]["messages"][1]["content"].count("[Source ") == 4


def test_sources_are_deduplicated(ready):
    train(ready, "Long", " ".join(f"refund{i % 7} policy" for i in range(400)))
    assert ask(ready, "refund policy").json()["sources"] == ["Long"]


def test_missing_answer_has_no_sources(ready, llm):
    train(ready, "Doc", "alpha beta")
    llm.reply = NOT_FOUND_MESSAGE
    assert ask(ready, "unrelated").json() == {"answer": NOT_FOUND_MESSAGE, "sources": []}


def test_empty_knowledge_base_makes_no_provider_calls(client, llm):
    assert ask(client, "Hello?").json() == {"answer": EMPTY_KB_MESSAGE, "sources": []}
    assert llm.completions == [] and llm.embeddings == []


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "  "}, {"question": "x" * 2001}])
def test_invalid_questions(client, payload):
    assert client.post("/api/public/chat", json=payload).status_code == 422


def test_public_errors_hide_provider_details(ready, llm):
    train(ready, "Doc", "alpha")
    llm.completion_error = RuntimeError("secret internal detail")
    response = ask(ready, "alpha")
    assert response.status_code == 502
    assert "secret internal detail" not in response.text


def test_empty_model_reply_is_an_error(ready, llm):
    train(ready, "Doc", "alpha")
    llm.reply = None
    assert ask(ready, "alpha").status_code == 502


def test_chat_without_configuration(ready):
    train(ready, "Doc", "alpha")
    save(ready, openai_api_key="")
    assert ask(ready, "alpha").status_code == 503


def test_provider_switch_applies_without_restart(ready, llm):
    train(ready, "Doc", "alpha")
    save(
        ready,
        chat_provider="anthropic",
        chat_model="claude-haiku-5-5",
        anthropic_api_key="sk-ant-secret-cd34",
    )
    assert ask(ready, "alpha").status_code == 200
    assert llm.completions[-1]["model"] == "anthropic/claude-haiku-5-5"


def test_knowledge_base_survives_restart(tmp_path, monkeypatch, llm):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ADMIN_PASSWORD", PASSWORD)
    monkeypatch.setenv("MASTER_SECRET_KEY", MASTER)
    with TestClient(create_app()) as first:
        save(first, openai_api_key="sk-openai-secret-ab12")
        train(first, "Persistent", "alpha beta")
    with TestClient(create_app()) as second:
        assert ask(second, "alpha").json()["sources"] == ["Persistent"]


def test_rate_limit(tmp_path, monkeypatch, llm):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ADMIN_PASSWORD", PASSWORD)
    monkeypatch.setenv("MASTER_SECRET_KEY", MASTER)
    monkeypatch.setenv("CHAT_RATE_LIMIT_PER_MINUTE", "2")
    with TestClient(create_app()) as c:
        responses = [ask(c, "hi") for _ in range(3)]
    assert [r.status_code for r in responses] == [200, 200, 429]
    assert responses[-1].headers["retry-after"] == "60"


def test_rate_limiter_window_expires(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("app.main.time.monotonic", lambda: now[0])
    limiter = RateLimiter(1)
    assert limiter.allow("a") and not limiter.allow("a") and limiter.allow("b")
    now[0] += 61
    assert limiter.allow("a")
    assert set(limiter._hits) == {"a"}


def test_public_body_limit(client):
    body = json.dumps({"question": "x" * 70_000})
    headers = {"Content-Type": "application/json"}
    assert client.post("/api/public/chat", content=body, headers=headers).status_code == 413

    def stream():
        yield body.encode()

    response = client.post("/api/public/chat", content=stream(), headers=headers)
    assert response.status_code == 413


# --------------------------------------------------------------------------- reset


def test_reset_clears_everything(ready):
    train(ready, "Refund Policy", "Refunds within 30 days.")
    assert ask(ready, "refunds").json()["sources"] == ["Refund Policy"]
    response = ready.delete("/api/admin/reset", headers=AUTH)
    assert response.status_code == 200
    assert response.json() == {"status": "reset", "documents_removed": 1}
    assert ready.get("/api/admin/documents", headers=AUTH).json() == {
        "documents": [],
        "total_chunks": 0,
    }
    assert ask(ready, "refunds").json() == {"answer": EMPTY_KB_MESSAGE, "sources": []}


def test_reset_is_idempotent(client):
    for _ in range(2):
        assert client.delete("/api/admin/reset", headers=AUTH).json()["documents_removed"] == 0


def test_embedding_model_change_requires_reset(ready, llm):
    train(ready, "Doc", "alpha")
    result = save(ready, embed_model="text-embedding-3-large")
    assert "Reset the knowledge base" in result["warnings"][0]
    llm.dimension = 512
    assert train(ready, "Doc 2", "beta").status_code == 409
    assert ask(ready, "alpha").status_code == 503
    ready.delete("/api/admin/reset", headers=AUTH)
    assert train(ready, "Doc 2", "beta").status_code == 201
    assert ask(ready, "beta").json()["sources"] == ["Doc 2"]


def test_dimension_change_behind_the_same_model_name(ready, llm):
    train(ready, "Doc", "alpha")
    llm.dimension = 64
    assert train(ready, "Doc 2", "beta").status_code == 409
    assert ask(ready, "alpha").status_code == 503
