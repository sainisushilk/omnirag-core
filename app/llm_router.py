"""Provider routing for chat and embeddings through LiteLLM."""

from __future__ import annotations

import logging
import os
from typing import Any

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm  # noqa: E402

from app.database import SECRET_KEYS, SecretDecryptionError  # noqa: E402

logger = logging.getLogger("omnirag.llm")

litellm.drop_params = True
litellm.suppress_debug_info = True

CHAT_PROVIDERS: tuple[str, ...] = ("openai", "groq", "gemini", "anthropic", "ollama", "custom")
EMBED_PROVIDERS: tuple[str, ...] = ("openai", "gemini", "ollama", "custom")

# LiteLLM's OpenAI client refuses an empty key, while many self-hosted
# OpenAI-compatible servers (LM Studio, vLLM) do not require one.
KEYLESS_PLACEHOLDER = "not-required"

CHAT_TIMEOUT_SECONDS = 90
TEST_TIMEOUT_SECONDS = 30
EMBED_TIMEOUT_SECONDS = 90
EMBED_BATCH_SIZE = 64
CHAT_MAX_TOKENS = 2048
CHAT_TEMPERATURE = 0.1
TEST_PROMPT = "Reply with OK"


class LLMConfigError(Exception):
    """The provider configuration is incomplete or invalid."""


class LLMProviderError(Exception):
    """The provider rejected the request or could not be reached."""


def _required(settings: dict[str, Any], key: str, label: str) -> str:
    value = settings.get(key, "")
    if isinstance(value, SecretDecryptionError):
        raise LLMConfigError(str(value))
    value = value.strip()
    if not value:
        raise LLMConfigError(f"{label} is not configured.")
    return value


def _optional_secret(settings: dict[str, Any], key: str) -> str:
    value = settings.get(key, "")
    if isinstance(value, SecretDecryptionError):
        raise LLMConfigError(str(value))
    return value.strip() or KEYLESS_PLACEHOLDER


def _prefixed(prefix: str, model: str) -> str:
    return model if model.startswith(f"{prefix}/") else f"{prefix}/{model}"


def build_chat_params(settings: dict[str, Any]) -> dict[str, Any]:
    """Translate saved settings into LiteLLM completion arguments."""
    provider = settings.get("chat_provider", "")
    if provider not in CHAT_PROVIDERS:
        raise LLMConfigError(f"Unsupported chat provider: {provider!r}.")
    model = _required(settings, "chat_model", "Chat model")

    if provider == "ollama":
        return {
            "model": _prefixed("ollama_chat", model),
            "api_base": _required(settings, "ollama_base_url", "Ollama base URL").rstrip("/"),
        }
    if provider == "custom":
        # The prefix is always added because the model ID itself may start with
        # "openai/" on gateways such as OpenRouter.
        return {
            "model": f"openai/{model}",
            "api_base": _required(settings, "custom_llm_base_url", "Custom LLM base URL").rstrip("/"),
            "api_key": _optional_secret(settings, "custom_llm_api_key"),
        }
    return {
        "model": _prefixed(provider, model),
        "api_key": _required(settings, f"{provider}_api_key", f"{provider.capitalize()} API key"),
    }


def build_embed_params(settings: dict[str, Any]) -> dict[str, Any]:
    """Translate saved settings into LiteLLM embedding arguments."""
    provider = settings.get("embed_provider", "")
    if provider not in EMBED_PROVIDERS:
        raise LLMConfigError(f"Unsupported embedding provider: {provider!r}.")
    model = _required(settings, "embed_model", "Embedding model")

    if provider == "ollama":
        return {
            "model": _prefixed("ollama", model),
            "api_base": _required(settings, "ollama_base_url", "Ollama base URL").rstrip("/"),
        }
    if provider == "custom":
        base_url = _required(settings, "custom_embed_base_url", "Custom embedding base URL")
        return {
            "model": f"openai/{model}",
            "api_base": base_url.rstrip("/"),
            "api_key": _optional_secret(settings, "custom_embed_api_key"),
            "encoding_format": "float",
        }
    params: dict[str, Any] = {
        "model": _prefixed(provider, model),
        "api_key": _required(settings, f"{provider}_api_key", f"{provider.capitalize()} API key"),
    }
    if provider == "openai":
        params["encoding_format"] = "float"
    return params


def embedding_signature(settings: dict[str, Any]) -> str:
    """Identify the embedding space; vectors from different signatures are incomparable."""
    return f"{settings.get('embed_provider', '')}:{settings.get('embed_model', '')}"


def _redact(message: str, settings: dict[str, Any]) -> str:
    for key in SECRET_KEYS:
        value = settings.get(key)
        if isinstance(value, str) and value:
            message = message.replace(value, "[redacted]")
    return message


def _provider_error(exc: Exception, settings: dict[str, Any]) -> LLMProviderError:
    message = _redact(str(exc).strip() or type(exc).__name__, settings)
    return LLMProviderError(message)


def _read(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


def _completion_text(response: Any) -> str:
    choices = _read(response, "choices") or []
    if not choices:
        return ""
    message = _read(choices[0], "message")
    content = _read(message, "content") if message is not None else None
    return (content or "").strip()


def chat(settings: dict[str, Any], messages: list[dict[str, str]]) -> str:
    """Run a chat completion and return its text."""
    params: dict[str, Any] = {
        **build_chat_params(settings),
        "messages": messages,
        "max_tokens": CHAT_MAX_TOKENS,
        "timeout": CHAT_TIMEOUT_SECONDS,
        "num_retries": 1,
    }
    try:
        try:
            response = litellm.completion(**params, temperature=CHAT_TEMPERATURE)
        except Exception as exc:
            # Some reasoning models accept only their default temperature.
            if "temperature" not in str(exc).lower():
                raise
            response = litellm.completion(**params)
    except Exception as exc:
        raise _provider_error(exc, settings) from exc
    text = _completion_text(response)
    if not text:
        raise LLMProviderError("The model returned an empty response.")
    return text


def test_connection(settings: dict[str, Any]) -> str:
    """Send a minimal prompt and return the reply, or raise the exact provider error."""
    params = build_chat_params(settings)
    try:
        response = litellm.completion(
            **params,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            timeout=TEST_TIMEOUT_SECONDS,
            num_retries=0,
        )
    except Exception as exc:
        raise _provider_error(exc, settings) from exc
    return _completion_text(response)


def embed(settings: dict[str, Any], texts: list[str]) -> list[list[float]]:
    """Embed texts in batches, preserving order."""
    params = build_embed_params(settings)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        try:
            response = litellm.embedding(
                **params, input=batch, timeout=EMBED_TIMEOUT_SECONDS, num_retries=1
            )
        except Exception as exc:
            raise _provider_error(exc, settings) from exc
        data = list(_read(response, "data") or [])
        if all(_read(item, "index") is not None for item in data):
            data.sort(key=lambda item: _read(item, "index"))
        batch_vectors = [list(_read(item, "embedding") or []) for item in data]
        if len(batch_vectors) != len(batch) or not all(batch_vectors):
            raise LLMProviderError(
                f"The embedding provider returned {len(batch_vectors)} vectors for "
                f"{len(batch)} inputs."
            )
        vectors.extend(batch_vectors)
    return vectors
