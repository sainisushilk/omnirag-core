"""SQLite persistence: runtime settings, encrypted secrets and the document log."""

from __future__ import annotations

import base64
import logging
import os
import secrets
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger("omnirag.database")

MASK_PREFIX = "********"
KDF_ITERATIONS = 390_000

SECRET_KEYS: tuple[str, ...] = (
    "openai_api_key",
    "groq_api_key",
    "gemini_api_key",
    "anthropic_api_key",
    "custom_llm_api_key",
    "custom_embed_api_key",
)

DEFAULT_SETTINGS: dict[str, str] = {
    "chat_provider": "openai",
    "chat_model": "gpt-4o-mini",
    "embed_provider": "openai",
    "embed_model": "text-embedding-3-small",
    "openai_api_key": "",
    "groq_api_key": "",
    "gemini_api_key": "",
    "anthropic_api_key": "",
    "ollama_base_url": "http://host.docker.internal:11434",
    "custom_llm_base_url": "",
    "custom_llm_api_key": "",
    "custom_embed_base_url": "",
    "custom_embed_api_key": "",
    "bot_title": "AI Assistant",
    "brand_color": "#4f46e5",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    source      TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('pdf', 'url', 'text')),
    chunk_count INTEGER NOT NULL,
    char_count  INTEGER NOT NULL,
    created_at  TEXT NOT NULL
);
"""


class SecretDecryptionError(Exception):
    """A stored secret cannot be decrypted with the current MASTER_SECRET_KEY."""

    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(
            f"The stored value of '{key}' cannot be decrypted. MASTER_SECRET_KEY has changed; "
            "restore the original key or enter this API key again in the Admin Panel."
        )


def load_or_create_secret(env_name: str, path: Path, length: int) -> str:
    """Return a secret from the environment, or from a generated owner-only file."""
    from_env = os.environ.get(env_name, "").strip()
    if from_env:
        return from_env
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        existing = path.read_text(encoding="utf-8").strip()
        if not existing:
            raise RuntimeError(f"{path} is empty. Set {env_name} or delete the file.") from None
        return existing
    value = secrets.token_urlsafe(length)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)
    logger.warning("%s is not set; a random value was generated and stored in %s", env_name, path)
    return value


def mask_secret(value: str) -> str:
    """Mask a secret, revealing the last four characters only when it is long enough."""
    if not value:
        return ""
    return MASK_PREFIX + (value[-4:] if len(value) >= 12 else "")


def is_masked(value: str) -> bool:
    return value.startswith(MASK_PREFIX)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    """Thread-safe access to settings, secrets and document metadata."""

    def __init__(self, path: Path, master_secret: str) -> None:
        if not master_secret:
            raise ValueError("MASTER_SECRET_KEY must not be empty")
        self.path = path
        self._write_lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)
        self._fernet = self._build_fernet(master_secret)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _build_fernet(self, master_secret: str) -> Fernet:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('kdf_salt', ?)",
                (secrets.token_hex(16),),
            )
            salt_hex = conn.execute("SELECT value FROM meta WHERE key = 'kdf_salt'").fetchone()[0]
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=bytes.fromhex(salt_hex),
            iterations=KDF_ITERATIONS,
        )
        return Fernet(base64.urlsafe_b64encode(kdf.derive(master_secret.encode("utf-8"))))

    def ping(self) -> None:
        with self._connect() as conn:
            conn.execute("SELECT 1").fetchone()

    # ------------------------------------------------------------------ secrets

    def encrypt(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode("utf-8")).decode("ascii")

    def decrypt(self, ciphertext: str) -> str:
        return self._fernet.decrypt(ciphertext.encode("ascii")).decode("utf-8")

    # ----------------------------------------------------------------- settings

    def _stored_settings(self) -> dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM settings").fetchall()
        merged = dict(DEFAULT_SETTINGS)
        merged.update({row["key"]: row["value"] for row in rows if row["key"] in DEFAULT_SETTINGS})
        return merged

    def get_settings(self) -> dict[str, Any]:
        """Return settings with decrypted secrets.

        A secret that cannot be decrypted is represented by a SecretDecryptionError
        instance, so it only causes a failure when that specific secret is needed.
        """
        result: dict[str, Any] = dict(self._stored_settings())
        for key in SECRET_KEYS:
            if not result[key]:
                continue
            try:
                result[key] = self.decrypt(result[key])
            except (InvalidToken, UnicodeError):
                logger.error("Stored secret %s cannot be decrypted", key)
                result[key] = SecretDecryptionError(key)
        return result

    def get_masked_settings(self) -> dict[str, str]:
        masked: dict[str, str] = {}
        for key, value in self.get_settings().items():
            if isinstance(value, SecretDecryptionError):
                masked[key] = MASK_PREFIX
            elif key in SECRET_KEYS:
                masked[key] = mask_secret(value)
            else:
                masked[key] = value
        return masked

    def merge_settings(self, updates: dict[str, str | None]) -> dict[str, Any]:
        """Overlay submitted values on the saved ones, ignoring masked secrets."""
        merged = self.get_settings()
        merged.update(self._effective_updates(updates))
        return merged

    @staticmethod
    def _effective_updates(updates: dict[str, str | None]) -> dict[str, str]:
        effective: dict[str, str] = {}
        for key, value in updates.items():
            if key not in DEFAULT_SETTINGS or value is None:
                continue
            if key in SECRET_KEYS and is_masked(value):
                continue
            effective[key] = value
        return effective

    def save_settings(self, updates: dict[str, str | None]) -> None:
        now = utc_now()
        rows = [
            (key, self.encrypt(value) if key in SECRET_KEYS and value else value, now)
            for key, value in self._effective_updates(updates).items()
        ]
        with self._write_lock, self._connect() as conn:
            conn.executemany(
                "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                "updated_at = excluded.updated_at",
                rows,
            )

    # --------------------------------------------------------------------- meta

    def get_meta(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._write_lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def delete_meta(self, key: str) -> None:
        with self._write_lock, self._connect() as conn:
            conn.execute("DELETE FROM meta WHERE key = ?", (key,))

    # ---------------------------------------------------------------- documents

    def add_document(
        self,
        document_id: str,
        title: str,
        source: str,
        source_type: str,
        chunk_count: int,
        char_count: int,
    ) -> dict[str, Any]:
        document = {
            "id": document_id,
            "title": title,
            "source": source,
            "source_type": source_type,
            "chunk_count": chunk_count,
            "char_count": char_count,
            "created_at": utc_now(),
        }
        with self._write_lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO documents "
                "(id, title, source, source_type, chunk_count, char_count, created_at) VALUES "
                "(:id, :title, :source, :source_type, :chunk_count, :char_count, :created_at)",
                document,
            )
        return document

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY created_at DESC, rowid DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def count_documents(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def delete_document(self, document_id: str) -> bool:
        with self._write_lock, self._connect() as conn:
            cursor = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            return cursor.rowcount > 0

    def clear_documents(self) -> int:
        with self._write_lock, self._connect() as conn:
            return conn.execute("DELETE FROM documents").rowcount
