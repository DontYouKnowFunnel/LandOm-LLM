"""Shared retrieval helpers for Qdrant + OpenAI embeddings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

from rag_pipeline.langsmith_tracking import traced_embedding_create


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDED_QDRANT_DIR = PROJECT_ROOT / "runs/qdrant"
OPENAI_TIMEOUT_SECONDS = 120.0
QDRANT_TIMEOUT_SECONDS = 10.0
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_EMBEDDING_CACHE: dict[tuple[str, str], list[float]] = {}


def openai_client() -> OpenAI:
    return llm_client("openai")


def llm_client(provider: str = "openai") -> OpenAI:
    load_dotenv(PROJECT_ROOT / ".env")
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env before retrieval.")
        return OpenAI(api_key=api_key, timeout=OPENAI_TIMEOUT_SECONDS)
    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to .env before retrieval.")
        return OpenAI(
            api_key=api_key,
            base_url=GROQ_BASE_URL,
            timeout=OPENAI_TIMEOUT_SECONDS,
        )
    raise RuntimeError(f"Unsupported LLM provider: {provider}")


def qdrant_client(
    *,
    path: Path | None = None,
    url: str | None = None,
    api_key: str | None = None,
) -> QdrantClient:
    """Create a Qdrant client.

    Runtime prefers a remote/server Qdrant when QDRANT_URL is set. If it is not
    set, it falls back to the local embedded Qdrant path for local development.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    resolved_url = url or os.getenv("QDRANT_URL")
    resolved_api_key = api_key or os.getenv("QDRANT_API_KEY") or None
    if resolved_url:
        return QdrantClient(
            url=resolved_url,
            api_key=resolved_api_key,
            timeout=QDRANT_TIMEOUT_SECONDS,
        )

    raw_path_value = path or DEFAULT_EMBEDDED_QDRANT_DIR
    raw_path = Path(raw_path_value)
    if not raw_path.is_absolute():
        raw_path = PROJECT_ROOT / raw_path
    return QdrantClient(path=str(raw_path))


def embed_query(
    client: OpenAI,
    query_text: str,
    model: str,
    provider: str = "openai",
) -> list[float]:
    cache_key = (model, query_text)
    if cache_key in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[cache_key]
    response = traced_embedding_create(
        client=client,
        request_kwargs={
            "model": model,
            "input": [query_text],
            "encoding_format": "float",
        },
        provider=provider,
        model=model,
        workflow="rag.optimization",
        stage="rag.embedding.query",
    )
    embedding = response.data[0].embedding
    _EMBEDDING_CACHE[cache_key] = embedding
    return embedding


def overlap(query_values: list[str], candidate_values: list[str]) -> float:
    if not query_values:
        return 0.0
    query_set = set(query_values)
    candidate_set = set(candidate_values)
    return len(query_set & candidate_set) / len(query_set)


def confidence_bonus(payload: dict[str, Any]) -> float:
    confidence = payload.get("rule_metadata", {}).get("confidence")
    if confidence == "high":
        return 1.0
    if confidence == "medium":
        return 0.5
    return 0.0
