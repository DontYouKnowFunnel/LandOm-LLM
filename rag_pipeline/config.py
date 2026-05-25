"""Central runtime configuration for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass


ModelSpec = tuple[str, str]


@dataclass(frozen=True)
class AIModelConfig:
    embedding: ModelSpec = ("openai", "text-embedding-3-small")
    feature_extraction: ModelSpec = ("openai", "gpt-5.4-mini")
    recommendation: ModelSpec = ("openai", "gpt-5.4-mini")


AI_MODELS = AIModelConfig()
