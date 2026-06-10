"""Central runtime configuration for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ModelSpec = tuple[str, str]
ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh"]


@dataclass(frozen=True)
class AIModelConfig:
    embedding: ModelSpec = ("openai", "text-embedding-3-small")
    feature_extraction: ModelSpec = ("openai", "gpt-5.4-mini")
    recommendation: ModelSpec = ("openai", "gpt-5.4-mini")
    code_generation: ModelSpec = ("openai", "gpt-5.4-mini")


AI_MODELS = AIModelConfig()

REASONING_EFFORTS: dict[str, ReasoningEffort] = {
    "feature_extraction": "low",
    "recommendation": "medium",
    "wireframe": "low",
    "code_generation": "medium",
}
