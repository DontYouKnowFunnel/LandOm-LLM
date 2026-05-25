"""Central model configuration for the funnel pipeline."""

from __future__ import annotations

from dataclasses import dataclass


ModelSpec = tuple[str, str]


@dataclass(frozen=True)
class AIModelConfig:
    # To switch back to OpenAI:
    # funnel_analysis: ModelSpec = ("openai", "gpt-5.4-mini")
    funnel_analysis: ModelSpec = (
        "groq",
        "meta-llama/llama-4-scout-17b-16e-instruct",
    )


AI_MODELS = AIModelConfig()
