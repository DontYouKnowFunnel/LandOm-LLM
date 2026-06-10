"""Problem Pattern retrieval over the local Qdrant vector DB."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from rag_pipeline.config import AI_MODELS
from rag_pipeline.features import ExtractedFeatures
from rag_pipeline.retrieval_utils import (
    DEFAULT_EMBEDDED_QDRANT_DIR,
    confidence_bonus,
    embed_query,
    openai_client,
    overlap,
    qdrant_client,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_PROBLEM_WEIGHTS = {
    "vector": 0.6349,
    "html": 0.1587,
    "behavior": 0.0794,
    "persona": 0.0794,
    "confidence": 0.0476,
}


@dataclass
class ProblemSearchResult:
    card_id: str
    vector_score: float
    final_score: float
    overlap_scores: dict[str, float]
    payload: dict[str, Any]


def make_problem_filter(section_label: str | None) -> models.Filter | None:
    if not section_label:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(
                key="section_label",
                match=models.MatchValue(value=section_label),
            )
        ]
    )


def resolve_problem_weights(weights: dict[str, float] | None = None) -> dict[str, float]:
    resolved = dict(DEFAULT_PROBLEM_WEIGHTS)
    if weights is None:
        return resolved
    unknown = sorted(set(weights) - set(DEFAULT_PROBLEM_WEIGHTS))
    if unknown:
        raise ValueError(f"Unknown problem reranker weight keys: {', '.join(unknown)}")
    resolved.update({key: float(value) for key, value in weights.items()})
    return resolved


def rerank_problem_result(
    *,
    vector_score: float,
    payload: dict[str, Any],
    features: ExtractedFeatures,
    weights: dict[str, float] | None = None,
) -> ProblemSearchResult:
    weights = resolve_problem_weights(weights)
    html_overlap = overlap(features.html_features, payload.get("html_features", []))
    behavior_overlap = overlap(features.behavior_clusters, payload.get("behavior_clusters", []))
    persona_overlap = overlap(features.persona_traits, payload.get("persona_traits", []))
    confidence = confidence_bonus(payload)
    final_score = (
        weights["vector"] * vector_score
        + weights["html"] * html_overlap
        + weights["behavior"] * behavior_overlap
        + weights["persona"] * persona_overlap
        + weights["confidence"] * confidence
    )
    return ProblemSearchResult(
        card_id=str(payload.get("card_id") or payload.get("id")),
        vector_score=vector_score,
        final_score=final_score,
        overlap_scores={
            "html_feature_overlap": html_overlap,
            "behavior_cluster_overlap": behavior_overlap,
            "persona_trait_overlap": persona_overlap,
            "confidence_bonus": confidence,
        },
        payload=payload,
    )


class ProblemRetriever:
    def __init__(
        self,
        *,
        qdrant_path: Path | None = DEFAULT_EMBEDDED_QDRANT_DIR,
        collection_name: str = "problem_patterns_en",
        embedding_model: str = AI_MODELS.embedding[1],
        embedding_provider: str = AI_MODELS.embedding[0],
        weights: dict[str, float] | None = None,
        qdrant: QdrantClient | None = None,
        openai: Any | None = None,
    ) -> None:
        self.qdrant = qdrant or qdrant_client(path=qdrant_path)
        self.openai = openai or openai_client()
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.weights = resolve_problem_weights(weights)

    def search(
        self,
        features: ExtractedFeatures,
        *,
        top_k: int = 5,
        candidate_k: int = 200,
        use_section_filter: bool = False,
    ) -> list[ProblemSearchResult]:
        query_vector = embed_query(
            self.openai,
            features.query_text_en,
            self.embedding_model,
            self.embedding_provider,
        )
        results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=make_problem_filter(features.section_label) if use_section_filter else None,
            limit=max(candidate_k, top_k),
            with_payload=True,
        )
        reranked = [
            rerank_problem_result(
                vector_score=float(result.score),
                payload=result.payload or {},
                features=features,
                weights=self.weights,
            )
            for result in results
        ]
        reranked.sort(key=lambda result: result.final_score, reverse=True)
        return reranked[:top_k]
