"""Intervention Evidence retrieval over the local Qdrant vector DB."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from rag_pipeline.config import AI_MODELS
from rag_pipeline.features import ExtractedFeatures, label_en, labels_with_descriptions_en
from rag_pipeline.problem_retriever import ProblemSearchResult
from rag_pipeline.retrieval_utils import (
    DEFAULT_EMBEDDED_QDRANT_DIR,
    confidence_bonus,
    embed_query,
    openai_client,
    overlap,
    qdrant_client,
)


DEFAULT_INTERVENTION_WEIGHTS = {
    "vector": 0.6364,
    "problem_case": 0.1818,
    "html": 0.1818,
    "behavior": 0.0,
    "persona": 0.0,
    "confidence": 0.0,
}


@dataclass
class InterventionSearchResult:
    card_id: str
    vector_score: float
    final_score: float
    overlap_scores: dict[str, float]
    payload: dict[str, Any]
    source_problem: dict[str, Any]


def make_intervention_filter(
    *,
    problem_case: str | None,
    section_label: str | None,
) -> models.Filter | None:
    conditions: list[models.FieldCondition] = []
    if problem_case:
        conditions.append(
            models.FieldCondition(key="problem_case", match=models.MatchValue(value=problem_case))
        )
    if section_label:
        conditions.append(
            models.FieldCondition(
                key="applicable_section_labels",
                match=models.MatchValue(value=section_label),
            )
        )
    if not conditions:
        return None
    return models.Filter(must=conditions)


def resolve_intervention_weights(weights: dict[str, float] | None = None) -> dict[str, float]:
    resolved = dict(DEFAULT_INTERVENTION_WEIGHTS)
    if weights is None:
        return resolved
    unknown = sorted(set(weights) - set(DEFAULT_INTERVENTION_WEIGHTS))
    if unknown:
        raise ValueError(f"Unknown intervention reranker weight keys: {', '.join(unknown)}")
    resolved.update({key: float(value) for key, value in weights.items()})
    return resolved


def build_intervention_query_text_en(
    *,
    problem_payload: dict[str, Any],
    features: ExtractedFeatures,
) -> str:
    problem_case = str(problem_payload.get("problem_case", ""))
    problem_boundary = str(
        problem_payload.get("taxonomy_boundary_en")
        or problem_payload.get("problem_boundary_en")
        or ""
    )
    return " ".join(
        [
            "Retrieval task: find intervention evidence for a diagnosed landing-page problem.",
            f"Problem to solve: {problem_case}, {label_en(problem_case)}.",
            f"Problem boundary: {problem_boundary}",
            f"Applicable funnel: {features.section_label}.",
            f"HTML features to address: {labels_with_descriptions_en(features.html_features)}.",
        ]
    )


def rerank_intervention_result(
    *,
    vector_score: float,
    payload: dict[str, Any],
    features: ExtractedFeatures,
    problem_payload: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> InterventionSearchResult:
    weights = resolve_intervention_weights(weights)
    problem_case_match = 1.0 if payload.get("problem_case") == problem_payload.get("problem_case") else 0.0
    html_overlap = overlap(features.html_features, payload.get("required_html_features") or payload.get("html_features", []))
    behavior_overlap = overlap(
        features.behavior_clusters,
        payload.get("target_behavior_clusters") or payload.get("behavior_clusters", []),
    )
    persona_overlap = overlap(
        features.persona_traits,
        payload.get("applicable_persona_traits") or payload.get("persona_traits", []),
    )
    confidence = confidence_bonus(payload)
    final_score = (
        weights["vector"] * vector_score
        + weights["problem_case"] * problem_case_match
        + weights["html"] * html_overlap
        + weights["behavior"] * behavior_overlap
        + weights["persona"] * persona_overlap
        + weights["confidence"] * confidence
    )
    return InterventionSearchResult(
        card_id=str(payload.get("card_id") or payload.get("id")),
        vector_score=vector_score,
        final_score=final_score,
        overlap_scores={
            "problem_case_match": problem_case_match,
            "required_html_feature_overlap": html_overlap,
            "behavior_symptom_fit": behavior_overlap,
            "persona_trait_fit": persona_overlap,
            "confidence_bonus": confidence,
        },
        payload=payload,
        source_problem={
            "card_id": problem_payload.get("card_id") or problem_payload.get("id"),
            "problem_case": problem_payload.get("problem_case"),
            "section_label": problem_payload.get("section_label"),
        },
    )


class InterventionRetriever:
    def __init__(
        self,
        *,
        qdrant_path: Path | None = DEFAULT_EMBEDDED_QDRANT_DIR,
        collection_name: str = "intervention_evidence_en",
        embedding_model: str = AI_MODELS.embedding[1],
        weights: dict[str, float] | None = None,
        qdrant: QdrantClient | None = None,
        openai: Any | None = None,
    ) -> None:
        self.qdrant = qdrant or qdrant_client(path=qdrant_path)
        self.openai = openai or openai_client()
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.weights = resolve_intervention_weights(weights)

    def search_for_problem(
        self,
        problem: ProblemSearchResult,
        features: ExtractedFeatures,
        *,
        top_k: int = 5,
        candidate_k: int = 200,
        use_problem_case_filter: bool = False,
        use_section_filter: bool = False,
        diversify_by_intervention: bool = True,
    ) -> list[InterventionSearchResult]:
        query_text = build_intervention_query_text_en(
            problem_payload=problem.payload,
            features=features,
        )
        query_vector = embed_query(self.openai, query_text, self.embedding_model)
        results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=make_intervention_filter(
                problem_case=problem.payload.get("problem_case") if use_problem_case_filter else None,
                section_label=features.section_label if use_section_filter else None,
            ),
            limit=max(candidate_k, top_k),
            with_payload=True,
        )
        reranked = [
            rerank_intervention_result(
                vector_score=float(result.score),
                payload=result.payload or {},
                features=features,
                problem_payload=problem.payload,
                weights=self.weights,
            )
            for result in results
        ]
        reranked.sort(key=lambda result: result.final_score, reverse=True)
        if not diversify_by_intervention:
            return reranked[:top_k]

        diversified: list[InterventionSearchResult] = []
        seen: set[str] = set()
        for result in reranked:
            intervention = str(result.payload.get("intervention") or result.card_id)
            if intervention in seen:
                continue
            seen.add(intervention)
            diversified.append(result)
            if len(diversified) >= top_k:
                break
        return diversified
