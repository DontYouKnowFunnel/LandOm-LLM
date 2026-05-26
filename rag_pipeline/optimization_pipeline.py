"""End-to-end landing-page optimization pipeline.

This module is the runtime path used by the backend optimization API:
section HTML + persona + visitor behavior logs -> structured features ->
Problem RAG -> Revision RAG -> recommendation generation.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from rag_pipeline.behavior_normalizer import normalize_visitor_behavior_data
from rag_pipeline.config import AI_MODELS
from rag_pipeline.features import ExtractedFeatures
from rag_pipeline.intervention_retriever import InterventionRetriever
from rag_pipeline.llm_feature_extractor import extract_features_with_llm_html
from rag_pipeline.problem_retriever import ProblemRetriever, ProblemSearchResult
from rag_pipeline.recommendation_generator import generate_recommendation
from rag_pipeline.retrieval_utils import openai_client, qdrant_client


PROBLEM_COLLECTION_NAME = "problem_patterns_en"
REVISION_COLLECTION_NAME = "intervention_evidence_en"
PROBLEM_TOP_K = 5
SELECTED_PROBLEM_TOP_K = 3
PROBLEM_CANDIDATE_K = 200
REVISION_TOP_K = 5
REVISION_CANDIDATE_K = 200
RECOMMENDATION_TOP_N = 3
MAX_LLM_HTML_FEATURES = 6
USE_PROBLEM_SECTION_FILTER = False
USE_REVISION_PROBLEM_FILTER = False
USE_REVISION_SECTION_FILTER = False
MIN_PROBLEM_FINAL_SCORE = 0.75
MIN_REVISION_FINAL_SCORE = 0.75
logger = logging.getLogger("uvicorn.error")


def log_stage(
    message: str,
    *,
    project_id: int | None = None,
    section_id: int | None = None,
    section_name: str | None = None,
    elapsed_seconds: float | None = None,
    **extra: Any,
) -> None:
    parts = [message]
    if project_id is not None:
        parts.append(f"projectId={project_id}")
    if section_id is not None:
        parts.append(f"sectionId={section_id}")
    if section_name:
        parts.append(f"sectionName={section_name}")
    if elapsed_seconds is not None:
        parts.append(f"elapsed={elapsed_seconds:.2f}s")
    for key, value in extra.items():
        if value is not None:
            parts.append(f"{key}={value}")
    logger.info(" ".join(parts))


def compact_problem(result: ProblemSearchResult, language: str = "en") -> dict[str, Any]:
    payload = result.payload
    return {
        "card_id": result.card_id,
        "final_score": round(result.final_score, 4),
        "vector_score": round(result.vector_score, 4),
        "overlap_scores": {key: round(value, 4) for key, value in result.overlap_scores.items()},
        "problem_case": payload.get("problem_case"),
        "section_label": payload.get("section_label"),
        "problem_description": payload.get(f"problem_description_{language}"),
        "barrier": payload.get(f"barrier_{language}"),
        "persona_traits": payload.get("persona_traits", []),
        "behavior_clusters": payload.get("behavior_clusters", []),
        "html_features": payload.get("html_features", []),
        "llm_context": payload.get(f"llm_context_{language}"),
        "confidence": payload.get("rule_metadata", {}).get("confidence"),
        "source_reference": payload.get("source_reference", {}),
    }


def compact_intervention(result: Any, language: str = "en") -> dict[str, Any]:
    payload = result.payload
    return {
        "card_id": result.card_id,
        "source_problem": result.source_problem,
        "final_score": round(result.final_score, 4),
        "vector_score": round(result.vector_score, 4),
        "overlap_scores": {key: round(value, 4) for key, value in result.overlap_scores.items()},
        "problem_case": payload.get("problem_case"),
        "intervention": payload.get("intervention"),
        "intervention_description": payload.get(f"intervention_description_{language}"),
        "mechanism": payload.get(f"mechanism_{language}"),
        "counterfactual_question": payload.get(f"counterfactual_question_{language}"),
        "expected_effect_direction": payload.get("expected_effect_direction", {}),
        "risk": payload.get(f"risk_{language}"),
        "evidence_summary": payload.get(f"evidence_summary_{language}"),
        "applicable_section_labels": payload.get("applicable_section_labels")
        or ([payload.get("section_label")] if payload.get("section_label") else []),
        "applicable_behavior_clusters": payload.get("applicable_behavior_clusters")
        or payload.get("target_behavior_clusters")
        or payload.get("behavior_clusters", []),
        "applicable_persona_traits": payload.get("applicable_persona_traits")
        or payload.get("persona_traits", []),
        "required_html_features": payload.get("required_html_features")
        or payload.get("html_features", []),
        "llm_context": payload.get(f"llm_context_{language}"),
        "confidence": payload.get("rule_metadata", {}).get("confidence"),
        "source_reference": payload.get("source_reference", {}),
    }


def dedupe_problems(problems: list[ProblemSearchResult]) -> list[ProblemSearchResult]:
    selected: list[ProblemSearchResult] = []
    seen: set[tuple[str, str]] = set()
    for problem in problems:
        key = (
            str(problem.payload.get("problem_case")),
            str(problem.payload.get("section_label")),
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(problem)
    return selected


def run_optimization(
    *,
    section_html: str,
    section_name: str,
    persona: str | None,
    visitor_behavior_data: dict[str, Any] | list[dict[str, Any]] | None,
    project_id: int | None = None,
    section_id: int | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    log_stage(
        "optimization started",
        project_id=project_id,
        section_id=section_id,
        section_name=section_name,
        html_chars=len(section_html),
    )
    events = normalize_visitor_behavior_data(visitor_behavior_data or {})
    log_stage(
        "behavior normalization completed",
        project_id=project_id,
        section_id=section_id,
        section_name=section_name,
        event_count=len(events),
    )
    shared_openai = openai_client()
    shared_qdrant = qdrant_client()
    _, embedding_model = AI_MODELS.embedding
    _, feature_model = AI_MODELS.feature_extraction

    feature_started_at = time.perf_counter()
    log_stage(
        "feature extraction started",
        project_id=project_id,
        section_id=section_id,
        section_name=section_name,
        model=feature_model,
    )
    features = extract_features_with_llm_html(
        html=section_html,
        section_label=section_name,
        persona_text=persona,
        behavior_events=events,
        client=shared_openai,
        model=feature_model,
        max_features=MAX_LLM_HTML_FEATURES,
    )
    log_stage(
        "feature extraction completed",
        project_id=project_id,
        section_id=section_id,
        section_name=section_name,
        elapsed_seconds=time.perf_counter() - feature_started_at,
        persona_traits=len(features.persona_traits),
        behavior_clusters=len(features.behavior_clusters),
        html_features=len(features.html_features),
    )

    problem_retriever = ProblemRetriever(
        collection_name=PROBLEM_COLLECTION_NAME,
        embedding_model=embedding_model,
        qdrant=shared_qdrant,
        openai=shared_openai,
    )
    problem_started_at = time.perf_counter()
    log_stage(
        "problem retrieval started",
        project_id=project_id,
        section_id=section_id,
        section_name=section_name,
        collection=PROBLEM_COLLECTION_NAME,
        candidate_k=PROBLEM_CANDIDATE_K,
    )
    problems = problem_retriever.search(
        features,
        top_k=PROBLEM_TOP_K,
        candidate_k=PROBLEM_CANDIDATE_K,
        use_section_filter=USE_PROBLEM_SECTION_FILTER,
    )
    selected_problems = dedupe_problems(problems)[:SELECTED_PROBLEM_TOP_K]
    log_stage(
        "problem retrieval completed",
        project_id=project_id,
        section_id=section_id,
        section_name=section_name,
        elapsed_seconds=time.perf_counter() - problem_started_at,
        retrieved=len(problems),
        selected=len(selected_problems),
    )

    intervention_retriever = InterventionRetriever(
        collection_name=REVISION_COLLECTION_NAME,
        embedding_model=embedding_model,
        qdrant=shared_qdrant,
        openai=shared_openai,
    )
    interventions_by_problem = []
    revision_started_at = time.perf_counter()
    log_stage(
        "revision retrieval started",
        project_id=project_id,
        section_id=section_id,
        section_name=section_name,
        collection=REVISION_COLLECTION_NAME,
        problem_count=len(selected_problems),
        candidate_k=REVISION_CANDIDATE_K,
    )
    for problem in selected_problems:
        interventions = intervention_retriever.search_for_problem(
            problem,
            features,
            top_k=REVISION_TOP_K,
            candidate_k=REVISION_CANDIDATE_K,
            use_problem_case_filter=USE_REVISION_PROBLEM_FILTER,
            use_section_filter=USE_REVISION_SECTION_FILTER,
            diversify_by_intervention=True,
        )
        compacted = [compact_intervention(result, "en") for result in interventions]
        interventions_by_problem.append(
            {
                "problem": compact_problem(problem, "en"),
                "interventions": compacted,
            }
        )

    selected_interventions = select_top_intervention_per_problem(
        interventions_by_problem,
        max_items=RECOMMENDATION_TOP_N,
    )
    log_stage(
        "revision retrieval completed",
        project_id=project_id,
        section_id=section_id,
        section_name=section_name,
        elapsed_seconds=time.perf_counter() - revision_started_at,
        problem_count=len(interventions_by_problem),
        selected_interventions=len(selected_interventions),
    )
    retrieval = build_retrieval_payload(
        section_name=section_name,
        persona=persona,
        features=features,
        behavior_event_count=len(events),
        problems=problems,
        selected_problems=selected_problems,
        interventions_by_problem=interventions_by_problem,
        selected_interventions=selected_interventions,
    )
    _, recommendation_model = AI_MODELS.recommendation
    recommendation_started_at = time.perf_counter()
    log_stage(
        "recommendation generation started",
        project_id=project_id,
        section_id=section_id,
        section_name=section_name,
        model=recommendation_model,
        max_recommendations=RECOMMENDATION_TOP_N,
    )
    recommendation = generate_recommendation(
        retrieval=retrieval,
        section_html=section_html,
        top_n=RECOMMENDATION_TOP_N,
        model=recommendation_model,
    )
    log_stage(
        "recommendation generation completed",
        project_id=project_id,
        section_id=section_id,
        section_name=section_name,
        elapsed_seconds=time.perf_counter() - recommendation_started_at,
        recommendation_count=len(recommendation.get("recommendations", [])),
    )
    log_stage(
        "optimization completed",
        project_id=project_id,
        section_id=section_id,
        section_name=section_name,
        elapsed_seconds=time.perf_counter() - started_at,
    )
    return {
        "retrieval": retrieval,
        "recommendation": recommendation,
    }


def build_retrieval_payload(
    *,
    section_name: str,
    persona: str | None,
    features: ExtractedFeatures,
    behavior_event_count: int,
    problems: list[ProblemSearchResult],
    selected_problems: list[ProblemSearchResult],
    interventions_by_problem: list[dict[str, Any]],
    selected_interventions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "input": {
            "section_label": features.section_label,
            "section_name_raw": section_name,
            "persona_text": persona or "",
            "behavior_event_count": behavior_event_count,
        },
        "features": {
            "structured_feature_extractor": "llm",
            "persona_traits": features.persona_traits,
            "behavior_clusters": features.behavior_clusters,
            "html_features": features.html_features,
            "query_text_en": features.query_text_en,
            "extraction_notes": features.extraction_notes,
        },
        "problem_retrieval": {
            "collection": PROBLEM_COLLECTION_NAME,
            "candidate_k": PROBLEM_CANDIDATE_K,
            "results": [compact_problem(result, "en") for result in problems],
            "selected_for_intervention": [
                compact_problem(result, "en") for result in selected_problems
            ],
        },
        "intervention_retrieval": {
            "collection": REVISION_COLLECTION_NAME,
            "candidate_k": REVISION_CANDIDATE_K,
            "by_problem": interventions_by_problem,
            "selected_interventions": selected_interventions,
            "deduped_top_interventions": selected_interventions,
            "selection_policy": {
                "mode": "top_1_intervention_per_problem",
                "max_items": RECOMMENDATION_TOP_N,
                "min_problem_score": MIN_PROBLEM_FINAL_SCORE,
                "min_revision_score": MIN_REVISION_FINAL_SCORE,
                "fallback": "return_top_scored_intervention_when_all_candidates_are_filtered",
            },
        },
    }
