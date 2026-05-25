"""End-to-end landing-page optimization pipeline.

This module is the runtime path used by the backend optimization API:
section HTML + persona + visitor behavior logs -> structured features ->
Problem RAG -> Revision RAG -> recommendation generation.
"""

from __future__ import annotations

import os
from typing import Any

from rag_pipeline.behavior_normalizer import normalize_visitor_behavior_data
from rag_pipeline.features import ExtractedFeatures
from rag_pipeline.intervention_retriever import InterventionRetriever
from rag_pipeline.llm_feature_extractor import extract_features_with_llm_html
from rag_pipeline.problem_retriever import ProblemRetriever, ProblemSearchResult
from rag_pipeline.recommendation_generator import generate_recommendation
from rag_pipeline.retrieval_utils import openai_client, qdrant_client


MIN_PROBLEM_FINAL_SCORE = 0.75
MIN_REVISION_FINAL_SCORE = 0.75


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


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


def select_top_intervention_per_problem(
    interventions_by_problem: list[dict[str, Any]],
    *,
    max_items: int,
    min_problem_score: float = MIN_PROBLEM_FINAL_SCORE,
    min_revision_score: float = MIN_REVISION_FINAL_SCORE,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    fallback: dict[str, Any] | None = None
    seen_problem_cases: set[str] = set()
    for group in interventions_by_problem:
        problem = group.get("problem") or {}
        problem_case = str(problem.get("problem_case") or "")
        if not problem_case or problem_case in seen_problem_cases:
            continue
        interventions = group.get("interventions") or []
        top_intervention = interventions[0] if interventions else None
        if top_intervention is not None and fallback is None:
            fallback = top_intervention
        if (
            float(problem.get("final_score") or 0.0) < min_problem_score
            or top_intervention is None
            or float(top_intervention.get("final_score") or 0.0) < min_revision_score
        ):
            continue

        for intervention in interventions:
            selected.append(intervention)
            seen_problem_cases.add(problem_case)
            break

        if len(selected) >= max_items:
            break
    return selected or ([fallback] if fallback is not None else [])


def normalize_provider(value: str | None) -> str:
    provider = (value or "openai").strip().lower()
    if provider in {"openai", "groq", "ollama"}:
        return provider
    return "openai"


def run_optimization(
    *,
    section_html: str,
    section_name: str,
    persona: str | None,
    visitor_behavior_data: dict[str, Any] | list[dict[str, Any]] | None,
) -> dict[str, Any]:
    events = normalize_visitor_behavior_data(visitor_behavior_data or {})
    shared_openai = openai_client()
    shared_qdrant = qdrant_client()
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    features = extract_features_with_llm_html(
        html=section_html,
        section_label=section_name,
        persona_text=persona,
        behavior_events=events,
        client=shared_openai,
        model=os.getenv("OPENAI_FEATURE_MODEL") or os.getenv("LLM_FEATURE_MODEL") or "gpt-5.4-mini",
        max_features=env_int("MAX_LLM_HTML_FEATURES", 6),
    )

    problem_retriever = ProblemRetriever(
        collection_name=os.getenv("QDRANT_PROBLEM_COLLECTION", "problem_patterns_en"),
        embedding_model=embedding_model,
        qdrant=shared_qdrant,
        openai=shared_openai,
    )
    problems = problem_retriever.search(
        features,
        top_k=env_int("PROBLEM_TOP_K", 5),
        candidate_k=env_int("PROBLEM_CANDIDATE_K", 200),
        use_section_filter=env_bool("USE_PROBLEM_SECTION_FILTER", False),
    )
    selected_problems = dedupe_problems(problems)[: env_int("SELECTED_PROBLEM_TOP_K", 3)]

    intervention_retriever = InterventionRetriever(
        collection_name=os.getenv("QDRANT_REVISION_COLLECTION", "intervention_evidence_en"),
        embedding_model=embedding_model,
        qdrant=shared_qdrant,
        openai=shared_openai,
    )
    interventions_by_problem = []
    for problem in selected_problems:
        interventions = intervention_retriever.search_for_problem(
            problem,
            features,
            top_k=env_int("REVISION_TOP_K", 5),
            candidate_k=env_int("REVISION_CANDIDATE_K", 200),
            use_problem_case_filter=env_bool("USE_REVISION_PROBLEM_FILTER", False),
            use_section_filter=env_bool("USE_REVISION_SECTION_FILTER", False),
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
        max_items=env_int("RECOMMENDATION_TOP_N", 3),
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
    recommendation = generate_recommendation(
        retrieval=retrieval,
        section_html=section_html,
        top_n=env_int("RECOMMENDATION_TOP_N", 3),
        mode=os.getenv("RECOMMENDATION_GENERATION_MODE", "llm"),
        provider=normalize_provider(os.getenv("RECOMMENDATION_PROVIDER") or os.getenv("LLM_PROVIDER")),
        model=os.getenv("OPENAI_RECOMMENDATION_MODEL") or os.getenv("LLM_MODEL"),
        base_url=os.getenv("LLM_BASE_URL"),
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
            "collection": os.getenv("QDRANT_PROBLEM_COLLECTION", "problem_patterns_en"),
            "candidate_k": env_int("PROBLEM_CANDIDATE_K", 200),
            "results": [compact_problem(result, "en") for result in problems],
            "selected_for_intervention": [
                compact_problem(result, "en") for result in selected_problems
            ],
        },
        "intervention_retrieval": {
            "collection": os.getenv("QDRANT_REVISION_COLLECTION", "intervention_evidence_en"),
            "candidate_k": env_int("REVISION_CANDIDATE_K", 200),
            "by_problem": interventions_by_problem,
            "selected_interventions": selected_interventions,
            "deduped_top_interventions": selected_interventions,
            "selection_policy": {
                "mode": "top_1_intervention_per_problem",
                "max_items": env_int("RECOMMENDATION_TOP_N", 3),
                "min_problem_score": MIN_PROBLEM_FINAL_SCORE,
                "min_revision_score": MIN_REVISION_FINAL_SCORE,
                "fallback": "return_top_scored_intervention_when_all_candidates_are_filtered",
            },
        },
    }
