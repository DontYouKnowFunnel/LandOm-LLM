"""LLM-based structured input feature extraction for RAG retrieval."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Any

from openai import OpenAI

from rag_pipeline.config import AI_MODELS, REASONING_EFFORTS
from rag_pipeline.features import (
    BEHAVIOR_CLUSTERS,
    HTML_FEATURES,
    PERSONA_TRAITS,
    ExtractedFeatures,
    build_query_text_en,
    extract_features,
    labels_en,
)
from rag_pipeline.html_preprocessor import preprocess_html_for_llm
from rag_pipeline.langsmith_tracking import traced_chat_completion
from rag_pipeline.retrieval_utils import llm_client
from rag_pipeline.taxonomy_definitions import (
    behavior_cluster_definitions_text,
    html_feature_definitions_text,
    persona_trait_definitions_text,
)


def extract_json_text(raw: str) -> str:
    stripped = raw.strip()
    fence_matches = re.findall(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    candidates = [candidate.strip() for candidate in fence_matches if candidate.strip()]
    first_object = stripped.find("{")
    last_object = stripped.rfind("}")
    if first_object != -1 and last_object != -1 and first_object < last_object:
        candidates.append(stripped[first_object : last_object + 1].strip())
    candidates.append(stripped)
    for candidate in candidates:
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"LLM feature extractor returned non-JSON output:\n{raw}")


def section_snapshot(html: str) -> dict[str, Any]:
    preprocessed = preprocess_html_for_llm(html)
    return {
        "compact_html": preprocessed.compact_html,
        "original_chars": preprocessed.original_chars,
        "compact_chars": preprocessed.compact_chars,
        "truncated": preprocessed.truncated,
    }


def summarize_behavior_events(events: list[dict[str, Any]], *, max_events: int = 40) -> dict[str, Any]:
    event_counts = Counter(str(event.get("event") or event.get("type") or "").lower() for event in events)
    session_ids = [str(event.get("sessionId") or "") for event in events]
    session_counts = Counter(value for value in session_ids if value)
    max_depth = 0.0
    max_percentage = 0.0
    section_pings: Counter[str] = Counter()
    click_targets: list[str] = []
    input_fields: list[str] = []
    exits: list[dict[str, Any]] = []
    sessions: dict[str, dict[str, Any]] = defaultdict(lambda: {"events": 0, "durationSeconds": None})

    for event in events:
        session_id = str(event.get("sessionId") or "unknown")
        sessions[session_id]["events"] += 1
        if event.get("sessionDurationSeconds") is not None:
            sessions[session_id]["durationSeconds"] = event.get("sessionDurationSeconds")
        event_name = str(event.get("event") or event.get("type") or "").lower()
        try:
            depth = float(event.get("maxDepth", 0) or 0)
            if depth > 1:
                depth = depth / 100.0
            max_depth = max(max_depth, depth)
        except (TypeError, ValueError):
            pass
        try:
            percentage = float(event.get("percentage", 0) or 0)
            if percentage > 1:
                percentage = percentage / 100.0
            max_percentage = max(max_percentage, percentage)
        except (TypeError, ValueError):
            pass
        if event_name == "ping" and event.get("sectionId"):
            section_pings[str(event.get("sectionId"))] += 1
        if event_name == "click":
            target = str(event.get("targetId") or event.get("cssSelector") or "").strip()
            if target and target not in click_targets:
                click_targets.append(target[:160])
        if event_name == "input":
            field = str(event.get("fieldId") or event.get("cssSelector") or "").strip()
            if field and field not in input_fields:
                input_fields.append(field[:160])
        if event_name == "exit":
            exits.append(
                {
                    "lastElementId": event.get("lastElementId"),
                    "maxDepth": event.get("maxDepth"),
                    "cssSelector": event.get("cssSelector"),
                }
            )

    sampled_events = []
    for event in events[:max_events]:
        sampled_events.append(
            {
                key: event.get(key)
                for key in (
                    "sessionId",
                    "event",
                    "timestamp",
                    "sectionId",
                    "sectionName",
                    "percentage",
                    "maxDepth",
                    "targetId",
                    "fieldId",
                    "lastElementId",
                    "cssSelector",
                    "sessionDurationSeconds",
                )
                if event.get(key) is not None
            }
        )

    return {
        "event_count": len(events),
        "session_count": len(session_counts),
        "event_counts": dict(event_counts),
        "sessions": sessions,
        "max_scroll_depth": round(max(max_depth, max_percentage), 4),
        "section_ping_counts": dict(section_pings),
        "click_targets": click_targets[:12],
        "input_fields": input_fields[:8],
        "exit_events": exits[:8],
        "sampled_events": sampled_events,
        "sample_truncated": len(events) > max_events,
    }


def build_persona_prompt(*, persona_text: str | None, section_label: str, max_traits: int) -> str:
    return f"""
You are a persona classifier for a landing-page RAG system.

Task:
Given the user-provided service persona, classify the visitor into controlled persona trait labels.
Use only the allowed labels and definitions. Choose the smallest sufficient set.

Important rules:
- Classify the persona text with the allowed labels, not with free-form labels.
- Do not infer traits that are not supported by the persona text.
- If the persona is broad or first-time, low_awareness can be selected.
- Return at most {max_traits} labels.
- Return valid JSON only.

Allowed persona trait labels:
{persona_trait_definitions_text()}

Funnel section: {section_label}
Persona text:
{persona_text or ""}

Required JSON schema:
{{
  "persona_traits": ["allowed_label"],
  "evidence": [
    {{
      "trait": "allowed_label",
      "reason": "short reason grounded in the persona text",
      "quote_or_signal": "short quote or signal"
    }}
  ],
  "confidence": "high|medium|low"
}}
""".strip()


def build_behavior_prompt(
    *,
    events: list[dict[str, Any]],
    section_label: str,
    max_clusters: int,
) -> str:
    summary = summarize_behavior_events(events)
    return f"""
You are a visitor behavior classifier for a landing-page RAG system.

Task:
Given normalized visitor behavior logs for one funnel-stage section, classify the behavior into controlled behavior cluster labels.
Use only the allowed labels and definitions. Choose the smallest sufficient set.

Important rules:
- Convert raw events such as click, scroll, ping, input, visibility, and exit into funnel-level behavior patterns.
- Do not classify from page content; classify from the event sequence and session summary.
- A click does not imply conversion unless a conversion/submit/purchase event is present.
- Repeated ping events in the same section can indicate section_stall.
- Deep scroll or long engagement followed by exit can indicate deep_engagement_exit.
- Return at most {max_clusters} labels.
- Return valid JSON only.

Allowed behavior cluster labels:
{behavior_cluster_definitions_text()}

Funnel section: {section_label}
Behavior log summary:
{json.dumps(summary, ensure_ascii=False, indent=2)}

Required JSON schema:
{{
  "behavior_clusters": ["allowed_label"],
  "evidence": [
    {{
      "cluster": "allowed_label",
      "reason": "short reason grounded in the behavior log",
      "quote_or_signal": "short event/count/depth signal"
    }}
  ],
  "confidence": "high|medium|low"
}}
""".strip()


def build_html_prompt(
    *,
    html: str,
    section_label: str,
    persona_traits: list[str],
    behavior_clusters: list[str],
    max_features: int,
) -> str:
    snapshot = section_snapshot(html)
    return f"""
You are an HTML feature classifier for a landing-page RAG system.

Task:
Given one funnel-stage HTML section, classify which controlled HTML feature labels apply.
Use only the allowed labels and definitions. Choose the smallest sufficient set.

Important rules:
- The input is a section-level fragment, not necessarily a full landing page.
- Do not mark a feature just because the concept is absent from this section if that feature is irrelevant to the specified funnel stage.
- Prefer direct evidence from the preprocessed HTML: DOM order, headings, copy, CTA, forms, media, attributes, and preserved class/id/style cues.
- Use persona and behavior as context, but do not invent HTML problems that are not visible in the section.
- Return at most {max_features} labels.
- Return valid JSON only.

Allowed HTML feature labels:
{html_feature_definitions_text()}

Funnel section: {section_label}
Persona context: {labels_en(persona_traits)}
Behavior context: {labels_en(behavior_clusters)}

Preprocessed HTML fragment:
{snapshot["compact_html"]}

Preprocessing metadata:
{json.dumps({key: value for key, value in snapshot.items() if key != "compact_html"}, ensure_ascii=False, indent=2)}

Required JSON schema:
{{
  "html_features": ["allowed_label"],
  "evidence": [
    {{
      "feature": "allowed_label",
      "reason": "short reason grounded in the section",
      "quote_or_signal": "short quote or observable signal"
    }}
  ],
  "confidence": "high|medium|low"
}}
""".strip()


def build_structured_feature_prompt(
    *,
    html: str,
    section_label: str,
    persona_text: str | None,
    events: list[dict[str, Any]],
    max_traits: int,
    max_clusters: int,
    max_features: int,
    persona_traits_override: list[str] | None = None,
    behavior_clusters_override: list[str] | None = None,
    html_features_override: list[str] | None = None,
) -> str:
    snapshot = section_snapshot(html)
    behavior_summary = summarize_behavior_events(events)
    fixed_values = {
        "persona_traits": persona_traits_override or [],
        "behavior_clusters": behavior_clusters_override or [],
        "html_features": html_features_override or [],
    }
    return f"""
You are a structured feature extractor for a landing-page RAG system.

Task:
Given one funnel-stage HTML section, a user-provided service persona, and normalized visitor behavior logs,
classify all three inputs into the controlled retrieval labels used by the RAG DB.

Important rules:
- Use only the allowed labels and definitions below. Do not invent free-form labels.
- Choose the smallest sufficient set for each field.
- Persona traits must be grounded in the persona text.
- Behavior clusters must be grounded in the event sequence and behavior summary, not in page content.
- HTML feature labels must be grounded in the preprocessed HTML fragment.
- The input HTML is a section-level fragment, not necessarily a full landing page.
- Prefer direct HTML evidence from DOM order, headings, copy, CTA, forms, media, attributes, and preserved class/id/style cues.
- Use persona and behavior as context for prioritizing HTML issues, but do not invent HTML problems that are not visible in the section.
- Do not mark a feature just because the concept is absent from this section if that feature is irrelevant to the specified funnel stage.
- A click does not imply conversion unless a conversion/submit/purchase event is present.
- Repeated ping events in the same section can indicate section_stall.
- Deep scroll or long engagement followed by exit can indicate deep_engagement_exit.
- If fixed labels are provided, copy those labels exactly into the corresponding output field and do not add unsupported labels.
- Return at most {max_traits} persona traits, {max_clusters} behavior clusters, and {max_features} HTML features.
- Return valid JSON only.

Allowed persona trait labels:
{persona_trait_definitions_text()}

Allowed behavior cluster labels:
{behavior_cluster_definitions_text()}

Allowed HTML feature labels:
{html_feature_definitions_text()}

Funnel section: {section_label}

Fixed labels supplied by caller:
{json.dumps(fixed_values, ensure_ascii=False, indent=2)}

Persona text:
{persona_text or ""}

Behavior log summary:
{json.dumps(behavior_summary, ensure_ascii=False, indent=2)}

Preprocessed HTML fragment:
{snapshot["compact_html"]}

Preprocessing metadata:
{json.dumps({key: value for key, value in snapshot.items() if key != "compact_html"}, ensure_ascii=False, indent=2)}

Required JSON schema:
{{
  "persona_traits": ["allowed_label"],
  "behavior_clusters": ["allowed_label"],
  "html_features": ["allowed_label"],
  "evidence": {{
    "persona": [
      {{
        "trait": "allowed_label",
        "reason": "short reason grounded in the persona text",
        "quote_or_signal": "short quote or signal"
      }}
    ],
    "behavior": [
      {{
        "cluster": "allowed_label",
        "reason": "short reason grounded in the behavior log",
        "quote_or_signal": "short event/count/depth signal"
      }}
    ],
    "html": [
      {{
        "feature": "allowed_label",
        "reason": "short reason grounded in the section",
        "quote_or_signal": "short quote or observable signal"
      }}
    ]
  }},
  "confidence": {{
    "persona": "high|medium|low",
    "behavior": "high|medium|low",
    "html": "high|medium|low"
  }}
}}
""".strip()


def validate_label_payload(
    payload: dict[str, Any],
    *,
    label_key: str,
    evidence_label_key: str,
    allowed_labels: set[str],
    max_labels: int,
) -> tuple[list[str], list[dict[str, str]], str]:
    raw_labels = payload.get(label_key, [])
    if not isinstance(raw_labels, list):
        raw_labels = []
    labels = []
    for value in raw_labels:
        label = str(value).strip()
        if label in allowed_labels and label not in labels:
            labels.append(label)
    labels = labels[:max_labels]

    raw_evidence = payload.get("evidence", [])
    evidence = []
    if isinstance(raw_evidence, list):
        for item in raw_evidence:
            if not isinstance(item, dict):
                continue
            label = str(item.get(evidence_label_key, "")).strip()
            if label not in labels:
                continue
            evidence.append(
                {
                    evidence_label_key: label,
                    "reason": str(item.get("reason", "")).strip()[:500],
                    "quote_or_signal": str(item.get("quote_or_signal", "")).strip()[:300],
                }
            )
    confidence = str(payload.get("confidence", "medium")).strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    return labels, evidence, confidence


def validate_labels(values: Any, *, allowed_labels: set[str], max_labels: int) -> list[str]:
    if not isinstance(values, list):
        return []
    labels = []
    for value in values:
        label = str(value).strip()
        if label in allowed_labels and label not in labels:
            labels.append(label)
    return labels[:max_labels]


def validate_nested_evidence(
    values: Any,
    *,
    evidence_label_key: str,
    labels: list[str],
) -> list[dict[str, str]]:
    evidence = []
    if not isinstance(values, list):
        return evidence
    for item in values:
        if not isinstance(item, dict):
            continue
        label = str(item.get(evidence_label_key, "")).strip()
        if label not in labels:
            continue
        evidence.append(
            {
                evidence_label_key: label,
                "reason": str(item.get("reason", "")).strip()[:500],
                "quote_or_signal": str(item.get("quote_or_signal", "")).strip()[:300],
            }
        )
    return evidence


def confidence_value(value: Any) -> str:
    confidence = str(value or "medium").strip().lower()
    return confidence if confidence in {"high", "medium", "low"} else "medium"


def validate_llm_payload(payload: dict[str, Any], *, max_features: int) -> tuple[list[str], list[dict[str, str]], str]:
    return validate_label_payload(
        payload,
        label_key="html_features",
        evidence_label_key="feature",
        allowed_labels=HTML_FEATURES,
        max_labels=max_features,
    )


def validate_structured_feature_payload(
    payload: dict[str, Any],
    *,
    max_traits: int,
    max_clusters: int,
    max_features: int,
    persona_traits_override: list[str] | None = None,
    behavior_clusters_override: list[str] | None = None,
    html_features_override: list[str] | None = None,
) -> tuple[list[str], list[str], list[str], dict[str, Any]]:
    persona_traits = (
        validate_labels(persona_traits_override, allowed_labels=PERSONA_TRAITS, max_labels=max_traits)
        if persona_traits_override
        else validate_labels(payload.get("persona_traits"), allowed_labels=PERSONA_TRAITS, max_labels=max_traits)
    )
    behavior_clusters = (
        validate_labels(behavior_clusters_override, allowed_labels=BEHAVIOR_CLUSTERS, max_labels=max_clusters)
        if behavior_clusters_override
        else validate_labels(payload.get("behavior_clusters"), allowed_labels=BEHAVIOR_CLUSTERS, max_labels=max_clusters)
    )
    html_features = (
        validate_labels(html_features_override, allowed_labels=HTML_FEATURES, max_labels=max_features)
        if html_features_override
        else validate_labels(payload.get("html_features"), allowed_labels=HTML_FEATURES, max_labels=max_features)
    )

    raw_evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
    raw_confidence = payload.get("confidence") if isinstance(payload.get("confidence"), dict) else {}
    notes = {
        "structured_feature_source": "llm_combined",
        "persona_source": "override" if persona_traits_override else "llm_combined",
        "behavior_source": "override" if behavior_clusters_override else "llm_combined",
        "html_feature_source": "override" if html_features_override else "llm_combined",
        "llm_persona_confidence": confidence_value(raw_confidence.get("persona")),
        "llm_behavior_confidence": confidence_value(raw_confidence.get("behavior")),
        "llm_feature_confidence": confidence_value(raw_confidence.get("html")),
        "llm_persona_evidence": validate_nested_evidence(
            raw_evidence.get("persona"),
            evidence_label_key="trait",
            labels=persona_traits,
        ),
        "llm_behavior_evidence": validate_nested_evidence(
            raw_evidence.get("behavior"),
            evidence_label_key="cluster",
            labels=behavior_clusters,
        ),
        "llm_feature_evidence": validate_nested_evidence(
            raw_evidence.get("html"),
            evidence_label_key="feature",
            labels=html_features,
        ),
        "llm_structured_feature_raw": payload,
    }
    return persona_traits, behavior_clusters, html_features, notes


def infer_persona_traits_with_llm(
    *,
    persona_text: str | None,
    section_label: str,
    client: OpenAI | None = None,
    model: str = AI_MODELS.feature_extraction[1],
    provider: str = AI_MODELS.feature_extraction[0],
    max_traits: int = 4,
) -> tuple[list[str], dict[str, Any]]:
    client = client or llm_client(provider)
    prompt = build_persona_prompt(
        persona_text=persona_text,
        section_label=section_label,
        max_traits=max_traits,
    )
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You classify a landing-page visitor persona into a controlled label set. Return valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
    }
    if provider == "openai":
        request_kwargs["reasoning_effort"] = REASONING_EFFORTS["feature_extraction"]
    response = traced_chat_completion(
        client=client,
        request_kwargs=request_kwargs,
        provider=provider,
        model=model,
        workflow="rag.optimization",
        stage="rag.feature.persona",
        prompt_name="rag.feature.persona.v1",
        metadata={"section_label": section_label},
    )
    raw = response.choices[0].message.content or ""
    parsed = json.loads(extract_json_text(raw))
    traits, evidence, confidence = validate_label_payload(
        parsed,
        label_key="persona_traits",
        evidence_label_key="trait",
        allowed_labels=PERSONA_TRAITS,
        max_labels=max_traits,
    )
    notes = {
        "persona_source": "llm",
        "llm_persona_model": model,
        "llm_persona_confidence": confidence,
        "llm_persona_evidence": evidence,
        "llm_persona_raw": parsed,
    }
    return traits, notes


def infer_behavior_clusters_with_llm(
    *,
    events: list[dict[str, Any]],
    section_label: str,
    client: OpenAI | None = None,
    model: str = AI_MODELS.feature_extraction[1],
    provider: str = AI_MODELS.feature_extraction[0],
    max_clusters: int = 3,
) -> tuple[list[str], dict[str, Any]]:
    client = client or llm_client(provider)
    prompt = build_behavior_prompt(
        events=events,
        section_label=section_label,
        max_clusters=max_clusters,
    )
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You classify landing-page visitor behavior logs into a controlled label set. Return valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
    }
    if provider == "openai":
        request_kwargs["reasoning_effort"] = REASONING_EFFORTS["feature_extraction"]
    response = traced_chat_completion(
        client=client,
        request_kwargs=request_kwargs,
        provider=provider,
        model=model,
        workflow="rag.optimization",
        stage="rag.feature.behavior",
        prompt_name="rag.feature.behavior.v1",
        metadata={"section_label": section_label, "event_count": len(events)},
    )
    raw = response.choices[0].message.content or ""
    parsed = json.loads(extract_json_text(raw))
    clusters, evidence, confidence = validate_label_payload(
        parsed,
        label_key="behavior_clusters",
        evidence_label_key="cluster",
        allowed_labels=BEHAVIOR_CLUSTERS,
        max_labels=max_clusters,
    )
    notes = {
        "behavior_source": "llm",
        "llm_behavior_model": model,
        "llm_behavior_confidence": confidence,
        "llm_behavior_evidence": evidence,
        "llm_behavior_raw": parsed,
        "llm_behavior_summary": summarize_behavior_events(events),
    }
    return clusters, notes


def infer_html_features_with_llm(
    *,
    html: str,
    section_label: str,
    persona_traits: list[str],
    behavior_clusters: list[str],
    client: OpenAI | None = None,
    model: str = AI_MODELS.feature_extraction[1],
    provider: str = AI_MODELS.feature_extraction[0],
    max_features: int = 6,
) -> tuple[list[str], dict[str, Any]]:
    client = client or llm_client(provider)
    prompt = build_html_prompt(
        html=html,
        section_label=section_label,
        persona_traits=persona_traits,
        behavior_clusters=behavior_clusters,
        max_features=max_features,
    )
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You classify landing-page section HTML into a controlled label set. Return valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
    }
    if provider == "openai":
        request_kwargs["reasoning_effort"] = REASONING_EFFORTS["feature_extraction"]
    response = traced_chat_completion(
        client=client,
        request_kwargs=request_kwargs,
        provider=provider,
        model=model,
        workflow="rag.optimization",
        stage="rag.feature.html",
        prompt_name="rag.feature.html.v1",
        metadata={"section_label": section_label, "html_chars": len(html)},
    )
    raw = response.choices[0].message.content or ""
    parsed = json.loads(extract_json_text(raw))
    features, evidence, confidence = validate_llm_payload(parsed, max_features=max_features)
    notes = {
        "html_feature_source": "llm",
        "llm_feature_model": model,
        "llm_feature_confidence": confidence,
        "llm_feature_evidence": evidence,
        "llm_feature_raw": parsed,
    }
    return features, notes


def infer_structured_features_with_llm(
    *,
    html: str,
    section_label: str,
    persona_text: str | None,
    events: list[dict[str, Any]],
    persona_traits_override: list[str] | None = None,
    behavior_clusters_override: list[str] | None = None,
    html_features_override: list[str] | None = None,
    client: OpenAI | None = None,
    model: str = AI_MODELS.feature_extraction[1],
    provider: str = AI_MODELS.feature_extraction[0],
    max_traits: int = 4,
    max_clusters: int = 3,
    max_features: int = 6,
) -> tuple[list[str], list[str], list[str], dict[str, Any]]:
    client = client or llm_client(provider)
    prompt = build_structured_feature_prompt(
        html=html,
        section_label=section_label,
        persona_text=persona_text,
        events=events,
        max_traits=max_traits,
        max_clusters=max_clusters,
        max_features=max_features,
        persona_traits_override=persona_traits_override,
        behavior_clusters_override=behavior_clusters_override,
        html_features_override=html_features_override,
    )
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You classify landing-page HTML, persona, and visitor behavior into controlled "
                    "retrieval labels. Return valid JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    if provider == "openai":
        request_kwargs["reasoning_effort"] = REASONING_EFFORTS["feature_extraction"]
    response = traced_chat_completion(
        client=client,
        request_kwargs=request_kwargs,
        provider=provider,
        model=model,
        workflow="rag.optimization",
        stage="rag.feature.structured",
        prompt_name="rag.feature.structured.v1",
        metadata={
            "section_label": section_label,
            "html_chars": len(html),
            "event_count": len(events),
        },
    )
    raw = response.choices[0].message.content or ""
    parsed = json.loads(extract_json_text(raw))
    persona_traits, behavior_clusters, html_features, notes = validate_structured_feature_payload(
        parsed,
        max_traits=max_traits,
        max_clusters=max_clusters,
        max_features=max_features,
        persona_traits_override=persona_traits_override,
        behavior_clusters_override=behavior_clusters_override,
        html_features_override=html_features_override,
    )
    notes.update(
        {
            "llm_structured_feature_model": model,
            "llm_behavior_summary": summarize_behavior_events(events),
        }
    )
    return persona_traits, behavior_clusters, html_features, notes


def extract_features_with_llm_html(
    *,
    html: str,
    section_label: str,
    persona_text: str | None = None,
    persona_traits_override: list[str] | None = None,
    behavior_events: list[dict[str, Any]] | None = None,
    behavior_clusters_override: list[str] | None = None,
    html_features_override: list[str] | None = None,
    client: OpenAI | None = None,
    model: str = AI_MODELS.feature_extraction[1],
    provider: str = AI_MODELS.feature_extraction[0],
    max_features: int = 6,
) -> ExtractedFeatures:
    base_features = extract_features(
        html=html,
        section_label=section_label,
        persona_text=persona_text,
        persona_traits_override=persona_traits_override,
        behavior_events=behavior_events,
        behavior_clusters_override=behavior_clusters_override,
    )
    notes = {
        key: value
        for key, value in base_features.extraction_notes.items()
        if key in {"headings", "ctas", "word_count"}
    }
    behavior_events = behavior_events or []
    persona_traits, behavior_clusters, llm_html_features, structured_notes = (
        infer_structured_features_with_llm(
            html=html,
            section_label=base_features.section_label,
            persona_text=persona_text,
            events=behavior_events,
            persona_traits_override=persona_traits_override,
            behavior_clusters_override=behavior_clusters_override,
            html_features_override=html_features_override,
            client=client,
            model=model,
            provider=provider,
            max_features=max_features,
        )
    )
    notes.update(structured_notes)
    if not persona_traits:
        notes["llm_persona_empty"] = True
    if not behavior_clusters:
        notes["llm_behavior_empty"] = True
    if not llm_html_features:
        notes["llm_feature_empty"] = True

    query_text_en = build_query_text_en(
        section_label=base_features.section_label,
        persona_traits=persona_traits,
        behavior_clusters=behavior_clusters,
        html_features=llm_html_features,
    )
    return replace(
        base_features,
        persona_traits=persona_traits,
        behavior_clusters=behavior_clusters,
        html_features=llm_html_features,
        query_text_en=query_text_en,
        extraction_notes=notes,
    )
