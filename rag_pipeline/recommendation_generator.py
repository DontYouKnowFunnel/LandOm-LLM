#!/usr/bin/env python3
"""Generate concrete landing-page recommendations from RAG retrieval output."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_pipeline.html_preprocessor import preprocess_html_for_llm


DEFAULT_RETRIEVAL_JSON = PROJECT_ROOT / "run/rag_retrieval_smoke_override.json"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "run/rag_recommendation.json"
DEFAULT_OUTPUT_MD = PROJECT_ROOT / "run/rag_recommendation.md"


PERSONA_TRAIT_DESCRIPTIONS_KO = {
    "low_awareness": "서비스를 처음 접했거나 핵심 가치를 빠르게 판단해야 하는 사용자",
    "time_constrained": "긴 설명을 읽기보다 첫 화면에서 빠르게 판단하려는 사용자",
    "price_sensitive": "비용 대비 효용과 가격 정당성을 중요하게 보는 사용자",
    "risk_averse": "가입, 결제, 개인정보 제공, 도입 실패 가능성을 조심스럽게 보는 사용자",
    "trust_sensitive": "후기, 고객 사례, 보안, 브랜드 신뢰 같은 근거를 확인하려는 사용자",
    "comparison_oriented": "대안과 비교해 왜 이 서비스를 선택해야 하는지 알고 싶은 사용자",
    "nontechnical_user": "기술적 기능보다 쉬운 사용 방식과 업무 효과를 먼저 이해해야 하는 사용자",
    "technical_user": "작동 방식, 연동, 구현 가능성, 세부 기능 근거를 확인하려는 사용자",
    "team_buyer": "개인 효율보다 팀 협업, 도입 효과, 조직 내 사용성을 중요하게 보는 사용자",
    "individual_user": "개인 작업 흐름과 즉시 얻는 효용을 중심으로 판단하는 사용자",
    "enterprise_buyer": "보안, 관리, 확장성, 구매 절차를 함께 고려하는 조직 구매자",
    "small_business_owner": "운영 부담, 비용, 빠른 실행 가능성을 중요하게 보는 소규모 사업자",
    "founder_operator": "성장, 실행 속도, 리소스 절감을 중요하게 보는 창업자/운영자",
}


BEHAVIOR_CLUSTER_DESCRIPTIONS_KO = {
    "quick_exit": "해당 섹션에서 오래 머무르지 않고 빠르게 이탈하는 경향",
    "passive_browsing": "뚜렷한 클릭 없이 내용을 훑어보는 경향",
    "interaction_without_conversion": "클릭이나 입력 시도는 있지만 전환까지 이어지지 않는 경향",
    "deep_engagement_exit": "깊게 읽거나 스크롤했지만 최종 행동 없이 이탈하는 경향",
    "section_stall": "특정 섹션에서 오래 머무르지만 다음 행동으로 이어지지 않는 경향",
    "shallow_scan": "섹션을 깊게 읽기보다 짧게 훑고 지나가는 경향",
}


HTML_FEATURE_DESCRIPTIONS_KO = {
    "unclear_value_proposition": "첫 화면에서 사용자가 얻는 핵심 결과가 즉시 드러나지 않음",
    "feature_oriented_headline": "핵심 문구가 사용자 결과보다 기능명이나 제품 구조 중심으로 읽힘",
    "generic_cta": "버튼 문구가 클릭 후 얻게 되는 결과나 다음 단계를 충분히 설명하지 못함",
    "weak_action_motivation": "지금 행동해야 할 이유가 약하게 전달됨",
    "low_cta_value_proximity": "CTA 근처에 가치, 근거, 안심 정보가 충분히 붙어 있지 않음",
    "no_social_proof": "고객, 후기, 수치, 사례 같은 사회적 증거가 부족함",
    "weak_trust_signal": "전환 전에 필요한 신뢰 근거가 약하게 보임",
    "missing_credibility_marker": "전문성, 고객군, 성과 수치 등 신뢰 마커가 부족함",
    "unclear_pricing": "가격 기준, 플랜 차이, 과금 조건을 판단하기 어려움",
    "weak_value_justification": "가격이나 행동 요구에 비해 제공 가치가 충분히 설명되지 않음",
    "no_risk_reversal": "무료 체험, 해지, 환불, 보안, 개인정보 같은 리스크 완화 정보가 부족함",
    "missing_objection_handling": "구매 전 자연스럽게 생기는 질문이나 반론을 해소하는 정보가 부족함",
    "unclear_category_explanation": "서비스 카테고리나 작동 방식을 처음 보는 사용자가 이해하기 어려움",
    "audience_not_explicit": "누구를 위한 서비스인지 역할, 팀, 산업, 상황이 충분히 드러나지 않음",
    "generic_positioning": "대안과 비교했을 때 이 서비스만의 선택 이유가 약하게 보임",
    "missing_use_case_context": "실제 사용 상황이나 적용 장면이 부족함",
    "feature_list_without_benefits": "기능 목록은 있지만 각 기능이 만드는 사용자 이득이 약하게 연결됨",
    "insufficient_product_context": "제품이 실제로 어떻게 쓰이는지 상상할 만한 맥락이 부족함",
    "no_visual_product_demo": "화면, 결과물, 사용 흐름 같은 시각적 제품 이해 단서가 부족함",
    "high_information_density": "한 섹션에 정보가 많아 핵심 판단 포인트가 묻힐 가능성이 있음",
    "poor_information_hierarchy": "핵심 메시지, 근거, 상세 정보의 우선순위가 분명하지 않음",
    "weak_differentiation": "경쟁 대안 대비 차별점이 충분히 설명되지 않음",
    "missing_onboarding_reassurance": "도입, 설정, 전환, 학습 부담을 낮추는 안내가 부족함",
    "unclear_conversion_entry_value": "가입, 데모, 문의 등 전환 진입 행동의 가치가 불명확함",
    "unclear_next_step_after_cta": "CTA를 누른 뒤 어떤 일이 일어나는지 예측하기 어려움",
    "missing_friction_reassurance": "전환 행동의 시간, 노력, 부담을 낮추는 안심 정보가 부족함",
    "unclear_form_value": "폼을 작성해야 하는 이유와 사용자가 받는 보상이 불명확함",
    "too_many_form_fields": "입력해야 할 항목이 많아 시작 부담이 커질 수 있음",
    "missing_privacy_reassurance": "개인정보 입력 시 용도, 보안, 스팸 방지에 대한 안심 문구가 부족함",
}


EXPECTED_EFFECT_KO = {
    "dwell_time": "섹션 체류 시간",
    "stage_exit": "해당 단계 이탈",
    "comprehension_friction": "가치 이해 마찰",
    "cta_interaction": "CTA 상호작용",
    "conversion_entry": "전환 진입",
    "form_start": "폼 시작",
    "form_completion": "폼 완료",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_html_summary_from_html(html: str, max_chars: int = 2200) -> dict[str, Any]:
    preprocessed = preprocess_html_for_llm(html)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    headings = [clean_text(tag.get_text(" ")) for tag in soup.find_all(["h1", "h2", "h3"])]
    ctas = []
    for tag in soup.find_all(["a", "button"]):
        text = clean_text(tag.get_text(" ") or tag.get("aria-label", "") or tag.get("title", ""))
        if text:
            ctas.append(text[:120])

    text = clean_text(soup.get_text(" "))
    section_copy_snippets = []
    for value in headings[:5]:
        if value and value not in section_copy_snippets:
            section_copy_snippets.append(value)
    for value in ctas[:5]:
        if value and value not in section_copy_snippets:
            section_copy_snippets.append(value)

    return {
        "available": True,
        "html_file": None,
        "structure_preserved_html": preprocessed.compact_html,
        "preprocessing_metadata": {
            "original_chars": preprocessed.original_chars,
            "compact_chars": preprocessed.compact_chars,
            "truncated": preprocessed.truncated,
        },
        "section_copy_snippets": section_copy_snippets[:8],
        "action_texts": ctas[:8],
        "visible_text_excerpt": text[:max_chars],
    }


def extract_html_summary(path: Path | None, max_chars: int = 2200) -> dict[str, Any]:
    if not path or not path.exists():
        return {"available": False}

    html = path.read_text(encoding="utf-8", errors="ignore")
    summary = extract_html_summary_from_html(html, max_chars=max_chars)
    summary["html_file"] = str(path)
    return summary


def pick_primary_problem(payload: dict[str, Any]) -> dict[str, Any]:
    selected = payload.get("problem_retrieval", {}).get("selected_for_intervention") or []
    if selected:
        return selected[0]
    results = payload.get("problem_retrieval", {}).get("results") or []
    return results[0] if results else {}


def pick_interventions(payload: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    intervention_retrieval = payload.get("intervention_retrieval", {})
    items = (
        intervention_retrieval.get("selected_interventions")
        or intervention_retrieval.get("deduped_top_interventions")
        or []
    )
    selected: list[dict[str, Any]] = []
    seen_problem_cases: set[str] = set()
    for item in items:
        problem_case = str(item.get("problem_case") or "")
        if problem_case and problem_case in seen_problem_cases:
            continue
        if problem_case:
            seen_problem_cases.add(problem_case)
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def describe_values(values: list[str], descriptions: dict[str, str]) -> list[str]:
    return [descriptions.get(value, value.replace("_", " ")) for value in values]


def selected_taxonomy_meanings(
    *,
    persona_traits: list[str],
    behavior_clusters: list[str],
    html_features: list[str],
) -> dict[str, dict[str, str]]:
    return {
        "persona_traits": {
            value: PERSONA_TRAIT_DESCRIPTIONS_KO.get(value, value.replace("_", " "))
            for value in persona_traits
        },
        "behavior_clusters": {
            value: BEHAVIOR_CLUSTER_DESCRIPTIONS_KO.get(value, value.replace("_", " "))
            for value in behavior_clusters
        },
        "html_features": {
            value: HTML_FEATURE_DESCRIPTIONS_KO.get(value, value.replace("_", " "))
            for value in html_features
        },
    }


def compact_generation_context(
    *,
    retrieval: dict[str, Any],
    html_summary: dict[str, Any],
    top_n: int,
) -> dict[str, Any]:
    features = retrieval.get("features", {})
    primary_problem = pick_primary_problem(retrieval)
    interventions = pick_interventions(retrieval, top_n)
    persona_traits = features.get("persona_traits", [])
    behavior_clusters = features.get("behavior_clusters", [])
    html_features = features.get("html_features", [])
    problem_lookup: dict[str, dict[str, Any]] = {}
    problem_retrieval = retrieval.get("problem_retrieval", {})
    for problem in (
        problem_retrieval.get("selected_for_intervention", [])
        + problem_retrieval.get("results", [])
    ):
        problem_case = str(problem.get("problem_case") or "")
        if problem_case and problem_case not in problem_lookup:
            problem_lookup[problem_case] = problem
    for group in retrieval.get("intervention_retrieval", {}).get("by_problem", []):
        problem = group.get("problem") or {}
        problem_case = str(problem.get("problem_case") or "")
        if problem_case and problem_case not in problem_lookup:
            problem_lookup[problem_case] = problem

    candidate_interventions = []
    for item in interventions:
        problem_case = str(item.get("problem_case") or "")
        matched_problem = problem_lookup.get(problem_case, {})
        candidate_interventions.append(
            {
                "intervention_title_ko": label_ko(item.get("intervention", "")),
                "problem_name_ko": label_ko(problem_case),
                "problem_description": matched_problem.get("problem_description")
                or primary_problem.get("problem_description"),
                "problem_barrier": matched_problem.get("barrier") or primary_problem.get("barrier"),
                "problem_supporting_context": matched_problem.get("llm_context"),
                "intervention_description": item.get("intervention_description"),
                "mechanism": item.get("mechanism"),
                "expected_effect_direction": item.get("expected_effect_direction", {}),
                "risk": item.get("risk"),
                "evidence_summary": item.get("evidence_summary"),
                "final_score": item.get("final_score"),
                "source_observed_pattern": item.get("source_reference", {}).get("observed_pattern_original"),
            }
        )

    return {
        "input": retrieval.get("input", {}),
        "section_label": features.get("section_label") or retrieval.get("input", {}).get("section_label"),
        "persona_text": retrieval.get("input", {}).get("persona_text", ""),
        "user_facing_input_interpretation": {
            "persona": describe_values(persona_traits, PERSONA_TRAIT_DESCRIPTIONS_KO),
            "behavior": describe_values(behavior_clusters, BEHAVIOR_CLUSTER_DESCRIPTIONS_KO),
            "section_observations": describe_values(html_features, HTML_FEATURE_DESCRIPTIONS_KO),
        },
        "html_summary": html_summary,
        "primary_problem": {
            "problem_name_ko": label_ko(primary_problem.get("problem_case", "")),
            "problem_description": primary_problem.get("problem_description"),
            "barrier": primary_problem.get("barrier"),
            "final_score": primary_problem.get("final_score"),
            "llm_context": primary_problem.get("llm_context"),
            "source_observed_pattern": primary_problem.get("source_reference", {}).get("observed_pattern_original"),
        },
        "candidate_interventions": candidate_interventions,
        "taxonomy_meanings_for_reasoning_only": selected_taxonomy_meanings(
            persona_traits=persona_traits,
            behavior_clusters=behavior_clusters,
            html_features=html_features,
        ),
        "internal_trace_do_not_quote": {
            "persona_traits": persona_traits,
            "behavior_clusters": behavior_clusters,
            "html_features": html_features,
            "primary_problem_case": primary_problem.get("problem_case"),
            "intervention_ids": [item.get("intervention") for item in interventions],
        },
    }


def resolve_client(provider: str, model: str | None, base_url: str | None) -> tuple[OpenAI, str]:
    load_dotenv(PROJECT_ROOT / ".env")
    timeout = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "90"))
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        return OpenAI(
            api_key=api_key,
            base_url=base_url or os.getenv("OPENAI_BASE_URL") or None,
            timeout=timeout,
        ), (
            model or os.getenv("OPENAI_RECOMMENDATION_MODEL") or "gpt-5.4-mini"
        )
    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set.")
        return OpenAI(
            api_key=api_key,
            base_url=base_url or "https://api.groq.com/openai/v1",
            timeout=timeout,
        ), (
            model or os.getenv("GROQ_RECOMMENDATION_MODEL") or "meta-llama/llama-4-scout-17b-16e-instruct"
        )
    if provider == "ollama":
        return OpenAI(
            api_key=os.getenv("OLLAMA_API_KEY") or "ollama",
            base_url=base_url or os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434/v1",
            timeout=timeout,
        ), (model or os.getenv("OLLAMA_MODEL") or "llama2:latest")
    raise RuntimeError(f"Unsupported provider: {provider}")


def recommendation_prompt(context: dict[str, Any]) -> str:
    max_recommendation_count = len(context.get("candidate_interventions") or [])
    return f"""
You are generating landing-page improvement recommendations from retrieved RAG evidence.

Rules:
- Write in Korean only. Do not use non-Korean words unless they are brand names, product names, or original UI text from the input.
- The input unit is one funnel-stage section HTML fragment, not the entire page.
- Ground every recommendation in the provided Problem Pattern and Intervention Evidence.
- Use the structure-preserved HTML in the context as the current section source. Reflect observable hierarchy, repeated blocks, CTA placement, media/form/action elements, class/id hints, and inline layout/style hints when making recommendations.
- For each recommendation, clearly state the current problem it addresses in the "problem" field.
- When writing implementation_direction, refer to actual editable areas visible in the current section structure. Do not invent current elements that are not present.
- Do not claim proven conversion lift. Say "expected direction" or "plausible effect".
- Be concrete enough that a designer, marketer, or frontend engineer can revise the section.
- Do not output code. The goal is improvement recommendations, not HTML generation.
- For each recommendation, provide concrete changes, implementation direction, copy direction, layout direction, expected behavior change, and caveat.
- The recommendations array may contain 1 to {max_recommendation_count} items. Include only recommendations that are meaningfully distinct and actionable.
- If multiple candidate interventions point to different useful edits, return them as separate recommendations instead of collapsing everything into one broad recommendation.
- Do not force weak or repetitive recommendations just to fill the maximum count.
- Rank recommendations sequentially by expected usefulness for the current section.
- Use the current section content and persona context when writing example copy. Avoid generic placeholder copy.
- Prefer small, implementable edits over broad redesign advice unless the evidence clearly requires structural change.
- Treat taxonomy labels and extractor fields as internal reasoning data only.
- Do not expose snake_case labels such as feature_oriented_headline, quick_exit, value_comprehension_gap, or intervention ids in user-facing fields.
- Do not mention extractor container names such as "headings", "ctas", "html_features", "behavior_clusters", or "persona_traits".
- Explain the meaning of internal categories in natural Korean, e.g. "기능 중심으로 읽히는 문구" rather than "feature_oriented_headline".
- If you refer to current section content, describe the user-facing issue, not the extraction artifact.
- Do not mix in other languages such as Hindi, Japanese, or Chinese. If you need the meaning of "ambiguous", write "불명확함".
- Return valid JSON only. No markdown fence.

Required JSON schema:
{{
  "summary": "one-sentence diagnosis",
  "diagnosis": {{
    "funnel_stage": "...",
    "problem_name_ko": "...",
    "why_this_is_a_problem": "...",
    "supporting_signals": ["..."],
    "confidence_note": "..."
  }},
  "recommendations": [
    {{
      "rank": 1,
      "title": "...",
      "problem": "current landing-page problem addressed by this recommendation",
      "what_to_change": ["..."],
      "implementation_direction": "concrete non-code edit direction",
      "copy_direction": ["headline or main copy direction", "supporting copy direction", "CTA copy direction"],
      "layout_direction": "layout direction",
      "expected_behavior_change": "expected direction of visitor behavior change",
      "risk_or_caveat": "..."
    }}
  ],
  "priority_order": ["..."],
  "validation_plan": ["..."],
  "_trace": {{
    "primary_problem_case": "internal id only",
    "intervention_ids": ["internal ids only"]
  }}
}}

RAG context:
{json.dumps(context, ensure_ascii=False, indent=2)}
""".strip()


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
    raise RuntimeError(f"Model output is not valid JSON:\n{raw}")


def generate_with_llm(
    *,
    context: dict[str, Any],
    provider: str,
    model: str | None,
    base_url: str | None,
) -> dict[str, Any]:
    client, resolved_model = resolve_client(provider, model, base_url)
    request_kwargs: dict[str, Any] = {
        "model": resolved_model,
        "messages": [
            {
                "role": "system",
                "content": "You produce grounded, concise CRO recommendations as valid JSON.",
            },
            {"role": "user", "content": recommendation_prompt(context)},
        ],
    }
    if provider == "openai":
        request_kwargs["reasoning_effort"] = "medium"
    response = client.chat.completions.create(**request_kwargs)
    content = response.choices[0].message.content or ""
    parsed = json.loads(extract_json_text(content))
    parsed["_generation_metadata"] = {
        "mode": "llm",
        "provider": provider,
        "model": resolved_model,
    }
    return parsed


INTERVENTION_TITLES_KO = {
    "headline_value_reframing": "헤드라인을 사용자 결과 중심으로 재구성",
    "cta_specificity_improvement": "CTA의 클릭 후 결과를 구체화",
    "low_friction_cta_reassurance": "CTA 주변에 낮은 시작 부담을 명시",
    "early_testimonial_placement": "초기 구간에 관련성 높은 신뢰 근거 배치",
    "logo_or_customer_proof": "고객/로고 기반 신뢰 신호 보강",
    "quantified_social_proof": "정량적 성과 근거 제시",
    "package_difference_clarification": "요금제 차이를 비교 가능하게 정리",
    "value_justification_near_price": "가격 근처에 가치 근거 배치",
    "free_trial_or_refund_reassurance": "결제/가입 리스크 완화 정보 추가",
    "faq_objection_handling": "남은 반론을 FAQ로 해소",
    "audience_self_identification": "사용자가 자신에게 맞는 경로를 식별하게 구성",
    "benefit_oriented_feature_copy": "기능 설명을 이득 중심으로 변환",
    "feature_to_outcome_mapping": "기능과 사용자 결과를 1:1로 매핑",
    "visual_feature_walkthrough": "제품 화면/사용 흐름을 시각적으로 설명",
    "use_case_linked_feature_explanation": "기능을 실제 use case와 연결",
    "onboarding_reassurance": "도입/설정 부담을 낮추는 정보 추가",
    "information_hierarchy_restructuring": "정보 우선순위와 위계를 재구성",
    "competitive_positioning_clarification": "대안 대비 차별점을 명확화",
    "entry_action_value_clarification": "전환 진입 행동의 가치와 다음 단계를 명확화",
}


PROBLEM_NAMES_KO = {
    "value_comprehension_gap": "초기 가치 이해 부족",
    "weak_cta_affordance": "CTA 행동 가치 불명확",
    "trust_gap": "신뢰 근거 부족",
    "pricing_value_uncertainty": "가격 판단 불확실성",
    "risk_uncertainty": "전환 리스크 불확실성",
    "unresolved_objection": "남은 반론/질문 미해소",
    "low_relevance_perception": "사용자 관련성 인식 부족",
    "feature_benefit_gap": "기능과 이득의 연결 부족",
    "product_understanding_gap": "제품 사용 방식 이해 부족",
    "cognitive_overload": "정보량/우선순위 부담",
    "weak_differentiation": "차별점 부족",
    "adoption_friction": "도입 부담",
    "conversion_entry_friction": "전환 진입 마찰",
}


def label_ko(value: str) -> str:
    return PROBLEM_NAMES_KO.get(value) or INTERVENTION_TITLES_KO.get(value) or value.replace("_", " ")


def public_label_replacements() -> dict[str, str]:
    replacements: dict[str, str] = {}
    replacements.update(PERSONA_TRAIT_DESCRIPTIONS_KO)
    replacements.update(BEHAVIOR_CLUSTER_DESCRIPTIONS_KO)
    replacements.update(HTML_FEATURE_DESCRIPTIONS_KO)
    replacements.update(PROBLEM_NAMES_KO)
    replacements.update(INTERVENTION_TITLES_KO)
    return replacements


def sanitize_public_text(text: str) -> str:
    sanitized = text
    for raw_label, public_text in public_label_replacements().items():
        sanitized = sanitized.replace(raw_label, public_text)
    artifact_replacements = {
        "headings에": "현재 섹션 문구에서",
        "headings": "현재 섹션 문구",
        "ctas": "CTA 문구",
        "html_features": "섹션 구조 관찰",
        "behavior_clusters": "행동 신호",
        "persona_traits": "페르소나 특성",
    }
    for raw, replacement in artifact_replacements.items():
        sanitized = sanitized.replace(raw, replacement)
    non_korean_replacements = {
        "अस्पष्ट": "불명확",
        "отдель개": "개별",
    }
    for raw, replacement in non_korean_replacements.items():
        sanitized = sanitized.replace(raw, replacement)
    sanitized = re.sub(r"[\u0900-\u097F]+", "불명확", sanitized)
    return sanitized


def sanitize_generated_result(value: Any, *, in_trace: bool = False) -> Any:
    if isinstance(value, str):
        return value if in_trace else sanitize_public_text(value)
    if isinstance(value, list):
        return [sanitize_generated_result(item, in_trace=in_trace) for item in value]
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            next_in_trace = in_trace or key in {"_trace", "internal_trace_do_not_quote", "_generation_metadata"}
            if not next_in_trace and key in {"primary_problem_case", "intervention"}:
                continue
            sanitized[key] = sanitize_generated_result(item, in_trace=next_in_trace)
        return sanitized
    return value


def direction_from_expected_effect(effect: dict[str, str]) -> list[str]:
    mapping = {
        "increase": "증가",
        "decrease": "감소",
    }
    directions = []
    for key, value in effect.items():
        metric = EXPECTED_EFFECT_KO.get(key, key.replace("_", " "))
        directions.append(f"{metric} {mapping.get(value, value)}")
    return directions or ["해당 퍼널 단계에서 이탈 가능성 감소"]


def join_directions(values: list[str]) -> str:
    return ", ".join(value.rstrip(".") for value in values if value).strip() + "."


def generate_template(context: dict[str, Any]) -> dict[str, Any]:
    problem = context["primary_problem"]
    interventions = context["candidate_interventions"][:3]
    trace = context.get("internal_trace_do_not_quote", {})
    problem_case = trace.get("primary_problem_case") or "unknown_problem"
    problem_name = problem.get("problem_name_ko") or label_ko(problem_case)
    section = context.get("section_label") or "GENERIC"
    interpretation = context.get("user_facing_input_interpretation", {})
    signals = []
    for value in interpretation.get("persona", []):
        signals.append(f"페르소나: {value}")
    for value in interpretation.get("behavior", []):
        signals.append(f"행동 신호: {value}")
    for value in interpretation.get("section_observations", []):
        signals.append(f"섹션 관찰: {value}")

    recommendations = []
    for index, item in enumerate(interventions, start=1):
        title = item.get("intervention_title_ko") or "메시지와 행동 유도 개선"
        recommendations.append(
            {
                "rank": index,
                "title": title,
                "problem": item.get("problem_description")
                or problem.get("problem_description")
                or f"{problem_name} 문제가 의심됩니다.",
                "what_to_change": [
                    f"{section} 섹션의 핵심 문구를 '{problem_name}' 장벽을 낮추는 방향으로 수정합니다.",
                    f"검색된 개입 원리인 '{title}'에 맞춰 현재 섹션의 메시지와 행동 유도를 조정합니다.",
                ],
                "implementation_direction": (
                    "현재 섹션에서 사용자가 먼저 보는 문구와 행동 경로를 기준으로 수정합니다. "
                    "핵심 가치, 보조 근거, CTA가 한 화면 안에서 함께 읽히도록 작은 구조 변경을 우선 적용합니다."
                ),
                "copy_direction": [
                    "기능명보다 사용자가 얻는 결과와 상황을 먼저 드러내는 문장으로 바꿉니다.",
                    "누구에게 어떤 문제가 줄어드는지, 왜 지금 이 섹션에서 계속 읽어야 하는지를 짧게 보강합니다.",
                    "클릭 후 얻게 되는 결과나 다음 단계를 CTA 문구에 반영합니다.",
                ],
                "layout_direction": "핵심 가치 문구, 보조 근거, CTA가 한 화면 안에서 함께 보이도록 배치합니다.",
                "expected_behavior_change": join_directions(
                    direction_from_expected_effect(item.get("expected_effect_direction", {}))
                ),
                "risk_or_caveat": "페르소나의 실제 의도와 맞지 않으면 메시지가 일반적인 마케팅 문구처럼 보일 수 있습니다.",
            }
        )

    return {
        "summary": f"{section} 섹션에서 {problem_name} 문제가 우선 의심되며, 상위 개선 방향은 {recommendations[0]['title'] if recommendations else '메시지 명확화'}입니다.",
        "diagnosis": {
            "funnel_stage": section,
            "problem_name_ko": problem_name,
            "why_this_is_a_problem": problem.get("problem_description")
            or "현재 입력 섹션의 메시지/구조가 사용자의 다음 행동을 충분히 돕지 못할 가능성이 있습니다.",
            "supporting_signals": signals,
            "confidence_note": f"Problem retrieval score={problem.get('final_score')}. 이 값은 전환 개선 확률이 아니라 검색/재랭킹 점수입니다.",
        },
        "recommendations": recommendations,
        "priority_order": [item["title"] for item in recommendations],
        "validation_plan": [
            "동일 섹션에서 체류 시간, 스크롤 깊이, CTA 클릭률, 단계 이탈률을 전후 비교합니다.",
            "개선안은 한 번에 하나의 핵심 개입만 적용해 어떤 변화가 효과를 냈는지 분리합니다.",
        ],
        "_trace": {
            "primary_problem_case": problem_case,
            "intervention_ids": trace.get("intervention_ids", []),
        },
        "_generation_metadata": {"mode": "template"},
    }


def generate_recommendation(
    *,
    retrieval: dict[str, Any],
    section_html: str,
    top_n: int = 3,
    mode: str = "llm",
    provider: str = "openai",
    model: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    html_summary = extract_html_summary_from_html(section_html)
    context = compact_generation_context(
        retrieval=retrieval,
        html_summary=html_summary,
        top_n=top_n,
    )
    if mode == "llm":
        result = generate_with_llm(
            context=context,
            provider=provider,
            model=model,
            base_url=base_url,
        )
    elif mode == "template":
        result = generate_template(context)
    else:
        raise RuntimeError(f"Unsupported recommendation generation mode: {mode}")
    return sanitize_generated_result(result)


def render_markdown(result: dict[str, Any], context: dict[str, Any]) -> str:
    lines = [
        "# Landing Page Improvement Recommendation",
        "",
        f"## Summary",
        "",
        result.get("summary", ""),
        "",
        "## Diagnosis",
        "",
    ]
    diagnosis = result.get("diagnosis", {})
    for key in ("funnel_stage", "problem_name_ko", "why_this_is_a_problem"):
        if diagnosis.get(key):
            lines.append(f"- {key}: {diagnosis[key]}")
    if diagnosis.get("supporting_signals"):
        lines.append("- supporting_signals:")
        for signal in diagnosis["supporting_signals"]:
            lines.append(f"  - {signal}")
    if diagnosis.get("confidence_note"):
        lines.append(f"- confidence_note: {diagnosis['confidence_note']}")

    lines.extend(["", "## Recommendations", ""])
    for item in result.get("recommendations", []):
        lines.append(f"### {item.get('rank')}. {item.get('title')}")
        if item.get("problem"):
            lines.append(f"- problem: {item['problem']}")
        for change in item.get("what_to_change", []):
            lines.append(f"- change: {change}")
        if item.get("implementation_direction"):
            lines.append(f"- implementation_direction: {item['implementation_direction']}")
        copy_direction = item.get("copy_direction", [])
        if copy_direction:
            if isinstance(copy_direction, dict):
                for label, value in copy_direction.items():
                    lines.append(f"- copy_direction_{label}: {value}")
            else:
                lines.append("- copy_direction:")
                for value in copy_direction:
                    lines.append(f"  - {value}")
        layout_direction = item.get("layout_direction")
        if layout_direction:
            if isinstance(layout_direction, list):
                for layout in layout_direction:
                    lines.append(f"- layout_direction: {layout}")
            else:
                lines.append(f"- layout_direction: {layout_direction}")
        expected_behavior_change = item.get("expected_behavior_change")
        if expected_behavior_change:
            if isinstance(expected_behavior_change, list):
                for effect in expected_behavior_change:
                    lines.append(f"- expected_behavior_change: {effect}")
            else:
                lines.append(f"- expected_behavior_change: {expected_behavior_change}")
        if item.get("risk_or_caveat"):
            lines.append(f"- risk_or_caveat: {item['risk_or_caveat']}")
        lines.append("")

    if result.get("validation_plan"):
        lines.extend(["## Validation Plan", ""])
        for item in result["validation_plan"]:
            lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Input Snapshot",
            "",
            f"- section_label: {context.get('section_label')}",
        ]
    )
    interpretation = context.get("user_facing_input_interpretation", {})
    for label, values in (
        ("persona", interpretation.get("persona", [])),
        ("behavior", interpretation.get("behavior", [])),
        ("section_observations", interpretation.get("section_observations", [])),
    ):
        if values:
            lines.append(f"- {label}:")
            for value in values:
                lines.append(f"  - {value}")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-json", type=Path, default=DEFAULT_RETRIEVAL_JSON)
    parser.add_argument("--html-file", type=Path)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--mode", choices=("template", "llm"), default="template")
    parser.add_argument("--provider", choices=("openai", "groq", "ollama"), default="openai")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    args = parser.parse_args()

    retrieval = load_json(args.retrieval_json)
    html_file = args.html_file
    if html_file is None:
        raw_html_file = retrieval.get("input", {}).get("html_file")
        html_file = Path(raw_html_file) if raw_html_file else None
        if html_file and not html_file.is_absolute():
            html_file = PROJECT_ROOT / html_file

    html_summary = extract_html_summary(html_file)
    context = compact_generation_context(
        retrieval=retrieval,
        html_summary=html_summary,
        top_n=args.top_n,
    )
    if args.mode == "llm":
        result = generate_with_llm(
            context=context,
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
        )
    else:
        result = generate_template(context)
    result = sanitize_generated_result(result)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(result, context), encoding="utf-8")
    print(f"Wrote recommendation JSON: {args.output_json}")
    print(f"Wrote recommendation report: {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
