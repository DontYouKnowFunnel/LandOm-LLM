"""Generate lightweight Tailwind wireframes for optimization recommendations."""

from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from rag_pipeline.config import AI_MODELS, REASONING_EFFORTS
from rag_pipeline.html_preprocessor import preprocess_html_for_llm
from rag_pipeline.langsmith_tracking import traced_chat_completion
from rag_pipeline.retrieval_utils import llm_client


WIREFRAME_SYSTEM_PROMPT = """
You are a senior frontend engineer creating low-fidelity landing-page wireframes.

Your task is to generate a lightweight HTML + TailwindCSS wireframe for each provided
optimization recommendation, using the original funnel section HTML as context.

Hard rules:
- Generate one wireframe per recommendation and keep the original recommendation ranks.
- Use only semantic HTML with Tailwind utility classes. Do not include CSS blocks, external
  scripts, CDN links, images, web fonts, JavaScript, or tracking snippets.
- Keep the result intentionally simple: structure, hierarchy, copy placement, CTA placement,
  proof/reassurance placeholders, and responsive layout only. Do not create a polished redesign.
- Apply only the specific recommendation. Do not merge unrelated recommendations into a single
  wireframe.
- Match the dominant user-facing language of the original HTML. If the original HTML language is
  unclear, use the recommendation's language. Preserve brand names, product names, URLs, ids,
  data attributes, and technical terms.
- Do not invent unverifiable claims, metrics, testimonials, customer names, certifications, or
  guarantees. If proof is recommended but not present in the original HTML, use modest placeholder
  copy such as "고객 사례 입력", "성과 수치 입력", or the same idea in the original language.
- Preserve the input scope. If the provided HTML is a section fragment, return a section fragment.
  If it is a full document, return a full document.
- Preserve important ids, form attributes, aria labels, and data attributes when they are visible
  in the compact source and still relevant to the wireframe.
- Return valid JSON only. Do not wrap the response in markdown.

Required JSON schema:
{
  "wireframes": [
    {
      "rank": 1,
      "wireframe": "HTML string with Tailwind utility classes"
    }
  ]
}
""".strip()


def generate_wireframes(
    *,
    recommendations: list[dict[str, Any]],
    section_html: str,
    model: str | None = None,
    client: OpenAI | None = None,
) -> dict[int, str]:
    """Return a rank -> wireframe mapping for the provided recommendations."""
    if not recommendations:
        return {}

    resolved_client, provider, resolved_model = resolve_client(model, client)
    request_kwargs: dict[str, Any] = {
        "model": resolved_model,
        "messages": [
            {"role": "system", "content": WIREFRAME_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": wireframe_user_prompt(
                    recommendations=recommendations,
                    section_html=section_html,
                ),
            },
        ],
    }
    if provider == "openai":
        request_kwargs["reasoning_effort"] = REASONING_EFFORTS["wireframe"]
    response = traced_chat_completion(
        client=resolved_client,
        request_kwargs=request_kwargs,
        provider=provider,
        model=resolved_model,
        workflow="rag.optimization",
        stage="rag.wireframe.generate",
        prompt_name="rag.wireframe.v1",
        metadata={"recommendation_count": len(recommendations)},
    )
    content = response.choices[0].message.content or ""
    parsed = json.loads(extract_json_text(content))
    return parse_wireframe_map(parsed)


def attach_wireframes(
    *,
    recommendations: list[dict[str, Any]],
    section_html: str,
    model: str | None = None,
    client: OpenAI | None = None,
) -> list[dict[str, Any]]:
    """Attach generated wireframes while preserving recommendation content and order."""
    wireframes = generate_wireframes(
        recommendations=recommendations,
        section_html=section_html,
        model=model,
        client=client,
    )
    return [
        recommendation_with_wireframe(
            recommendation,
            wireframes.get(normalize_rank(recommendation.get("rank")), ""),
        )
        for recommendation in recommendations
    ]


def recommendation_with_wireframe(
    recommendation: dict[str, Any],
    wireframe: str,
) -> dict[str, Any]:
    """Insert the wireframe field near the title for stable API output shape."""
    updated: dict[str, Any] = {}
    inserted = False
    for key, value in recommendation.items():
        updated[key] = value
        if key == "title":
            updated["wireframe"] = wireframe
            inserted = True
    if not inserted:
        updated["wireframe"] = wireframe
    return updated


def attach_empty_wireframes(recommendations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [recommendation_with_wireframe(recommendation, "") for recommendation in recommendations]


def resolve_client(model: str | None, client: OpenAI | None) -> tuple[OpenAI, str, str]:
    provider, configured_model = AI_MODELS.code_generation
    if client is not None:
        return client, provider, (model or configured_model)

    return llm_client(provider), provider, (model or configured_model)


def wireframe_user_prompt(
    *,
    recommendations: list[dict[str, Any]],
    section_html: str,
) -> str:
    compact = preprocess_html_for_llm(section_html, max_chars=12000)
    return f"""
Generate Tailwind HTML wireframes for the following recommendations.

Original funnel section compact HTML:
```html
{compact.compact_html}
```

Optimization recommendations:
{json.dumps(recommendations, ensure_ascii=False, indent=2)}
""".strip()


def parse_wireframe_map(parsed: dict[str, Any]) -> dict[int, str]:
    wireframes = parsed.get("wireframes")
    if not isinstance(wireframes, list):
        raise RuntimeError("wireframe generation response must contain a wireframes list")

    result: dict[int, str] = {}
    for item in wireframes:
        if not isinstance(item, dict):
            continue
        rank = normalize_rank(item.get("rank"))
        wireframe = str(item.get("wireframe", "")).strip()
        if rank is not None and wireframe:
            result[rank] = wireframe
    return result


def normalize_rank(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_json_text(raw: str) -> str:
    stripped = raw.strip()
    fence_matches = re.findall(
        r"```(?:json)?\s*(.*?)```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
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
