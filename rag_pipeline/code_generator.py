"""Generate improved landing-page HTML/CSS from optimization recommendations."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from rag_pipeline.config import AI_MODELS
from rag_pipeline.retrieval_utils import OPENAI_TIMEOUT_SECONDS, PROJECT_ROOT


CODEGEN_SYSTEM_PROMPT = """
You are a senior frontend engineer specializing in conversion-oriented landing pages.

Your task is to generate one improved HTML/CSS result by applying the given landing-page
optimization recommendation or selected recommendation list to the provided funnel section code.

Hard rules:
- Preserve the existing design system first: typography scale, color palette, spacing rhythm,
  button/card/form styles, responsive behavior, class naming style, CSS variables, and component
  conventions must remain consistent with the original HTML and CSS.
- Apply only changes that are supported by the optimization recommendations. Do not perform a
  broad redesign, visual rebrand, or unrelated cleanup.
- Keep the original section structure as much as possible. Make targeted edits to hierarchy,
  copy, CTA placement, proof elements, comparison blocks, FAQ, or reassurance text only when the
  recommendation calls for it.
- Reuse existing classes, tokens, custom properties, media queries, and component patterns where
  possible. Add new classes only when necessary, and make them fit the existing naming style.
- Do not invent unverifiable claims, metrics, testimonials, customer names, certifications, or
  guarantees. If a recommendation asks for proof but no concrete proof is provided, add a modest
  placeholder-like structure that the product team can replace, without fabricating facts.
- Do not introduce external scripts, external stylesheets, CDN links, tracking snippets, web fonts,
  or image URLs.
- Produce accessible, semantic HTML. Preserve important attributes such as ids, data attributes,
  form names, aria labels, and analytics hooks unless a recommendation directly requires a change.
- Preserve the input scope. If the provided HTML is a section fragment, return a section fragment;
  if it is a full document, return a full document.
- The CSS must be a complete replacement for the provided CSS, not a partial diff. Keep unrelated
  existing rules unless they conflict with the recommendation.
- The returned HTML and CSS should be ready to replace the provided funnel section code.
- Return valid JSON only. Do not wrap the response in markdown.

Required JSON schema:
{
  "html": "improved HTML string",
  "css": "improved CSS string"
}
""".strip()


def generate_codegen(
    *,
    optimization_plan: Any,
    html: str,
    css: str,
    selected_recommendation_ranks: list[int] | None = None,
    model: str | None = None,
) -> dict[str, str]:
    client, resolved_model = resolve_client(model)
    response = client.chat.completions.create(
        model=resolved_model,
        messages=[
            {"role": "system", "content": CODEGEN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": codegen_user_prompt(
                    optimization_plan=optimization_plan,
                    html=html,
                    css=css,
                    selected_recommendation_ranks=selected_recommendation_ranks,
                ),
            },
        ],
        reasoning_effort="medium",
    )
    content = response.choices[0].message.content or ""
    parsed = json.loads(extract_json_text(content))
    return {
        "html": str(parsed.get("html", "")).strip(),
        "css": str(parsed.get("css", "")).strip(),
    }


def resolve_client(model: str | None) -> tuple[OpenAI, str]:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(
        api_key=api_key,
        timeout=OPENAI_TIMEOUT_SECONDS,
    ), (model or AI_MODELS.code_generation[1])


def codegen_user_prompt(
    *,
    optimization_plan: Any,
    html: str,
    css: str,
    selected_recommendation_ranks: list[int] | None,
) -> str:
    selection_text = "Apply all recommendations in the optimization plan."
    if selected_recommendation_ranks:
        selection_text = (
            "Apply only recommendations whose rank is included in this list: "
            f"{selected_recommendation_ranks}."
        )

    return f"""
Generate improved replacement HTML and CSS for this funnel section.

Selection rule:
{selection_text}

Optimization recommendation JSON or selected recommendation list:
{json.dumps(optimization_plan, ensure_ascii=False, indent=2)}

Existing funnel HTML:
```html
{html}
```

Existing funnel CSS:
```css
{css}
```
""".strip()


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
