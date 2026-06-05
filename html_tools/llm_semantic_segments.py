from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from bs4 import Tag
from openai import OpenAI

from .hash_id import generate_hash
from .segment_compress_denoised import compress_segment_node_denoised
from .selector_lookup import build_css_selector
from .segments import direct_tag_children, is_noise_node, node_text
from .spec import CompressionSpec
from .transform import iter_kept_nodes, parse_html_root


DEFAULT_PROVIDER = "groq"
DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
LLM_SEGMENTATION_MAX_RETRIES = 3
MAX_PROMPT_CANDIDATES = 120
PRESERVE_EARLY_CANDIDATES = 55

CANDIDATE_TAGS = {"main", "section", "article", "header", "aside", "form", "div"}
STRUCTURAL_HINT_TOKENS = {
    "hero",
    "section",
    "feature",
    "benefit",
    "value",
    "problem",
    "solution",
    "pricing",
    "price",
    "plan",
    "faq",
    "cta",
    "proof",
    "testimonial",
    "review",
    "case",
    "use",
    "target",
    "content",
    "container",
    "wrapper",
    "elementor",
    "e-con",
    "wixui-section",
    "column-strip",
}
SKIP_CLASS_TOKENS = {
    "elementor-widget-container",
    "e-con-inner",
    "e-grid",
    "e-n-tabs-content",
    "swiper",
    "carousel",
    "slider",
    "tab-content",
    "tabs-content",
    "modal",
    "popup",
    "overlay",
    "cookie",
    "nav",
    "gnb",
}
ANCESTOR_SKIP_TOKENS = {
    "e-grid",
    "e-n-tabs-content",
    "swiper",
    "carousel",
    "slider",
    "tab-content",
    "tabs-content",
    "tabpanel",
    "modal",
    "popup",
    "overlay",
    "cookie",
}


@dataclass(frozen=True)
class SegmentCandidate:
    node: Tag
    path: List[int]
    node_id: str
    selector: str
    page_order: int
    tag: str
    depth: int
    text: str
    heading_count: int
    action_count: int
    score: int
    kind: str
    compressed_preview: str


def _class_text(node: Tag) -> str:
    classes = node.get("class", [])
    if isinstance(classes, list):
        return " ".join(str(item) for item in classes).lower()
    return str(classes).lower()


def _identity_text(node: Tag) -> str:
    return " ".join([str(node.get("id", "")), _class_text(node), str(node.get("role", "")), node.name or ""]).lower()


def _has_any_token(text: str, tokens: Iterable[str]) -> bool:
    return any(token in text for token in tokens)


def _is_descendant_or_self(node: Tag, ancestor: Tag) -> bool:
    current: Any = node
    while isinstance(current, Tag):
        if current is ancestor:
            return True
        current = current.parent
    return False


def _is_wrapper_like(node: Tag, text_len: int) -> bool:
    identity = _identity_text(node)
    has_grid_child = any(
        isinstance(child, Tag) and _has_any_token(_identity_text(child), {"e-grid"})
        for child in direct_tag_children(node)
    )
    has_accordion_child = any(
        isinstance(child, Tag) and _has_any_token(_identity_text(child), {"accordion"})
        for child in node.find_all(list(CANDIDATE_TAGS))
    )
    text_preview = node_text(node, max_len=180).lower()
    if _has_any_token(identity, {"e-con"}) and has_grid_child:
        return False
    if has_accordion_child and (text_preview.startswith("faq") or text_preview.startswith("자주 묻는 질문")):
        return False

    children = [
        child
        for child in direct_tag_children(node)
        if child.name in CANDIDATE_TAGS and len(node_text(child, max_len=500)) >= 40
    ]
    descendant_blocks = [
        child
        for child in node.find_all(list(CANDIDATE_TAGS))
        if child is not node and len(node_text(child, max_len=500)) >= 80
    ]
    if len(children) >= 3 and node.name in {"main", "article", "div"} and text_len >= 1200:
        return True
    if len(descendant_blocks) >= 5 and node.name in {"main", "article", "div"} and text_len >= 1200:
        return True
    if len(children) == 1 and node.name == "div":
        child_identity = _identity_text(children[0])
        if _has_any_token(child_identity, {"e-grid", "accordion"}):
            return False
        child_text_len = len(node_text(children[0], max_len=2000))
        if child_text_len >= int(text_len * 0.85):
            return True
    return False


def _candidate_score(node: Tag, text_len: int, heading_count: int, action_count: int) -> int:
    identity = _identity_text(node)
    meaningful_children = [
        child
        for child in direct_tag_children(node)
        if child.name in CANDIDATE_TAGS and len(node_text(child, max_len=500)) >= 40
    ]
    score = min(text_len, 700)
    score += heading_count * 90
    score += action_count * 20
    score += len(meaningful_children) * 35
    if node.name in {"section", "article", "form", "header"}:
        score += 80
    if node.name == "section":
        score += 160
    if _has_any_token(identity, STRUCTURAL_HINT_TOKENS):
        score += 70
    return score


def _candidate_kind(node: Tag) -> str:
    identity = _identity_text(node)
    if node.name == "section":
        return "page_section"
    if _has_any_token(identity, {"wixui-section", "column-strip"}):
        return "page_section"
    if _has_any_token(identity, {"e-parent", "e-con-boxed"}):
        return "elementor_parent_section"
    if _has_any_token(identity, {"e-con"}):
        return "elementor_container"
    if node.name in {"form", "header", "aside"}:
        return f"{node.name}_block"
    return "generic_block"


def _has_section_ancestor(node: Tag) -> bool:
    current = node.parent
    while isinstance(current, Tag):
        if current.name == "section":
            return True
        current = current.parent
    return False


def _should_keep_candidate(node: Tag) -> bool:
    if node.name not in CANDIDATE_TAGS:
        return False
    if is_noise_node(node):
        return False

    identity = _identity_text(node)
    if _has_any_token(identity, SKIP_CLASS_TOKENS):
        return False
    current = node.parent
    while isinstance(current, Tag):
        if _has_any_token(_identity_text(current), ANCESTOR_SKIP_TOKENS):
            return False
        current = current.parent
    if node.find_parent("nav") is not None:
        return False

    text_len = len(node_text(node, max_len=1200))
    heading_count = len(node.find_all(["h1", "h2", "h3"]))
    action_count = len(node.find_all(["a", "button", "input"]))
    if text_len < 35 and heading_count == 0 and action_count == 0:
        return False
    if (
        node.name == "div"
        and text_len < 260
        and heading_count == 0
        and action_count == 0
        and not _has_any_token(identity, {"e-con", "hero", "section", "content"})
    ):
        return False
    if node.name == "div" and text_len < 220 and action_count == 0 and "%" in node_text(node, max_len=260):
        return False
    if node.name == "div" and text_len < 130 and action_count > 0 and not _has_any_token(identity, {"e-con"}):
        return False
    if _is_wrapper_like(node, text_len):
        return False
    return True


def _primary_llm_scope(root: Tag) -> Tag:
    main = root.find("main")
    if isinstance(main, Tag):
        return main
    return root


def _has_enough_real_sections(root: Tag, scope: Tag, spec: CompressionSpec) -> bool:
    count = 0
    for node, _, _ in iter_kept_nodes(root, spec):
        if not _is_descendant_or_self(node, scope):
            continue
        if node.name != "section" or _has_section_ancestor(node):
            continue
        if len(node_text(node, max_len=500)) < 35:
            continue
        if is_noise_node(node):
            continue
        count += 1
        if count >= 3:
            return True
    return False


def _should_keep_section_mode_candidate(node: Tag, scope: Tag) -> bool:
    if node.name != "section":
        return False
    if _has_section_ancestor(node):
        return False
    if not _is_descendant_or_self(node, scope):
        return False
    if is_noise_node(node):
        return False
    text_len = len(node_text(node, max_len=1000))
    if text_len < 35:
        return False
    return True


def _build_candidates(root: Tag, spec: CompressionSpec) -> List[SegmentCandidate]:
    scope = _primary_llm_scope(root)
    section_mode = _has_enough_real_sections(root, scope, spec)
    candidates: List[SegmentCandidate] = []
    for node, path, _ in iter_kept_nodes(root, spec):
        if section_mode:
            if not _should_keep_section_mode_candidate(node, scope):
                continue
        else:
            if not _is_descendant_or_self(node, scope):
                continue
            if not _should_keep_candidate(node):
                continue

        text = node_text(node, max_len=500)
        heading_count = len(node.find_all(["h1", "h2", "h3"]))
        action_count = len(node.find_all(["a", "button", "input"]))
        score = _candidate_score(node, len(node_text(node, max_len=1200)), heading_count, action_count)
        kind = _candidate_kind(node)
        candidates.append(
            SegmentCandidate(
                node=node,
                path=list(path),
                node_id=generate_hash(path),
                selector=build_css_selector(node, root),
                page_order=len(candidates) + 1,
                tag=node.name,
                depth=len(path),
                text=text,
                heading_count=heading_count,
                action_count=action_count,
                score=score,
                kind=kind,
                compressed_preview=compress_segment_node_denoised(
                    node,
                    spec,
                    max_lines=10,
                    text_max_len=140,
                ),
            )
        )

    if len(candidates) <= MAX_PROMPT_CANDIDATES:
        return candidates

    preserved = candidates[:PRESERVE_EARLY_CANDIDATES]
    preserved_ids = {candidate.node_id for candidate in preserved}
    remaining_slots = max(0, MAX_PROMPT_CANDIDATES - len(preserved))
    strongest_ids = preserved_ids | {
        candidate.node_id
        for candidate in sorted(
            [candidate for candidate in candidates if candidate.node_id not in preserved_ids],
            key=lambda item: item.score,
            reverse=True,
        )[:remaining_slots]
    }
    return [candidate for candidate in candidates if candidate.node_id in strongest_ids]


def _prompt_candidate(candidate: SegmentCandidate) -> Dict[str, Any]:
    return {
        "root_node_id": candidate.node_id,
        "order": candidate.page_order,
        "tag": candidate.tag,
        "candidate_kind": candidate.kind,
        "depth": candidate.depth,
        "heading_count": candidate.heading_count,
        "action_count": candidate.action_count,
        "text_excerpt": candidate.text,
        "html_preview": candidate.compressed_preview,
    }


def _build_prompt(candidates: List[SegmentCandidate]) -> str:
    candidate_json = json.dumps([_prompt_candidate(candidate) for candidate in candidates], ensure_ascii=False, indent=2)
    return f"""
너는 랜딩페이지 HTML을 의미 단위 세그먼트로 나누는 분석기다.
아래 후보 DOM 노드 중에서 랜딩페이지 흐름상 하나의 목적과 메시지를 대표하는 root 노드 ID만 선택하라.

선택 기준:
- 각 세그먼트는 hero, 문제 제기, 타깃, 기능, 가치 제안, 사용 사례, 사회적 증거, 가격, FAQ, CTA처럼 하나의 독립된 퍼널 역할을 가져야 한다.
- 후보에 page_section 또는 elementor_parent_section이 있으면, 단일 카드나 내부 텍스트 조각보다 그 후보를 우선 선택한다.
- 반복 카드 여러 개가 하나의 기능/사용사례/장점 묶음을 이룬다면 개별 카드가 아니라 그 묶음을 담는 상위 후보를 선택한다.
- 내부 후보가 여러 역할을 잘 나누고 있다면, 여러 역할을 한 번에 포함하는 전체 페이지 wrapper는 선택하지 않는다.
- 단일 카드, 버튼 묶음, 이미지 wrapper, 슬라이더 wrapper, 탭 내부 콘텐츠, 장식용 컨테이너, 중복되는 부모/자식 노드는 선택하지 않는다.
- 페이지 전체를 설명하기에 필요한 주요 세그먼트는 가능한 4~12개 사이로 선택한다.
- 후보에 없는 ID를 만들지 말고, 반드시 아래 root_node_id 중에서만 선택한다.
- 결과는 페이지에 등장하는 순서대로 정렬한다.

반환 형식:
{{"segments":[{{"root_node_id":"후보_ID","reason":"이 노드를 의미 단위 세그먼트로 선택한 짧은 이유"}}]}}

후보 DOM 노드:
{candidate_json}
""".strip()


def _extract_json_object(raw: str) -> Any:
    stripped = raw.strip()
    if not stripped:
        return {}

    fence_matches = re.findall(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    candidates = [candidate.strip() for candidate in fence_matches if candidate.strip()]

    first_object = stripped.find("{")
    last_object = stripped.rfind("}")
    if first_object != -1 and last_object != -1 and first_object < last_object:
        candidates.append(stripped[first_object : last_object + 1].strip())

    first_array = stripped.find("[")
    last_array = stripped.rfind("]")
    if first_array != -1 and last_array != -1 and first_array < last_array:
        candidates.append(stripped[first_array : last_array + 1].strip())

    for candidate in candidates or [stripped]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return {}


def _selected_ids_from_response(raw: str) -> List[str]:
    parsed = _extract_json_object(raw)
    if isinstance(parsed, dict):
        items = parsed.get("segments", [])
    else:
        items = parsed

    selected: List[str] = []
    if not isinstance(items, list):
        return selected

    for item in items:
        if not isinstance(item, dict):
            continue
        node_id = item.get("root_node_id") or item.get("id")
        if node_id is None:
            continue
        selected.append(str(node_id))
    return selected


def _node_contains(left: Tag, right: Tag) -> bool:
    if left is right:
        return True
    current: Any = right.parent
    while isinstance(current, Tag):
        if current is left:
            return True
        current = current.parent
    return False


def _dedupe_and_order(
    selected_ids: List[str],
    candidates: List[SegmentCandidate],
) -> List[SegmentCandidate]:
    by_id = {candidate.node_id: candidate for candidate in candidates}
    selected = [by_id[node_id] for node_id in selected_ids if node_id in by_id]

    unique_by_id: Dict[str, SegmentCandidate] = {}
    for candidate in selected:
        unique_by_id.setdefault(candidate.node_id, candidate)

    accepted: List[SegmentCandidate] = []
    for candidate in sorted(unique_by_id.values(), key=lambda item: item.depth):
        overlaps = any(
            _node_contains(candidate.node, other.node) or _node_contains(other.node, candidate.node)
            for other in accepted
        )
        if not overlaps:
            accepted.append(candidate)

    return sorted(accepted, key=lambda item: item.page_order)


def _segment_from_candidate(candidate: SegmentCandidate, index: int, spec: CompressionSpec) -> Dict[str, Any]:
    return {
        "id": candidate.node_id,
        "dom_id": str(candidate.node.get("id")) if candidate.node.get("id") is not None else None,
        "selector": candidate.selector,
        "page_order": index,
        "section_index": index,
        "tag": candidate.tag,
        "depth": candidate.depth,
        "heading_count": candidate.heading_count,
        "action_count": candidate.action_count,
        "text": candidate.text,
        "compressed_segment": compress_segment_node_denoised(
            candidate.node,
            spec,
            max_lines=36,
            text_max_len=180,
        ),
    }


def _client_for_provider(provider: str) -> OpenAI:
    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set.")
        return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        return OpenAI(api_key=api_key)

    raise RuntimeError(f"Unsupported provider for LLM segmentation: {provider}")


def _request_segmentation(client: OpenAI, model: str, prompt: str) -> str:
    last_error: Exception | None = None
    for attempt in range(1, LLM_SEGMENTATION_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_error = exc
            if attempt == LLM_SEGMENTATION_MAX_RETRIES:
                break
            time.sleep(1.5 * attempt)

    raise RuntimeError(f"LLM semantic segmentation request failed after retries: {last_error}") from last_error


def extract_page_segments_with_llm(
    html: str,
    spec: Optional[CompressionSpec] = None,
    *,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
) -> List[Dict[str, Any]]:
    active_spec = spec or CompressionSpec()
    root = parse_html_root(html, active_spec)
    candidates = _build_candidates(root, active_spec)
    if not candidates:
        return []

    prompt = _build_prompt(candidates)
    client = _client_for_provider(provider)
    raw = _request_segmentation(client, model, prompt)
    selected_ids = _selected_ids_from_response(raw)
    selected_candidates = _dedupe_and_order(selected_ids, candidates)

    return [
        _segment_from_candidate(candidate, index, active_spec)
        for index, candidate in enumerate(selected_candidates, start=1)
    ]
