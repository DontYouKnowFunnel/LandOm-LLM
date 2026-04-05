from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from bs4 import Tag

from .hash_id import generate_hash
from .selector_lookup import build_css_selector, normalize_funnel_name
from .segments import (
    compress_segment_node,
    direct_tag_children,
    expand_segment_children,
    is_noise_node,
    meaningful_block_children,
    node_text,
    select_primary_content_root,
)
from .spec import CompressionSpec
from .transform import iter_kept_nodes, parse_html_root


STRUCTURAL_TAGS = {"section", "div", "article", "aside", "header", "footer", "nav", "main"}
HEADING_TAGS = ["h1", "h2", "h3"]
ACTION_TAGS = ["a", "button", "input"]
MAX_LOCAL_CHILDREN = 15
MAX_REFINED_BASES_PER_PAGE = 2
TRUST_SIGNAL_RE = re.compile(
    r"\b(trusted by|organizations?|companies?|customers?|teams?|people|users?|brands?)\b",
    re.IGNORECASE,
)


def descendant_heading_count(node: Tag) -> int:
    return len(node.find_all(HEADING_TAGS))


def descendant_action_count(node: Tag) -> int:
    return len(node.find_all(ACTION_TAGS))


def descendant_h1_count(node: Tag) -> int:
    return len(node.find_all("h1"))


def descendant_image_count(node: Tag) -> int:
    return len(node.find_all("img"))


def has_descendant_id_fragment(node: Tag, fragments: Sequence[str]) -> bool:
    lowered_fragments = tuple(fragment.lower() for fragment in fragments)
    for tag in node.find_all(True):
        tag_id = tag.get("id")
        if not tag_id:
            continue
        lowered_id = str(tag_id).lower()
        if any(fragment in lowered_id for fragment in lowered_fragments):
            return True
    return False


def is_meaningful_local_subsection(node: Tag) -> bool:
    if is_noise_node(node):
        return False

    name = node.name.lower()
    text_len = len(node_text(node, max_len=280))
    child_count = len(direct_tag_children(node))
    heading_count = descendant_heading_count(node)
    action_count = descendant_action_count(node)

    if name not in STRUCTURAL_TAGS and "-" not in name and child_count < 2:
        return False

    return text_len >= 24 or heading_count > 0 or child_count >= 2 or action_count >= 2


def local_subsection_children(node: Tag) -> List[Tag]:
    return [child for child in direct_tag_children(node) if is_meaningful_local_subsection(child)]


def descend_single_child_wrappers(node: Tag) -> Tag:
    current = node
    while True:
        children = local_subsection_children(current)
        if len(children) != 1:
            return current
        child = children[0]
        if child.name not in {"div", "main", "article", "section", "aside"} and "-" not in child.name:
            return current
        current = child


def child_strength(node: Tag) -> int:
    name = node.name.lower()
    text_len = len(node_text(node, max_len=240))
    child_count = len(local_subsection_children(node))
    heading_count = descendant_heading_count(node)
    action_count = descendant_action_count(node)

    score = 0
    if name in {"section", "article", "aside"}:
        score += 3
    elif name in {"div", "main"}:
        score += 1
    if "-" in name:
        score += 2
    if heading_count > 0:
        score += 2
    if action_count > 0:
        score += 1
    if child_count > 0:
        score += 1
    if text_len >= 120:
        score += 2
    elif text_len >= 60:
        score += 1
    return score


def is_leafish_child(node: Tag) -> bool:
    name = node.name.lower()
    text_len = len(node_text(node, max_len=180))
    heading_count = descendant_heading_count(node)
    action_count = descendant_action_count(node)
    child_count = len(local_subsection_children(node))

    if name in {"section", "article", "aside"} or "-" in name:
        return False
    return heading_count == 0 and action_count <= 1 and child_count == 0 and text_len < 80


def is_hero_like_intro_child(node: Tag) -> bool:
    text = node_text(node, max_len=320)
    return descendant_h1_count(node) >= 1 and descendant_action_count(node) >= 1 and len(text) >= 80


def is_social_proof_like_row(node: Tag) -> bool:
    text = node_text(node, max_len=240)
    heading_count = descendant_heading_count(node)
    action_count = descendant_action_count(node)
    image_count = descendant_image_count(node)

    if heading_count > 0 or action_count > 1:
        return False

    return bool(TRUST_SIGNAL_RE.search(text)) or image_count >= 3


def select_targeted_children_for_refine(
    base_item: Dict[str, Any],
    candidate_root: Tag,
    children: Sequence[Tag],
) -> Optional[List[Tag]]:
    if base_item["page_order"] > 2:
        return None
    if len(children) != 2:
        return None
    if candidate_root.name not in {"div", "section"}:
        return None

    intro_child, trust_child = children
    if not is_hero_like_intro_child(intro_child):
        return None
    if not is_social_proof_like_row(trust_child):
        return None
    if not has_descendant_id_fragment(intro_child, ("sectionhero", "hero_")):
        return None
    if not has_descendant_id_fragment(trust_child, ("trusted", "trsuted", "socialproof", "proof")):
        return None
    return [intro_child, trust_child]


def ratio_against_second(value: int, second_value: int) -> float:
    if value <= 0:
        return 0.0
    if second_value <= 0:
        return float(value)
    return value / second_value


def metric_shares(base_items: Sequence[Dict[str, Any]], key: str) -> tuple[dict[str, float], dict[str, float]]:
    values = {item["id"]: int(item[key]) for item in base_items}
    total = sum(values.values())
    ordered = sorted(values.values(), reverse=True)
    second_value = ordered[1] if len(ordered) > 1 else 0
    shares = {
        item_id: (value / total if total > 0 else 0.0)
        for item_id, value in values.items()
    }
    ratios = {
        item_id: ratio_against_second(value, second_value)
        for item_id, value in values.items()
    }
    return shares, ratios


def dominance_profile(base_items: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    heading_shares, heading_ratios = metric_shares(base_items, "heading_count")
    action_shares, action_ratios = metric_shares(base_items, "action_count")
    text_shares, text_ratios = metric_shares(base_items, "text_len")

    profile: Dict[str, Dict[str, float]] = {}
    for item in base_items:
        item_id = item["id"]
        heading_count = item["heading_count"]
        action_count = item["action_count"]
        text_len = item["text_len"]

        heading_dominant = heading_count >= 3 and (
            heading_shares[item_id] >= 0.34 or heading_ratios[item_id] >= 1.8
        )
        action_dominant = action_count >= 2 and (
            action_shares[item_id] >= 0.34 or action_ratios[item_id] >= 1.8
        )
        text_dominant = text_len >= 180 and (
            text_shares[item_id] >= 0.34 or text_ratios[item_id] >= 1.7
        )

        dominance_score = int(heading_dominant) + int(action_dominant) + int(text_dominant)
        combined_share = heading_shares[item_id] + action_shares[item_id] + text_shares[item_id]

        profile[item_id] = {
            "heading_share": heading_shares[item_id],
            "action_share": action_shares[item_id],
            "text_share": text_shares[item_id],
            "heading_ratio": heading_ratios[item_id],
            "action_ratio": action_ratios[item_id],
            "text_ratio": text_ratios[item_id],
            "dominance_score": dominance_score,
            "combined_share": combined_share,
        }

    return profile


def select_dominant_base_ids(base_items: Sequence[Dict[str, Any]]) -> set[str]:
    if not base_items:
        return set()
    if len(base_items) == 1:
        return {base_items[0]["id"]}

    profile = dominance_profile(base_items)
    candidates = [
        (
            metrics["dominance_score"],
            metrics["combined_share"],
            item["id"],
        )
        for item in base_items
        for metrics in [profile[item["id"]]]
        if metrics["dominance_score"] >= 2
    ]
    candidates.sort(reverse=True)
    return {
        item_id
        for _, _, item_id in candidates[:MAX_REFINED_BASES_PER_PAGE]
    }


def should_refine_dominant_segment(
    node: Tag,
    children: Sequence[Tag],
    *,
    single_segment_page: bool,
) -> bool:
    if len(children) < 2 or len(children) > MAX_LOCAL_CHILDREN:
        return False

    strong_children = [child for child in children if child_strength(child) >= 4]
    if len(strong_children) < 2:
        return False

    heading_children = sum(1 for child in children if descendant_heading_count(child) > 0)
    section_like_children = sum(
        1
        for child in children
        if child.name in {"section", "article", "aside"} or "-" in child.name
    )
    leafish_children = sum(1 for child in children if is_leafish_child(child))

    if leafish_children > len(children) // 2:
        return False

    if single_segment_page:
        return heading_children >= 2 or section_like_children >= 2 or len(children) >= 4

    if len(children) <= 3 and section_like_children == 0 and heading_children < 3:
        return False
    if section_like_children >= 2:
        return True
    if heading_children >= 3:
        return True
    if len(children) >= 4 and heading_children >= 2:
        return True
    return False


def build_segment_item(
    node: Tag,
    path: Sequence[int],
    root_selector_base: Tag,
    spec: CompressionSpec,
    *,
    page_order: int,
    origin_id: str,
    origin_selector: str,
) -> Dict[str, Any]:
    text = node_text(node, max_len=500)
    return {
        "id": generate_hash(path),
        "dom_id": str(node.get("id")) if node.get("id") is not None else None,
        "selector": build_css_selector(node, root_selector_base),
        "page_order": page_order,
        "section_index": page_order,
        "tag": node.name,
        "path": list(path),
        "parent_path": list(path[:-1]),
        "depth": len(path),
        "text": text,
        "text_len": len(text),
        "heading_count": descendant_heading_count(node),
        "action_count": descendant_action_count(node),
        "origin_id": origin_id,
        "origin_selector": origin_selector,
        "compressed_segment": compress_segment_node(node, spec),
    }


def extract_page_segments_targeted_refine(html: str, spec: Optional[CompressionSpec] = None) -> List[Dict[str, Any]]:
    active_spec = spec or CompressionSpec()
    root = parse_html_root(html, active_spec)
    root_selector_base = root
    node_path_map = {id(node): path for node, path, _ in iter_kept_nodes(root, active_spec)}
    content_root = select_primary_content_root(root)

    base_nodes: List[Tag] = []
    for child in meaningful_block_children(content_root):
        base_nodes.extend(expand_segment_children(child))

    prepared_base_items: List[Dict[str, Any]] = []
    for index, base_node in enumerate(base_nodes, start=1):
        base_path = node_path_map.get(id(base_node))
        if base_path is None:
            continue
        base_selector = build_css_selector(base_node, root_selector_base)
        prepared_base_items.append(
            build_segment_item(
                base_node,
                base_path,
                root_selector_base,
                active_spec,
                page_order=index,
                origin_id=generate_hash(base_path),
                origin_selector=base_selector,
            )
        )

    dominant_base_ids = select_dominant_base_ids(prepared_base_items)
    single_segment_page = len(prepared_base_items) == 1

    segments: List[Dict[str, Any]] = []
    page_order = 1
    base_by_id = {item["id"]: item for item in prepared_base_items}

    for base_node in base_nodes:
        base_path = node_path_map.get(id(base_node))
        if base_path is None:
            continue

        base_id = generate_hash(base_path)
        base_item = base_by_id[base_id]
        candidate_root = descend_single_child_wrappers(base_node)
        candidate_children = local_subsection_children(candidate_root)

        targeted_children = select_targeted_children_for_refine(base_item, candidate_root, candidate_children)
        if targeted_children is not None:
            refined_items: List[Dict[str, Any]] = []
            for child in targeted_children:
                child_path = node_path_map.get(id(child))
                if child_path is None:
                    continue
                refined_items.append(
                    build_segment_item(
                        child,
                        child_path,
                        root_selector_base,
                        active_spec,
                        page_order=page_order,
                        origin_id=base_id,
                        origin_selector=base_item["selector"],
                    )
                )
                page_order += 1

            if len(refined_items) == 2:
                segments.extend(refined_items)
                continue

        if base_id in dominant_base_ids and should_refine_dominant_segment(
            candidate_root,
            candidate_children,
            single_segment_page=single_segment_page,
        ):
            refined_items: List[Dict[str, Any]] = []
            for child in candidate_children:
                child_path = node_path_map.get(id(child))
                if child_path is None:
                    continue
                refined_items.append(
                    build_segment_item(
                        child,
                        child_path,
                        root_selector_base,
                        active_spec,
                        page_order=page_order,
                        origin_id=base_id,
                        origin_selector=base_item["selector"],
                    )
                )
                page_order += 1

            if 2 <= len(refined_items) <= MAX_LOCAL_CHILDREN:
                segments.extend(refined_items)
                continue

        kept_item = dict(base_item)
        kept_item["page_order"] = page_order
        kept_item["section_index"] = page_order
        segments.append(kept_item)
        page_order += 1

    return segments


def segments_to_prompt_input_targeted_refine(segments: List[Dict[str, Any]]) -> str:
    prompt_items = []
    for segment in segments:
        prompt_items.append(
            {
                "id": segment["id"],
                "dom_id": segment.get("dom_id"),
                "page_order": segment["page_order"],
                "section_index": segment["section_index"],
                "tag": segment["tag"],
                "compressed_segment": segment["compressed_segment"],
            }
        )
    return json.dumps(prompt_items, ensure_ascii=False, indent=2)


def is_weak_segment(item: Dict[str, Any]) -> bool:
    return item["heading_count"] == 0 and item["action_count"] <= 1 and item["text_len"] < 180


def can_merge_adjacent_same_funnel(prev: Dict[str, Any], curr: Dict[str, Any]) -> bool:
    if prev["funnel"] != curr["funnel"]:
        return False
    if prev["page_order"] + 1 != curr["page_order"]:
        return False
    if prev["origin_id"] != curr["origin_id"]:
        return False
    if prev["heading_count"] > 0 and curr["heading_count"] > 0:
        return False
    if not (is_weak_segment(prev) or is_weak_segment(curr)):
        return False
    return True


def merge_adjacent_same_funnel_items(
    funnel_items: Sequence[Dict[str, Any]],
    segments: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    segment_by_id = {segment["id"]: segment for segment in segments}
    enriched: List[Dict[str, Any]] = []

    for item in funnel_items:
        segment = segment_by_id[item["id"]]
        enriched.append(
            {
                "id": item["id"],
                "funnel": normalize_funnel_name(item["funnel"]),
                "selector": segment["selector"],
                "page_order": segment["page_order"],
                "origin_id": segment["origin_id"],
                "origin_selector": segment["origin_selector"],
                "heading_count": segment["heading_count"],
                "action_count": segment["action_count"],
                "text_len": segment["text_len"],
            }
        )

    merged: List[Dict[str, Any]] = []
    for item in enriched:
        if merged and can_merge_adjacent_same_funnel(merged[-1], item):
            merged[-1]["page_order"] = item["page_order"]
            merged[-1]["heading_count"] = max(merged[-1]["heading_count"], item["heading_count"])
            merged[-1]["action_count"] = max(merged[-1]["action_count"], item["action_count"])
            merged[-1]["text_len"] += item["text_len"]
            merged[-1]["selector"] = merged[-1]["origin_selector"]
            continue
        merged.append(dict(item))

    return [{"funnel": item["funnel"], "selector": item["selector"]} for item in merged]
