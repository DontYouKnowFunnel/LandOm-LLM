from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Dict, List, Optional

from bs4 import Tag

from .hash_id import generate_hash
from .segment_compress_denoised_soft import compress_segment_node_denoised_soft
from .selector_lookup import build_css_selector
from .spec import CompressionSpec
from .transform import attrs_to_string, clean_text, filter_attrs, iter_kept_nodes, parse_html_root


BLOCK_TAGS = {"section", "div", "article", "aside", "header", "footer", "nav", "main"}
SKIP_TAGS = {"script", "style", "noscript", "svg", "path"}
NOISE_TOKENS = {
    "cookie",
    "consent",
    "onetrust",
    "route-announcer",
    "veepn",
    "lock-screen",
    "banner",
}


def direct_tag_children(node: Tag) -> List[Tag]:
    return [child for child in node.children if isinstance(child, Tag) and child.name not in SKIP_TAGS]


def node_text(node: Tag, max_len: int = 500) -> str:
    return clean_text(node.get_text(" ", strip=True), max_len=max_len)


def node_content_score(node: Tag) -> int:
    text_len = len(node_text(node, max_len=800))
    children = direct_tag_children(node)
    block_children = sum(1 for child in children if child.name in BLOCK_TAGS)
    heading_count = len(node.find_all(["h1", "h2", "h3"]))
    action_count = len(node.find_all(["a", "button", "input"]))
    return text_len + (block_children * 120) + (heading_count * 60) + (action_count * 8)


def is_noise_node(node: Tag) -> bool:
    node_id = str(node.get("id", "")).lower()
    classes = " ".join(node.get("class", []) if isinstance(node.get("class"), list) else [str(node.get("class", ""))]).lower()
    name = node.name.lower()
    combined = " ".join([node_id, classes, name])

    if any(token in combined for token in NOISE_TOKENS):
        return True

    if "-" in name and len(node_text(node, max_len=120)) == 0:
        return True

    return False


def is_meaningful_block(node: Tag) -> bool:
    if node.name not in BLOCK_TAGS:
        return False
    if is_noise_node(node):
        return False

    text_len = len(node_text(node, max_len=240))
    child_count = len(direct_tag_children(node))
    heading_count = len(node.find_all(["h1", "h2", "h3"]))
    return text_len >= 20 or heading_count > 0 or child_count >= 2


def meaningful_block_children(node: Tag) -> List[Tag]:
    return [child for child in direct_tag_children(node) if is_meaningful_block(child)]


def should_unwrap_single_child(parent: Tag, child: Tag) -> bool:
    if child.name not in {"div", "main"}:
        return False
    if child.get("id") in {"root", "__next", "__nuxt"}:
        return True
    if len(meaningful_block_children(child)) >= 2:
        return True
    if len(direct_tag_children(parent)) == 1:
        return True
    return False


def select_primary_content_root(root: Tag) -> Tag:
    main = root.find("main")
    current = main if isinstance(main, Tag) else root

    if current is root:
        children = [child for child in direct_tag_children(root) if not is_noise_node(child)]
        if children:
            current = max(children, key=node_content_score)

    while True:
        children = meaningful_block_children(current)
        if len(children) == 1 and should_unwrap_single_child(current, children[0]):
            current = children[0]
            continue
        break

    return current


def expand_segment_children(node: Tag) -> List[Tag]:
    children = meaningful_block_children(node)
    if len(children) >= 2 and node.name in {"div", "main"} and len(node_text(node, max_len=120)) < 40:
        return children
    return [node]


def compress_segment_node(
    node: Tag,
    spec: CompressionSpec,
    *,
    max_lines: int = 36,
    text_max_len: int = 180,
) -> str:
    return compress_segment_node_denoised_soft(
        node,
        spec,
        max_lines=max_lines,
        text_max_len=text_max_len,
    )


def extract_page_segments(html: str, spec: Optional[CompressionSpec] = None) -> List[Dict[str, Any]]:
    active_spec = spec or CompressionSpec()
    root = parse_html_root(html, active_spec)
    root_selector_base = root

    node_path_map = {id(node): path for node, path, _ in iter_kept_nodes(root, active_spec)}
    content_root = select_primary_content_root(root)

    raw_segments: List[Tag] = []
    for child in meaningful_block_children(content_root):
        raw_segments.extend(expand_segment_children(child))

    segments: List[Dict[str, Any]] = []
    for index, node in enumerate(raw_segments, start=1):
        path = node_path_map.get(id(node))
        if path is None:
            continue

        segment_text = node_text(node, max_len=500)
        segments.append(
            {
                "id": generate_hash(path),
                "dom_id": str(node.get("id")) if node.get("id") is not None else None,
                "selector": build_css_selector(node, root_selector_base),
                "page_order": index,
                "section_index": index,
                "tag": node.name,
                "depth": len(path),
                "heading_count": len(node.find_all(["h1", "h2", "h3"])),
                "action_count": len(node.find_all(["a", "button", "input"])),
                "text": segment_text,
                "compressed_segment": compress_segment_node(node, active_spec),
            }
        )

    return segments


def segments_to_prompt_input(segments: List[Dict[str, Any]]) -> str:
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
