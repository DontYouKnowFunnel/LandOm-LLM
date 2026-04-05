from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

from bs4 import Tag

from .spec import CompressionSpec
from .transform import attrs_to_string, clean_text, filter_attrs, iter_kept_nodes


SKIP_TAGS = {"br", "hr", "source", "track", "meta", "link"}
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
ACTION_TAGS = {"a", "button", "input", "select", "textarea", "form", "label"}
TEXT_TAGS = {"p", "li", "blockquote", "dd", "dt", "strong", "em", "span"}
STRUCTURAL_TAGS = {"section", "article", "aside", "main", "header"}
NOISE_TEXT_PATTERNS = [
    re.compile(r"^\[\s*\]$"),
    re.compile(r"^\$\s*/\$\s*$"),
    re.compile(r"^<style>", re.IGNORECASE),
    re.compile(r"upload in progress", re.IGNORECASE),
    re.compile(r"currently loaded video", re.IGNORECASE),
]
GENERIC_ACTION_TEXTS = {
    "learn more",
    "get started",
    "start free",
    "start for free",
    "read more",
    "discover",
    "explore",
    "contact sales",
    "book demo",
    "watch demo",
}


def normalize_signature(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"\s+", " ", lowered).strip()
    lowered = re.sub(r"[^a-z0-9가-힣 ]+", "", lowered)
    return lowered[:120]


def is_noise_text(text: str) -> bool:
    normalized = clean_text(text, max_len=200)
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in NOISE_TEXT_PATTERNS)


def classify_candidate(node: Tag, text: str, attrs: Dict, path: Sequence[int]) -> Optional[dict]:
    tag = node.name.lower()
    role = str(attrs.get("role", "")).lower()

    if tag in SKIP_TAGS:
        return None
    if is_noise_text(text):
        return None

    has_text = bool(text)
    has_attrs = bool(attrs)

    if role == "tab":
        return {"category": "tab", "score": 65 + min(len(text), 40) // 8}

    if tag in HEADING_TAGS:
        base = 120 if tag == "h1" else 110 if tag == "h2" else 95
        return {"category": "heading", "score": base + min(len(text), 140) // 12}

    if tag in ACTION_TAGS:
        score = 82 + min(len(text), 120) // 16
        if normalize_signature(text) in GENERIC_ACTION_TEXTS:
            score -= 4
        return {"category": "action", "score": score}

    if tag in TEXT_TAGS and has_text:
        score = 72 + min(len(text), 220) // 10
        return {"category": "text", "score": score}

    if tag in STRUCTURAL_TAGS:
        score = 50 if has_text else 32
        return {"category": "structural", "score": score - min(len(path), 6)}

    if tag == "div":
        if has_text:
            score = 50 + min(len(text), 180) // 14
            return {"category": "text", "score": score - min(len(path), 6)}
        if has_attrs or len(path) <= 2:
            return {"category": "structural", "score": 24 - min(len(path), 6)}
        return None

    if has_text:
        return {"category": "other", "score": 36 + min(len(text), 180) // 18}
    if has_attrs and len(path) <= 3:
        return {"category": "structural", "score": 18 - min(len(path), 6)}
    return None


def category_limit(category: str, max_lines: int) -> int:
    if category == "heading":
        return min(10, max(4, max_lines // 3))
    if category == "action":
        return min(10, max(4, max_lines // 3))
    if category == "text":
        return min(22, max(10, (max_lines * 3) // 5))
    if category == "structural":
        return min(10, max(4, max_lines // 4))
    if category == "tab":
        return 4
    return min(8, max(3, max_lines // 5))


def line_signature(tag: str, text: str) -> str:
    if not text:
        return f"{tag}:__empty__"
    return f"{tag}:{normalize_signature(text)}"


def repeat_limit(category: str, signature: str) -> int:
    generic = signature.split(":", 1)[-1]
    if category == "tab":
        return 1
    if category == "action" and generic in GENERIC_ACTION_TEXTS:
        return 2
    return 3


def compress_segment_node_denoised_soft(
    node: Tag,
    spec: CompressionSpec,
    *,
    max_lines: int = 36,
    text_max_len: int = 180,
) -> str:
    segment_spec = replace(spec, text_max_len=text_max_len)
    candidates: List[dict] = []

    for order, (subnode, path, text) in enumerate(iter_kept_nodes(node, segment_spec)):
        attrs = filter_attrs(subnode.attrs, segment_spec)
        info = classify_candidate(subnode, text, attrs, path)
        if info is None:
            continue

        indent = "  " * min(len(path), 5)
        tag_str = f"{indent}<{subnode.name}"
        attr_str = attrs_to_string(attrs)
        if attr_str:
            tag_str += f" {attr_str}"
        tag_str += ">"

        line = tag_str if not text else f'{tag_str} text="{clean_text(text, max_len=text_max_len)}"'
        candidates.append(
            {
                "order": order,
                "path": list(path),
                "tag": subnode.name.lower(),
                "text": clean_text(text, max_len=text_max_len),
                "line": line,
                "category": info["category"],
                "score": info["score"],
            }
        )

    if not candidates:
        return ""

    selected_orders = set()
    counts_by_category: Counter[str] = Counter()
    counts_by_signature: Counter[str] = Counter()

    for item in candidates[:2]:
        selected_orders.add(item["order"])
        counts_by_category[item["category"]] += 1
        counts_by_signature[line_signature(item["tag"], item["text"])] += 1

    ranked = sorted(candidates[2:], key=lambda item: (-item["score"], item["order"]))
    omitted = 0

    for item in ranked:
        if len(selected_orders) >= max_lines:
            break

        category = item["category"]
        if counts_by_category[category] >= category_limit(category, max_lines):
            continue

        signature = line_signature(item["tag"], item["text"])
        if counts_by_signature[signature] >= repeat_limit(category, signature):
            omitted += 1
            continue

        selected_orders.add(item["order"])
        counts_by_category[category] += 1
        counts_by_signature[signature] += 1

    selected = [item for item in candidates if item["order"] in selected_orders]
    lines = [item["line"] for item in selected[:max_lines]]

    if len(selected) < len(candidates) or omitted > 0:
        lines.append("...")

    return "\n".join(lines)
