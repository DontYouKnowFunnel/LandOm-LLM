from typing import Any, Dict, List, Optional, Sequence

from bs4 import Tag

from .hash_id import generate_hash
from .spec import CompressionSpec
from .transform import iter_kept_nodes, parse_html_root


def build_css_selector(node: Tag, root: Tag) -> str:
    """대상 노드에서 루트까지 올라가며 안정적인 CSS selector를 생성합니다."""
    parts = []
    current = node

    while isinstance(current, Tag):
        if current is root.parent:
            break

        element_id = current.get("id")
        if element_id:
            safe_id = str(element_id).replace('"', '\\"')
            parts.append(f'{current.name}[id="{safe_id}"]')
            break

        position = 1
        for prev in current.previous_siblings:
            if isinstance(prev, Tag) and prev.name == current.name:
                position += 1
        parts.append(f"{current.name}:nth-of-type({position})")

        if current is root:
            break
        current = current.parent

    return " > ".join(reversed(parts))


def find_selector_by_generated_id(
    html: str,
    target_id: str,
    spec: CompressionSpec,
) -> Optional[str]:
    """생성 해시 ID에 해당하는 노드를 찾아 CSS selector를 반환합니다."""
    root = parse_html_root(html, spec)

    for node, path, _ in iter_kept_nodes(root, spec):
        if generate_hash(path) == target_id:
            return build_css_selector(node, root)

    return None


def find_selector_by_dom_id(
    html: str,
    target_id: str,
    spec: CompressionSpec,
) -> Optional[str]:
    """실제 DOM id 속성과 일치하는 노드를 찾아 CSS selector를 반환합니다."""
    root = parse_html_root(html, spec)

    for node, _, _ in iter_kept_nodes(root, spec):
        element_id = node.get("id")
        if element_id is not None and str(element_id) == target_id:
            return build_css_selector(node, root)

    return None


def find_selector_for_target_id(
    html: str,
    target_id: str,
    spec: CompressionSpec,
) -> Optional[str]:
    """생성 해시 ID를 우선 사용하고, 실패하면 실제 DOM id로 재시도합니다."""
    selector = find_selector_by_generated_id(html=html, target_id=target_id, spec=spec)
    if selector:
        return selector
    return find_selector_by_dom_id(html=html, target_id=target_id, spec=spec)


def normalize_funnel_name(name: Optional[str]) -> Optional[str]:
    """Funnel 라벨 표기 차이를 정규화합니다."""
    if not name:
        return name
    if name == "CTA_SECTION":
        return "CTA"
    return name


def map_funnel_items_to_selectors(
    html: str,
    funnel_items: Sequence[Dict[str, Any]],
    spec: CompressionSpec,
    id_key: str = "id",
) -> List[Dict[str, Any]]:
    """Funnel 항목 목록을 `funnel`과 `selector`만 포함한 형태로 매핑합니다."""
    output: List[Dict[str, Any]] = []

    for item in funnel_items:
        mapped: Dict[str, Any] = {
            "funnel": normalize_funnel_name(item.get("funnel")),
            "selector": None,
        }
        raw_id = item.get(id_key)
        target_id = str(raw_id) if raw_id is not None else ""

        if not target_id:
            output.append(mapped)
            continue

        selector = find_selector_for_target_id(
            html=html,
            target_id=target_id,
            spec=spec,
        )

        mapped["selector"] = selector
        output.append(mapped)

    return output
