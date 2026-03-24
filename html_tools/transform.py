import re
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from bs4 import BeautifulSoup, NavigableString, Tag

from .spec import CompressionSpec


def clean_text(text: str, max_len: int = 1024) -> str:
    """줄바꿈/탭/연속 공백을 한 칸으로 압축하고 최대 길이로 잘라 반환합니다."""
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized[:max_len]


def is_name_allowed(
    name: str,
    allowlist: Optional[Iterable[str]] = None,
    denylist: Optional[Iterable[str]] = None,
) -> bool:
    """태그/속성 이름이 allowlist/denylist 정책을 통과하는지 판별합니다."""
    if allowlist is not None and name not in allowlist:
        return False
    if denylist is not None and name in denylist:
        return False
    return True


def is_tailwind_class(cls: str, common_tailwind: Iterable[str]) -> bool:
    """클래스 문자열이 Tailwind 유틸리티로 간주되는지 판별합니다."""
    return ":" in cls or "-" in cls or cls in common_tailwind


def filter_class_value(value, spec: CompressionSpec):
    """class 값에서 Tailwind 관련 클래스를 제거하고 결과를 반환합니다."""
    if not spec.remove_tailwind_classes:
        return value

    if isinstance(value, list):
        cleaned = [c for c in value if not is_tailwind_class(c, spec.common_tailwind)]
        return cleaned if cleaned else None

    if isinstance(value, str):
        cleaned = [c for c in value.split() if not is_tailwind_class(c, spec.common_tailwind)]
        if not cleaned:
            return None
        return cleaned

    return value


def is_attribute_allowed(attr_name: str, spec: CompressionSpec) -> bool:
    """속성명이 스펙의 allow/deny/prefix 규칙에 따라 허용되는지 판별합니다."""
    if attr_name in spec.denied_attrs:
        return False

    if spec.allowed_attrs is not None and attr_name in spec.allowed_attrs:
        return True

    if any(attr_name.startswith(prefix) for prefix in spec.allow_attr_prefixes):
        return True

    return spec.allowed_attrs is None


def filter_attrs(attrs: Dict, spec: CompressionSpec) -> Dict:
    """노드 속성 딕셔너리를 정책에 맞게 필터링해 반환합니다."""
    filtered = {}

    for key, value in attrs.items():
        if not is_attribute_allowed(key, spec):
            continue

        if key == "class":
            class_value = filter_class_value(value, spec)
            if class_value:
                filtered[key] = class_value
            continue

        filtered[key] = value

    return filtered


def attrs_to_string(attrs: Dict) -> str:
    """속성 딕셔너리를 `k=\"v\"` 형식의 문자열로 직렬화합니다."""
    parts = []
    for k, v in attrs.items():
        if isinstance(v, list):
            v = " ".join(v)
        parts.append(f'{k}="{v}"')
    return " ".join(parts)


def get_direct_text(node: Tag, spec: CompressionSpec) -> str:
    """현재 노드의 직계 텍스트만 추출해 정규화한 문자열을 반환합니다."""
    texts: List[str] = []
    for child in node.children:
        if isinstance(child, NavigableString):
            t = str(child).strip()
            if t:
                texts.append(t)
    return clean_text(" ".join(texts), max_len=spec.text_max_len)


def should_keep_node(node: Tag, text: str, spec: CompressionSpec) -> bool:
    """태그 정책/빈 텍스트 정책을 적용해 노드 유지 여부를 결정합니다."""
    if not is_name_allowed(node.name, spec.allowed_tags, spec.denied_tags):
        return False

    if spec.remove_empty_text and not text and node.name not in spec.structural_tags:
        return False

    return True


def preprocess_soup(soup: BeautifulSoup, spec: CompressionSpec) -> None:
    """압축 전에 제거 대상 태그(script/style 등)를 DOM에서 삭제합니다."""
    for tag_name in spec.pre_remove_tags:
        for tag in soup.find_all(tag_name):
            tag.decompose()


def parse_html_root(html: str, spec: CompressionSpec) -> Tag:
    """HTML 문자열을 파싱하고 전처리한 뒤 순회의 시작 루트(body 또는 soup)를 반환합니다."""
    soup = BeautifulSoup(html, "html.parser")
    preprocess_soup(soup, spec)
    return soup.body if soup.body else soup


def iter_kept_nodes(root: Tag, spec: CompressionSpec) -> Iterator[Tuple[Tag, List[int], str]]:
    """정책을 통과한 노드를 `(node, path, [text])` 형태로 순회하며 생성합니다."""
    # node: 실제 Element 객체
    # path: 상대적인 경로(좌표)
    # text: Element가 가지고 있는 Text (Optional)
    def traverse(node: Tag, path: List[int]):
        if not isinstance(node, Tag):
            return

        text = get_direct_text(node, spec)
        if not should_keep_node(node, text, spec):
            return

        yield node, path, text

        children = [c for c in node.children if isinstance(c, Tag)]
        for idx, child in enumerate(children):
            yield from traverse(child, path + [idx])

    yield from traverse(root, [])
