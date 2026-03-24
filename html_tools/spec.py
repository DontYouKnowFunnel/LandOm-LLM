from dataclasses import dataclass, field
from typing import Optional, Set


DEFAULT_STRUCTURAL_TAGS = {
    "section", "main", "header", "footer", "nav", "article", "aside"
}

DEFAULT_TAILWIND_TAGS = {
    "absolute", "relative", "fixed", "sticky",
    "flex", "grid", "block", "inline", "hidden",
    "font-serif", "font-sans", "font-mono",
    "select-none", "select-auto",
    "whitespace-nowrap", "whitespace-pre",
    "transition", "transition-colors", "transition-all",
}

REMOVAL_STRUCTURAL_TAGS = {"script", "style", "svg", "path"}

@dataclass
class CompressionSpec:
    """HTML 압축/역추적 전 과정에 적용할 필터링 및 정규화 정책 모음입니다."""
    remove_empty_text: bool = False
    text_max_len: int = 1024

    # Node keep/remove policy (allowlist + denylist)
    allowed_tags: Optional[Set[str]] = None
    denied_tags: Set[str] = field(default_factory=set)

    # Node가 비어있어도 유지할 구조 태그
    structural_tags: Set[str] = field(default_factory=lambda: set(DEFAULT_STRUCTURAL_TAGS))

    # 사전 제거 태그 (DOM cleanup)
    pre_remove_tags: Set[str] = field(default_factory=lambda: set(REMOVAL_STRUCTURAL_TAGS))

    # Attribute keep/remove policy
    allowed_attrs: Optional[Set[str]] = field(default_factory=lambda: {"id", "class", "role"})
    denied_attrs: Set[str] = field(default_factory=set)
    allow_attr_prefixes: Set[str] = field(default_factory=set)

    # class 정리 정책
    remove_tailwind_classes: bool = True
    common_tailwind: Set[str] = field(default_factory=lambda: set(DEFAULT_TAILWIND_TAGS))
