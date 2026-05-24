"""Compact HTML preprocessing for LLM-based landing-page feature extraction.

This module does not infer landing-page problems. It only removes noisy code
and preserves observable section structure in a shorter HTML-like form.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import escape
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from bs4 import BeautifulSoup, Comment, NavigableString, Tag


DROP_TAGS = {
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "canvas",
    "meta",
    "link",
}

VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}

UNWRAP_TAGS = {"html", "head", "body"}

STRUCTURAL_TAGS = {
    "section",
    "main",
    "header",
    "footer",
    "nav",
    "article",
    "aside",
    "div",
    "ul",
    "ol",
    "li",
    "form",
    "fieldset",
    "legend",
    "details",
    "summary",
    "dialog",
}

CONTENT_TAGS = {
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "span",
    "strong",
    "em",
    "b",
    "i",
    "small",
    "blockquote",
    "q",
    "cite",
    "label",
    "figcaption",
}

ACTION_TAGS = {"a", "button", "input", "textarea", "select", "option"}
MEDIA_TAGS = {"img", "picture", "video", "source", "iframe"}
KEEP_TAGS = STRUCTURAL_TAGS | CONTENT_TAGS | ACTION_TAGS | MEDIA_TAGS

BASE_ALLOWED_ATTRS = {
    "id",
    "class",
    "role",
    "aria-label",
    "aria-expanded",
    "aria-controls",
    "aria-haspopup",
    "aria-current",
    "aria-describedby",
    "aria-labelledby",
    "href",
    "type",
    "name",
    "placeholder",
    "required",
    "checked",
    "selected",
    "disabled",
    "alt",
    "title",
    "for",
    "value",
    "style",
    "width",
    "height",
    "poster",
    "src",
}

SEMANTIC_TOKEN_RE = re.compile(
    r"("
    r"hero|headline|title|subtitle|subhead|copy|content|container|wrapper|"
    r"cta|button|btn|primary|secondary|signup|demo|trial|contact|submit|"
    r"price|pricing|plan|tier|billing|checkout|"
    r"testimonial|review|quote|customer|client|logo|trust|proof|social|"
    r"faq|question|answer|accordion|collapse|tab|tabs|modal|dialog|popup|"
    r"carousel|slider|swiper|dropdown|menu|"
    r"card|grid|list|row|column|col|feature|benefit|value|usecase|use-case|"
    r"form|field|input|email|privacy|security|guarantee|refund|"
    r"product|preview|screenshot|dashboard|image|video|media|"
    r"sticky|fixed|floating|hidden|visible"
    r")",
    re.IGNORECASE,
)

LAYOUT_OR_TYPE_TOKEN_RE = re.compile(
    r"^(grid|flex|inline-flex|block|inline-block|hidden|sticky|fixed|relative|absolute|"
    r"grid-cols-\d+|col-span-\d+|row-span-\d+|flex-col|flex-row|items-[a-z-]+|"
    r"justify-[a-z-]+|gap-\d+|space-[xy]-\d+|text-(xs|sm|base|lg|xl|[2-9]xl)|"
    r"font-(thin|light|normal|medium|semibold|bold|extrabold|black)|"
    r"leading-[a-z0-9-]+|tracking-[a-z0-9-]+|"
    r"w-\d+|h-\d+|max-w-[a-z0-9-]+|min-h-[a-z0-9-]+)$",
    re.IGNORECASE,
)

HASHLIKE_TOKEN_RE = re.compile(
    r"^(css|sc|jsx|framer|chakra|mantine|mui|emotion|astro|svelte|"
    r"_[a-z0-9]+|[a-z]{1,3}-[a-z0-9]{5,}|[a-z0-9_-]{12,})$",
    re.IGNORECASE,
)

TRACKING_ATTR_RE = re.compile(
    r"(analytics|tracking|track|gtm|ga-|google|facebook|fbq|pixel|segment|"
    r"amplitude|mixpanel|hotjar|clarity|sentry|posthog|utm)",
    re.IGNORECASE,
)

INTERACTION_ATTR_RE = re.compile(
    r"(modal|dialog|tab|accordion|collapse|carousel|slider|swiper|dropdown|"
    r"menu|popover|tooltip|cta|button|form)",
    re.IGNORECASE,
)

STYLE_PROPS_TO_KEEP = {
    "display",
    "position",
    "font-size",
    "font-weight",
    "text-transform",
    "visibility",
    "opacity",
    "grid-template-columns",
    "grid-template-rows",
    "flex-direction",
    "align-items",
    "justify-content",
}


@dataclass(frozen=True)
class PreprocessResult:
    compact_html: str
    original_chars: int
    compact_chars: int
    truncated: bool


def clean_text(text: str, *, max_len: int) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) > max_len:
        return normalized[: max_len - 1].rstrip() + "..."
    return normalized


def is_hashlike_token(token: str) -> bool:
    if HASHLIKE_TOKEN_RE.match(token):
        return True
    if re.search(r"[a-z]{2,}-[a-z0-9]{6,}", token, flags=re.IGNORECASE):
        return True
    return False


def filter_token_list(values: Any) -> list[str]:
    if isinstance(values, str):
        tokens = values.split()
    elif isinstance(values, list):
        tokens = [str(value) for value in values]
    else:
        tokens = [str(values)]

    kept: list[str] = []
    seen: set[str] = set()
    for raw in tokens:
        token = raw.strip()
        if not token or len(token) > 64:
            continue
        lowered = token.lower()
        is_semantic = bool(SEMANTIC_TOKEN_RE.search(lowered))
        is_layout_or_type = bool(LAYOUT_OR_TYPE_TOKEN_RE.match(lowered))
        if not is_semantic and is_hashlike_token(lowered):
            continue
        if not (is_semantic or is_layout_or_type):
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        kept.append(token)
    return kept[:12]


def filter_id(value: Any) -> str | None:
    token = str(value or "").strip()
    if not token or len(token) > 64:
        return None
    lowered = token.lower()
    if SEMANTIC_TOKEN_RE.search(lowered):
        return token
    if is_hashlike_token(lowered):
        return None
    return None


def sanitize_url(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.startswith(("data:", "blob:", "javascript:")):
        return None
    if len(raw) > 220:
        raw = raw[:220]

    parsed = urlsplit(raw)
    if parsed.scheme in {"http", "https"}:
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path[:120], "", ""))
    if raw.startswith("#"):
        return raw[:80]
    if parsed.path:
        return urlunsplit(("", "", parsed.path[:140], "", ""))
    return raw[:120]


def filter_inline_style(value: Any) -> str | None:
    raw = str(value or "")
    kept: list[str] = []
    for declaration in raw.split(";"):
        if ":" not in declaration:
            continue
        name, css_value = declaration.split(":", 1)
        name = name.strip().lower()
        css_value = css_value.strip()
        if name not in STYLE_PROPS_TO_KEEP or not css_value or len(css_value) > 80:
            continue
        if "url(" in css_value.lower():
            continue
        kept.append(f"{name}: {css_value}")
    return "; ".join(kept[:8]) if kept else None


def should_keep_data_attr(name: str, value: Any) -> bool:
    if TRACKING_ATTR_RE.search(name):
        return False
    if INTERACTION_ATTR_RE.search(name):
        return True
    value_text = str(value or "")
    if len(value_text) > 120:
        return False
    return bool(INTERACTION_ATTR_RE.search(value_text))


def filter_attrs(tag: Tag) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for name, value in tag.attrs.items():
        attr = str(name).strip().lower()
        if not attr or attr.startswith("on"):
            continue
        if TRACKING_ATTR_RE.search(attr):
            continue
        if attr.startswith("data-") and not should_keep_data_attr(attr, value):
            continue
        if attr not in BASE_ALLOWED_ATTRS and not attr.startswith("data-"):
            continue

        if attr == "class":
            classes = filter_token_list(value)
            if classes:
                attrs[attr] = " ".join(classes)
            continue

        if attr == "id":
            filtered_id = filter_id(value)
            if filtered_id:
                attrs[attr] = filtered_id
            continue

        if attr == "style":
            filtered_style = filter_inline_style(value)
            if filtered_style:
                attrs[attr] = filtered_style
            continue

        if attr in {"href", "src", "poster"}:
            sanitized = sanitize_url(value)
            if sanitized:
                attrs[attr] = sanitized
            continue

        if attr == "value" and tag.name not in {"button", "option"}:
            input_type = str(tag.get("type", "")).lower()
            if input_type not in {"button", "submit", "reset"}:
                continue

        if isinstance(value, list):
            value_text = " ".join(str(item) for item in value)
        elif value is True or value == "":
            attrs[attr] = True
            continue
        else:
            value_text = str(value)

        value_text = clean_text(value_text, max_len=160)
        if value_text:
            attrs[attr] = value_text
    return attrs


def attrs_to_html(attrs: dict[str, Any]) -> str:
    parts: list[str] = []
    for name, value in attrs.items():
        if value is True:
            parts.append(name)
        else:
            parts.append(f'{name}="{escape(str(value), quote=True)}"')
    return (" " + " ".join(parts)) if parts else ""


def preprocess_soup(soup: BeautifulSoup) -> None:
    for comment in soup.find_all(string=lambda value: isinstance(value, Comment)):
        comment.extract()
    for tag in soup.find_all(DROP_TAGS):
        tag.decompose()
    for tag in soup.find_all("input"):
        if str(tag.get("type", "")).lower() == "hidden":
            tag.decompose()


def collect_renderable_children(
    node: Tag,
    *,
    depth: int,
    text_max_len: int,
    max_depth: int,
) -> list[str]:
    rendered: list[str] = []
    for child in node.children:
        if isinstance(child, NavigableString):
            text = clean_text(str(child), max_len=text_max_len)
            if text:
                rendered.append("  " * depth + escape(text))
            continue
        if isinstance(child, Tag):
            rendered_child = render_tag(
                child,
                depth=depth,
                text_max_len=text_max_len,
                max_depth=max_depth,
            )
            if rendered_child:
                rendered.extend(rendered_child)
    return rendered


def should_keep_tag(tag: Tag, attrs: dict[str, Any], child_lines: list[str], direct_text: str, depth: int) -> bool:
    name = tag.name.lower()
    if name in VOID_TAGS:
        return bool(attrs)
    if name in ACTION_TAGS or name in MEDIA_TAGS:
        return bool(attrs or child_lines or direct_text)
    if name in STRUCTURAL_TAGS:
        return bool(child_lines or attrs or direct_text)
    if name in CONTENT_TAGS:
        return bool(child_lines or attrs or direct_text)
    if attrs and (child_lines or direct_text):
        return True
    if child_lines and depth <= 2:
        return True
    return False


def render_tag(
    tag: Tag,
    *,
    depth: int,
    text_max_len: int,
    max_depth: int,
) -> list[str]:
    name = tag.name.lower()
    if name in DROP_TAGS:
        return []
    if name in UNWRAP_TAGS:
        return collect_renderable_children(tag, depth=depth, text_max_len=text_max_len, max_depth=max_depth)
    if depth > max_depth:
        text = clean_text(tag.get_text(" "), max_len=text_max_len)
        return ["  " * depth + escape(text)] if text else []

    attrs = filter_attrs(tag)
    child_lines = collect_renderable_children(
        tag,
        depth=depth + 1,
        text_max_len=text_max_len,
        max_depth=max_depth,
    )
    direct_text = clean_text(
        " ".join(str(child) for child in tag.children if isinstance(child, NavigableString)),
        max_len=text_max_len,
    )
    if not should_keep_tag(tag, attrs, child_lines, direct_text, depth):
        return child_lines

    indent = "  " * depth
    open_tag = f"{indent}<{name}{attrs_to_html(attrs)}>"
    if name in VOID_TAGS:
        return [open_tag]
    if not child_lines:
        return [open_tag, f"{indent}</{name}>"]
    return [open_tag, *child_lines, f"{indent}</{name}>"]


def clamp_lines(lines: list[str], max_chars: int) -> tuple[list[str], bool]:
    output: list[str] = []
    total = 0
    truncated = False
    for line in lines:
        next_total = total + len(line) + 1
        if next_total > max_chars:
            truncated = True
            break
        output.append(line)
        total = next_total
    if truncated:
        output.append("<!-- compact HTML truncated -->")
    return output, truncated


def preprocess_html_for_llm(
    html: str,
    *,
    max_chars: int = 16000,
    text_max_len: int = 280,
    max_depth: int = 10,
) -> PreprocessResult:
    """Return compact HTML that keeps observable structure without judging it."""
    soup = BeautifulSoup(html or "", "html.parser")
    preprocess_soup(soup)
    roots = list(soup.body.children) if soup.body else list(soup.children)
    lines: list[str] = []
    for root in roots:
        if isinstance(root, NavigableString):
            text = clean_text(str(root), max_len=text_max_len)
            if text:
                lines.append(escape(text))
            continue
        if isinstance(root, Tag):
            lines.extend(render_tag(root, depth=0, text_max_len=text_max_len, max_depth=max_depth))

    if not lines:
        text = clean_text(soup.get_text(" "), max_len=min(text_max_len, max_chars))
        if text:
            lines = [escape(text)]

    lines, truncated = clamp_lines(lines, max_chars)
    compact_html = "\n".join(lines).strip()
    return PreprocessResult(
        compact_html=compact_html,
        original_chars=len(html or ""),
        compact_chars=len(compact_html),
        truncated=truncated,
    )
