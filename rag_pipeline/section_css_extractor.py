"""Extract CSS rules relevant to a section HTML fragment."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from bs4 import BeautifulSoup

try:
    import tinycss2
except ModuleNotFoundError:  # pragma: no cover - runtime fallback before deps are installed.
    tinycss2 = None  # type: ignore[assignment]


NESTED_RULE_AT_KEYWORDS = {"media", "supports", "container"}
PRESERVED_AT_KEYWORDS = {"keyframes", "-webkit-keyframes"}
IDENTIFIER_DELIMITERS = set(" \t\r\n\f.#:[>+~,{})=|^$*\"'")


@dataclass(frozen=True)
class SectionSelectorContext:
    tags: set[str]
    classes: set[str]
    ids: set[str]


def extract_relevant_section_css(section_html: str, css: str) -> str:
    """Return only CSS rules that plausibly affect the given section HTML.

    The function is intentionally conservative: if parsing fails or matching
    becomes ambiguous, it returns the original CSS rather than risk removing
    required styling.
    """
    if not css.strip() or not section_html.strip() or tinycss2 is None:
        return css

    context = selector_context_from_html(section_html)
    if not context.tags and not context.classes and not context.ids:
        return css

    try:
        rules = tinycss2.parse_stylesheet(css, skip_comments=True, skip_whitespace=True)
        filtered = filter_rules(rules, context)
    except Exception:
        return css

    rendered = "\n".join(rule for rule in filtered if rule.strip()).strip()
    return rendered or css


def selector_context_from_html(section_html: str) -> SectionSelectorContext:
    soup = BeautifulSoup(section_html, "html.parser")
    tags: set[str] = set()
    classes: set[str] = set()
    ids: set[str] = set()

    for node in soup.find_all(True):
        if node.name:
            tags.add(str(node.name).lower())

        raw_id = node.get("id")
        if raw_id:
            ids.add(str(raw_id))

        raw_classes = node.get("class")
        if isinstance(raw_classes, list):
            classes.update(str(value) for value in raw_classes if str(value).strip())
        elif raw_classes:
            classes.update(value for value in str(raw_classes).split() if value)

    return SectionSelectorContext(tags=tags, classes=classes, ids=ids)


def filter_rules(rules: list[Any], context: SectionSelectorContext) -> list[str]:
    rendered: list[str] = []

    for rule in rules:
        rule_type = getattr(rule, "type", "")
        if rule_type == "qualified-rule":
            css = render_qualified_rule(rule, context)
            if css:
                rendered.append(css)
        elif rule_type == "at-rule":
            css = render_at_rule(rule, context)
            if css:
                rendered.append(css)

    return rendered


def render_qualified_rule(rule: Any, context: SectionSelectorContext) -> str | None:
    selector = tinycss2.serialize(rule.prelude).strip()
    content = tinycss2.serialize(rule.content).strip()
    if not selector or not content:
        return None

    if is_global_foundation_rule(selector, content) or selector_matches_context(selector, context):
        return f"{selector}{{{content}}}"
    return None


def render_at_rule(rule: Any, context: SectionSelectorContext) -> str | None:
    keyword = str(getattr(rule, "at_keyword", "")).lower()
    if keyword == "import":
        return None

    prelude = tinycss2.serialize(getattr(rule, "prelude", [])).strip()
    content = getattr(rule, "content", None)

    if keyword in NESTED_RULE_AT_KEYWORDS and content is not None:
        nested_rules = tinycss2.parse_rule_list(
            content,
            skip_comments=True,
            skip_whitespace=True,
        )
        nested = filter_rules(nested_rules, context)
        if not nested:
            return None
        prelude_text = f" {prelude}" if prelude else ""
        return f"@{keyword}{prelude_text}{{\n" + "\n".join(nested) + "\n}"

    if keyword in PRESERVED_AT_KEYWORDS:
        prelude_text = f" {prelude}" if prelude else ""
        if content is None:
            return f"@{keyword}{prelude_text};"
        return f"@{keyword}{prelude_text}{{{tinycss2.serialize(content).strip()}}}"

    return None


def selector_matches_context(selector_text: str, context: SectionSelectorContext) -> bool:
    for selector in split_selector_list(selector_text):
        classes, ids = extract_class_and_id_selectors(selector)
        if classes & context.classes:
            return True
        if ids & context.ids:
            return True
        if selector_matches_tag(selector, context.tags):
            return True
    return False


def is_global_foundation_rule(selector_text: str, content: str) -> bool:
    selectors = split_selector_list(selector_text)
    if not selectors:
        return False

    if all(is_global_foundation_selector(selector) for selector in selectors):
        return True

    normalized_selectors = {normalize_selector(selector) for selector in selectors}
    token_selectors = {":root", "html", "body", ".dark"}
    return bool(normalized_selectors & token_selectors) and has_custom_property_declaration(content)


def is_global_foundation_selector(selector: str) -> bool:
    normalized = normalize_selector(selector)
    if normalized in {
        "*",
        ":before",
        ":after",
        "::before",
        "::after",
        ":root",
        "html",
        "body",
        ":host",
    }:
        return True
    if normalized.startswith("html:") or normalized.startswith("body:"):
        return True
    return normalized in {"*, :before, :after", "*,:before,:after"}


def has_custom_property_declaration(content: str) -> bool:
    return bool(re.search(r"(^|[;{\s])--[-_a-zA-Z0-9]+\s*:", content))


def split_selector_list(selector_text: str) -> list[str]:
    selectors: list[str] = []
    start = 0
    bracket_depth = 0
    paren_depth = 0
    quote: str | None = None
    escaped = False

    for index, char in enumerate(selector_text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
            continue
        if char == "[":
            bracket_depth += 1
            continue
        if char == "]" and bracket_depth:
            bracket_depth -= 1
            continue
        if char == "(":
            paren_depth += 1
            continue
        if char == ")" and paren_depth:
            paren_depth -= 1
            continue
        if char == "," and bracket_depth == 0 and paren_depth == 0:
            selectors.append(selector_text[start:index].strip())
            start = index + 1

    tail = selector_text[start:].strip()
    if tail:
        selectors.append(tail)
    return selectors


def extract_class_and_id_selectors(selector: str) -> tuple[set[str], set[str]]:
    classes: set[str] = set()
    ids: set[str] = set()
    index = 0

    while index < len(selector):
        char = selector[index]
        if char not in {".", "#"}:
            index += 1
            continue

        identifier, next_index = read_css_identifier(selector, index + 1)
        if identifier:
            if char == ".":
                classes.add(identifier)
            else:
                ids.add(identifier)
        index = max(next_index, index + 1)

    return classes, ids


def read_css_identifier(value: str, start: int) -> tuple[str, int]:
    raw: list[str] = []
    index = start

    while index < len(value):
        char = value[index]
        if char == "\\":
            escape, index = read_css_escape(value, index)
            raw.append(escape)
            continue
        if char in IDENTIFIER_DELIMITERS:
            break
        raw.append(char)
        index += 1

    return "".join(raw), index


def read_css_escape(value: str, start: int) -> tuple[str, int]:
    index = start + 1
    if index >= len(value):
        return "\\", index

    hex_match = re.match(r"[0-9a-fA-F]{1,6}", value[index:])
    if hex_match:
        hex_value = hex_match.group(0)
        index += len(hex_value)
        if index < len(value) and value[index].isspace():
            index += 1
        try:
            return chr(int(hex_value, 16)), index
        except ValueError:
            return hex_value, index

    return value[index], index + 1


def selector_matches_tag(selector: str, tags: set[str]) -> bool:
    if not tags:
        return False
    for tag in tags:
        if re.search(rf"(^|[\s>+~,(]){re.escape(tag)}(?=($|[\s.#:[>+~),]))", selector, re.I):
            return True
    return False


def normalize_selector(selector: str) -> str:
    return re.sub(r"\s+", " ", selector.strip())
