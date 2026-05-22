"""Feature extraction for PBS-CF RAG retrieval.

The production system will receive persona and behavior logs directly. For
experiments, this module can also infer reasonable defaults from the HTML and
optional free-text persona notes.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from rag_pipeline.behavior_normalizer import normalize_visitor_behavior_data


VALID_SECTION_LABELS = {
    "HERO",
    "FEATURE",
    "CTA_SECTION",
    "SOCIAL_PROOF",
    "VALUE_PROP",
    "USE_CASE",
    "PRICING",
    "FAQ",
    "TARGET",
    "PROBLEM",
    "GENERIC",
}


SECTION_ALIASES = {
    "CTA": "CTA_SECTION",
    "CALL_TO_ACTION": "CTA_SECTION",
    "SOCIALPROOF": "SOCIAL_PROOF",
    "VALUE": "VALUE_PROP",
    "VALUE_PROPOSITION": "VALUE_PROP",
}


PERSONA_TRAITS = {
    "low_awareness",
    "time_constrained",
    "price_sensitive",
    "risk_averse",
    "trust_sensitive",
    "comparison_oriented",
    "nontechnical_user",
    "technical_user",
    "team_buyer",
    "individual_user",
    "enterprise_buyer",
    "small_business_owner",
    "founder_operator",
}


BEHAVIOR_CLUSTERS = {
    "quick_exit",
    "passive_browsing",
    "interaction_without_conversion",
    "deep_engagement_exit",
    "section_stall",
    "shallow_scan",
}


HTML_FEATURES = {
    "unclear_value_proposition",
    "feature_oriented_headline",
    "generic_cta",
    "weak_action_motivation",
    "low_cta_value_proximity",
    "no_social_proof",
    "weak_trust_signal",
    "missing_credibility_marker",
    "unclear_pricing",
    "weak_value_justification",
    "no_risk_reversal",
    "missing_objection_handling",
    "unclear_category_explanation",
    "audience_not_explicit",
    "generic_positioning",
    "missing_use_case_context",
    "feature_list_without_benefits",
    "insufficient_product_context",
    "no_visual_product_demo",
    "high_information_density",
    "poor_information_hierarchy",
    "weak_differentiation",
    "missing_onboarding_reassurance",
    "unclear_conversion_entry_value",
    "unclear_next_step_after_cta",
    "missing_friction_reassurance",
    "unclear_form_value",
    "too_many_form_fields",
    "missing_privacy_reassurance",
}


LABEL_EN_OVERRIDES = {
    "low_awareness": "low awareness",
    "time_constrained": "time constrained",
    "price_sensitive": "price sensitive",
    "risk_averse": "risk averse",
    "trust_sensitive": "trust sensitive",
    "comparison_oriented": "comparison oriented",
    "nontechnical_user": "nontechnical user",
    "technical_user": "technical user",
    "team_buyer": "team buyer",
    "individual_user": "individual user",
    "enterprise_buyer": "enterprise buyer",
    "small_business_owner": "small business owner",
    "founder_operator": "founder operator",
    "quick_exit": "quick exit",
    "passive_browsing": "passive browsing",
    "interaction_without_conversion": "interaction without conversion",
    "deep_engagement_exit": "deep engagement followed by exit",
    "section_stall": "stalling within a section",
    "shallow_scan": "shallow scanning",
    "unclear_value_proposition": "unclear value proposition",
    "feature_oriented_headline": "feature-oriented headline",
    "generic_cta": "generic CTA",
    "weak_action_motivation": "weak motivation to act",
    "low_cta_value_proximity": "CTA placed far from value proof",
    "no_social_proof": "missing social proof",
    "weak_trust_signal": "weak trust signal",
    "missing_credibility_marker": "missing credibility marker",
    "unclear_pricing": "unclear pricing",
    "weak_value_justification": "weak value justification",
    "no_risk_reversal": "missing risk reversal",
    "missing_objection_handling": "missing objection handling",
    "unclear_category_explanation": "unclear category explanation",
    "audience_not_explicit": "unclear target audience",
    "generic_positioning": "generic positioning",
    "missing_use_case_context": "missing use-case context",
    "feature_list_without_benefits": "feature list without benefits",
    "insufficient_product_context": "insufficient product context",
    "no_visual_product_demo": "missing visual product demo",
    "high_information_density": "high information density",
    "poor_information_hierarchy": "poor information hierarchy",
    "weak_differentiation": "weak differentiation",
    "missing_onboarding_reassurance": "missing onboarding reassurance",
    "unclear_conversion_entry_value": "unclear conversion-entry value",
    "unclear_next_step_after_cta": "unclear next step after CTA",
    "missing_friction_reassurance": "missing friction reassurance",
    "unclear_form_value": "unclear form value",
    "too_many_form_fields": "too many form fields",
    "missing_privacy_reassurance": "missing privacy reassurance",
}


LABEL_DESCRIPTION_EN_OVERRIDES = {
    "low_awareness": "The visitor has limited category or product awareness and needs a plain explanation.",
    "time_constrained": "The visitor needs quick scanning and fast time-to-value.",
    "price_sensitive": "The visitor is sensitive to price, budget, plan value, or payment commitment.",
    "risk_averse": "The visitor is cautious about commitment, switching cost, lock-in, security, or failure.",
    "trust_sensitive": "The visitor needs credibility, proof, safety, or authority signals before acting.",
    "comparison_oriented": "The visitor is comparing alternatives and needs clear differentiation.",
    "nontechnical_user": "The visitor is likely not technical and needs accessible product explanation.",
    "technical_user": "The visitor can evaluate technical details such as APIs, workflows, data, or integrations.",
    "team_buyer": "The visitor is buying or evaluating for a team or shared workflow.",
    "individual_user": "The visitor is buying or evaluating for personal or solo use.",
    "enterprise_buyer": "The visitor represents a larger organization with procurement, security, or scale concerns.",
    "small_business_owner": "The visitor runs a small business and values practical ROI and simplicity.",
    "founder_operator": "The visitor is a founder or operator seeking growth, speed, or leverage.",
    "quick_exit": "The visitor leaves quickly before meaningful exploration.",
    "passive_browsing": "The visitor views the section without notable clicks or conversion intent.",
    "interaction_without_conversion": "The visitor clicks, focuses a field, or interacts but does not convert.",
    "deep_engagement_exit": "The visitor scrolls or stays deeply but exits without converting.",
    "section_stall": "The visitor stays repeatedly in the same section, suggesting uncertainty or friction.",
    "shallow_scan": "The visitor scans lightly with limited depth or low engagement.",
    "unclear_value_proposition": "The section does not quickly make the core user outcome or value clear.",
    "feature_oriented_headline": "The main copy reads like product mechanics, features, modules, or category terms rather than user outcomes.",
    "generic_cta": "The CTA text is generic and does not clarify the result or next step after clicking.",
    "weak_action_motivation": "The section gives little reason to act now or continue to the next step.",
    "low_cta_value_proximity": "The CTA is not close enough to value proof, reassurance, or a concrete benefit.",
    "no_social_proof": "The section lacks customer proof, testimonials, logos, metrics, reviews, or representative examples.",
    "weak_trust_signal": "Credibility, safety, expertise, or proof signals are weak for the requested action.",
    "missing_credibility_marker": "The section lacks concrete credibility markers such as customer count, named customer, certification, track record, or expert identity.",
    "unclear_pricing": "Pricing, plan basis, package differences, or payment conditions are hard to understand.",
    "weak_value_justification": "The section asks for money or commitment without clearly explaining the value received.",
    "no_risk_reversal": "The section lacks reassurance such as free trial, no credit card, refund, cancellation, privacy, or security.",
    "missing_objection_handling": "Likely objections or decision questions are not answered near this funnel step.",
    "unclear_category_explanation": "A low-awareness user may not understand the category, terminology, or what the service actually does.",
    "audience_not_explicit": "The intended user role, team, industry, company size, or use case is not explicit.",
    "generic_positioning": "The positioning is broad or category-generic and does not explain why this option is different.",
    "missing_use_case_context": "The section lacks concrete use cases, scenarios, jobs-to-be-done, or role-specific context.",
    "feature_list_without_benefits": "Features are listed without explaining the user benefit, result, or job each feature supports.",
    "insufficient_product_context": "The section does not provide enough concrete product usage context, examples, outputs, or workflow explanation.",
    "no_visual_product_demo": "The section lacks visual product proof such as screenshots, demo frames, interface previews, or output examples.",
    "high_information_density": "The section contains too much information for quick scanning.",
    "poor_information_hierarchy": "The section does not clearly separate primary message, supporting proof, details, and next action.",
    "weak_differentiation": "The section does not make the product's difference from alternatives clear.",
    "missing_onboarding_reassurance": "The section does not reduce concerns about setup, adoption, migration, learning, or time-to-value.",
    "unclear_conversion_entry_value": "The value of starting signup, demo request, waitlist, contact, or form entry is unclear.",
    "unclear_next_step_after_cta": "The user cannot predict what happens after clicking the CTA.",
    "missing_friction_reassurance": "The section does not reduce perceived effort, time cost, sales pressure, or commitment around conversion.",
    "unclear_form_value": "The reason to complete a form and the value received in return are unclear.",
    "too_many_form_fields": "The form appears to ask for too much information for the current perceived value.",
    "missing_privacy_reassurance": "The section asks for input or data without privacy, data-use, spam, or security reassurance.",
}


GENERIC_CTA_TEXTS = {
    "get started",
    "start",
    "learn more",
    "try it",
    "try now",
    "continue",
    "submit",
    "sign up",
    "contact us",
    "book now",
    "request",
}


OUTCOME_WORDS = {
    "save",
    "reduce",
    "increase",
    "grow",
    "convert",
    "rank",
    "revenue",
    "faster",
    "automate",
    "without",
    "clear",
    "better",
    "predictable",
    "secure",
    "trusted",
    "launch",
    "scale",
}


FEATURE_WORDS = {
    "platform",
    "dashboard",
    "workflow",
    "automation",
    "analytics",
    "ai",
    "api",
    "integration",
    "sync",
    "module",
    "tool",
    "software",
    "system",
}


TRUST_WORDS = {
    "trusted",
    "customers",
    "reviews",
    "stars",
    "testimonial",
    "case study",
    "certified",
    "secure",
    "soc 2",
    "gdpr",
}


RISK_REVERSAL_WORDS = {
    "free trial",
    "no credit card",
    "cancel anytime",
    "refund",
    "money back",
    "privacy",
    "secure",
    "takes",
    "minutes",
}


@dataclass
class HtmlExtraction:
    text: str
    headings: list[str]
    ctas: list[str]
    link_texts: list[str]
    form_inputs: list[str]
    image_alts: list[str]
    word_count: int


@dataclass
class ExtractedFeatures:
    section_label: str
    persona_traits: list[str]
    behavior_clusters: list[str]
    html_features: list[str]
    query_text_en: str
    extraction_notes: dict[str, Any] = field(default_factory=dict)


def normalize_section_label(section_label: str | None) -> str:
    normalized = str(section_label or "GENERIC").strip().upper()
    normalized = SECTION_ALIASES.get(normalized, normalized)
    return normalized if normalized in VALID_SECTION_LABELS else "GENERIC"


def label_en(value: str) -> str:
    return LABEL_EN_OVERRIDES.get(value, value.replace("_", " ").replace("-", " "))


def labels_en(values: list[str]) -> str:
    return ", ".join(label_en(value) for value in values)


def label_description_en(value: str) -> str:
    return LABEL_DESCRIPTION_EN_OVERRIDES.get(value, "")


def label_with_description_en(value: str) -> str:
    label = label_en(value)
    description = label_description_en(value).strip()
    if not description:
        return label
    return f"{label}: {description.rstrip('.')}"


def labels_with_descriptions_en(values: list[str]) -> str:
    return "; ".join(label_with_description_en(value) for value in values)


def parse_label_list(raw: str | None, allowed: set[str]) -> list[str]:
    if not raw:
        return []
    values = [value.strip() for value in re.split(r"[,;\n]", raw) if value.strip()]
    normalized = [value for value in values if value in allowed]
    unknown = sorted(set(values) - set(normalized))
    if unknown:
        raise ValueError(f"Unknown labels: {', '.join(unknown)}")
    return unique_preserve_order(normalized)


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_html(html: str) -> HtmlExtraction:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    headings = [clean_text(tag.get_text(" ")) for tag in soup.find_all(["h1", "h2", "h3"])]
    ctas = []
    for tag in soup.find_all(["button", "a"]):
        text = clean_text(tag.get_text(" ") or tag.get("aria-label", "") or tag.get("title", ""))
        if text:
            ctas.append(text[:120])
    link_texts = [
        clean_text(tag.get_text(" "))
        for tag in soup.find_all("a")
        if clean_text(tag.get_text(" "))
    ]
    form_inputs = []
    for tag in soup.find_all(["input", "textarea", "select"]):
        label = tag.get("placeholder") or tag.get("name") or tag.get("id") or tag.get("type") or tag.name
        form_inputs.append(str(label))
    image_alts = [clean_text(tag.get("alt", "")) for tag in soup.find_all("img") if clean_text(tag.get("alt", ""))]
    text = clean_text(soup.get_text(" "))
    return HtmlExtraction(
        text=text,
        headings=headings,
        ctas=ctas,
        link_texts=link_texts,
        form_inputs=form_inputs,
        image_alts=image_alts,
        word_count=len(re.findall(r"\w+", text)),
    )


def contains_any(text: str, words: set[str] | tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in words)


def add_feature(
    features: set[str],
    reasons: dict[str, str],
    feature: str,
    reason: str,
) -> None:
    if feature in HTML_FEATURES:
        features.add(feature)
        reasons.setdefault(feature, reason)


def infer_html_features(html: str, section_label: str) -> tuple[list[str], dict[str, str], HtmlExtraction]:
    extracted = extract_html(html)
    text_lower = extracted.text.lower()
    heading_text = " ".join(extracted.headings).lower()
    cta_text = " ".join(extracted.ctas).lower()
    features: set[str] = set()
    reasons: dict[str, str] = {}

    if section_label in {"HERO", "VALUE_PROP", "PROBLEM"}:
        if not extracted.headings or not contains_any(heading_text, OUTCOME_WORDS):
            add_feature(features, reasons, "unclear_value_proposition", "No clear outcome-oriented heading was detected.")
        if contains_any(heading_text, FEATURE_WORDS) and not contains_any(heading_text, OUTCOME_WORDS):
            add_feature(features, reasons, "feature_oriented_headline", "Heading appears to emphasize product mechanics over user outcomes.")

    generic_ctas = [cta for cta in extracted.ctas if cta.lower().strip() in GENERIC_CTA_TEXTS]
    if generic_ctas:
        add_feature(features, reasons, "generic_cta", f"Generic CTA text detected: {generic_ctas[:3]}.")
        add_feature(features, reasons, "weak_action_motivation", "CTA text does not clearly state the result of clicking.")

    if section_label in {"HERO", "CTA_SECTION", "PRICING"} and extracted.ctas:
        cta_context_has_value = contains_any(text_lower, OUTCOME_WORDS | RISK_REVERSAL_WORDS | TRUST_WORDS)
        if not cta_context_has_value:
            add_feature(features, reasons, "low_cta_value_proximity", "CTA appears without nearby value, proof, or reassurance cues.")

    if not contains_any(text_lower, TRUST_WORDS):
        add_feature(features, reasons, "no_social_proof", "No obvious social proof or trust cue was detected.")
        add_feature(features, reasons, "weak_trust_signal", "Credibility signals appear weak or absent.")

    if section_label == "PRICING" or "$" in extracted.text or "pricing" in text_lower:
        if not contains_any(text_lower, {"per", "month", "year", "seat", "plan", "trial", "free"}):
            add_feature(features, reasons, "unclear_pricing", "Pricing cues were detected, but package/value details look thin.")
        if not contains_any(text_lower, OUTCOME_WORDS | RISK_REVERSAL_WORDS):
            add_feature(features, reasons, "weak_value_justification", "Pricing appears without nearby value justification or risk reduction.")

    if not contains_any(text_lower, RISK_REVERSAL_WORDS):
        add_feature(features, reasons, "no_risk_reversal", "No trial, cancellation, refund, security, or low-friction reassurance was detected.")

    if section_label in {"FAQ", "PRICING", "CTA_SECTION"} and not soup_like_has_faq(extracted.text):
        add_feature(features, reasons, "missing_objection_handling", "No clear FAQ or objection-handling copy was detected.")

    if not contains_any(text_lower, {"for ", "teams", "founders", "developers", "marketers", "sales", "enterprise", "agencies"}):
        add_feature(features, reasons, "audience_not_explicit", "Target audience cues are weak or absent.")

    if extracted.word_count > 180:
        add_feature(features, reasons, "high_information_density", "The section contains a large amount of text.")
    if len(extracted.headings) <= 1 and extracted.word_count > 100:
        add_feature(features, reasons, "poor_information_hierarchy", "Long text with few headings suggests weak hierarchy.")

    if section_label in {"FEATURE", "VALUE_PROP"}:
        if contains_any(text_lower, FEATURE_WORDS) and not contains_any(text_lower, OUTCOME_WORDS):
            add_feature(features, reasons, "feature_list_without_benefits", "Feature language appears without clear benefit mapping.")
        has_product_context = bool(extracted.image_alts) or contains_any(
            text_lower,
            {
                "screenshot",
                "demo",
                "preview",
                "workflow",
                "walkthrough",
                "example",
                "screen",
                "output",
                "step-by-step",
            },
        )
        if not has_product_context:
            add_feature(features, reasons, "insufficient_product_context", "No concrete product example, workflow, or usage context was detected.")
            add_feature(features, reasons, "no_visual_product_demo", "No visual product demo or screenshot cue was detected.")

    conversion_ctas = [
        cta
        for cta in extracted.ctas
        if contains_any(
            cta,
            {
                "sign up",
                "signup",
                "start trial",
                "free trial",
                "request demo",
                "book demo",
                "contact sales",
                "join waitlist",
                "get started",
            },
        )
    ]
    if conversion_ctas:
        if not contains_any(text_lower, {"what happens", "takes", "minutes", "no credit card", "demo", "consultation"}):
            add_feature(features, reasons, "unclear_next_step_after_cta", "Conversion CTA exists, but the post-click path is not clearly explained.")
        if not contains_any(text_lower, RISK_REVERSAL_WORDS):
            add_feature(features, reasons, "missing_friction_reassurance", "Conversion CTA exists without clear friction-reducing reassurance.")
        if not contains_any(text_lower, OUTCOME_WORDS):
            add_feature(features, reasons, "unclear_conversion_entry_value", "Conversion CTA exists, but the value of entering conversion is unclear.")

    if extracted.form_inputs:
        if len(extracted.form_inputs) >= 4:
            add_feature(features, reasons, "too_many_form_fields", "Form has several input fields.")
        if not contains_any(text_lower, {"privacy", "secure", "spam", "no credit card"}):
            add_feature(features, reasons, "missing_privacy_reassurance", "Form exists without obvious privacy or data-use reassurance.")
        if not contains_any(text_lower, OUTCOME_WORDS):
            add_feature(features, reasons, "unclear_form_value", "Form exists, but the value exchange is unclear.")

    if len(features) < 2:
        add_feature(features, reasons, "generic_positioning", "Few specific differentiation or audience cues were detected.")

    ordered = sorted(features)
    return ordered, reasons, extracted


def soup_like_has_faq(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in ("faq", "frequently asked", "question", "how does", "can i", "do you"))


def infer_persona_traits(persona_text: str | None, html_text: str) -> list[str]:
    combined = f"{persona_text or ''} {html_text}".lower()
    traits: list[str] = []
    checks = [
        ("enterprise_buyer", ("enterprise", "security", "compliance", "procurement", "sales team")),
        ("team_buyer", ("team", "teams", "collaboration", "workspace", "organization")),
        ("individual_user", ("personal", "individual", "creator", "solo", "freelancer")),
        ("small_business_owner", ("small business", "agency", "shopify", "local business")),
        ("founder_operator", ("founder", "startup", "operator", "build", "launch")),
        ("technical_user", ("developer", "api", "engineering", "code", "sdk")),
        ("nontechnical_user", ("no code", "marketer", "sales", "ops", "human resources")),
        ("price_sensitive", ("budget", "cheap", "affordable", "price", "pricing", "cost")),
        ("risk_averse", ("risk", "secure", "privacy", "compliance", "refund", "cancel")),
        ("trust_sensitive", ("trust", "review", "customers", "case study", "secure")),
        ("comparison_oriented", ("compare", "alternative", "vs", "competitor", "best")),
        ("time_constrained", ("fast", "quick", "minutes", "save time", "busy")),
    ]
    for trait, keywords in checks:
        if any(keyword in combined for keyword in keywords):
            traits.append(trait)
    if "low_awareness" not in traits:
        traits.insert(0, "low_awareness")
    if "time_constrained" not in traits:
        traits.append("time_constrained")
    return unique_preserve_order([trait for trait in traits if trait in PERSONA_TRAITS])[:4]


def load_behavior_log(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = normalize_visitor_behavior_data(payload)
    if events:
        return events
    raise ValueError("Behavior log must be a JSON list, an object with `events`, or an object with `sessions`.")


def infer_behavior_clusters(
    events: list[dict[str, Any]],
    *,
    html_features: list[str],
    fallback: list[str] | None = None,
) -> list[str]:
    if fallback:
        return unique_preserve_order([value for value in fallback if value in BEHAVIOR_CLUSTERS])
    if not events:
        if any(
            feature in html_features
            for feature in (
                "unclear_conversion_entry_value",
                "unclear_next_step_after_cta",
                "missing_friction_reassurance",
                "too_many_form_fields",
            )
        ):
            return ["interaction_without_conversion", "passive_browsing"]
        if any(feature in html_features for feature in ("unclear_value_proposition", "generic_cta")):
            return ["quick_exit", "shallow_scan"]
        return ["passive_browsing"]

    event_names = [str(event.get("event") or event.get("type") or "").lower() for event in events]
    counter = Counter(event_names)
    max_depth = max((float(event.get("maxDepth", event.get("percentage", 0)) or 0) for event in events), default=0.0)
    if max_depth > 1:
        max_depth = max_depth / 100.0
    clicked = counter["click"] > 0
    input_focused = counter["input"] > 0
    ping_sections = [
        str(event.get("sectionId"))
        for event in events
        if str(event.get("event") or event.get("type") or "").lower() == "ping" and event.get("sectionId")
    ]
    repeated_ping = Counter(ping_sections).most_common(1)
    has_conversion = any(
        str(event.get("event") or event.get("type") or "").lower() in {"conversion", "submit", "purchase"}
        for event in events
    )

    clusters: list[str] = []
    if clicked or input_focused:
        if not has_conversion:
            clusters.append("interaction_without_conversion")
    if repeated_ping and repeated_ping[0][1] >= 3:
        clusters.append("section_stall")
    if max_depth < 0.25 and not clicked and not input_focused:
        clusters.append("quick_exit")
    if max_depth < 0.45:
        clusters.append("shallow_scan")
    if max_depth >= 0.75 and not has_conversion:
        clusters.append("deep_engagement_exit")
    if not clusters:
        clusters.append("passive_browsing")
    return unique_preserve_order([cluster for cluster in clusters if cluster in BEHAVIOR_CLUSTERS])[:3]


def build_query_text_en(
    *,
    section_label: str,
    persona_traits: list[str],
    behavior_clusters: list[str],
    html_features: list[str],
) -> str:
    return " ".join(
        [
            f"Funnel: {section_label}.",
            f"Persona traits: {labels_with_descriptions_en(persona_traits)}.",
            f"Behavior symptoms: {labels_with_descriptions_en(behavior_clusters)}.",
            f"HTML features: {labels_with_descriptions_en(html_features)}.",
        ]
    )


def extract_features(
    *,
    html: str,
    section_label: str,
    persona_text: str | None = None,
    persona_traits_override: list[str] | None = None,
    behavior_events: list[dict[str, Any]] | None = None,
    behavior_clusters_override: list[str] | None = None,
    html_features_override: list[str] | None = None,
) -> ExtractedFeatures:
    normalized_section = normalize_section_label(section_label)
    inferred_html_features, html_reasons, extracted_html = infer_html_features(html, normalized_section)
    html_features = html_features_override or inferred_html_features
    persona_traits = persona_traits_override or infer_persona_traits(persona_text, extracted_html.text)
    behavior_clusters = infer_behavior_clusters(
        behavior_events or [],
        html_features=html_features,
        fallback=behavior_clusters_override,
    )
    query_text_en = build_query_text_en(
        section_label=normalized_section,
        persona_traits=persona_traits,
        behavior_clusters=behavior_clusters,
        html_features=html_features,
    )
    return ExtractedFeatures(
        section_label=normalized_section,
        persona_traits=persona_traits,
        behavior_clusters=behavior_clusters,
        html_features=html_features,
        query_text_en=query_text_en,
        extraction_notes={
            "html_feature_reasons": html_reasons,
            "headings": extracted_html.headings[:8],
            "ctas": extracted_html.ctas[:12],
            "word_count": extracted_html.word_count,
            "persona_source": "override" if persona_traits_override else "heuristic",
            "behavior_source": "override" if behavior_clusters_override else ("events" if behavior_events else "html_default"),
            "html_feature_source": "override" if html_features_override else "heuristic",
        },
    )
