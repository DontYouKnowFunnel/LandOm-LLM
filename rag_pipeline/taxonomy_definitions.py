"""Shared controlled-label definitions for RAG feature mapping prompts.

These definitions mirror the label criteria used when the current Problem and
Revision DB records were mapped into controlled retrieval features.
"""

from __future__ import annotations

from rag_pipeline.features import (
    BEHAVIOR_CLUSTERS,
    HTML_FEATURES,
    PERSONA_TRAITS,
    label_description_en,
)


HTML_FEATURE_DEFINITIONS_EN = {
    label: label_description_en(label) or label.replace("_", " ")
    for label in HTML_FEATURES
}


PERSONA_TRAIT_DEFINITIONS_EN = {
    "low_awareness": "Visitor has limited category/product awareness and needs plain explanation.",
    "time_constrained": "Visitor needs quick scanning and fast time-to-value.",
    "price_sensitive": "Visitor is sensitive to price, budget, plan value, or payment commitment.",
    "risk_averse": "Visitor is cautious about commitment, switching cost, lock-in, security, or failure.",
    "trust_sensitive": "Visitor needs credibility, proof, safety, or authority signals before acting.",
    "comparison_oriented": "Visitor is comparing alternatives and needs clear differentiation.",
    "nontechnical_user": "Visitor is likely not technical and needs accessible product explanation.",
    "technical_user": "Visitor can evaluate technical details such as API, workflow, data, or integration.",
    "team_buyer": "Visitor is buying or evaluating for a team/workflow.",
    "individual_user": "Visitor is buying or evaluating for personal or solo use.",
    "enterprise_buyer": "Visitor represents a larger organization with procurement, security, or scale concerns.",
    "small_business_owner": "Visitor runs a small business and values practical ROI and simplicity.",
    "founder_operator": "Visitor is a founder/operator seeking growth, speed, or leverage.",
}


BEHAVIOR_CLUSTER_DEFINITIONS_EN = {
    "quick_exit": "Leaves quickly before meaningful exploration.",
    "passive_browsing": "Views the section without notable clicks or conversion intent.",
    "interaction_without_conversion": "Clicks, focuses a field, or interacts but does not convert.",
    "deep_engagement_exit": "Scrolls or stays deeply but exits without converting.",
    "section_stall": "Stays repeatedly in the same section, suggesting uncertainty or friction.",
    "shallow_scan": "Scans lightly with limited depth or low engagement.",
}


def label_definitions_text(labels: set[str], definitions: dict[str, str]) -> str:
    return "\n".join(
        f"- {label}: {definitions.get(label, label.replace('_', ' '))}"
        for label in sorted(labels)
    )


def html_feature_definitions_text() -> str:
    return label_definitions_text(HTML_FEATURES, HTML_FEATURE_DEFINITIONS_EN)


def persona_trait_definitions_text() -> str:
    return label_definitions_text(PERSONA_TRAITS, PERSONA_TRAIT_DEFINITIONS_EN)


def behavior_cluster_definitions_text() -> str:
    return label_definitions_text(BEHAVIOR_CLUSTERS, BEHAVIOR_CLUSTER_DEFINITIONS_EN)
