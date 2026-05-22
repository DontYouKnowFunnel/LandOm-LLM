#!/usr/bin/env python3
"""Query Qdrant RAG collections with an OpenAI embedding."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient, models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from rag_pipeline.retrieval_utils import qdrant_client as make_qdrant_client  # noqa: E402


def openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env before querying.")
    return OpenAI(api_key=api_key)


def embed_query(client: OpenAI, query: str, model: str) -> list[float]:
    response = client.embeddings.create(
        model=model,
        input=[query],
        encoding_format="float",
    )
    return response.data[0].embedding


def make_filter(args: argparse.Namespace) -> models.Filter | None:
    conditions: list[models.FieldCondition] = []
    if args.section_label:
        if args.collection.startswith("problem"):
            key = "section_label"
        else:
            key = "applicable_section_labels"
        conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=args.section_label)))
    if args.problem_case:
        conditions.append(
            models.FieldCondition(key="problem_case", match=models.MatchValue(value=args.problem_case))
        )
    if not conditions:
        return None
    return models.Filter(must=conditions)


def compact_payload(payload: dict[str, Any], language: str) -> dict[str, Any]:
    context_field = f"llm_context_{language}"
    embedding_field = f"embedding_text_{language}"
    return {
        "card_id": payload.get("card_id") or payload.get("id"),
        "db_type": payload.get("db_type") or payload.get("type"),
        "section_label": payload.get("section_label") or payload.get("applicable_section_labels"),
        "problem_case": payload.get("problem_case"),
        "intervention": payload.get("intervention"),
        "behavior_clusters": payload.get("behavior_clusters")
        or payload.get("applicable_behavior_clusters")
        or payload.get("target_behavior_clusters"),
        "html_features": payload.get("html_features") or payload.get("required_html_features"),
        embedding_field: payload.get(embedding_field),
        context_field: payload.get(context_field),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="problem_patterns_en")
    parser.add_argument("--qdrant-path", type=Path, default=PROJECT_ROOT / "run/qdrant")
    parser.add_argument("--qdrant-url", help="Remote Qdrant URL. Defaults to QDRANT_URL.")
    parser.add_argument("--qdrant-api-key", help="Remote Qdrant API key. Defaults to QDRANT_API_KEY.")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--language", choices=("en", "ko"), default="en")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--section-label")
    parser.add_argument("--problem-case")
    parser.add_argument("--full-payload", action="store_true")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    client = openai_client()
    qdrant = make_qdrant_client(
        path=args.qdrant_path,
        url=args.qdrant_url,
        api_key=args.qdrant_api_key,
    )
    query_vector = embed_query(client, args.query, args.embedding_model)

    results = qdrant.search(
        collection_name=args.collection,
        query_vector=query_vector,
        query_filter=make_filter(args),
        limit=args.top_k,
    )

    for index, result in enumerate(results, 1):
        payload = result.payload or {}
        rendered = payload if args.full_payload else compact_payload(payload, args.language)
        print(f"\n## {index}. score={result.score:.4f}")
        print(json.dumps(rendered, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
