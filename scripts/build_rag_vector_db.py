#!/usr/bin/env python3
"""Embed RAG JSONL records and store them in Qdrant collections.

Default usage builds two Qdrant collections from the search-ready DBs:

    .venv/bin/python scripts/build_rag_vector_db.py --recreate

Set QDRANT_URL and QDRANT_API_KEY to upload to a remote Qdrant server.

For English landing-page experiments, keep the default `--language en`. The
script embeds `embedding_text_en` and stores the full card as Qdrant payload so
the generation step can later use `llm_context_en` or `llm_context_ko`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient, models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from rag_pipeline.retrieval_utils import qdrant_client as make_qdrant_client  # noqa: E402


DEFAULT_PROBLEM_PATH = PROJECT_ROOT / "data/rag/problem_patterns_en.jsonl"
DEFAULT_INTERVENTION_PATH = PROJECT_ROOT / "data/rag/revision_evidence_en.jsonl"
DEFAULT_QDRANT_PATH = PROJECT_ROOT / "run/qdrant"


@dataclass(frozen=True)
class CollectionSpec:
    name: str
    source_path: Path
    db_type: str


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def batched(values: list[Any], batch_size: int) -> list[list[Any]]:
    return [values[index : index + batch_size] for index in range(0, len(values), batch_size)]


def openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env before embedding.")
    return OpenAI(api_key=api_key)


def embed_texts(
    client: OpenAI,
    texts: list[str],
    *,
    model: str,
    batch_size: int,
    sleep_seconds: float,
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for batch_index, batch in enumerate(batched(texts, batch_size), 1):
        response = client.embeddings.create(
            model=model,
            input=batch,
            encoding_format="float",
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors.extend([item.embedding for item in ordered])
        print(f"  embedded batch {batch_index}: {len(batch)} texts")
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return vectors


def ensure_collection(
    qdrant: QdrantClient,
    *,
    collection_name: str,
    vector_size: int,
    recreate: bool,
) -> None:
    try:
        exists = qdrant.collection_exists(collection_name)
    except AttributeError:
        try:
            qdrant.get_collection(collection_name)
            exists = True
        except Exception:
            exists = False

    if recreate and exists:
        qdrant.delete_collection(collection_name)
        exists = False

    if not exists:
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        return

    try:
        info = qdrant.get_collection(collection_name)
    except Exception:
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        return

    existing_size = info.config.params.vectors.size
    if existing_size != vector_size:
        raise RuntimeError(
            f"Collection {collection_name} already exists with vector size {existing_size}, "
            f"but current embeddings have size {vector_size}. Re-run with --recreate."
        )


def point_id(collection_name: str, record_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"landom-rag:{collection_name}:{record_id}"))


def payload_for_record(
    record: dict[str, Any],
    *,
    db_type: str,
    language: str,
    embedding_field: str,
    context_field: str,
    embedding_model: str,
) -> dict[str, Any]:
    payload = dict(record)
    payload.update(
        {
            "card_id": record.get("id"),
            "db_type": db_type,
            "language": language,
            "embedding_field": embedding_field,
            "context_field": context_field,
            "embedding_model": embedding_model,
        }
    )
    if db_type == "intervention_evidence" and "target_behavior_clusters" in payload:
        payload["applicable_behavior_clusters"] = payload["target_behavior_clusters"]
    return payload


def upsert_collection(
    qdrant: QdrantClient,
    openai: OpenAI,
    spec: CollectionSpec,
    *,
    language: str,
    embedding_model: str,
    batch_size: int,
    upsert_batch_size: int,
    recreate: bool,
    sleep_seconds: float,
) -> int:
    embedding_field = f"embedding_text_{language}"
    context_field = f"llm_context_{language}"
    records = load_jsonl(spec.source_path)
    texts: list[str] = []
    usable_records: list[dict[str, Any]] = []
    for record in records:
        text = str(record.get(embedding_field, "")).strip()
        if not text:
            continue
        texts.append(text)
        usable_records.append(record)

    if not usable_records:
        raise RuntimeError(f"No records with {embedding_field} found in {spec.source_path}")

    print(f"\n[{spec.name}] records={len(usable_records)} embedding_field={embedding_field}")
    vectors = embed_texts(
        openai,
        texts,
        model=embedding_model,
        batch_size=batch_size,
        sleep_seconds=sleep_seconds,
    )
    vector_size = len(vectors[0])
    ensure_collection(
        qdrant,
        collection_name=spec.name,
        vector_size=vector_size,
        recreate=recreate,
    )

    points: list[models.PointStruct] = []
    for record, vector in zip(usable_records, vectors):
        record_id = str(record.get("id") or point_id(spec.name, json.dumps(record, sort_keys=True)))
        points.append(
            models.PointStruct(
                id=point_id(spec.name, record_id),
                vector=vector,
                payload=payload_for_record(
                    record,
                    db_type=spec.db_type,
                    language=language,
                    embedding_field=embedding_field,
                    context_field=context_field,
                    embedding_model=embedding_model,
                ),
            )
        )

    for batch_index, point_batch in enumerate(batched(points, upsert_batch_size), 1):
        qdrant.upsert(collection_name=spec.name, points=point_batch)
        print(f"  upserted batch {batch_index}: {len(point_batch)} points")

    return len(points)


def search_smoke_test(
    qdrant: QdrantClient,
    openai: OpenAI,
    *,
    collection_name: str,
    query: str,
    embedding_model: str,
    top_k: int,
) -> None:
    vector = embed_texts(
        openai,
        [query],
        model=embedding_model,
        batch_size=1,
        sleep_seconds=0,
    )[0]
    results = qdrant.search(collection_name=collection_name, query_vector=vector, limit=top_k)
    print(f"\nSmoke search: {collection_name}")
    print(f"Query: {query}")
    for index, result in enumerate(results, 1):
        payload = result.payload or {}
        label = payload.get("problem_case") or payload.get("intervention") or payload.get("card_id")
        print(f"  {index}. score={result.score:.4f} label={label} id={payload.get('card_id')}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=("en", "ko"), default="en")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--qdrant-path", type=Path, default=DEFAULT_QDRANT_PATH)
    parser.add_argument("--qdrant-url", help="Remote Qdrant URL. Defaults to QDRANT_URL.")
    parser.add_argument("--qdrant-api-key", help="Remote Qdrant API key. Defaults to QDRANT_API_KEY.")
    parser.add_argument("--problem-jsonl", type=Path, default=DEFAULT_PROBLEM_PATH)
    parser.add_argument("--intervention-jsonl", type=Path, default=DEFAULT_INTERVENTION_PATH)
    parser.add_argument("--problem-collection", default="problem_patterns_en")
    parser.add_argument("--intervention-collection", default="intervention_evidence_en")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--upsert-batch-size", type=int, default=64)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--no-smoke-test", action="store_true")
    args = parser.parse_args()

    if args.language == "ko":
        if args.problem_collection == "problem_patterns_en":
            args.problem_collection = "problem_patterns_ko"
        if args.intervention_collection == "intervention_evidence_en":
            args.intervention_collection = "intervention_evidence_ko"

    load_dotenv(PROJECT_ROOT / ".env")
    client = openai_client()
    qdrant = make_qdrant_client(
        path=args.qdrant_path,
        url=args.qdrant_url,
        api_key=args.qdrant_api_key,
    )

    specs = [
        CollectionSpec(args.problem_collection, args.problem_jsonl, "problem_pattern"),
        CollectionSpec(args.intervention_collection, args.intervention_jsonl, "intervention_evidence"),
    ]

    counts: dict[str, int] = {}
    for spec in specs:
        counts[spec.name] = upsert_collection(
            qdrant,
            client,
            spec,
            language=args.language,
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            upsert_batch_size=args.upsert_batch_size,
            recreate=args.recreate,
            sleep_seconds=args.sleep_seconds,
        )

    if not args.no_smoke_test:
        if args.language == "en":
            problem_query = (
                "Funnel: HERO. Persona traits: low awareness, time constrained. "
                "Behavior symptoms: quick exit. HTML features: unclear value proposition, generic CTA."
            )
            intervention_query = (
                "Retrieval task: find intervention evidence for a diagnosed landing-page problem. "
                "Problem to solve: value comprehension gap. Applicable funnel: HERO. "
                "Structural issue cues: unclear value proposition, feature-oriented headline."
            )
        else:
            problem_query = (
                "퍼널: HERO. 페르소나: low_awareness, time_constrained. "
                "행동 증상: quick_exit. HTML 특징: unclear_value_proposition, generic_cta."
            )
            intervention_query = (
                "해결 문제: value_comprehension_gap. 적용 퍼널: HERO. "
                "필요 HTML 특징: unclear_value_proposition, feature_oriented_headline."
            )
        search_smoke_test(
            qdrant,
            client,
            collection_name=args.problem_collection,
            query=problem_query,
            embedding_model=args.embedding_model,
            top_k=5,
        )
        search_smoke_test(
            qdrant,
            client,
            collection_name=args.intervention_collection,
            query=intervention_query,
            embedding_model=args.embedding_model,
            top_k=5,
        )

    print("\nDone.")
    for collection_name, count in counts.items():
        print(f"- {collection_name}: {count} points")
    print(f"- qdrant_path: {args.qdrant_path}")
    print(f"- embedding_model: {args.embedding_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
