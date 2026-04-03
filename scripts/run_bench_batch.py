from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from funnel_pipeline.run_funnel_langgraph import (  # noqa: E402
    extract_json_text,
    normalize_llm_output,
    resolve_client_config,
)
from html_tools.segments_targeted_refine import (  # noqa: E402
    extract_page_segments_targeted_refine,
    merge_adjacent_same_funnel_items,
    segments_to_prompt_input_targeted_refine,
)
from html_tools.spec import CompressionSpec  # noqa: E402
from html_tools.transform import extract_body_html  # noqa: E402

PROMPT_PATH = PROJECT_ROOT / "prompts" / "html_to_funnel_prompt.txt"
DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-5.4",
}
OUTPUT_SUBDIRS = {
    "groq": "llama",
    "openai": "gpt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the main benchmark pipeline with a selected provider and model."
    )
    parser.add_argument("--provider", choices=["groq", "openai"], default="groq")
    parser.add_argument("--model", default=None)
    parser.add_argument("--input-dir", default="LandOm-LLM-Bench/input")
    parser.add_argument("--gold-dir", default="LandOm-LLM-Bench/output")
    parser.add_argument("--output-root", default="run/bench_eval")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--indices", nargs="+", default=None)
    parser.add_argument("--prompt-path", default=str(PROMPT_PATH))
    return parser.parse_args()


def sorted_html_paths(input_dir: Path) -> list[Path]:
    def sort_key(path: Path) -> tuple[int, str]:
        stem = path.stem
        return (int(stem), stem) if stem.isdigit() else (10**9, stem)

    return sorted(input_dir.glob("*.html"), key=sort_key)


def filter_html_paths(paths: list[Path], args: argparse.Namespace) -> list[Path]:
    if args.indices:
        wanted = set(args.indices)
        paths = [path for path in paths if path.stem in wanted]
    if args.limit is not None:
        paths = paths[: args.limit]
    return paths


def call_llm(provider: str, prompt: str, model: str) -> str:
    api_key, resolved_model, base_url = resolve_client_config(
        {
            "provider": provider,
            "model": model,
            "base_url": None,
        }
    )
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    request_kwargs: dict[str, Any] = {
        "model": resolved_model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if provider == "openai":
        request_kwargs["reasoning_effort"] = "medium"
    response = client.chat.completions.create(**request_kwargs)
    return (response.choices[0].message.content or "").strip()


def run_prediction(
    provider: str,
    html: str,
    prompt_template: str,
    model: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    body_html = extract_body_html(html)
    segments = extract_page_segments_targeted_refine(body_html, spec=CompressionSpec())
    prompt_input = segments_to_prompt_input_targeted_refine(segments)

    prompt = prompt_template.replace("{{SEGMENT_COUNT}}", str(len(segments)))
    prompt = prompt.replace("{{INPUT_SEGMENTS}}", prompt_input)

    raw = call_llm(provider, prompt, model=model)
    parsed = json.loads(extract_json_text(raw))
    normalized = normalize_llm_output(parsed, segments)
    merged = merge_adjacent_same_funnel_items(normalized, segments)
    return normalized, merged


def write_prediction_outputs(
    output_root: Path,
    idx: str,
    raw_items: list[dict[str, Any]],
    pred_items: list[dict[str, Any]],
) -> None:
    raw_dir = output_root / "raw_funnels"
    pred_dir = output_root / "pred"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{idx}.json").write_text(
        json.dumps(raw_items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (pred_dir / f"{idx}.json").write_text(
        json.dumps(pred_items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def grade_all(input_dir: Path, gold_dir: Path, pred_dir: Path, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "LandOm-LLM-Bench" / "grade_funnels.py"),
            "--input-dir",
            str(input_dir),
            "--gold-dir",
            str(gold_dir),
            "--pred-dir",
            str(pred_dir),
            "--out-csv",
            str(report_dir / "grade.csv"),
            "--out-json",
            str(report_dir / "grade.json"),
        ],
        check=True,
    )


def main() -> None:
    load_dotenv()
    args = parse_args()
    input_dir = PROJECT_ROOT / args.input_dir
    gold_dir = PROJECT_ROOT / args.gold_dir
    model = args.model or DEFAULT_MODELS[args.provider]
    subdir = OUTPUT_SUBDIRS[args.provider]
    output_root = PROJECT_ROOT / args.output_root / subdir
    prompt_template = Path(args.prompt_path).read_text(encoding="utf-8")

    html_paths = filter_html_paths(sorted_html_paths(input_dir), args)
    for html_path in html_paths:
        idx = html_path.stem
        html = html_path.read_text(encoding="utf-8")
        raw_items, pred_items = run_prediction(args.provider, html, prompt_template, model)
        write_prediction_outputs(output_root, idx, raw_items, pred_items)
        print(f"[{subdir}] saved sample {idx}")

    grade_all(input_dir, gold_dir, output_root / "pred", output_root / "reports")


if __name__ == "__main__":
    main()
