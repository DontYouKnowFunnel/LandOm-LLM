import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from funnel_pipeline.run_funnel_langgraph import build_graph
from html_tools.selector_lookup import map_funnel_items_to_selectors
from html_tools.spec import CompressionSpec


DEFAULT_INPUT_HTML = PROJECT_ROOT / "examples" / "input" / "slack.html"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "run" / "slack_llama_backend_payload.json"
DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def run_funnel_analysis(
    html: str,
    *,
    provider: str = "groq",
    model: str = DEFAULT_MODEL,
    base_url: str | None = None,
) -> list[dict[str, Any]]:
    """Run the main funnel pipeline and return normalized funnel items."""
    workflow = build_graph()
    result = workflow.invoke(
        {
            "input_html": html,
            "provider": provider,
            "model": model,
            "base_url": base_url,
        }
    )
    return json.loads(result["funnel_json_text"])


def build_backend_payload(
    html: str,
    funnel_items: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Convert normalized funnel items into the backend response shape."""
    mapped_items = map_funnel_items_to_selectors(
        html=html,
        funnel_items=funnel_items,
        spec=CompressionSpec(),
        id_key="id",
    )

    funnels: list[dict[str, Any]] = []
    for step_order, item in enumerate(mapped_items, start=1):
        funnels.append(
            {
                "stepOrder": step_order,
                "name": item["funnel"],
                "selector": item["selector"],
            }
        )

    return {"funnels": funnels}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the funnel pipeline with Groq llama and emit the backend payload shape."
    )
    parser.add_argument(
        "--input-html",
        default=str(DEFAULT_INPUT_HTML),
        help="Path to the source landing-page HTML file.",
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_OUTPUT_JSON),
        help="Path to save the backend payload JSON.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Groq llama model name.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional Groq-compatible base URL override.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    args = parse_args()

    html = Path(args.input_html).read_text(encoding="utf-8")
    funnel_items = run_funnel_analysis(
        html,
        provider="groq",
        model=args.model,
        base_url=args.base_url,
    )
    payload = build_backend_payload(html, funnel_items)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
