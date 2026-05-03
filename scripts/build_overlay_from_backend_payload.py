import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from funnel_pipeline.run_funnel_selector_mapping import build_overlay_js


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an overlay JS file directly from the backend payload shape."
    )
    parser.add_argument(
        "--payload-json",
        default=str(PROJECT_ROOT / "run" / "slack_llama_backend_payload.json"),
        help="Path to the backend payload JSON file.",
    )
    parser.add_argument(
        "--output-js",
        default=str(PROJECT_ROOT / "run" / "slack_llama_backend_payload_overlay.js"),
        help="Path to save the overlay JS.",
    )
    return parser.parse_args()


def convert_payload_to_overlay_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_funnels = payload.get("funnels")
    if not isinstance(raw_funnels, list):
        raise RuntimeError("Payload must contain a `funnels` array.")

    overlay_items: list[dict[str, Any]] = []
    for item in raw_funnels:
        if not isinstance(item, dict):
            continue

        name = str(item.get("name") or "").strip().upper()
        selector = item.get("selector")
        overlay_items.append(
            {
                "funnel": name,
                "selector": selector,
            }
        )

    return overlay_items


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.payload_json).read_text(encoding="utf-8"))
    overlay_items = convert_payload_to_overlay_items(payload)

    output_js = build_overlay_js(overlay_items)
    output_path = Path(args.output_js)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_js, encoding="utf-8")

    print(f"Saved JS: {output_path}")


if __name__ == "__main__":
    main()
