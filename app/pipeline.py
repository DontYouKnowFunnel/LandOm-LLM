import json
import os
from pathlib import Path
from typing import Any

import funnel_pipeline.run_funnel_langgraph as _funnel_flow
from html_tools.selector_lookup import map_funnel_items_to_selectors
from html_tools.spec import CompressionSpec

# 기존 모듈은 cwd 기반으로 prompts 파일을 읽으므로 절대경로로 보정.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_funnel_flow.PROMPT_PATH = _PROJECT_ROOT / "prompts" / "html_to_funnel_prompt.txt"


def run(
    html: str,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> list[dict[str, Any]]:
    workflow = _funnel_flow.build_graph()
    result = workflow.invoke(
        {
            "input_html": html,
            "provider": provider or os.getenv("LLM_PROVIDER", "openai"),
            "model": model or os.getenv("LLM_MODEL"),
            "base_url": base_url or os.getenv("LLM_BASE_URL"),
        }
    )
    funnel_items = json.loads(result["funnel_json_text"])
    return map_funnel_items_to_selectors(
        html=html,
        funnel_items=funnel_items,
        spec=CompressionSpec(),
        id_key="id",
    )
