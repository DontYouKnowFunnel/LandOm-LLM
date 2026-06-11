import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv()

import funnel_pipeline.run_funnel_langgraph as _funnel_flow
from funnel_pipeline.config import AI_MODELS
from html_tools.selector_lookup import map_funnel_items_to_selectors
from html_tools.spec import CompressionSpec
from rag_pipeline.langsmith_tracking import trace_tags, trace_workflow

# 기존 모듈은 cwd 기반으로 prompts 파일을 읽으므로 절대경로로 보정.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_funnel_flow.PROMPT_PATH = _PROJECT_ROOT / "prompts" / "html_to_funnel_prompt.txt"


def run(
    html: str,
    provider: str | None = None,
    model: str | None = None,
) -> list[dict[str, Any]]:
    resolved_provider = provider or AI_MODELS.funnel_analysis[0]
    workflow_name = "funnel.analysis"
    with trace_workflow(
        name=workflow_name,
        inputs={
            "html_chars": len(html),
            "provider": resolved_provider,
            "model": model,
        },
        metadata={"provider": resolved_provider, "model": model},
        tags=trace_tags(workflow=workflow_name),
    ):
        workflow = _funnel_flow.build_graph()
        result = workflow.invoke(
            {
                "input_html": html,
                "provider": resolved_provider,
                "model": model,
            }
        )
    funnel_items = json.loads(result["funnel_json_text"])
    return map_funnel_items_to_selectors(
        html=html,
        funnel_items=funnel_items,
        spec=CompressionSpec(),
        id_key="id",
    )
