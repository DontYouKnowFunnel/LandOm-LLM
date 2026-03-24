import argparse
import json
import os
import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from html_tools.compress import dom_to_compressed_lines
from html_tools.spec import CompressionSpec


PROMPT_PATH = Path("prompts/html_to_funnel_prompt.txt")


class FunnelState(TypedDict, total=False):
    input_html: str
    prompt_template: str
    compressed_html: str
    prompt: str
    llm_raw_output: str
    funnel_json_text: str


def load_prompt_node(state: FunnelState) -> FunnelState:
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    return {"prompt_template": prompt_template}


def compress_html_node(state: FunnelState) -> FunnelState:
    spec = CompressionSpec()
    compressed_html = dom_to_compressed_lines(state["input_html"], spec=spec)
    return {"compressed_html": compressed_html}


def compose_prompt_node(state: FunnelState) -> FunnelState:
    prompt = state["prompt_template"].replace("{{INPUT_HTML}}", state["compressed_html"])
    return {"prompt": prompt}


def call_openai_node(state: FunnelState) -> FunnelState:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        reasoning_effort="medium",
        messages=[
            {"role": "user", "content": state["prompt"]},
        ],
    )
    content = response.choices[0].message.content or ""
    return {"llm_raw_output": content.strip()}


def validate_and_format_json_node(state: FunnelState) -> FunnelState:
    raw = state["llm_raw_output"].strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model output is not valid JSON: {exc}\nOutput:\n{raw}") from exc

    formatted = json.dumps(parsed, ensure_ascii=False, indent=2)
    return {"funnel_json_text": formatted}


def build_graph():
    graph = StateGraph(FunnelState)
    graph.add_node("load_prompt", load_prompt_node)
    graph.add_node("compress_html", compress_html_node)
    graph.add_node("compose_prompt", compose_prompt_node)
    graph.add_node("call_openai", call_openai_node)
    graph.add_node("validate_json", validate_and_format_json_node)

    graph.add_edge(START, "load_prompt")
    graph.add_edge("load_prompt", "compress_html")
    graph.add_edge("compress_html", "compose_prompt")
    graph.add_edge("compose_prompt", "call_openai")
    graph.add_edge("call_openai", "validate_json")
    graph.add_edge("validate_json", END)
    return graph.compile()


def parse_args():
    parser = argparse.ArgumentParser(
        description="LangGraph workflow: HTML -> compressed HTML -> funnel JSON text via OpenAI"
    )
    parser.add_argument("--input-html", default="input.html", help="Path to the source HTML file")
    parser.add_argument("--output", default="funnel.json", help="Path to save JSON text")
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    input_html = Path(args.input_html).read_text(encoding="utf-8")
    workflow = build_graph()

    result = workflow.invoke(
        {
            "input_html": input_html,
        }
    )

    output_path = Path(args.output)
    output_path.write_text(result["funnel_json_text"], encoding="utf-8")
    print(result["funnel_json_text"])
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
