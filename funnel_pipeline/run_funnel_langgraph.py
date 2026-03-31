import argparse
import json
import os
import re
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
from html_tools.transform import extract_body_html


PROMPT_PATH = Path("prompts/html_to_funnel_prompt.txt")


class FunnelState(TypedDict, total=False):
    input_html: str
    body_html: str
    prompt_template: str
    compressed_html: str
    prompt: str
    provider: str
    model: str
    base_url: str
    llm_raw_output: str
    funnel_json_text: str


def load_prompt_node(state: FunnelState) -> FunnelState:
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    return {"prompt_template": prompt_template}


def extract_body_node(state: FunnelState) -> FunnelState:
    body_html = extract_body_html(state["input_html"])
    return {"body_html": body_html}


def compress_html_node(state: FunnelState) -> FunnelState:
    spec = CompressionSpec()
    compressed_html = dom_to_compressed_lines(state["body_html"], spec=spec)
    return {"compressed_html": compressed_html}


def compose_prompt_node(state: FunnelState) -> FunnelState:
    prompt = state["prompt_template"].replace("{{INPUT_HTML}}", state["compressed_html"])
    return {"prompt": prompt}


def resolve_client_config(state: FunnelState) -> tuple[str, str, str | None]:
    provider = state["provider"]

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
        default_model = "gpt-5.4-mini"
        default_base_url = os.getenv("OPENAI_BASE_URL")
    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
        default_model = "meta-llama/llama-4-scout-17b-16e-instruct"
        default_base_url = "https://api.groq.com/openai/v1"
    else:
        raise RuntimeError(f"Unsupported provider: {provider}")

    model = state.get("model") or default_model
    base_url = state.get("base_url") or default_base_url
    return api_key, model, base_url


def call_llm_node(state: FunnelState) -> FunnelState:
    api_key, model, base_url = resolve_client_config(state)
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)
    request_kwargs = {
        "model": model,
        "messages": [
            {"role": "user", "content": state["prompt"]},
        ],
    }
    if state["provider"] == "openai":
        request_kwargs["reasoning_effort"] = "medium"

    response = client.chat.completions.create(**request_kwargs)
    content = response.choices[0].message.content or ""
    return {"llm_raw_output": content.strip()}


def extract_json_text(raw: str) -> str:
    stripped = raw.strip()
    if not stripped:
        return stripped

    fence_matches = re.findall(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    candidates = [candidate.strip() for candidate in fence_matches if candidate.strip()]

    first_array = stripped.find("[")
    last_array = stripped.rfind("]")
    if first_array != -1 and last_array != -1 and first_array < last_array:
        candidates.append(stripped[first_array : last_array + 1].strip())

    first_object = stripped.find("{")
    last_object = stripped.rfind("}")
    if first_object != -1 and last_object != -1 and first_object < last_object:
        candidates.append(stripped[first_object : last_object + 1].strip())

    for candidate in candidates:
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue

    return stripped


def validate_and_format_json_node(state: FunnelState) -> FunnelState:
    raw = extract_json_text(state["llm_raw_output"])
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model output is not valid JSON: {exc}\nOutput:\n{raw}") from exc

    formatted = json.dumps(parsed, ensure_ascii=False, indent=2)
    return {"funnel_json_text": formatted}


def build_graph():
    graph = StateGraph(FunnelState)
    graph.add_node("load_prompt", load_prompt_node)
    graph.add_node("extract_body", extract_body_node)
    graph.add_node("compress_html", compress_html_node)
    graph.add_node("compose_prompt", compose_prompt_node)
    graph.add_node("call_llm", call_llm_node)
    graph.add_node("validate_json", validate_and_format_json_node)

    graph.add_edge(START, "load_prompt")
    graph.add_edge("load_prompt", "extract_body")
    graph.add_edge("extract_body", "compress_html")
    graph.add_edge("compress_html", "compose_prompt")
    graph.add_edge("compose_prompt", "call_llm")
    graph.add_edge("call_llm", "validate_json")
    graph.add_edge("validate_json", END)
    return graph.compile()


def parse_args():
    parser = argparse.ArgumentParser(
        description="LangGraph workflow: HTML -> compressed HTML -> funnel JSON text via OpenAI-compatible LLMs"
    )
    parser.add_argument("--input-html", default="input.html", help="Path to the source HTML file")
    parser.add_argument("--output", default="funnel.json", help="Path to save JSON text")
    parser.add_argument(
        "--provider",
        choices=("openai", "groq"),
        default=os.getenv("LLM_PROVIDER", "openai"),
        help="LLM provider to call",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL"),
        help="Model name override for the selected provider",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BASE_URL"),
        help="Optional OpenAI-compatible API base URL override",
    )
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    input_html = Path(args.input_html).read_text(encoding="utf-8")
    workflow = build_graph()

    result = workflow.invoke(
        {
            "input_html": input_html,
            "provider": args.provider,
            "model": args.model,
            "base_url": args.base_url,
        }
    )

    output_path = Path(args.output)
    output_path.write_text(result["funnel_json_text"], encoding="utf-8")
    print(result["funnel_json_text"])
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
