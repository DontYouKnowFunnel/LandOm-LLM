"""LangSmith tracing helpers for OpenAI-compatible model calls."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LANGSMITH_PROJECT = "trackerLLM"
_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}

try:  # LangSmith is optional so local scripts still run without observability.
    import langsmith as ls
except Exception:  # pragma: no cover - exercised when dependency is absent.
    ls = None  # type: ignore[assignment]


def configure_langsmith_from_env() -> bool:
    """Enable LangSmith tracing when API key configuration is present."""
    load_dotenv(PROJECT_ROOT / ".env")
    if ls is None:
        return False

    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        return False

    if _env_is_false("LANGSMITH_TRACING") or _env_is_false("LANGCHAIN_TRACING_V2"):
        return False

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", DEFAULT_LANGSMITH_PROJECT)
    os.environ.setdefault("LANGCHAIN_PROJECT", os.environ["LANGSMITH_PROJECT"])
    return True


def _env_is_false(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and value.strip().lower() in _FALSY


def _compact_metadata(values: dict[str, Any] | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in (values or {}).items():
        if value is None:
            continue
        if isinstance(value, str) and len(value) > 500:
            metadata[key] = f"{value[:500]}..."
        else:
            metadata[key] = value
    return metadata


def model_metadata(
    *,
    provider: str,
    model: str,
    workflow: str,
    stage: str,
    prompt_name: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Metadata LangSmith needs for model identity and cost lookup."""
    return _compact_metadata(
        {
            "ls_provider": provider,
            "ls_model_name": model,
            "workflow": workflow,
            "stage": stage,
            "prompt_name": prompt_name,
            **(extra or {}),
        }
    )


def trace_tags(
    *,
    provider: str | None = None,
    model: str | None = None,
    workflow: str | None = None,
    stage: str | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    tags = list(extra or [])
    for prefix, value in (
        ("provider", provider),
        ("model", model),
        ("workflow", workflow),
        ("stage", stage),
    ):
        if value:
            tags.append(f"{prefix}:{value}")
    return tags


@contextmanager
def trace_workflow(
    *,
    name: str,
    inputs: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Iterator[Any | None]:
    """Create a parent LangSmith workflow span when tracing is configured."""
    if not configure_langsmith_from_env():
        yield None
        return

    with ls.trace(
        name=name,
        run_type="chain",
        inputs=inputs,
        metadata=_compact_metadata(metadata),
        tags=tags or [],
    ) as run_tree:
        yield run_tree


def traced_chat_completion(
    *,
    client: Any,
    request_kwargs: dict[str, Any],
    provider: str,
    model: str,
    workflow: str,
    stage: str,
    prompt_name: str,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Any:
    """Trace an OpenAI-compatible chat completion and preserve the raw response."""
    if not configure_langsmith_from_env():
        return client.chat.completions.create(**request_kwargs)

    trace_metadata = model_metadata(
        provider=provider,
        model=model,
        workflow=workflow,
        stage=stage,
        prompt_name=prompt_name,
        extra=metadata,
    )
    trace_inputs = {
        "messages": request_kwargs.get("messages", []),
        "model": model,
        "parameters": {
            key: value
            for key, value in request_kwargs.items()
            if key not in {"messages", "model"}
        },
    }
    with ls.trace(
        name=stage,
        run_type="llm",
        inputs=trace_inputs,
        metadata=trace_metadata,
        tags=tags
        or trace_tags(provider=provider, model=model, workflow=workflow, stage=stage),
    ) as run_tree:
        response = client.chat.completions.create(**request_kwargs)
        usage_metadata = usage_metadata_from_response(response)
        outputs = chat_completion_outputs(response, usage_metadata)
        if usage_metadata:
            run_tree.set(usage_metadata=usage_metadata)
        run_tree.end(outputs=outputs)
        return response


def traced_embedding_create(
    *,
    client: Any,
    request_kwargs: dict[str, Any],
    provider: str,
    model: str,
    workflow: str,
    stage: str,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Trace an embedding request with token usage when the provider returns it."""
    if not configure_langsmith_from_env():
        return client.embeddings.create(**request_kwargs)

    trace_metadata = model_metadata(
        provider=provider,
        model=model,
        workflow=workflow,
        stage=stage,
        extra=metadata,
    )
    trace_inputs = {
        "model": model,
        "input": request_kwargs.get("input"),
        "parameters": {
            key: value
            for key, value in request_kwargs.items()
            if key not in {"input", "model"}
        },
    }
    with ls.trace(
        name=stage,
        run_type="embedding",
        inputs=trace_inputs,
        metadata=trace_metadata,
        tags=trace_tags(provider=provider, model=model, workflow=workflow, stage=stage),
    ) as run_tree:
        response = client.embeddings.create(**request_kwargs)
        usage_metadata = usage_metadata_from_response(response)
        if usage_metadata:
            run_tree.set(usage_metadata=usage_metadata)
        run_tree.end(outputs={"usage_metadata": usage_metadata})
        return response


def usage_metadata_from_response(response: Any) -> dict[str, Any]:
    usage = _get(response, "usage")
    if not usage:
        return {}

    input_tokens = _get(usage, "prompt_tokens", _get(usage, "input_tokens"))
    output_tokens = _get(usage, "completion_tokens", _get(usage, "output_tokens"))
    total_tokens = _get(usage, "total_tokens")
    usage_metadata: dict[str, Any] = {}
    if input_tokens is not None:
        usage_metadata["input_tokens"] = int(input_tokens)
    if output_tokens is not None:
        usage_metadata["output_tokens"] = int(output_tokens)
    if total_tokens is not None:
        usage_metadata["total_tokens"] = int(total_tokens)

    input_details = _token_details(
        _get(usage, "prompt_tokens_details", _get(usage, "input_tokens_details")),
        mapping={"cached_tokens": "cache_read"},
    )
    if input_details:
        usage_metadata["input_token_details"] = input_details

    output_details = _token_details(
        _get(usage, "completion_tokens_details", _get(usage, "output_tokens_details")),
        mapping={"reasoning_tokens": "reasoning"},
    )
    if output_details:
        usage_metadata["output_token_details"] = output_details

    return usage_metadata


def _token_details(details: Any, *, mapping: dict[str, str]) -> dict[str, int]:
    if not details:
        return {}
    result: dict[str, int] = {}
    for source_key, target_key in mapping.items():
        value = _get(details, source_key)
        if value is not None:
            result[target_key] = int(value)
    return result


def chat_completion_outputs(response: Any, usage_metadata: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "id": _get(response, "id"),
        "model": _get(response, "model"),
        "choices": _jsonable(_get(response, "choices")),
        "usage_metadata": usage_metadata,
    }
    content = first_message_content(response)
    if content is not None:
        output["messages"] = [{"role": "assistant", "content": content}]
    return output


def first_message_content(response: Any) -> str | None:
    choices = _get(response, "choices")
    if not choices:
        return None
    first_choice = choices[0]
    message = _get(first_choice, "message")
    content = _get(message, "content")
    if content is None:
        return None
    return str(content)


def _get(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return _jsonable(value.dict())
    return str(value)
