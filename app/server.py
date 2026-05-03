import logging
import os
from typing import Any

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from app.pipeline import run as run_pipeline

logger = logging.getLogger(__name__)

app = FastAPI(title="LandOm-LLM Funnel API")


class AnalyzeRequest(BaseModel):
    apiKey: str = Field(..., min_length=1)
    html: str = Field(..., min_length=1)


@app.post("/api/v1/funnels/analyze", status_code=status.HTTP_202_ACCEPTED)
def analyze(req: AnalyzeRequest, background_tasks: BackgroundTasks) -> None:
    callback_url = os.getenv("BACKEND_CALLBACK_URL")
    if not callback_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="BACKEND_CALLBACK_URL is not configured",
        )
    background_tasks.add_task(_process_and_callback, req.apiKey, req.html, callback_url)


def _process_and_callback(api_key: str, html: str, callback_url: str) -> None:
    try:
        items = run_pipeline(html)
    except Exception:
        logger.exception("funnel pipeline failed for apiKey=%s", api_key)
        return

    funnels = [
        {
            "stepOrder": idx,
            "name": item.get("funnel"),
            "selector": item.get("selector"),
        }
        for idx, item in enumerate(items, start=1)
    ]
    payload: dict[str, Any] = {"apiKey": api_key, "funnels": funnels}

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(callback_url, json=payload)
            response.raise_for_status()
    except Exception:
        logger.exception(
            "callback POST failed url=%s apiKey=%s", callback_url, api_key
        )
