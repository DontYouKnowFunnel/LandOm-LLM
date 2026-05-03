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
    projectId: int = Field(..., gt=0)
    html: str = Field(..., min_length=1)


@app.post("/api/v1/funnels/analyze", status_code=status.HTTP_202_ACCEPTED)
def analyze(req: AnalyzeRequest, background_tasks: BackgroundTasks) -> None:
    base_url = os.getenv("BACKEND_BASE_URL")
    if not base_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="BACKEND_BASE_URL is not configured",
        )
    background_tasks.add_task(_process_and_callback, req.projectId, req.html, base_url)


def _process_and_callback(project_id: int, html: str, base_url: str) -> None:
    try:
        items = run_pipeline(html)
    except Exception:
        logger.exception("funnel pipeline failed for projectId=%s", project_id)
        return

    funnels = [
        {
            "stepOrder": idx,
            "name": item.get("funnel"),
            "selector": item.get("selector"),
        }
        for idx, item in enumerate(items, start=1)
    ]
    payload: dict[str, Any] = {"funnels": funnels}
    callback_url = f"{base_url.rstrip('/')}/api/v1/projects/{project_id}/analytics/section"

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(callback_url, json=payload)
            response.raise_for_status()
    except Exception:
        logger.exception(
            "callback POST failed url=%s projectId=%s", callback_url, project_id
        )
