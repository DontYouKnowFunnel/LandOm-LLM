import logging
import os
from typing import Any

import httpx
from dotenv import load_dotenv
load_dotenv()

from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from app.pipeline import run as run_pipeline
from rag_pipeline.code_generator import generate_codegen
from rag_pipeline.optimization_pipeline import run_optimization
from rag_pipeline.section_css_extractor import extract_relevant_section_css

logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)


app = FastAPI(title="LandOm-LLM Funnel API")


class AnalyzeRequest(BaseModel):
    projectId: int = Field(..., gt=0)
    html: str = Field(..., min_length=1)


class OptimizationRequest(BaseModel):
    projectId: int = Field(..., gt=0)
    sectionId: int = Field(..., gt=0)
    sectionName: str = Field(..., min_length=1)
    sectionHtml: str = Field(..., min_length=1)
    visitorBehaviorData: dict[str, Any] = Field(default_factory=dict)
    persona: str | None = None


class CodegenRequest(BaseModel):
    projectId: int = Field(..., gt=0)
    sectionId: int = Field(..., gt=0)
    sectionHtml: str = Field(..., min_length=1)
    sectionCss: str = Field(...)
    optimizationPlans: list[dict[str, Any]] = Field(..., min_length=1)

    class Config:
        extra = "forbid"


@app.post("/api/v1/funnels/analyze", status_code=status.HTTP_202_ACCEPTED)
def analyze(req: AnalyzeRequest, background_tasks: BackgroundTasks) -> None:
    callback_origin = os.getenv("BACKEND_BASE_URL")
    if not callback_origin:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="BACKEND_BASE_URL is not configured",
        )
    background_tasks.add_task(_process_and_callback, req.projectId, req.html, callback_origin)


@app.post("/api/v1/funnels/optimize", status_code=status.HTTP_202_ACCEPTED)
def optimize(req: OptimizationRequest, background_tasks: BackgroundTasks) -> None:
    callback_origin = os.getenv("BACKEND_BASE_URL")
    if not callback_origin:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="BACKEND_BASE_URL is not configured",
        )
    logger.info(
        "optimization request received projectId=%s sectionId=%s sectionName=%s "
        "persona=%r sectionHtmlLength=%s sectionHtmlPreview=%r "
        "visitorBehaviorSessionCount=%s visitorBehaviorData=%s",
        req.projectId,
        req.sectionId,
        req.sectionName,
        req.persona,
        len(req.sectionHtml),
        _preview(req.sectionHtml),
        _session_count(req.visitorBehaviorData),
        req.visitorBehaviorData,
    )

    background_tasks.add_task(_process_optimization_and_callback, req, callback_origin)


@app.post("/api/v1/funnels/codegen", status_code=status.HTTP_202_ACCEPTED)
def codegen(req: CodegenRequest, background_tasks: BackgroundTasks) -> None:
    callback_origin = os.getenv("BACKEND_BASE_URL")
    if not callback_origin:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="BACKEND_BASE_URL is not configured",
        )

    logger.info(
        "code generation request received projectId=%s sectionId=%s optimizationCount=%s "
        "sectionHtmlLength=%s sectionHtmlPreview=%r sectionCssLength=%s",
        req.projectId,
        req.sectionId,
        len(req.optimizationPlans),
        len(req.sectionHtml),
        _preview(req.sectionHtml),
        len(req.sectionCss),
    )

    background_tasks.add_task(_process_codegen_and_callback, req, callback_origin)


def _process_and_callback(project_id: int, html: str, callback_origin: str) -> None:
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
    callback_url = f"{callback_origin.rstrip('/')}/api/v1/projects/{project_id}/analytics/section"

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(callback_url, json=payload)
            response.raise_for_status()
    except Exception:
        logger.exception(
            "callback POST failed url=%s projectId=%s", callback_url, project_id
        )


def _process_optimization_and_callback(req: OptimizationRequest, callback_origin: str) -> None:
    try:
        result = run_optimization(
            section_html=req.sectionHtml,
            section_name=req.sectionName,
            persona=req.persona,
            visitor_behavior_data=req.visitorBehaviorData,
            project_id=req.projectId,
            section_id=req.sectionId,
        )
    except Exception:
        logger.exception(
            "optimization pipeline failed projectId=%s sectionId=%s",
            req.projectId,
            req.sectionId,
        )
        return

    recommendations = result["recommendation"].get("recommendations", [])
    _send_optimization_plan_callback(
        project_id=req.projectId,
        section_id=req.sectionId,
        recommendations=recommendations,
        callback_origin=callback_origin,
    )


def _process_codegen_and_callback(req: CodegenRequest, callback_origin: str) -> None:
    try:
        filtered_css = extract_relevant_section_css(req.sectionHtml, req.sectionCss)
        logger.info(
            "code generation css filtered projectId=%s sectionId=%s sectionCssLength=%s "
            "filteredSectionCssLength=%s",
            req.projectId,
            req.sectionId,
            len(req.sectionCss),
            len(filtered_css),
        )
        result = generate_codegen(
            optimization_plan=req.optimizationPlans,
            html=req.sectionHtml,
            css=filtered_css,
            project_id=req.projectId,
            section_id=req.sectionId,
        )
        if not result["html"]:
            raise RuntimeError("code generation returned empty html")
    except Exception:
        logger.exception(
            "code generation failed projectId=%s sectionId=%s",
            req.projectId,
            req.sectionId,
        )
        return

    _send_codegen_callback(
        project_id=req.projectId,
        section_id=req.sectionId,
        html=result["html"],
        css=result["css"],
        callback_origin=callback_origin,
    )


def _send_optimization_plan_callback(
    *,
    project_id: int,
    section_id: int,
    recommendations: list[dict[str, Any]],
    callback_origin: str,
) -> None:
    payload: dict[str, Any] = {
        "recommendations": recommendations,
    }
    callback_url = (
        f"{callback_origin.rstrip('/')}/api/v1/projects/{project_id}"
        f"/optimizations/{section_id}"
    )

    try:
        logger.info(
            "callback started projectId=%s sectionId=%s url=%s recommendationCount=%s",
            project_id,
            section_id,
            callback_url,
            len(recommendations),
        )
        with httpx.Client(timeout=10.0) as client:
            response = client.patch(callback_url, json=payload)
            response.raise_for_status()
        logger.info(
            "callback completed projectId=%s sectionId=%s statusCode=%s",
            project_id,
            section_id,
            response.status_code,
        )
    except Exception:
        logger.exception(
            "optimization callback PATCH failed url=%s projectId=%s sectionId=%s",
            callback_url,
            project_id,
            section_id,
        )


def _send_codegen_callback(
    *,
    project_id: int,
    section_id: int,
    html: str,
    css: str,
    callback_origin: str,
) -> None:
    payload: dict[str, Any] = {
        "html": html,
        "css": css,
    }
    callback_url = (
        f"{callback_origin.rstrip('/')}/api/v1/projects/{project_id}"
        f"/optimizations/{section_id}/codegen"
    )

    try:
        logger.info(
            "code generation callback started projectId=%s sectionId=%s "
            "url=%s htmlChars=%s cssChars=%s",
            project_id,
            section_id,
            callback_url,
            len(html),
            len(css),
        )
        with httpx.Client(timeout=10.0) as client:
            response = client.patch(callback_url, json=payload)
            response.raise_for_status()
        logger.info(
            "code generation callback completed projectId=%s sectionId=%s statusCode=%s",
            project_id,
            section_id,
            response.status_code,
        )
    except Exception:
        logger.exception(
            "code generation callback failed url=%s projectId=%s sectionId=%s",
            callback_url,
            project_id,
            section_id,
        )


def _preview(value: str, limit: int = 500) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def _session_count(visitor_behavior_data: dict[str, Any]) -> int:
    sessions = visitor_behavior_data.get("sessions")
    if isinstance(sessions, list):
        return len(sessions)
    return 0
