"""Normalize backend visitor behavior payloads into retrieval events."""

from __future__ import annotations

import json
from typing import Any


def parse_event_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value in (None, ""):
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"rawPayload": value}
        return parsed if isinstance(parsed, dict) else {"payload": parsed}
    return {"payload": value}


def normalize_event(raw_event: dict[str, Any], *, session_id: str | None = None) -> dict[str, Any]:
    event_type = str(
        raw_event.get("event")
        or raw_event.get("type")
        or raw_event.get("eventType")
        or ""
    ).strip().lower()
    payload = parse_event_payload(raw_event.get("payload"))
    normalized: dict[str, Any] = {
        "event": event_type,
        "type": event_type,
        "timestamp": raw_event.get("timestamp"),
        "sessionId": raw_event.get("sessionId") or session_id,
    }
    normalized.update(payload)
    css_selector = raw_event.get("cssSelector")
    if css_selector is not None:
        normalized["cssSelector"] = css_selector
    if event_type == "click" and not normalized.get("targetId") and css_selector:
        normalized["targetId"] = css_selector
    if event_type == "input" and not normalized.get("fieldId") and css_selector:
        normalized["fieldId"] = css_selector
    return {key: value for key, value in normalized.items() if value is not None}


def normalize_visitor_behavior_data(payload: Any) -> list[dict[str, Any]]:
    """Flatten API visitorBehaviorData/session payloads into event dicts.

    Supported inputs:
    - [{"event": "scroll", ...}]
    - {"events": [...]}
    - {"latestSessionLimit": 10, "sessions": [{"sessionId": "...", "events": [...]}]}
    - {"sessions": [...]}
    """
    if isinstance(payload, list):
        return [
            normalize_event(event)
            for event in payload
            if isinstance(event, dict)
        ]
    if not isinstance(payload, dict):
        return []

    if isinstance(payload.get("events"), list):
        return [
            normalize_event(event, session_id=str(payload.get("sessionId") or ""))
            for event in payload["events"]
            if isinstance(event, dict)
        ]

    events: list[dict[str, Any]] = []
    sessions = payload.get("sessions")
    if isinstance(sessions, list):
        for session in sessions:
            if not isinstance(session, dict):
                continue
            session_id = str(session.get("sessionId") or "")
            for event in session.get("events") or []:
                if isinstance(event, dict):
                    normalized = normalize_event(event, session_id=session_id)
                    normalized["sessionDurationSeconds"] = session.get("durationSeconds")
                    events.append(normalized)
    return events
