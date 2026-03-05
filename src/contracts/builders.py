"""Contract payload builders with trace_info."""

from __future__ import annotations

from typing import Any, Dict


def build_trace_info(run_id: str, step_id: str, attempt: int) -> Dict[str, str]:
    """Build trace_info block used by all step contracts."""
    return {
        "run_id": run_id,
        "step_id": step_id,
        "idempotency_key": f"{run_id}_{step_id}_attempt_{attempt}",
    }


def build_envelope_success(data: Dict[str, Any]) -> Dict[str, Any]:
    """Build a SUCCESS response envelope."""
    return {"status": "SUCCESS", "error": None, "data": data}


def build_envelope_failed(code: str, message: str, retryable: bool) -> Dict[str, Any]:
    """Build a FAILED response envelope."""
    return {
        "status": "FAILED",
        "error": {"code": code, "message": message, "retryable": retryable},
        "data": None,
    }
