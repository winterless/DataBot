"""Pipeline entry: start, resume, stop runs."""

from __future__ import annotations

from src.contracts import validate_payload
from src.utils.logger import get_logger, with_trace


def validate_eval_feedback_contract(run_id: str, payload: dict) -> None:
    """Example: orchestrator validates contracts via `src/contracts` only."""
    logger = with_trace(
        get_logger("databot.orchestrator"),
        trace_id=f"{run_id}:step_eval_01",
        run_id=run_id,
        step_id="step_eval_01",
    )
    validate_payload(payload, "eval_feedback.schema.json")
    logger.info("eval feedback contract validated")


def main() -> None:
    """CLI entry placeholder."""
    get_logger("databot.orchestrator").info("orchestrator skeleton is ready")


if __name__ == "__main__":
    main()
