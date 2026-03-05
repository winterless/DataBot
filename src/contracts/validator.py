"""JSON schema validation entrypoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from jsonschema import Draft202012Validator


_SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"


def load_schema(schema_file: str) -> Dict[str, Any]:
    """Load one schema file from the root-level `schemas/` directory."""
    schema_path = _SCHEMA_DIR / schema_file
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    return json.loads(schema_path.read_text(encoding="utf-8"))


def validate_payload(payload: Dict[str, Any], schema_file: str) -> None:
    """Validate payload against schema and raise on the first error."""
    schema = load_schema(schema_file)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)
    if errors:
        first = errors[0]
        path = ".".join(str(p) for p in first.path) or "<root>"
        raise ValueError(f"Contract validation failed at {path}: {first.message}")
