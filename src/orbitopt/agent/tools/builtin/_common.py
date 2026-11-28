"""JSON Schema shorthands shared by the built-in tool specs."""

from __future__ import annotations

from typing import Any

TEST_FILE = "tests/test_e14_04_tool_registry.py"


def number(description: str) -> dict:
    return {"type": "number", "description": description}


def integer(description: str) -> dict:
    return {"type": "integer", "description": description}


def numbers(description: str, min_items: int = 1, max_items: int = 64) -> dict:
    return {"type": "array", "items": {"type": "number"}, "minItems": min_items, "maxItems": max_items,
            "description": description}


def text(description: str, **extra: Any) -> dict:
    return {"type": "string", "description": description, **extra}


def inputs(properties: dict, required: tuple[str, ...] | None = None) -> dict:
    """A closed input object; every property is required unless ``required`` says otherwise."""
    return {"type": "object", "properties": properties,
            "required": list(properties if required is None else required), "additionalProperties": False}


def outputs(properties: dict) -> dict:
    """A closed output object whose every property is required."""
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def tests(*names: str) -> tuple[str, ...]:
    return tuple(f"{TEST_FILE}::{name}" for name in names)
