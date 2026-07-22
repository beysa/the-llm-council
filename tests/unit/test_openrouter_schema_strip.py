from __future__ import annotations

from llm_council.providers.openrouter import (
    _prepare_structured_output_schema,
    _strip_schema_metadata,
)


def test_strip_removes_number_range_keywords() -> None:
    """Anthropic/OpenAI structured output rejects minimum/maximum on numbers.

    OpenRouter forwards the schema untouched, so these keywords must be stripped
    or every structured call to those providers 400s (and a single/critique
    model failing surfaces as "Below minimum required providers (1)").
    """
    schema = {
        "type": "object",
        "properties": {
            "score": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "exclusiveMinimum": 0,
                "exclusiveMaximum": 10,
                "multipleOf": 0.1,
            }
        },
        "required": ["score"],
        "additionalProperties": False,
    }

    stripped = _strip_schema_metadata(schema)
    score = stripped["properties"]["score"]

    assert score == {"type": "number"}
    for keyword in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"):
        assert keyword not in score


def test_strip_removes_nested_and_array_range_keywords() -> None:
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1, "maximum": 100},
            }
        },
    }

    stripped = _strip_schema_metadata(schema)
    item_schema = stripped["properties"]["items"]["items"]

    assert item_schema == {"type": "integer"}


def test_strip_preserves_shape_keywords() -> None:
    """Only value constraints are dropped; structural keywords stay intact."""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    }

    stripped = _strip_schema_metadata(schema)

    assert stripped["type"] == "object"
    assert stripped["required"] == ["name"]
    assert stripped["additionalProperties"] is False
    assert stripped["properties"]["name"] == {"type": "string"}


def test_prepare_structured_output_schema_drops_range_keywords() -> None:
    schema = {
        "type": "object",
        "properties": {"score": {"type": "number", "minimum": 0, "maximum": 1}},
        "required": ["score"],
        "additionalProperties": False,
    }

    prepared, strict_mode = _prepare_structured_output_schema(schema, strict=True)

    assert "minimum" not in prepared["properties"]["score"]
    assert "maximum" not in prepared["properties"]["score"]
    # A fully-required, closed object with the range keywords gone stays strict-compatible.
    assert strict_mode is True
