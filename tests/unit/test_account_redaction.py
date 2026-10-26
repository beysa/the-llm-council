from __future__ import annotations

import time

import pytest

from llm_council.logging import redact_account_identity


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Owner (jane.doe@example.com) pays", "Owner ([REDACTED-email]) pays"),
        ("<j.doe+council@mail.example.co.uk>", "<[REDACTED-email]>"),
        ("+jane.doe@example.com", "+[REDACTED-email]"),
        ('user.email="jane.doe@example.com"', "user.email=[REDACTED]"),
        ("user.account_id=acct-synthetic-0001 next", "user.account_id=[REDACTED] next"),
    ],
)
def test_masks_addresses_and_account_fields(text: str, expected: str) -> None:
    assert redact_account_identity(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "No address here: rate limit reached, retry in 30s.",
        "npm i -g @openai/codex@0.160.1 react@18.3.1",
        "account_id=acct-synthetic-0001",  # a bare id is not bound to the telemetry field
        "scores = q @ k.transpose(-2, -1)",
        "+@pytest.mark.parametrize(",  # a decorator added in a diff
        "-@app.route('/')",
    ],
)
def test_leaves_other_text_unchanged(text: str) -> None:
    assert redact_account_identity(text) == text


@pytest.mark.parametrize("unit", ["a", "-", "a-", "+a"])
def test_long_runs_without_an_address_stay_linear(unit: str) -> None:
    # An unanchored local part rescans such a run from every position: 20 KB took 2.4 s.
    text = unit * (200_000 // len(unit))
    started = time.perf_counter()
    assert redact_account_identity(text) == text
    assert time.perf_counter() - started < 1.0
