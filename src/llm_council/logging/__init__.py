"""
Secret-safe logging for LLM Council.

Implements a redaction pipeline to prevent credential and prompt leakage.
"""

from __future__ import annotations

import re

# Codex telemetry (RUST_LOG=info and up) names the signed-in account in these fields.
_ACCOUNT_FIELD = re.compile(r'(\buser\.(?:email|account_id)=)"?[^"\s]+"?')
# The lookbehind lets a match start only where a local part starts; without it a long
# run of word characters with no "@" is rescanned from every position (quadratic time).
# The local part needs a letter or digit, so a diff line such as `+@pytest.mark.parametrize(`
# is no address; leading punctuation (a diff's +/-) is kept. A letters-only top-level
# domain leaves version pins such as react@18.3.1 alone.
_EMAIL_ADDRESS = re.compile(r"(?<![\w.%+-])([.%+-]*)\w[\w.%+-]*@(?:[\w-]+\.)+[^\W\d_]{2,}")


def redact_account_identity(text: str) -> str:
    """Mask e-mail addresses and the account fields of Codex telemetry.

    Claude Code gives the model the signed-in account's address, and Codex prints it
    in its telemetry; neither may reach another stage, a provider or an error report.
    """
    text = _ACCOUNT_FIELD.sub(r"\1[REDACTED]", text)
    return _EMAIL_ADDRESS.sub(r"\1[REDACTED-email]", text)
