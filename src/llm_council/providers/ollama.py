"""
Ollama Cloud provider adapter.

Ollama Cloud hosts large open-weight models behind an OpenAI-compatible endpoint
at ``https://ollama.com/v1``. The wire protocol is the standard OpenAI
``/chat/completions`` contract, so this adapter subclasses the OpenRouter
provider and reuses its request building, streaming, and response parsing. Only
the base URL, auth env var, request headers, and default model differ.

WHY THIS SEAT EXISTS
--------------------
The default council trio is ``claude`` (Anthropic) + ``codex`` (OpenAI) +
``sakana`` (Fugu). Fugu is itself a multi-agent orchestrator, so it adds
meta-review depth rather than an independent pretraining lineage -- see the
module docstring in ``sakana.py``. That leaves the council with two independent
lineages and no third. An Ollama seat running an open-weight reasoner
(``deepseek-v4-pro``, ``kimi-k3``, ``qwen3.5:122b``) is the first genuinely
uncorrelated lineage in the room, at zero marginal cost on a flat-rate Pro plan.

Council value scales with lineage independence, not seat count: two models that
share pretraining data and post-training methodology produce correlated errors,
which read as consensus and manufacture false confidence. So the model choice
matters more than the seat. ``gpt-oss:120b`` is a poor council pick despite being
Ollama's flagship -- it is OpenAI-lineage and would correlate with the ``codex``
seat, adding headcount without adding independence.

Intended use: an opt-in fourth seat
(``council run <subagent> --providers "claude,codex,sakana,ollama"``), NOT a
member of ``defaults.providers`` until a bake-off shows it produces real dissent
rather than agreement-noise.

Environment variables:
    OLLAMA_API_KEY:  Required. Create one at https://ollama.com/settings/keys.
    OLLAMA_BASE_URL: Optional. Override the API base URL -- e.g. point at a local
                     ``http://localhost:11434/v1`` server instead of the cloud.
    OLLAMA_MODEL:    Optional. Override the default model.

NOTE on structured output: reported as ``False`` deliberately, even though
Ollama's compatibility layer *does* accept ``response_format: json_schema``.
Council's heavy-schema subagents (``architect``, ``assessor``) pair a 14-15 field
schema with a 4000-token default output ceiling; a model that *complies* with the
schema gets truncated mid-object, its JSON fails to parse, and it is dropped from
the council silently -- ``success: true``, ``provider_errors: null``, empty
degradation report. Open-weight models are the most exposed to this, being weaker
at long strict-JSON adherence. Reporting ``False`` routes this seat through the
orchestrator's prompt-based JSON fallback (schema inlined into the prompt), which
is the same path that made the Sakana seat reliable.

This is a TEMPORARY guard, not a fix. A 2026-08-28 council review on these very
seats (kimi-k3 + deepseek-v4-pro + qwen3.5:397b) returned ``request_changes`` on
this decision, and its diagnosis is correct: the flag is then factually wrong, it
disables native structured output even for small schemas that would fit under the
ceiling, and it treats a symptom. A follow-up council (claude-opus-5 +
gpt-5.6-sol + deepseek-v4-pro, split 2-1 to keep False) converged on the sharper
point: a docstring calling the guard temporary does not make it temporary. Half
that gap is now closed in code -- see ``OllamaOutputTruncatedError``, which turns
a truncated response into a visible seat failure. The remaining fix is upstream
and outside this adapter's scope: derive the output budget from schema size.
False remains the safer default here only until that lands, because
True today would ship a seat that silently vanishes on exactly the heavy-schema
runs that matter most. Revisit with per-model json_schema compliance data, and
pass ``--max-tokens 24000`` on structured subagents regardless.

NOTE on ``reasoning_effort``: no sanitizing shim is needed here, unlike Fugu.
``ReasoningConfig.effort`` is typed ``Literal["low", "medium", "high", "none"]``
and Ollama's ``/v1/chat/completions`` accepts ``{low, medium, high, max, none}``
-- a strict superset -- so the base class pass-through is already safe.

NOTE on ``tool_choice``: Ollama's compatibility layer does not support it, so
``tool_use`` is reported ``False``. Council drafts do not request tools, and the
base ``_build_request_body`` only emits ``tool_choice`` when one is set.
``cache_control`` is model-prefix gated in the base class and never fires for
Ollama model tags.
"""

from __future__ import annotations

import contextlib
import os
import time
from typing import TYPE_CHECKING, ClassVar

import httpx

from llm_council.providers.base import DoctorResult, GenerateResponse, ProviderCapabilities
from llm_council.providers.openrouter import OpenRouterProvider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from llm_council.providers.base import GenerateRequest

# Truncation length for the response body appended to a re-raised HTTPStatusError.
# Bodies are typically short JSON error objects; 500 chars keeps the message readable
# in logs/degradation reports while still capturing the actual rejection reason.
_ERROR_BODY_MAX_CHARS = 500


class OllamaOutputTruncatedError(RuntimeError):
    """Raised when Ollama stops generating because the token budget ran out.

    Exists so a truncated answer becomes a VISIBLE seat failure rather than a
    plausible-looking partial draft that gets critiqued and scored as if complete.
    """


class OllamaProvider(OpenRouterProvider):
    """Ollama Cloud OpenAI-compatible provider adapter."""

    name: ClassVar[str] = "ollama"
    capabilities: ClassVar[ProviderCapabilities] = ProviderCapabilities(
        streaming=True,
        tool_use=False,  # /v1 compat layer does not support tool_choice
        structured_output=False,  # deliberate: prompt-based JSON fallback (see module docstring)
        multimodal=False,
        max_tokens=None,  # Varies by model
    )

    BASE_URL = "https://ollama.com/v1"
    DEFAULT_MODEL = "deepseek-v4-pro"
    ENV_API_KEY = "OLLAMA_API_KEY"
    # Generic per-seat model override, deliberately None on the pinned seats
    # below -- a single shared env var would collapse them all onto one model.
    ENV_MODEL: ClassVar[str | None] = "OLLAMA_MODEL"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        env_model = os.environ.get(self.ENV_MODEL) if self.ENV_MODEL else None
        self._api_key = api_key or os.environ.get(self.ENV_API_KEY)
        self._base_url = base_url or os.environ.get("OLLAMA_BASE_URL", self.BASE_URL)
        self._default_model = default_model or env_model or self.DEFAULT_MODEL
        self._http_client = http_client

    def _get_headers(self) -> dict[str, str]:
        """Build request headers (plain OpenAI bearer auth; no OpenRouter extras)."""
        if not self._api_key:
            raise ValueError(
                "Ollama API key not configured. "
                "Set the OLLAMA_API_KEY environment variable or pass api_key. "
                "Create a key at https://ollama.com/settings/keys."
            )
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def generate(
        self, request: GenerateRequest
    ) -> GenerateResponse | AsyncIterator[GenerateResponse]:
        """Generate a response, surfacing the Ollama HTTP error body on failure.

        ``httpx.HTTPStatusError.__str__`` never includes the response body, so a
        rejected-parameter or rate-limited 4xx would otherwise surface downstream as an
        opaque status code -- indistinguishable from any other failure, and silently
        degraded out of the council. Ollama Pro enforces undisclosed 5-hour session and
        weekly caps, so a quota rejection is a realistic failure mode here and its body
        text is the only place the reason appears. Only the non-streaming path is
        wrapped: a streaming request surfaces errors lazily during iteration, past
        this method own try/except.
        """
        try:
            response = await super().generate(request)
        except httpx.HTTPStatusError as exc:
            body_text = exc.response.text[:_ERROR_BODY_MAX_CHARS]
            raise httpx.HTTPStatusError(
                f"{exc}\nResponse body: {body_text}",
                request=exc.request,
                response=exc.response,
            ) from exc

        # Turn truncation into an explicit failure. A response cut off at the token
        # ceiling is still a well-formed HTTP 200 carrying a partial answer, so
        # without this it flows onward and gets critiqued and scored as though the
        # seat had finished -- the silent-degradation shape a 2026-08-28 council of
        # claude + gpt-5.6-sol + deepseek-v4-pro unanimously flagged. Streaming
        # responses are async iterators whose finish_reason is not knowable here.
        if isinstance(response, GenerateResponse) and response.finish_reason == "length":
            raise OllamaOutputTruncatedError(
                f"Ollama stopped at the token ceiling (finish_reason='length') for model "
                f"{response.model or request.model or self._default_model!r}; the draft is "
                f"incomplete. Raise --max-tokens (24000 is the standing default for "
                f"structured subagents) or shorten the task."
            )
        return response

    async def doctor(self) -> DoctorResult:
        """Health check against the Ollama Cloud models endpoint."""
        start_time = time.time()

        if not self._api_key:
            return DoctorResult(
                ok=False,
                message="OLLAMA_API_KEY environment variable not set",
                details={"error": "missing_api_key"},
            )

        try:
            client = await self._get_client()
            response = await client.get(
                f"{self._base_url}/models",
                headers=self._get_headers(),
            )
            response.raise_for_status()
            latency_ms = (time.time() - start_time) * 1000
            return DoctorResult(
                ok=True,
                message="Ollama Cloud API is accessible",
                latency_ms=latency_ms,
                details={"models_available": True},
            )
        except httpx.HTTPStatusError as e:
            latency_ms = (time.time() - start_time) * 1000
            return DoctorResult(
                ok=False,
                message=f"API error: {e.response.status_code}",
                latency_ms=latency_ms,
                details={
                    "status_code": e.response.status_code,
                    "body": e.response.text[:_ERROR_BODY_MAX_CHARS],
                },
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return DoctorResult(
                ok=False,
                message=f"Connection error: {str(e)}",
                latency_ms=latency_ms,
                details={"error": str(e)},
            )


class _OllamaSeat(OllamaProvider):
    """Base for a single-model Ollama seat with its own council identity.

    WHY DISTINCT PROVIDER NAMES ARE REQUIRED
    ----------------------------------------
    The obvious way to run three Ollama models is
    ``--providers "ollama,ollama,ollama" --models "a,b,c"``. It does not work,
    and it fails *silently*. ``Orchestrator._resolve_model_overrides`` returns a
    dict keyed by provider name, so three identical keys overwrite each other and
    only the LAST model survives -- one seat, not three. ``--dry-run`` still
    prints all three providers and all three models, so the collapse is invisible
    at every point a caller would think to check.

    Registering each model under its own provider name sidesteps the collapse and
    buys correct ledger attribution as a side effect: artifacts and token
    accounting are keyed by provider, so three seats all named ``ollama`` would be
    indistinguishable in the record. With distinct names, a bake-off can say which
    model actually dissented.

    ``ENV_MODEL`` is None here on purpose: the shared ``OLLAMA_MODEL`` variable
    would otherwise override all three seats at once and reintroduce the very
    collapse these classes exist to prevent.
    """

    ENV_MODEL: ClassVar[str | None] = None


class OllamaKimiProvider(_OllamaSeat):
    """Moonshot Kimi seat on Ollama Cloud."""

    name: ClassVar[str] = "ollama-kimi"
    DEFAULT_MODEL = "kimi-k3"


class OllamaDeepSeekProvider(_OllamaSeat):
    """DeepSeek seat on Ollama Cloud."""

    name: ClassVar[str] = "ollama-deepseek"
    DEFAULT_MODEL = "deepseek-v4-pro"


class OllamaQwenProvider(_OllamaSeat):
    """Qwen seat on Ollama Cloud.

    Note the size suffix is mandatory: ``qwen3.5`` alone is not a cloud tag,
    unlike ``deepseek-v4-pro`` which aliases to its dated snapshot.
    """

    name: ClassVar[str] = "ollama-qwen"
    DEFAULT_MODEL = "qwen3.5:397b"


def _register() -> None:
    """Register the Ollama providers with the global registry."""
    from llm_council.providers.registry import get_registry

    registry = get_registry()
    for adapter in (
        OllamaProvider,
        OllamaKimiProvider,
        OllamaDeepSeekProvider,
        OllamaQwenProvider,
    ):
        with contextlib.suppress(ValueError):
            registry.register_provider(adapter.name, adapter)


_register()
