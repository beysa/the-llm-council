from __future__ import annotations

import pytest

from llm_council.engine.orchestrator import Orchestrator
from llm_council.protocol.types import CouncilConfig
from llm_council.providers.base import GenerateRequest, GenerateResponse, Message, ReasoningConfig
from llm_council.providers.ollama import (
    OllamaDeepSeekProvider,
    OllamaKimiProvider,
    OllamaOutputTruncatedError,
    OllamaProvider,
    OllamaQwenProvider,
)
from llm_council.providers.openrouter import OpenRouterProvider
from llm_council.providers.registry import get_registry

SEATS = [
    (OllamaKimiProvider, "ollama-kimi", "kimi-k3"),
    (OllamaDeepSeekProvider, "ollama-deepseek", "deepseek-v4-pro"),
    (OllamaQwenProvider, "ollama-qwen", "qwen3.5:397b"),
]


def test_registered_under_ollama() -> None:
    """The seat is discovered via the pyproject entry point, not by import.

    Nothing in the package imports provider modules directly, so the
    ``llm_council.providers`` entry point is the only discovery path. Adding the
    module without reinstalling leaves the provider invisible to `council run`.
    """
    assert "ollama" in get_registry().list_providers()
    assert type(get_registry().get_provider("ollama")) is OllamaProvider


def test_structured_output_is_disabled() -> None:
    """Guards the seat against silent elimination on heavy-schema subagents.

    Ollama's compat layer does accept ``response_format: json_schema``, so this
    False is a deliberate choice, not a capability gap. Council's architect and
    assessor schemas (14-15 fields) against the 4000-token default output ceiling
    truncate a *compliant* model mid-object; the JSON then fails to parse and the
    seat is dropped with ``success: true`` and no provider_errors. Reporting False
    routes through the orchestrator's prompt-based fallback instead -- the same
    path that made the Sakana seat reliable. Flip only with bake-off data.
    """
    assert OllamaProvider.capabilities.structured_output is False


def test_tool_use_is_disabled() -> None:
    """Ollama's /v1 compat layer does not support tool_choice."""
    assert OllamaProvider.capabilities.tool_use is False


def test_headers_are_plain_bearer_without_openrouter_extras() -> None:
    """The base class injects HTTP-Referer and X-Title; Ollama must not receive them."""
    headers = OllamaProvider(api_key="sk-test")._get_headers()

    assert headers == {
        "Authorization": "Bearer sk-test",
        "Content-Type": "application/json",
    }


def test_missing_api_key_raises_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    with pytest.raises(ValueError) as excinfo:
        OllamaProvider()._get_headers()

    message = str(excinfo.value)
    assert "OLLAMA_API_KEY" in message
    assert "ollama.com/settings/keys" in message


def test_defaults_target_ollama_cloud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    provider = OllamaProvider(api_key="sk-test")

    assert provider._base_url == "https://ollama.com/v1"
    assert provider._default_model == "deepseek-v4-pro"


def test_env_overrides_base_url_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """OLLAMA_BASE_URL lets the same adapter drive a local server instead of the cloud."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3.5:122b")

    provider = OllamaProvider(api_key="sk-test")

    assert provider._base_url == "http://localhost:11434/v1"
    assert provider._default_model == "qwen3.5:122b"


@pytest.mark.parametrize("effort", ["low", "medium", "high", "none"])
def test_reasoning_effort_passes_through_unmodified(effort: str) -> None:
    """Documents why this adapter needs no reasoning_effort shim, unlike Sakana.

    ``ReasoningConfig.effort`` is typed Literal["low","medium","high","none"] and
    Ollama accepts {low, medium, high, max, none} -- a strict superset. Fugu, by
    contrast, rejects sub-"high" values with an opaque 400 and needs clamping. If
    ReasoningConfig ever gains a value outside Ollama's set, this parametrize
    should fail to cover it and a clamp becomes necessary.
    """
    request = GenerateRequest(
        messages=[Message(role="user", content="hi")],
        reasoning=ReasoningConfig(enabled=True, effort=effort),
    )

    body = OllamaProvider(api_key="sk-test")._build_request_body(request)

    assert body["reasoning_effort"] == effort


def test_request_body_omits_openrouter_only_fields() -> None:
    """cache_control is model-prefix gated upstream and must never reach Ollama."""
    request = GenerateRequest(
        messages=[Message(role="user", content="hi")],
        max_tokens=24000,
    )

    body = OllamaProvider(api_key="sk-test")._build_request_body(request)

    assert body["model"] == "deepseek-v4-pro"
    assert body["max_tokens"] == 24000
    assert "cache_control" not in body
    assert "response_format" not in body
    assert "tool_choice" not in body


@pytest.mark.asyncio
async def test_doctor_reports_missing_key_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    result = await OllamaProvider().doctor()

    assert result.ok is False
    assert result.details["error"] == "missing_api_key"


@pytest.mark.parametrize(("adapter", "provider_name", "model"), SEATS)
def test_named_seats_are_registered_with_pinned_models(
    adapter: type[OllamaProvider], provider_name: str, model: str
) -> None:
    """Tags verified against the live /v1/models catalogue on 2026-08-28."""
    assert adapter.name == provider_name
    assert model == adapter.DEFAULT_MODEL
    assert type(get_registry().get_provider(provider_name)) is adapter


def test_seat_names_are_distinct() -> None:
    """The whole reason the seats exist -- see test_duplicate_provider_names_collapse."""
    names = [name for _, name, _ in SEATS]
    assert len(set(names)) == len(names)


@pytest.mark.parametrize(("adapter", "_name", "model"), SEATS)
def test_shared_env_model_cannot_collapse_the_seats(
    adapter: type[OllamaProvider],
    _name: str,
    model: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single OLLAMA_MODEL must not override every seat onto one model.

    The base adapter honours OLLAMA_MODEL so the generic ``ollama`` seat stays
    steerable. If the pinned seats inherited that, exporting the variable would
    silently point kimi, deepseek and qwen at the same model -- recreating the
    exact collapse these classes exist to prevent, but harder to spot because the
    provider names would still differ.
    """
    monkeypatch.setenv("OLLAMA_MODEL", "gpt-oss:20b")

    assert adapter.ENV_MODEL is None
    assert adapter(api_key="sk-test")._default_model == model
    # ...while the generic seat still honours it.
    assert OllamaProvider(api_key="sk-test")._default_model == "gpt-oss:20b"


def test_duplicate_provider_names_collapse_but_distinct_names_do_not() -> None:
    """Characterizes the upstream behaviour that forced the named-seat design.

    ``_resolve_model_overrides`` returns a dict keyed by provider name, so
    ``--providers "ollama,ollama,ollama" --models "a,b,c"`` keeps only the last
    model: two seats vanish with no error, and ``--dry-run`` still prints all
    three. If this test ever fails because duplicates stop collapsing, the named
    seats are no longer load-bearing and could be reconsidered.
    """

    def resolve(providers: list[str], models: list[str]) -> dict[str, str]:
        orch = object.__new__(Orchestrator)
        orch._config = CouncilConfig(providers=providers, models=models)
        orch._provider_names = providers
        orch._subagent_config = {}
        orch._resolved_mode = None
        return orch._resolve_model_overrides()

    models = ["kimi-k3", "deepseek-v4-pro", "qwen3.5:397b"]

    collapsed = resolve(["ollama", "ollama", "ollama"], models)
    assert collapsed == {"ollama": "qwen3.5:397b"}, "duplicates should collapse to the last model"

    distinct = resolve([name for _, name, _ in SEATS], models)
    assert distinct == {
        "ollama-kimi": "kimi-k3",
        "ollama-deepseek": "deepseek-v4-pro",
        "ollama-qwen": "qwen3.5:397b",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["length"])
async def test_truncated_response_raises_instead_of_being_scored(
    finish_reason: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A response cut off at the token ceiling must fail loudly, not flow onward.

    Truncation arrives as a well-formed HTTP 200 carrying a partial answer, so
    without this guard the fragment is critiqued and scored as if the seat had
    finished. Flagged unanimously by a 2026-08-28 council of claude + gpt-5.6-sol
    + deepseek-v4-pro reviewing this adapter.
    """

    async def fake_generate(self: object, request: GenerateRequest) -> GenerateResponse:
        return GenerateResponse(text="half an ans", finish_reason=finish_reason, model="kimi-k3")

    monkeypatch.setattr(OpenRouterProvider, "generate", fake_generate)

    with pytest.raises(OllamaOutputTruncatedError) as excinfo:
        await OllamaProvider(api_key="sk-test").generate(
            GenerateRequest(messages=[Message(role="user", content="hi")])
        )

    assert "finish_reason='length'" in str(excinfo.value)
    assert "--max-tokens" in str(excinfo.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["stop", None])
async def test_complete_response_passes_through(
    finish_reason: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The truncation guard must not fire on a normally-terminated response."""

    async def fake_generate(self: object, request: GenerateRequest) -> GenerateResponse:
        return GenerateResponse(text="a whole answer", finish_reason=finish_reason)

    monkeypatch.setattr(OpenRouterProvider, "generate", fake_generate)

    result = await OllamaProvider(api_key="sk-test").generate(
        GenerateRequest(messages=[Message(role="user", content="hi")])
    )

    assert isinstance(result, GenerateResponse)
    assert result.text == "a whole answer"
