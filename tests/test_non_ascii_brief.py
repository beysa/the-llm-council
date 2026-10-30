"""A non-ASCII brief reaches every phase as written, whatever the machine's codepage is.

On a Turkish Windows machine text read with no encoding decodes as cp1254. "bulunamadı" in a
brief reached every phase as "bulunamadÄ±", and a critique reported that as a defect in the
code under review. Nothing here needs such a machine: the autouse fixture makes any host
behave like one, and EncodingWarning reports the call itself.
"""

from __future__ import annotations

import builtins
import importlib
import json
import subprocess
import sys
import textwrap
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import yaml
from typer.testing import CliRunner

from llm_council.cli.main import app
from llm_council.engine.orchestrator import CouncilResult, Orchestrator
from llm_council.providers.base import (
    DoctorResult,
    GenerateRequest,
    GenerateResponse,
    Message,
    ProviderAdapter,
    ProviderCapabilities,
)
from llm_council.providers.cli import codex as codex_module
from llm_council.providers.cli.claude_code import ClaudeCodeCLIProvider
from llm_council.providers.cli.codex import CodexCLIProvider
from llm_council.providers.ollama import OllamaKimiProvider
from llm_council.providers.registry import get_registry
from llm_council.subagents import SUBAGENTS_DIR, list_subagents, load_subagent

cli = importlib.import_module("llm_council.cli.main")

# Every byte of these decodes under cp1254, so the old read garbled them without a sound.
BRIEF = "Kanıt: bulunamadı ş ğ, sınır ≤ 40 — son"
# cp1254 leaves 0x9d and 0x9e undefined: "Ş", "Ğ" and "”" stopped the old read instead.
UNDECODABLE = "BULUNAMADI: Ş Ğ “son”"

SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "description": BRIEF,
}
PHASES = ("draft", "critique", "synthesis")
ROUTES = ("-i file", "-i -", "argument", "-f file")


@pytest.fixture(autouse=True)
def turkish_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Text opened with no encoding decodes as cp1254 here, as it does on such a machine."""
    real_open = builtins.open

    def locale_open(
        file: Any,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
        **kwargs: Any,
    ) -> Any:
        if "b" not in mode and encoding in (None, "locale"):
            encoding = "cp1254"
        return real_open(file, mode, buffering, encoding, errors, newline, **kwargs)

    monkeypatch.setattr(builtins, "open", locale_open)
    monkeypatch.setattr(Path, "open", locale_open)


@pytest.fixture(autouse=True)
def no_user_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_load_config_defaults", lambda: {})
    monkeypatch.setattr(cli, "_load_provider_configs", lambda: {})


def _invoke(args: list[str], stdin: bytes | None = None) -> Any:
    # A piped stdin and stdout use the locale's codepage, so that is what the runner gets.
    return CliRunner(charset="cp1254").invoke(app, args, input=stdin)


def _route(route: str, tmp_path: Path, content: bytes, text: str) -> tuple[list[str], bytes | None]:
    """The arguments and stdin that hand the same brief to the CLI along one route."""
    brief = tmp_path / "brief.md"
    brief.write_bytes(content)
    return {
        "-i file": (["-i", str(brief)], None),
        "-i -": (["-i", "-"], content),
        "argument": ([text], None),
        "-f file": (["Assess the attached file.", "-f", str(brief)], None),
    }[route]


class EchoProvider(ProviderAdapter):
    """Answers every phase, so a run goes all the way through draft, critique and synthesis."""

    name: ClassVar[str] = "non-ascii-echo"
    capabilities: ClassVar[ProviderCapabilities] = ProviderCapabilities(
        streaming=False,
        tool_use=False,
        structured_output=True,
        multimodal=False,
        max_tokens=4096,
    )

    async def generate(
        self, request: GenerateRequest
    ) -> GenerateResponse | AsyncIterator[GenerateResponse]:
        return GenerateResponse(text='{"ok": true}')

    async def supports(self, capability: str) -> bool:
        return capability == "structured_output"

    async def doctor(self) -> DoctorResult:
        return DoctorResult(ok=True, message="ok")


@pytest.fixture
def phase_prompts(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """Each phase's messages, taken where the orchestrator hands them to an adapter."""
    monkeypatch.setitem(get_registry()._providers, EchoProvider.name, EchoProvider)
    seen: dict[str, list[str]] = {}
    call_provider = Orchestrator._call_provider

    async def recording(
        self: Orchestrator,
        provider_name: str,
        adapter: ProviderAdapter,
        request: GenerateRequest,
        **kwargs: Any,
    ) -> GenerateResponse:
        seen.setdefault(str(kwargs.get("phase")), []).append(
            "\n".join(str(message.content) for message in request.messages)
        )
        return await call_provider(self, provider_name, adapter, request, **kwargs)

    monkeypatch.setattr(Orchestrator, "_call_provider", recording)
    return seen


@pytest.mark.parametrize("text", [BRIEF, UNDECODABLE], ids=["garbled", "stopped-the-run"])
@pytest.mark.parametrize("route", ROUTES)
def test_every_phase_gets_the_brief_as_written(
    tmp_path: Path, phase_prompts: dict[str, list[str]], route: str, text: str
) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps(SCHEMA), encoding="utf-8")
    args, stdin = _route(route, tmp_path, text.encode("utf-8"), text)

    invoked = _invoke(
        [
            "run",
            "planner",
            "--mode",
            "assess",
            *args,
            "--providers",
            EchoProvider.name,
            "--schema",
            str(schema),
            "--no-artifacts",
            "--json",
        ],
        stdin,
    )

    assert invoked.exit_code == 0, invoked.output
    assert sorted(phase_prompts) == sorted(PHASES)
    for phase in PHASES:
        for prompt in phase_prompts[phase]:
            assert text in prompt, f"{phase} did not get the brief as written"


@pytest.mark.parametrize("route", ["-i file", "-i -"])
def test_a_byte_order_mark_is_not_part_of_the_task(tmp_path: Path, route: str) -> None:
    # PowerShell's Out-File writes one, and str.strip() does not take it off.
    args, stdin = _route(route, tmp_path, b"\xef\xbb\xbf" + BRIEF.encode("utf-8"), BRIEF)
    with patch("llm_council.Council") as council:
        council.return_value.run = AsyncMock(return_value=CouncilResult(success=True))
        invoked = _invoke(["run", "planner", *args, "--json"], stdin)
    assert invoked.exit_code == 0, invoked.output
    assert council.return_value.run.await_args.kwargs["task"] == BRIEF


@pytest.mark.parametrize("route", ["-i file", "-i -"])
def test_windows_line_endings_are_still_normalised(tmp_path: Path, route: str) -> None:
    args, stdin = _route(route, tmp_path, b"first\r\nsecond\r\n", "")
    with patch("llm_council.Council") as council:
        council.return_value.run = AsyncMock(return_value=CouncilResult(success=True))
        invoked = _invoke(["run", "planner", *args, "--json"], stdin)
    assert invoked.exit_code == 0, invoked.output
    assert council.return_value.run.await_args.kwargs["task"] == "first\nsecond"


@pytest.mark.parametrize(
    "content,line",
    [
        ("first\nKanıt: bulunamadı".encode("cp1254"), 2),
        ("first\nsecond\nbulunamadı".encode()[:-1], 3),
        ("bulunamadı\n".encode() + "yok: ı".encode("cp1254"), 2),
    ],
    ids=["saved-as-cp1254", "cut-mid-character", "mixed"],
)
@pytest.mark.parametrize("route", ["-i file", "-i -"])
def test_a_brief_that_is_not_utf8_is_refused_before_any_call(
    tmp_path: Path, route: str, content: bytes, line: int
) -> None:
    """A guess can mislead a whole council; a refusal costs nothing."""
    args, stdin = _route(route, tmp_path, content, "")
    with patch("llm_council.Council") as council:
        invoked = _invoke(["run", "planner", *args, "--json"], stdin)
    assert invoked.exit_code == 1
    payload = json.loads(invoked.stdout)
    assert payload["execution_status"] == "failed"
    assert f"Input is not UTF-8 ({args[1]}, line {line})" in payload["error"]
    council.assert_not_called()


def test_a_custom_schema_keeps_its_text(tmp_path: Path) -> None:
    schema = tmp_path / "schema.json"
    schema.write_bytes(json.dumps(SCHEMA, ensure_ascii=False).encode("utf-8"))
    with patch("llm_council.Council") as council:
        council.return_value.run = AsyncMock(return_value=CouncilResult(success=True))
        invoked = _invoke(["run", "planner", "task", "--schema", str(schema), "--json"])
    assert invoked.exit_code == 0, invoked.output
    assert council.call_args.kwargs["config"].output_schema["description"] == BRIEF


@pytest.mark.parametrize("name", sorted(list_subagents()))
def test_a_subagent_prompt_reads_as_it_is_written(name: str) -> None:
    # critic, drafter and synthesizer each hold an em dash in their draft system prompt.
    assert load_subagent(name) == yaml.safe_load((SUBAGENTS_DIR / f"{name}.yaml").read_bytes())


def test_no_loader_uses_the_locale_encoding(tmp_path: Path) -> None:
    """EncodingWarning reports the call itself, so a file that is pure ASCII today counts too."""
    tools = tmp_path / "tool_registry.yaml"
    tools.write_text("tools: {}\n", encoding="utf-8")
    code = textwrap.dedent(
        """
        import sys
        import warnings
        from pathlib import Path

        from llm_council.registry.tool_registry import ToolRegistry
        from llm_council.schemas import list_schemas, load_schema
        from llm_council.subagents import list_subagents, load_subagent

        warnings.simplefilter("error", EncodingWarning)
        for name in list_subagents():
            load_subagent(name)
        for name in list_schemas():
            load_schema(name)
        ToolRegistry().load_from_yaml(Path(sys.argv[1]))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-X", "warn_default_encoding", "-c", code, str(tools)],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr


def test_a_dry_run_can_echo_a_brief_its_console_cannot_show(tmp_path: Path) -> None:
    # A cp1254 pipe has no "≤". The garbled text used to fit; the real text must not end the run.
    brief = tmp_path / "brief.md"
    brief.write_bytes(BRIEF.encode("utf-8"))
    invoked = _invoke(["run", "planner", "-i", str(brief), "--dry-run"])
    assert invoked.exit_code == 0, invoked.output
    assert "bulunamadı" in invoked.stdout
    assert "\\u2264" in invoked.stdout


def test_a_paid_result_is_not_lost_to_its_console() -> None:
    # Markdown carries the critique as the model wrote it, and json.dumps does not touch it.
    result = CouncilResult(success=True, output={"verdict": "ok"}, critique=BRIEF)
    with patch("llm_council.Council") as council:
        council.return_value.run = AsyncMock(return_value=result)
        invoked = _invoke(["run", "planner", "task", "--format", "markdown"])
    assert invoked.exit_code == 0, invoked.output
    assert "bulunamadı" in invoked.stdout
    assert "\\u2264" in invoked.stdout


# ---------------------------------------------------------------- where each seat's bytes leave
REQUEST = GenerateRequest(
    messages=[Message(role="system", content=BRIEF), Message(role="user", content=BRIEF)]
)


async def test_the_claude_seat_gets_utf8_on_stdin_and_text_in_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answer = b'{"type":"result","subtype":"success","is_error":false,"result":"ok"}'
    run_cli = AsyncMock(return_value=(0, answer, b""))
    monkeypatch.setattr(ClaudeCodeCLIProvider, "_run_cli", run_cli)
    monkeypatch.setattr(
        ClaudeCodeCLIProvider,
        "_check_cli_capabilities",
        AsyncMock(return_value="2.1.301 (Claude Code)"),
    )
    monkeypatch.setattr(
        ClaudeCodeCLIProvider, "_assert_unmanaged_environment", staticmethod(lambda _env: None)
    )
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    await ClaudeCodeCLIProvider(cli_path="claude").generate(REQUEST)

    command = run_cli.await_args.args[0]
    assert run_cli.await_args.kwargs["input_bytes"] == BRIEF.encode("utf-8")
    assert command[command.index("--system-prompt") + 1].endswith(BRIEF)


async def test_the_codex_seat_gets_utf8_in_both_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "empty-codex-home"))
    monkeypatch.setenv("CODEX_API_KEY", "synthetic-unit-key")
    monkeypatch.setattr(
        CodexCLIProvider, "_verify_native_contract", AsyncMock(return_value="codex-cli 0.160.1")
    )
    monkeypatch.setattr(
        CodexCLIProvider, "_generation_catalog", AsyncMock(return_value="/synthetic/models.json")
    )
    monkeypatch.setattr(codex_module, "terminate_process_tree", AsyncMock())
    seen: dict[str, bytes] = {}

    async def spawn(*args: Any, **kwargs: Any) -> Any:
        seen["stdin"] = kwargs["stdin"].read()
        option = next(arg for arg in args if str(arg).startswith("model_instructions_file="))
        seen["instructions"] = Path(json.loads(option.split("=", 1)[1])).read_bytes()
        Path(args[list(args).index("-o") + 1]).write_bytes(b"ANSWER")
        process = AsyncMock()
        process.returncode = 0
        process.communicate.return_value = (
            b'{"type":"item.completed","item":{"type":"agent_message","text":"ANSWER"}}\n'
            b'{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n',
            b"",
        )
        return process

    monkeypatch.setattr("asyncio.create_subprocess_exec", spawn)

    await CodexCLIProvider(cli_path="codex").generate(REQUEST)

    assert seen == {"stdin": BRIEF.encode("utf-8"), "instructions": BRIEF.encode("utf-8")}


async def test_the_kimi_seat_gets_utf8_json() -> None:
    bodies: list[bytes] = []

    def answer(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
        await OllamaKimiProvider(api_key="synthetic", http_client=client).generate(REQUEST)

    body = json.loads(bodies[0].decode("utf-8"))
    assert [message["content"] for message in body["messages"]] == [BRIEF, BRIEF]
