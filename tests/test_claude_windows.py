"""Claude Code adapter: fork-local Windows behaviour on top of upstream 0.8.4."""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import io
import json
import logging
import os
import struct
import subprocess
import sys
import sysconfig
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from llm_council.providers.base import GenerateRequest, Message, ReasoningConfig
from llm_council.providers.cli import _compatibility
from llm_council.providers.cli import claude_code as claude_module
from llm_council.providers.cli.claude_code import ClaudeCodeCLIProvider


class TestTheOfficialBaseUrlIsNotAGateway:
    """A Claude Code host exports ANTHROPIC_BASE_URL=https://api.anthropic.com to children."""

    @pytest.mark.parametrize("url", ["https://api.anthropic.com", "https://api.anthropic.com/"])
    def test_the_default_endpoint_keeps_subscription_auth(self, url: str) -> None:
        assert ClaudeCodeCLIProvider._resolve_auth({"ANTHROPIC_BASE_URL": url}) == "subscription"

    def test_any_other_endpoint_is_still_rejected(self) -> None:
        with pytest.raises(ValueError, match="gateway"):
            ClaudeCodeCLIProvider._resolve_auth({"ANTHROPIC_BASE_URL": "https://proxy.invalid"})

    def test_an_api_key_gateway_is_unchanged(self) -> None:
        environment = {"ANTHROPIC_API_KEY": "k", "ANTHROPIC_BASE_URL": "https://proxy.invalid"}
        assert ClaudeCodeCLIProvider._resolve_auth(environment) == "api_key"


def test_windows_system_variables_reach_the_child() -> None:
    env = ClaudeCodeCLIProvider()._get_minimal_env(
        {"SystemRoot": r"C:\Windows", "COMSPEC": "cmd.exe", "UNRELATED": "x"}
    )
    assert env == {"SystemRoot": r"C:\Windows", "COMSPEC": "cmd.exe"}


@pytest.mark.skipif(sys.platform != "win32", reason="Windows managed-policy locations")
class TestWindowsManagedPolicy:
    @staticmethod
    def _environment(tmp_path: Path) -> dict[str, str]:
        home = tmp_path / "home"
        home.mkdir()
        return {"USERPROFILE": str(home)}

    @pytest.fixture(autouse=True)
    def _no_machine_policy(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Point the machine-wide locations at empty test directories, and no opt-out.
        monkeypatch.setenv("ProgramFiles", str(tmp_path / "pf"))
        monkeypatch.setenv("ProgramData", str(tmp_path / "pd"))
        monkeypatch.delenv("COUNCIL_CLAUDE_ALLOW_REMOTE_SETTINGS", raising=False)

    def test_an_unmanaged_machine_is_admitted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(claude_module.sys, "platform", "win32")
        import winreg

        def _missing(*_args: object) -> None:
            raise FileNotFoundError

        monkeypatch.setattr(winreg, "OpenKey", _missing)
        ClaudeCodeCLIProvider._assert_unmanaged_environment(self._environment(tmp_path))

    @pytest.mark.parametrize("where", ["config", "program-files"])
    def test_a_policy_file_rejects(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
    ) -> None:
        environment = self._environment(tmp_path)
        if where == "config":
            target = Path(environment["USERPROFILE"]) / ".claude" / "managed-settings.json"
        else:
            target = tmp_path / "pf" / "ClaudeCode" / "managed-settings.json"
        target.parent.mkdir(parents=True)
        target.write_text("{}", encoding="utf-8")
        with pytest.raises(RuntimeError, match="managed-policy configuration present"):
            ClaudeCodeCLIProvider._assert_unmanaged_environment(environment)

    def test_a_policy_registry_key_rejects(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import winreg

        monkeypatch.setattr(winreg, "OpenKey", lambda *_args: object())
        monkeypatch.setattr(winreg, "CloseKey", lambda _key: None)
        with pytest.raises(RuntimeError, match="managed-policy configuration present"):
            ClaudeCodeCLIProvider._assert_unmanaged_environment(self._environment(tmp_path))

    # Claude Code (2.1.319) saves `{}` to remote-settings.json when its settings endpoint
    # answers 204/404, i.e. no remote settings are configured, and removes it at login/logout.
    @staticmethod
    def _remote_settings(
        environment: dict[str, str], content: bytes | None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Write remote-settings.json (None: a directory of that name); no registry policy."""
        import winreg

        def _missing(*_args: object) -> None:
            raise FileNotFoundError

        monkeypatch.setattr(winreg, "OpenKey", _missing)
        target = Path(environment["USERPROFILE"]) / ".claude" / "remote-settings.json"
        target.parent.mkdir(parents=True)
        if content is None:
            target.mkdir()
        else:
            target.write_bytes(content)

    @pytest.mark.parametrize("content", [b"{}", b"{}\r\n"], ids=["as-written", "newline"])
    def test_the_empty_remote_settings_sentinel_is_admitted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: bytes
    ) -> None:
        environment = self._environment(tmp_path)
        self._remote_settings(environment, content, monkeypatch)
        ClaudeCodeCLIProvider._assert_unmanaged_environment(environment)

    @pytest.mark.parametrize(
        "content",
        [b'{"hooks":{"SessionStart":[]}}', b"", b"[]", b"\xef\xbb\xbf{}", b"{}" + b" " * 64, None],
        ids=["settings", "empty-file", "not-an-object", "bom", "oversized", "directory"],
    )
    def test_any_other_remote_settings_still_rejects(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: bytes | None
    ) -> None:
        environment = self._environment(tmp_path)
        self._remote_settings(environment, content, monkeypatch)
        with pytest.raises(
            RuntimeError,
            match=r"present in remote-settings\.json \(not Claude Code's empty sentinel\)$",
        ):
            ClaudeCodeCLIProvider._assert_unmanaged_environment(environment)

    def test_a_link_named_remote_settings_still_rejects(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        environment = self._environment(tmp_path)
        sentinel = tmp_path / "elsewhere.json"
        sentinel.write_bytes(b"{}")
        link = Path(environment["USERPROFILE"]) / ".claude" / "remote-settings.json"
        link.parent.mkdir(parents=True)
        try:
            link.symlink_to(sentinel)
        except OSError:
            pytest.skip("creating a symbolic link needs Developer Mode or the privilege")
        with pytest.raises(RuntimeError, match="not Claude Code's empty sentinel"):
            ClaudeCodeCLIProvider._assert_unmanaged_environment(environment)

    @pytest.mark.parametrize(
        "change,refusal",
        [
            ("removed", None),  # Claude Code's login reset: the file is absent, so admitted
            ("settings", "not Claude Code's empty sentinel"),
            (
                "grown",
                "not Claude Code's empty sentinel",
            ),  # bigger than lstat said: never read past the cap
            ("unreadable", "could not be verified"),
        ],
    )
    def test_a_change_between_lstat_and_read(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        change: str,
        refusal: str | None,
    ) -> None:
        environment = self._environment(tmp_path)
        self._remote_settings(environment, b"{}", monkeypatch)
        opened = Path.open

        def change_first(path: Path, *args: Any, **kwargs: Any) -> Any:
            if path.name == "remote-settings.json":
                if change == "removed":
                    path.unlink()
                elif change == "unreadable":
                    raise PermissionError("synthetic read denied")
                else:
                    with open(path, "wb") as handle:  # the builtin, not the patched Path.open
                        handle.write(
                            b'{"hooks":{}}' if change == "settings" else b"{}" + b" " * 1000
                        )
            return opened(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", change_first)
        if refusal is None:
            ClaudeCodeCLIProvider._assert_unmanaged_environment(environment)
        else:
            with pytest.raises(RuntimeError, match=refusal):
                ClaudeCodeCLIProvider._assert_unmanaged_environment(environment)

    def test_the_opt_out_admits_remote_settings_unread_and_warns_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Read from Council's own environment: the child's environment, checked as well,
        # never carries the variable.
        monkeypatch.setenv("COUNCIL_CLAUDE_ALLOW_REMOTE_SETTINGS", "1")
        claude_module._warn_remote_settings_unread.cache_clear()
        environment = self._environment(tmp_path)
        self._remote_settings(environment, b'{"hooks":{"SessionStart":[]}}', monkeypatch)
        with caplog.at_level(logging.WARNING, logger=claude_module.__name__):
            for _ in range(3):  # a request checks three times
                ClaudeCodeCLIProvider._assert_unmanaged_environment(environment)
        warnings = [r for r in caplog.records if "remote-settings.json" in r.getMessage()]
        assert len(warnings) == 1
        assert "COUNCIL_CLAUDE_ALLOW_REMOTE_SETTINGS=1" in warnings[0].getMessage()

    def test_the_opt_out_covers_no_other_location(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("COUNCIL_CLAUDE_ALLOW_REMOTE_SETTINGS", "1")
        environment = self._environment(tmp_path)
        target = Path(environment["USERPROFILE"]) / ".claude" / "managed-settings.json"
        target.parent.mkdir(parents=True)
        target.write_text("{}", encoding="utf-8")
        with pytest.raises(RuntimeError, match="managed-policy configuration present$"):
            ClaudeCodeCLIProvider._assert_unmanaged_environment(environment)

    @pytest.mark.parametrize("value", ["0", "true", "yes"])
    def test_only_the_value_1_opts_out(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("COUNCIL_CLAUDE_ALLOW_REMOTE_SETTINGS", value)
        environment = self._environment(tmp_path)
        self._remote_settings(environment, b'{"hooks":{}}', monkeypatch)
        with pytest.raises(RuntimeError, match="present in remote-settings"):
            ClaudeCodeCLIProvider._assert_unmanaged_environment(environment)


# ------------------------------------------------------------------ the npm claude.cmd shim
# What npm writes for a package bin that is a native executable (Claude Code 2.1.301).
# Run as a batch file, cmd.exe ends an argument at its first line break and drops the rest.
NPM_SHIM = (
    "@ECHO off\r\n"
    "GOTO start\r\n"
    ":find_dp0\r\n"
    "SET dp0=%~dp0\r\n"
    "EXIT /b\r\n"
    ":start\r\n"
    "SETLOCAL\r\n"
    "CALL :find_dp0\r\n"
    '"%dp0%\\node_modules\\@anthropic-ai\\claude-code\\bin\\claude.exe"   %*\r\n'
)
SHIM_COMMAND = '"%dp0%\\node_modules\\@anthropic-ai\\claude-code\\bin\\claude.exe"   %*'
# What npm writes for a node script (codex 0.160.1): it starts node, not one executable.
NODE_SCRIPT_SHIM = NPM_SHIM.replace(
    SHIM_COMMAND,
    'endLocal & goto #_undefined_# 2>NUL || title %COMSPEC% & "%_prog%"  '
    '"%dp0%\\node_modules\\@openai\\codex\\bin\\codex.js" %*',
)


def _npm_layout(root: Path, shim: str = NPM_SHIM, executable: bytes = b"") -> tuple[Path, Path]:
    """`root/claude.cmd` and the executable npm installed beside it."""
    target = root.joinpath("node_modules", "@anthropic-ai", "claude-code", "bin", "claude.exe")
    target.parent.mkdir(parents=True)
    target.write_bytes(executable)
    path = root / "claude.cmd"
    path.write_bytes(shim.encode("utf-8"))
    return path, target


@pytest.mark.parametrize(
    "shim",
    [
        NPM_SHIM,
        NPM_SHIM.replace("\r\n", "\n"),
        NPM_SHIM.replace("@ECHO off", "@echo OFF").replace("%dp0%", "%DP0%"),
        NPM_SHIM.replace('"   %*', '" %*'),
        NPM_SHIM.removesuffix("\r\n"),
    ],
    ids=["npm", "lf-endings", "letter-case", "one-space", "no-final-newline"],
)
def test_an_npm_shim_names_the_executable_it_runs(tmp_path: Path, shim: str) -> None:
    path, target = _npm_layout(tmp_path, shim)
    assert _compatibility.npm_shim_target(str(path)) == str(target)


@pytest.mark.parametrize(
    "shim",
    [
        NODE_SCRIPT_SHIM,
        NPM_SHIM.replace("SETLOCAL\r\n", "SETLOCAL\r\nSET NODE_OPTIONS=--inspect\r\n"),
        NPM_SHIM.replace('"   %*', '" --debug %*'),
        NPM_SHIM.replace("bin\\claude.exe", "%BIN%\\claude.exe"),
        NPM_SHIM.replace("   %*", ""),
        NPM_SHIM.replace("claude-code", "claude-cöde"),
    ],
    ids=["node-script", "extra-line", "extra-argument", "variable", "no-forwarding", "non-ascii"],
)
def test_any_other_batch_file_names_no_executable(tmp_path: Path, shim: str) -> None:
    path, _ = _npm_layout(tmp_path, shim)
    assert _compatibility.npm_shim_target(str(path)) is None


def test_a_shim_whose_executable_is_gone_names_none(tmp_path: Path) -> None:
    path, target = _npm_layout(tmp_path)
    target.unlink()
    assert _compatibility.npm_shim_target(str(path)) is None


def _request(system: str = "ROLE\n\nRULES") -> GenerateRequest:
    return GenerateRequest(
        messages=[Message(role="system", content=system), Message(role="user", content="USER")]
    )


def test_on_windows_the_shim_executable_is_spawned_not_cmd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, target = _npm_layout(tmp_path)
    provider = ClaudeCodeCLIProvider(cli_path=str(path))
    monkeypatch.setattr(claude_module.sys, "platform", "win32")
    cmd = provider._build_command(_request(), auth_source="subscription")
    assert cmd[0] == str(target)
    assert cmd[cmd.index("--system-prompt") + 1].endswith(" ROLE\n\nRULES")
    assert provider._cli_path == str(path)  # the selected entrypoint is what is reported


@pytest.mark.parametrize(
    "name,text",
    [
        ("claude.cmd", NODE_SCRIPT_SHIM),
        ("claude.bat", "@echo off\r\nclaude.exe %*\r\n"),
        ("CLAUDE.CMD", NPM_SHIM.replace("claude-code", "elsewhere")),
    ],
    ids=["node-script-shim", "wrapper", "executable-gone"],
)
async def test_on_windows_any_other_batch_file_is_refused_before_a_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, text: str
) -> None:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    provider = ClaudeCodeCLIProvider(cli_path=str(path))
    monkeypatch.setattr(claude_module.sys, "platform", "win32")
    monkeypatch.setattr(
        ClaudeCodeCLIProvider, "_assert_unmanaged_environment", staticmethod(lambda _env: None)
    )

    async def spawn(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("nothing may be spawned through cmd.exe")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    with (
        patch.dict(os.environ, {"PATH": os.environ.get("PATH", "")}, clear=True),
        pytest.raises(RuntimeError, match="cmd.exe") as caught,
    ):
        await provider.generate(_request())
    assert repr(str(path)) in str(caught.value)


def test_outside_windows_a_cmd_file_is_spawned_as_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = _npm_layout(tmp_path)
    provider = ClaudeCodeCLIProvider(cli_path=str(path))
    monkeypatch.setattr(claude_module.sys, "platform", "linux")
    assert provider._build_command(_request(), auth_source="subscription")[0] == str(path)


def _console_script_launcher() -> bytes | None:
    """pip's console-script launcher: an .exe that runs the zip appended to it."""
    if sysconfig.get_platform() == "win-arm64":
        name = "t64-arm.exe"
    else:
        name = "t64.exe" if struct.calcsize("P") == 8 else "t32.exe"
    spec = importlib.util.find_spec("pip")
    if spec and spec.origin:
        launcher = Path(spec.origin).parent / "_vendor" / "distlib" / name
        if launcher.is_file():
            return launcher.read_bytes()
    spec = importlib.util.find_spec("ensurepip")
    if spec and spec.origin:
        for wheel in Path(spec.origin).parent.glob("_bundled/pip-*.whl"):
            with zipfile.ZipFile(wheel) as archive, contextlib.suppress(KeyError):
                return archive.read(f"pip/_vendor/distlib/{name}")
    return None


# The stand-in answers the two probes and returns the arguments it received as its result.
STAND_IN = """
import json, sys
args = sys.argv[1:]
if args == ["--version"]:
    out = "2.1.301 (Claude Code)\\n"
elif args == ["--help"]:
    out = HELP
else:
    sys.stdin.buffer.read()
    out = json.dumps({"type": "result", "subtype": "success", "is_error": False,
                      "result": json.dumps(args), "usage": {"input_tokens": 1}})
sys.stdout.buffer.write(out.encode("utf-8"))
"""
STAND_IN_HELP = "Usage: claude [options]\n\nOptions:\n" + "".join(
    f"  {flag}  Stand-in option\n"
    for flag in (
        "-p, --print",
        "--bare",
        "--safe-mode",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-chrome",
        "--no-session-persistence",
        *(
            f"{flag} <value>"
            for flag in (
                "--output-format",
                "--model",
                "--setting-sources",
                "--tools",
                "--mcp-config",
                "--settings",
                "--permission-mode",
                "--permission-prompts",
                "--system-prompt",
                "--effort",
            )
        ),
    )
)


@pytest.mark.skipif(sys.platform != "win32", reason="cmd.exe runs the npm shim only on Windows")
async def test_the_whole_system_prompt_and_every_later_argument_reach_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _console_script_launcher()
    if launcher is None:
        pytest.skip("no console-script launcher (pip) to build a stand-in claude.exe")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("__main__.py", f"HELP = {STAND_IN_HELP!r}\n{STAND_IN}")
    python = f'"{sys.executable}"' if " " in sys.executable else sys.executable
    shim, executable = _npm_layout(
        tmp_path, executable=launcher + f"#!{python}\r\n".encode() + archive.getvalue()
    )
    home = tmp_path / "home"
    home.mkdir()
    import winreg

    def _missing(*_args: object) -> None:
        raise FileNotFoundError

    monkeypatch.setattr(winreg, "OpenKey", _missing)  # no machine policy for this run
    system = 'ROLE: drafter.\n\nRULES: keep 100% of %PATH% and "quotes" verbatim.\nEnd λ \\'
    request = GenerateRequest(
        messages=[Message(role="system", content=system), Message(role="user", content="USER")],
        model="claude-opus-5",
        reasoning=ReasoningConfig(enabled=True, effort="low"),  # --effort after the prompt
    )
    environment = {
        key: os.environ[key]
        for key in ("SystemRoot", "COMSPEC", "PATHEXT", "PATH", "TEMP", "TMP")
        if key in os.environ
    }
    environment.update(
        USERPROFILE=str(home), ProgramFiles=str(tmp_path / "pf"), ProgramData=str(tmp_path / "pd")
    )
    provider = ClaudeCodeCLIProvider(cli_path=str(shim))
    with patch.dict(os.environ, environment, clear=True):
        response = await provider.generate(request)
        sent = provider._build_command(request, auth_source="subscription")
        # The same arguments through the shim, as Council sent them before: cut at the break.
        via_cmd = subprocess.run(
            [str(shim), *sent[1:]], input=b"USER", capture_output=True, timeout=60, check=True
        )

    received = json.loads(response.text)
    value = f"{claude_module._ACCOUNT_PRIVACY_INSTRUCTION} {system}"
    assert received == sent[1:]
    assert received[received.index("--system-prompt") + 1] == value
    assert received[-2:] == ["--effort", "low"]
    assert response.raw["native_identity"] == {
        "cli_path": str(shim),
        "cli_realpath": os.path.realpath(executable),
        "cli_version": "2.1.301 (Claude Code)",
    }
    assert response.raw["cli_path"] == str(shim)
    cut = json.loads(json.loads(via_cmd.stdout)["result"])
    assert cut[cut.index("--system-prompt") + 1] == value.splitlines()[0]
    assert "--effort" not in cut
