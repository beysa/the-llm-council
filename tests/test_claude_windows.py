"""Claude Code adapter: fork-local Windows behaviour on top of upstream 0.8.4."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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
        # Point the machine-wide locations at empty test directories.
        monkeypatch.setenv("ProgramFiles", str(tmp_path / "pf"))
        monkeypatch.setenv("ProgramData", str(tmp_path / "pd"))

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
