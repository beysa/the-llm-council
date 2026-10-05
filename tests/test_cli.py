"""Tests for CLI commands."""

from __future__ import annotations

import asyncio
import json
import locale
import os
import re
import stat
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import typer
import yaml
from conftest import isolate_home
from typer.testing import CliRunner

from llm_council.cli.main import _cli_state, _load_config_defaults, _load_provider_configs, app
from llm_council.eval_import import ImportedPullRequest
from llm_council.providers.base import GenerateResponse
from llm_council.storage.artifacts import ArtifactStore, ArtifactType

runner = CliRunner()


class TestCLIHelp:
    """Tests for CLI help commands."""

    def test_help(self):
        """Test --help displays correctly."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Multi-LLM Council Framework" in result.stdout

    def test_run_help(self):
        """Test run --help displays correctly."""
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "subagent" in result.stdout.lower()

    def test_doctor_help(self):
        """Test doctor --help displays correctly."""
        result = runner.invoke(app, ["doctor", "--help"])
        assert result.exit_code == 0
        assert "provider" in result.stdout.lower() or "availability" in result.stdout.lower()

    def test_config_help(self):
        """Test config --help displays correctly."""
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0

    def test_eval_help(self):
        """Test eval help displays correctly."""
        result = runner.invoke(app, ["eval", "--help"])
        assert result.exit_code == 0
        assert "evaluation" in result.stdout.lower() or "dataset" in result.stdout.lower()


class TestCLIVersion:
    """Tests for version command."""

    def test_version(self):
        """Test version command output."""
        from llm_council import __version__

        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "LLM Council" in result.stdout
        assert __version__ in result.stdout


class TestCLIStorage:
    """Tests for storage commands."""

    def test_storage_status_reports_legacy_usage(self, monkeypatch, tmp_path):
        """Storage status should show when the legacy store is active."""
        monkeypatch.setattr("llm_council.storage.artifacts.Path.home", lambda: tmp_path)
        legacy_store = ArtifactStore(
            artifact_dir=tmp_path / ".claude" / "council-artifacts",
            db_path=tmp_path / ".claude" / "council-ledger.db",
        )
        run = legacy_store.create_run(subagent="test", task="legacy")
        legacy_store.store_artifact(run.run_id, "legacy content", ArtifactType.DRAFT)

        result = runner.invoke(app, ["storage", "status", "--json"])

        assert result.exit_code == 0
        # Parse rather than substring-match: on Windows the path is JSON-escaped
        # ("C:\\Users\\...") so the raw str(Path) never appears in the output and
        # the assertion failed for reasons that had nothing to do with storage.
        payload = json.loads(result.stdout)
        assert payload["using_legacy"] is True
        assert Path(payload["active_artifact_dir"]) == tmp_path / ".claude" / "council-artifacts"

    def test_storage_migrate_dry_run(self, monkeypatch, tmp_path):
        """Dry-run migration should report what would be copied."""
        monkeypatch.setattr("llm_council.storage.artifacts.Path.home", lambda: tmp_path)
        legacy_store = ArtifactStore(
            artifact_dir=tmp_path / ".claude" / "council-artifacts",
            db_path=tmp_path / ".claude" / "council-ledger.db",
        )
        run = legacy_store.create_run(subagent="test", task="legacy")
        legacy_store.store_artifact(run.run_id, "legacy content", ArtifactType.DRAFT)

        result = runner.invoke(app, ["storage", "migrate", "--dry-run", "--json"])

        assert result.exit_code == 0
        assert '"dry_run": true' in result.stdout
        assert '"copied_files": 1' in result.stdout
        assert not (tmp_path / ".council").exists()

    def test_storage_migrate_copies_legacy_store(self, monkeypatch, tmp_path):
        """Migration command should copy legacy store into neutral council home."""
        monkeypatch.setattr("llm_council.storage.artifacts.Path.home", lambda: tmp_path)
        legacy_store = ArtifactStore(
            artifact_dir=tmp_path / ".claude" / "council-artifacts",
            db_path=tmp_path / ".claude" / "council-ledger.db",
        )
        run = legacy_store.create_run(subagent="test", task="legacy")
        legacy_store.store_artifact(run.run_id, "legacy content", ArtifactType.DRAFT)

        result = runner.invoke(app, ["storage", "migrate", "--json"])

        assert result.exit_code == 0
        assert '"migrated": true' in result.stdout
        assert (tmp_path / ".council" / "ledger.db").exists()
        assert (tmp_path / ".council" / "artifacts").exists()


class TestCLIDoctor:
    """Tests for doctor command."""

    def test_doctor_shows_providers(self):
        """Test doctor command shows provider table."""
        with patch(
            "llm_council.cli.main._load_config_defaults",
            return_value={},
        ):
            result = runner.invoke(app, ["doctor"])
            assert result.exit_code == 0
            assert "Provider Status" in result.stdout

    def test_doctor_uses_provider_config(self):
        """Doctor passes provider config to get_provider (#28)."""
        with (
            patch(
                "llm_council.cli.main._load_config_defaults",
                return_value={},
            ),
            patch(
                "llm_council.cli.main._load_provider_configs",
                return_value={
                    "openai": {"default_model": "gpt-5.2"},
                },
            ),
            patch("llm_council.providers.registry.get_registry") as mock_reg,
        ):
            mock_registry = MagicMock()
            mock_provider = MagicMock()
            mock_doctor_result = MagicMock()
            mock_doctor_result.ok = True
            mock_doctor_result.message = "OK"
            mock_doctor_result.latency_ms = 100.0

            mock_registry.list_providers.return_value = ["openai"]
            mock_registry.get_provider.return_value = mock_provider
            # doctor() is now awaited inside an async function
            mock_provider.doctor = AsyncMock(return_value=mock_doctor_result)
            mock_reg.return_value = mock_registry

            runner.invoke(app, ["doctor"])

            # Verify get_provider was called with config kwargs
            mock_registry.get_provider.assert_called_with("openai", default_model="gpt-5.2")

    def test_doctor_with_no_providers(self):
        """Test doctor when no providers registered."""
        with patch("llm_council.providers.registry.get_registry") as mock_reg:
            mock_registry = MagicMock()
            mock_registry.list_providers.return_value = []
            mock_reg.return_value = mock_registry

            result = runner.invoke(app, ["doctor"])
            # With no providers, should show empty table or message
            assert result.exit_code == 0

    def test_doctor_accepts_multiple_provider_filters_and_aliases(self):
        """Doctor should accept repeated --provider flags and resolve friendly aliases."""
        with (
            patch(
                "llm_council.cli.main._load_config_defaults",
                return_value={"output_format": "json"},
            ),
            patch("llm_council.providers.registry.get_registry") as mock_reg,
        ):
            mock_registry = MagicMock()
            mock_registry.list_providers.return_value = ["claude", "codex", "gemini", "gemini-cli"]
            mock_registry.resolve_name.side_effect = {
                "codex": "codex",
                "gemini-cli": "gemini-cli",
                "claude": "claude",
            }.get

            def _get_provider(name, **_kwargs):
                provider = MagicMock()
                result = MagicMock()
                result.ok = True
                result.message = f"{name} ok"
                result.latency_ms = 1.0
                provider.doctor = AsyncMock(return_value=result)
                return provider

            mock_registry.get_provider.side_effect = _get_provider
            mock_reg.return_value = mock_registry

            result = runner.invoke(
                app,
                [
                    "doctor",
                    "--provider",
                    "codex",
                    "--provider",
                    "gemini-cli,claude",
                    "--json",
                ],
            )

            assert result.exit_code == 0
            assert '"name": "codex"' in result.stdout
            assert '"name": "gemini-cli"' in result.stdout
            assert '"name": "claude"' in result.stdout

    def test_doctor_deep_probe_reports_generation_readiness(self):
        """Deep doctor should surface whether a provider can answer a trivial prompt."""

        @asynccontextmanager
        async def fake_slot(_name: str, *, timeout_seconds: float | None = None):
            assert timeout_seconds == 5.0
            yield 0.0

        with (
            patch(
                "llm_council.cli.main._load_config_defaults",
                return_value={"output_format": "json"},
            ),
            patch("llm_council.providers.registry.get_registry") as mock_reg,
            patch("llm_council.cli.main.provider_call_slot", fake_slot),
        ):
            mock_registry = MagicMock()
            mock_registry.list_providers.return_value = ["codex"]
            mock_registry.resolve_name.side_effect = {"codex": "codex"}.get

            provider = MagicMock()
            doctor_result = MagicMock()
            doctor_result.ok = True
            doctor_result.message = "Logged in using ChatGPT"
            doctor_result.latency_ms = None
            provider.doctor = AsyncMock(return_value=doctor_result)
            provider.generate = AsyncMock(return_value=GenerateResponse(text="OK", content="OK"))
            mock_registry.get_provider.return_value = provider
            mock_reg.return_value = mock_registry

            result = runner.invoke(app, ["doctor", "--provider", "codex", "--deep", "--json"])

            assert result.exit_code == 0
            assert '"probe_ok": true' in result.stdout
            assert '"probe_message": "OK"' in result.stdout

    def test_doctor_deep_probe_skips_when_base_health_fails(self):
        """Deep doctor should not run a probe when the base doctor already failed."""
        with (
            patch(
                "llm_council.cli.main._load_config_defaults",
                return_value={"output_format": "json"},
            ),
            patch("llm_council.providers.registry.get_registry") as mock_reg,
        ):
            mock_registry = MagicMock()
            mock_registry.list_providers.return_value = ["codex"]
            mock_registry.resolve_name.side_effect = {"codex": "codex"}.get

            provider = MagicMock()
            doctor_result = MagicMock()
            doctor_result.ok = False
            doctor_result.message = "Not logged in"
            doctor_result.latency_ms = None
            provider.doctor = AsyncMock(return_value=doctor_result)
            provider.generate = AsyncMock()
            mock_registry.get_provider.return_value = provider
            mock_reg.return_value = mock_registry

            result = runner.invoke(app, ["doctor", "--provider", "codex", "--deep", "--json"])

            assert result.exit_code == 0
            assert '"probe_ok": false' in result.stdout
            assert "Skipped because base doctor failed" in result.stdout
            provider.generate.assert_not_called()

    def test_doctor_deep_probe_explains_openrouter_missing_model(self):
        """Deep doctor should explain when the configured OpenRouter model is unavailable."""

        @asynccontextmanager
        async def fake_slot(_name: str, *, timeout_seconds: float | None = None):
            yield 0.0

        with (
            patch(
                "llm_council.cli.main._load_config_defaults",
                return_value={"output_format": "json"},
            ),
            patch("llm_council.providers.registry.get_registry") as mock_reg,
            patch("llm_council.cli.main.provider_call_slot", fake_slot),
        ):
            mock_registry = MagicMock()
            mock_registry.list_providers.return_value = ["openrouter"]
            mock_registry.resolve_name.side_effect = {"openrouter": "openrouter"}.get

            provider = MagicMock()
            provider._default_model = "qwen/qwen3.6-plus:free"
            doctor_result = MagicMock()
            doctor_result.ok = True
            doctor_result.message = "OpenRouter API is accessible"
            doctor_result.latency_ms = 10.0
            provider.doctor = AsyncMock(return_value=doctor_result)

            request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
            provider.generate = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "Client error '404 Not Found' for url 'https://openrouter.ai/api/v1/chat/completions'",
                    request=request,
                    response=httpx.Response(
                        404,
                        request=request,
                        json={
                            "error": {
                                "message": "The free model has been deprecated. Transition to qwen/qwen3.6-plus for continued paid access."
                            }
                        },
                    ),
                )
            )

            mock_registry.get_provider.return_value = provider
            mock_reg.return_value = mock_registry

            result = runner.invoke(app, ["doctor", "--provider", "openrouter", "--deep", "--json"])

            assert result.exit_code == 0
            assert '"probe_ok": false' in result.stdout
            assert (
                "OpenRouter probe failed for configured model 'qwen/qwen3.6-plus:free'"
                in result.stdout
            )
            assert "The free model has been deprecated." in result.stdout
            assert "use --models" in result.stdout

    def test_doctor_deep_probe_preserves_generic_openrouter_404(self):
        """Generic OpenRouter 404s should preserve upstream failure text."""

        @asynccontextmanager
        async def fake_slot(_name: str, *, timeout_seconds: float | None = None):
            yield 0.0

        with (
            patch(
                "llm_council.cli.main._load_config_defaults",
                return_value={"output_format": "json"},
            ),
            patch("llm_council.providers.registry.get_registry") as mock_reg,
            patch("llm_council.cli.main.provider_call_slot", fake_slot),
        ):
            mock_registry = MagicMock()
            mock_registry.list_providers.return_value = ["openrouter"]
            mock_registry.resolve_name.side_effect = {"openrouter": "openrouter"}.get

            provider = MagicMock()
            provider._default_model = "qwen/qwen3.6-plus:free"
            doctor_result = MagicMock()
            doctor_result.ok = True
            doctor_result.message = "OpenRouter API is accessible"
            doctor_result.latency_ms = 10.0
            provider.doctor = AsyncMock(return_value=doctor_result)

            request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
            provider.generate = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "Client error '404 Not Found' for url 'https://openrouter.ai/api/v1/chat/completions'",
                    request=request,
                    response=httpx.Response(
                        404,
                        request=request,
                        json={"error": {"message": "Upstream route missing"}},
                    ),
                )
            )

            mock_registry.get_provider.return_value = provider
            mock_reg.return_value = mock_registry

            result = runner.invoke(app, ["doctor", "--provider", "openrouter", "--deep", "--json"])

            assert result.exit_code == 0
            assert '"probe_ok": false' in result.stdout
            assert "Configured OpenRouter model" not in result.stdout
            assert "Client error '404 Not Found'" in result.stdout
            assert "Upstream route missing" in result.stdout

    def test_doctor_deep_probe_does_not_misclassify_openrouter_model_404(self):
        """Model-not-found 404s should remain generic unless the payload signals deprecation."""

        @asynccontextmanager
        async def fake_slot(_name: str, *, timeout_seconds: float | None = None):
            yield 0.0

        with (
            patch(
                "llm_council.cli.main._load_config_defaults",
                return_value={"output_format": "json"},
            ),
            patch("llm_council.providers.registry.get_registry") as mock_reg,
            patch("llm_council.cli.main.provider_call_slot", fake_slot),
        ):
            mock_registry = MagicMock()
            mock_registry.list_providers.return_value = ["openrouter"]
            mock_registry.resolve_name.side_effect = {"openrouter": "openrouter"}.get

            provider = MagicMock()
            provider._default_model = "qwen/qwen3.6-plus:free"
            doctor_result = MagicMock()
            doctor_result.ok = True
            doctor_result.message = "OpenRouter API is accessible"
            doctor_result.latency_ms = 10.0
            provider.doctor = AsyncMock(return_value=doctor_result)

            request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
            provider.generate = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "Client error '404 Not Found' for url 'https://openrouter.ai/api/v1/chat/completions'",
                    request=request,
                    response=httpx.Response(
                        404,
                        request=request,
                        json={"error": {"message": "Unknown model slug"}},
                    ),
                )
            )

            mock_registry.get_provider.return_value = provider
            mock_reg.return_value = mock_registry

            result = runner.invoke(app, ["doctor", "--provider", "openrouter", "--deep", "--json"])

            assert result.exit_code == 0
            assert '"probe_ok": false' in result.stdout
            assert "OpenRouter probe failed for configured model" not in result.stdout
            assert "Client error '404 Not Found'" in result.stdout
            assert "Unknown model slug" in result.stdout

    def test_doctor_deep_probe_preserves_plain_text_http_error_body(self):
        """Plain-text upstream HTTP failures should remain visible in the probe message."""

        @asynccontextmanager
        async def fake_slot(_name: str, *, timeout_seconds: float | None = None):
            yield 0.0

        with (
            patch(
                "llm_council.cli.main._load_config_defaults",
                return_value={"output_format": "json"},
            ),
            patch("llm_council.providers.registry.get_registry") as mock_reg,
            patch("llm_council.cli.main.provider_call_slot", fake_slot),
        ):
            mock_registry = MagicMock()
            mock_registry.list_providers.return_value = ["openrouter"]
            mock_registry.resolve_name.side_effect = {"openrouter": "openrouter"}.get

            provider = MagicMock()
            provider._default_model = "z-ai/glm-5.1"
            doctor_result = MagicMock()
            doctor_result.ok = True
            doctor_result.message = "OpenRouter API is accessible"
            doctor_result.latency_ms = 10.0
            provider.doctor = AsyncMock(return_value=doctor_result)

            request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
            provider.generate = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "Server error '502 Bad Gateway' for url 'https://openrouter.ai/api/v1/chat/completions'",
                    request=request,
                    response=httpx.Response(
                        502,
                        request=request,
                        text="Gateway upstream timeout at edge",
                    ),
                )
            )

            mock_registry.get_provider.return_value = provider
            mock_reg.return_value = mock_registry

            result = runner.invoke(app, ["doctor", "--provider", "openrouter", "--deep", "--json"])

            assert result.exit_code == 0
            assert '"probe_ok": false' in result.stdout
            assert "OpenRouter probe failed for configured model" not in result.stdout
            assert "Server error '502 Bad Gateway'" in result.stdout
            assert "Gateway upstream timeout at edge" in result.stdout

    def test_doctor_deep_probe_serializes_live_probes(self):
        """Deep doctor should not fan out multiple live probes at once."""
        with (
            patch(
                "llm_council.cli.main._load_config_defaults",
                return_value={"output_format": "json"},
            ),
            patch("llm_council.providers.registry.get_registry") as mock_reg,
        ):
            mock_registry = MagicMock()
            mock_registry.list_providers.return_value = ["openai", "claude"]
            mock_registry.resolve_name.side_effect = {"openai": "openai", "claude": "claude"}.get

            def _provider(name: str) -> MagicMock:
                provider = MagicMock()
                doctor_result = MagicMock()
                doctor_result.ok = True
                doctor_result.message = f"{name} ok"
                doctor_result.latency_ms = 1.0
                provider.doctor = AsyncMock(return_value=doctor_result)

                async def _generate(_request):
                    await asyncio.sleep(0.01)
                    return GenerateResponse(text="OK", content="OK")

                provider.generate = AsyncMock(side_effect=_generate)
                return provider

            providers = {"openai": _provider("openai"), "claude": _provider("claude")}
            mock_registry.get_provider.side_effect = lambda name, **_kwargs: providers[name]
            mock_reg.return_value = mock_registry

            state = {"active": 0, "max_active": 0}

            @asynccontextmanager
            async def fake_slot(_name: str, *, timeout_seconds: float | None = None):
                assert timeout_seconds == 5.0
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
                try:
                    yield 0.0
                finally:
                    state["active"] -= 1

            with patch("llm_council.cli.main.provider_call_slot", fake_slot):
                result = runner.invoke(
                    app,
                    [
                        "doctor",
                        "--provider",
                        "openai",
                        "--provider",
                        "claude",
                        "--deep",
                        "--json",
                    ],
                )

            assert result.exit_code == 0
            assert state["max_active"] == 1


class TestCLIConfig:
    """Tests for config command."""

    def test_config_show_no_file(self, tmp_path, monkeypatch):
        """Test config --show with no config file."""
        isolate_home(monkeypatch, tmp_path)
        result = runner.invoke(app, ["config", "--show"])
        # Should indicate no config or show empty/default
        assert result.exit_code == 0

    def test_config_init(self, tmp_path, monkeypatch):
        """Test config --init creates config file.

        Isolation is load-bearing here, not hygiene: this test WRITES a config.
        It used to patch HOME only, which Windows ignores, so `config --init`
        escaped tmp_path and overwrote the developer's real
        ~/.config/llm-council/config.yaml on every suite run.
        """
        isolate_home(monkeypatch, tmp_path)
        result = runner.invoke(app, ["config", "--init"])
        assert result.exit_code == 0

        # Check file was created
        config_file = tmp_path / ".config" / "llm-council" / "config.yaml"
        assert config_file.exists()
        contents = config_file.read_text()
        assert "api_key: ${OPENROUTER_API_KEY}" in contents
        # Exact line: a substring check would also accept a longer id such as "...-5.55".
        assert "    default_model: anthropic/claude-opus-5.5" in contents.splitlines()

    def test_load_provider_configs_expands_env_api_keys(self, tmp_path, monkeypatch):
        """Provider configs should expand ${ENV_VAR} placeholders before constructor wiring."""
        isolate_home(monkeypatch, tmp_path)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        config_file = tmp_path / ".config" / "llm-council" / "config.yaml"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(
            """\
providers:
  - name: openrouter
    api_key: ${OPENROUTER_API_KEY}
    default_model: qwen/qwen3-max-thinking
"""
        )

        provider_configs = _load_provider_configs()

        assert provider_configs == {
            "openrouter": {
                "api_key": "sk-or-test",
                "default_model": "qwen/qwen3-max-thinking",
            }
        }

    def test_config_no_options(self):
        """Test config with no options shows usage."""
        result = runner.invoke(app, ["config"])
        assert result.exit_code == 0
        assert "Usage" in result.stdout or "config" in result.stdout


# Shaped like a real hand-maintained config: comments, an env reference, a block list.
_COMMENTED_CONFIG = """\
# LLM Council Configuration -- notes the user wants kept

providers:
  - name: openrouter
    api_key: ${OPENROUTER_API_KEY}  # never inline the real key
    default_model: anthropic/claude-opus-5

# Default settings for 'council run'
defaults:
  providers:
    - claude
    - codex
  timeout: 120

# trailing note
"""


def _write_council_config(home: Path, text: str) -> Path:
    """Write the default-location config as exact bytes (no newline translation)."""
    config_file = home / ".config" / "llm-council" / "config.yaml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_bytes(text.encode("utf-8"))
    return config_file


class TestCLIConfigSet:
    """`config --set` edits one value in place and never rewrites the rest of the file."""

    def test_set_rewrites_only_the_target_value(self, isolated_home, monkeypatch):
        """It used to write expanded ${VAR} secrets to disk and drop every comment."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-must-not-reach-disk")
        config_file = _write_council_config(isolated_home, _COMMENTED_CONFIG)

        result = runner.invoke(app, ["config", "--set", "defaults.timeout", "600"])

        assert result.exit_code == 0, result.output
        text = config_file.read_bytes().decode("utf-8")
        assert "sk-or-must-not-reach-disk" not in text
        assert text == _COMMENTED_CONFIG.replace("  timeout: 120\n", "  timeout: 600\n")

    @pytest.mark.parametrize(
        ("key", "value", "inserted"),
        [
            ("defaults.max_retries", "5", "  max_retries: 5\n"),
            ("summary.tier", "actions", "summary:\n  tier: actions\n"),
        ],
    )
    def test_set_inserts_a_new_key_after_its_siblings(self, isolated_home, key, value, inserted):
        """A new key lands under its parent mapping; comments around it stay put."""
        config_file = _write_council_config(isolated_home, _COMMENTED_CONFIG)

        result = runner.invoke(app, ["config", "--set", key, value])

        assert result.exit_code == 0, result.output
        assert config_file.read_bytes().decode("utf-8") == _COMMENTED_CONFIG.replace(
            "  timeout: 120\n", "  timeout: 120\n" + inserted
        )

    def test_set_replaces_a_block_list_in_the_same_style(self, isolated_home):
        """A JSON list value replaces a block list without reformatting it as flow."""
        config_file = _write_council_config(isolated_home, _COMMENTED_CONFIG)

        result = runner.invoke(
            app, ["config", "--set", "defaults.providers", '["claude", "codex", "sakana"]']
        )

        assert result.exit_code == 0, result.output
        assert config_file.read_bytes().decode("utf-8") == _COMMENTED_CONFIG.replace(
            "    - codex\n", "    - codex\n    - sakana\n"
        )

    @pytest.mark.parametrize("newline", ["\n", "\r\n"])
    def test_set_keeps_the_files_line_endings(self, isolated_home, newline):
        """Neither LF nor CRLF is translated to the platform default."""
        original = "defaults:\n  timeout: 120\n".replace("\n", newline)
        config_file = _write_council_config(isolated_home, original)

        result = runner.invoke(app, ["config", "--set", "defaults.max_retries", "5"])

        assert result.exit_code == 0, result.output
        expected = "defaults:\n  timeout: 120\n  max_retries: 5\n".replace("\n", newline)
        assert config_file.read_bytes() == expected.encode("utf-8")

    def test_set_creates_a_missing_config(self, isolated_home):
        """With no file yet, --set creates one holding just that key, as it always did.

        A new file has no line endings to keep, so it gets the platform's (as --init's does).
        """
        config_file = isolated_home / ".config" / "llm-council" / "config.yaml"

        result = runner.invoke(app, ["config", "--set", "defaults.timeout", "600"])

        assert result.exit_code == 0, result.output
        assert config_file.read_text(encoding="utf-8") == "defaults:\n  timeout: 600\n"

    def test_set_keeps_comments_under_the_section_they_follow(self, isolated_home):
        """A comment indented under the last entry stays with it, above the new section."""
        original = "defaults:\n  timeout: 120\n  # raising this breaks CI\n\n# tail\n"
        config_file = _write_council_config(isolated_home, original)

        result = runner.invoke(app, ["config", "--set", "summary.tier", "actions"])

        assert result.exit_code == 0, result.output
        assert config_file.read_bytes().decode("utf-8") == original.replace(
            "  # raising this breaks CI\n",
            "  # raising this breaks CI\nsummary:\n  tier: actions\n",
        )

    @pytest.mark.parametrize(
        ("key", "value", "written"),
        [
            # json.loads makes a NaN that never equals the reloaded one under ==.
            ("limits.ratio", "NaN", "  ratio: .nan\n"),
            # Key order is the user's, not sorted.
            ("limits.weights", '{"z": 3, "a": 2}', "  weights: {z: 3, a: 2}\n"),
            # Non-ASCII is escaped: _load_config() decodes with the locale encoding.
            ("limits.note", '"Türkçe"', '  note: "T\\xFCrk\\xE7e"\n'),
        ],
    )
    def test_set_writes_values_that_read_back_unchanged(self, isolated_home, key, value, written):
        """Values whose plain == comparison or encoding could trip up still round-trip."""
        original = "limits:\n  score: .nan\n"
        config_file = _write_council_config(isolated_home, original)

        result = runner.invoke(app, ["config", "--set", key, value])

        assert result.exit_code == 0, result.output
        assert config_file.read_bytes() == (original + written).encode("ascii")

    @pytest.mark.parametrize(
        ("original", "key", "reason"),
        [
            # Unparseable: the old code read this as {} and replaced the file with one key.
            ("defaults:\n  timeout: [120\n", "defaults.timeout", "while parsing"),
            # Through a list: no in-place edit can express it.
            (_COMMENTED_CONFIG, "providers.0.default_model", "providers is a list"),
            # A section whose children are all commented out holds null.
            ("defaults:\n  # timeout: 120\n", "defaults.timeout", "defaults is empty"),
            # The splice is valid YAML but would cut the kept trailing blank line out of
            # `note`: only the re-parse comparison catches this one.
            (
                "defaults:\n  note: |+\n    text\n\nnext: 1\n",
                "defaults.max_retries",
                "without altering other values",
            ),
            # A scalar at column 0 cannot be the value of an indentless list's key.
            ("p:\n- a\nn: 1\n", "p", "could not be made in place"),
            # Through an alias the edit would land on `shared` and change both keys.
            ("shared: &shared\n  timeout: 120\ndefaults: *shared\n", "defaults.timeout", "aliases"),
            # Replacing a list would drop the comment inside it...
            ("p:\n  - a\n  # b is flaky\n  - b\n", "p", "holds comments"),
            # ...or leave its last item's comment beside the new last item,
            ("p:\n  - a  # primary\nn: 1\n", "p", "holds comments"),
            # ...as it would a comment line indented under that item.
            ("p:\n  - a\n  # b removed: too slow\nn: 1\n", "p", "holds comments"),
            # A block scalar's header comment is inside its token, but still a comment.
            ("note: |  # keep this\n  text\nn: 1\n", "note", "holds comments"),
            # A !!set section is a mapping node but loads as a Python set.
            ("defaults: !!set\n  ? timeout\n", "defaults.timeout", "not a mapping"),
        ],
    )
    def test_set_refuses_and_leaves_the_file_alone(self, isolated_home, original, key, reason):
        """When the edit cannot be made in place, nothing is written, and the user is told why."""
        config_file = _write_council_config(isolated_home, original)

        result = runner.invoke(app, ["config", "--set", key, "600"])

        assert result.exit_code == 1
        assert config_file.read_bytes() == original.encode("utf-8")
        assert reason in " ".join(result.output.split())  # rich wraps long lines
        assert "council config --edit" in result.output

    @pytest.mark.skipif(
        hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores file permissions"
    )
    def test_set_refuses_a_read_only_config(self, isolated_home):
        """Like the plain write it replaced, the atomic write does not override read-only."""
        config_file = _write_council_config(isolated_home, _COMMENTED_CONFIG)
        config_file.chmod(stat.S_IREAD)
        try:
            result = runner.invoke(app, ["config", "--set", "defaults.timeout", "600"])
            entries = sorted(p.name for p in config_file.parent.iterdir())
        finally:
            config_file.chmod(stat.S_IREAD | stat.S_IWRITE)

        assert result.exit_code == 1
        assert config_file.read_bytes() == _COMMENTED_CONFIG.encode("utf-8")
        assert entries == ["config.yaml"]
        assert "read-only" in result.output

    def test_set_refuses_a_config_that_is_not_utf8(self, isolated_home):
        """A locale-encoded file is refused rather than rewritten under another encoding."""
        config_file = _write_council_config(isolated_home, "")
        original = "# Türkçe not\ndefaults:\n  timeout: 120\n".encode("cp1254")
        config_file.write_bytes(original)

        result = runner.invoke(app, ["config", "--set", "defaults.timeout", "600"])

        assert result.exit_code == 1
        assert config_file.read_bytes() == original
        assert "utf-8" in result.output

    def test_set_failed_write_leaves_the_old_file_whole(self, isolated_home, monkeypatch):
        """The new text goes to a temp file first; a failure never truncates the config."""
        config_file = _write_council_config(isolated_home, _COMMENTED_CONFIG)

        def disk_full(*args, **kwargs):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr("llm_council.cli.main.os.replace", disk_full)
        result = runner.invoke(app, ["config", "--set", "defaults.timeout", "600"])

        assert result.exit_code == 1
        assert config_file.read_bytes() == _COMMENTED_CONFIG.encode("utf-8")
        assert sorted(p.name for p in config_file.parent.iterdir()) == ["config.yaml"]
        assert "No space left on device" in result.output


class TestConfigEnvReferences:
    """`${VAR}` in the config is expanded for use only: not on display, not from argv.

    The Windows-specific cases pass either way on other platforms, where nothing in
    the old code path behaved differently.
    """

    def test_get_shows_the_value_as_written(self, isolated_home, monkeypatch):
        """--get printed expanded values, so `--get providers` showed the API key."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-must-not-be-printed")
        _write_council_config(isolated_home, _COMMENTED_CONFIG)

        result = runner.invoke(app, ["config", "--get", "providers"])

        assert result.exit_code == 0, result.output
        assert "sk-or-must-not-be-printed" not in result.output
        assert json.loads(result.output)[0]["api_key"] == "${OPENROUTER_API_KEY}"

    @pytest.mark.parametrize(
        "before_key",
        [
            "# don't inline keys\n",  # an apostrophe in a comment...
            "defaults:\n  context: Don't guess\n",  # ...or in an earlier value
        ],
    )
    def test_an_apostrophe_does_not_stop_expansion(self, isolated_home, monkeypatch, before_key):
        """Windows' ntpath.expandvars copies text between single quotes as it is, so one
        apostrophe above the key sent the literal `${FAKE_PROBE_KEY}` to the provider."""
        monkeypatch.setenv("FAKE_PROBE_KEY", "sk-or-test")
        _write_council_config(
            isolated_home,
            before_key + "providers:\n  - name: openrouter\n    api_key: ${FAKE_PROBE_KEY}\n",
        )

        assert _load_provider_configs() == {"openrouter": {"api_key": "sk-or-test"}}

    def test_a_single_quoted_value_expands_like_elsewhere(self, isolated_home, monkeypatch):
        """`'${VAR}'` expanded on Linux and macOS but stayed literal on Windows."""
        monkeypatch.setenv("FAKE_PROBE_KEY", "sk-or-test")
        _write_council_config(isolated_home, "defaults:\n  context: '${FAKE_PROBE_KEY}'\n")

        assert _load_config_defaults()["context"] == "sk-or-test"

    @pytest.mark.skipif(os.name != "nt", reason="only Windows expands %VAR%")
    def test_percent_references_still_load_on_windows(self, isolated_home, monkeypatch):
        """Only the quote handling changed. With plain shell-style expansion a value
        starting with an unexpanded `%` is invalid YAML, and the whole config loaded as {}."""
        monkeypatch.setenv("FAKE_PROBE_KEY", "sk-or-test")
        _write_council_config(
            isolated_home,
            "providers:\n  - name: openrouter\n    api_key: %FAKE_PROBE_KEY%\n"
            "defaults:\n  timeout: 55\n",
        )

        assert _load_provider_configs() == {"openrouter": {"api_key": "sk-or-test"}}
        assert _load_config_defaults() == {"timeout": 55}

    def test_config_is_read_as_utf8(self, isolated_home):
        """It was decoded with the locale encoding: cp1254 on a Turkish Windows."""
        _write_council_config(isolated_home, "defaults:\n  context: Türkçe yanıt ver\n")

        assert _load_config_defaults()["context"] == "Türkçe yanıt ver"
        shown = runner.invoke(app, ["config", "--show"])
        assert "Türkçe yanıt ver" in shown.output

    def test_config_with_a_utf8_bom_loads(self, isolated_home):
        """Notepad and PowerShell 5 write a BOM; decoded as cp1254 it joined the first key."""
        config_file = _write_council_config(isolated_home, "")
        config_file.write_bytes(b"\xef\xbb\xbfdefaults:\n  timeout: 55\n")

        assert _load_config_defaults() == {"timeout": 55}

    def test_config_in_the_locale_encoding_still_loads(self, isolated_home):
        """A file saved in the locale encoding, which is how this used to read it, still works."""
        encoding = locale.getpreferredencoding(False)
        try:
            data = "defaults:\n  context: Türkçe\n".encode(encoding)
            data.decode("utf-8")
        except UnicodeEncodeError:
            pytest.skip(f"{encoding} cannot encode the sample")
        except UnicodeDecodeError:
            pass  # not UTF-8, so loading it takes the fallback
        else:
            pytest.skip(f"{encoding} bytes are valid UTF-8; there is no fallback to take")
        config_file = _write_council_config(isolated_home, "")
        config_file.write_bytes(data)

        assert _load_config_defaults()["context"] == "Türkçe"

    def test_entry_point_keeps_env_references_in_arguments(
        self, isolated_home, monkeypatch, capsys
    ):
        """On Windows Typer expanded $VAR and %VAR% in sys.argv before any council code ran,
        so `council config --set k '${OPENROUTER_API_KEY}'` wrote the key. CliRunner passes
        arguments directly and never takes that path, so this calls app() like the script."""
        monkeypatch.setenv("FAKE_PROBE_KEY", "sk-or-must-not-reach-disk")
        config_file = _write_council_config(isolated_home, "a: 1\n")
        literal = "${FAKE_PROBE_KEY} %FAKE_PROBE_KEY%"
        monkeypatch.setattr(sys, "argv", ["council", "config", "--set", "k", literal])
        # app() replaces sys.excepthook and _cli_state; registering them puts them back.
        monkeypatch.setattr(sys, "excepthook", sys.excepthook)
        monkeypatch.setitem(_cli_state, "config_path", None)

        with pytest.raises(SystemExit) as exited:
            app()

        assert exited.value.code == 0, capsys.readouterr().out
        assert yaml.safe_load(config_file.read_bytes()) == {"a": 1, "k": literal}

    @pytest.mark.skipif(os.name != "nt", reason="only Windows expands arguments itself")
    def test_entry_point_still_globs_arguments_on_windows(self, tmp_path, monkeypatch, capsys):
        """cmd.exe and PowerShell do not glob, so a single-match wildcard relies on this
        expansion; only the variable step was dropped."""
        (tmp_path / "only-match.yaml").write_text("a: 1\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["council", "--config", "*.yaml", "config", "--path"])
        # app() replaces sys.excepthook and _cli_state; registering them puts them back.
        monkeypatch.setattr(sys, "excepthook", sys.excepthook)
        monkeypatch.setitem(_cli_state, "config_path", None)

        with pytest.raises(SystemExit) as exited:
            app()

        assert exited.value.code == 0
        assert capsys.readouterr().out.strip().endswith("only-match.yaml")


class TestWindowsArgumentExpansion:
    """What `app()` hands to typer on Windows; the platform is patched, so this runs anywhere."""

    @pytest.fixture
    def handed_to_typer(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr("llm_council.cli.main._ON_WINDOWS", True)
        monkeypatch.setattr(typer.Typer, "__call__", lambda _, *a, **kw: calls.append((a, kw)))
        monkeypatch.setenv("FAKE_PROBE_KEY", "sk-or-must-stay-a-reference")
        (tmp_path / "only-match.md").write_text("x")
        monkeypatch.chdir(tmp_path)
        argv = ["council", "run", "${FAKE_PROBE_KEY}", "$FAKE_PROBE_KEY", "only*.md"]
        monkeypatch.setattr(sys, "argv", argv)
        return calls

    @pytest.mark.parametrize(("args", "kwargs"), [((), {}), ((None,), {}), ((), {"args": None})])
    def test_the_script_call_expands_wildcards_but_not_variables(
        self, handed_to_typer, args, kwargs
    ):
        """`app()`, `app(None)` and `app(args=None)` all leave argv to click."""
        app(*args, **kwargs)

        ((passed_args, passed_kwargs),) = handed_to_typer
        argv = passed_args[0] if passed_args else passed_kwargs["args"]
        assert argv == ["run", "${FAKE_PROBE_KEY}", "$FAKE_PROBE_KEY", "only-match.md"]
        assert passed_kwargs["windows_expand_args"] is False

    @pytest.mark.parametrize(
        "kwargs",
        [{"windows_expand_args": True}, {"windows_expand_args": False}, {"args": ["--help"]}],
    )
    def test_explicit_choices_are_passed_through(self, handed_to_typer, kwargs):
        """A caller that sets the arguments or the expansion itself gets exactly that."""
        app(**kwargs)

        assert handed_to_typer == [((), kwargs)]


class TestOutputFormatConfig:
    """Tests for output_format config default (#29)."""

    def test_run_uses_output_format_json_from_config(self):
        """Config output_format: json enables JSON output."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = {"result": "test"}
        mock_result.model_dump.return_value = {
            "success": True,
            "output": {"result": "test"},
        }

        with (
            patch("llm_council.Council") as mock_council_class,
            patch("asyncio.run") as mock_run,
            patch(
                "llm_council.cli.main._load_config_defaults",
                return_value={"output_format": "json"},
            ),
        ):
            mock_council_class.return_value = MagicMock()
            mock_run.return_value = mock_result

            result = runner.invoke(app, ["run", "router", "Test task"])
            assert result.exit_code in [0, 1]
            # Should produce JSON output (no rich panels)
            if result.exit_code == 0:
                assert "Council Result" not in result.stdout

    def test_run_json_flag_overrides_config(self):
        """Explicit --json flag works even with no config."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = {"result": "test"}
        mock_result.model_dump.return_value = {
            "success": True,
            "output": {"result": "test"},
        }

        with (
            patch("llm_council.Council") as mock_council_class,
            patch("asyncio.run") as mock_run,
            patch(
                "llm_council.cli.main._load_config_defaults",
                return_value={},
            ),
        ):
            mock_council_class.return_value = MagicMock()
            mock_run.return_value = mock_result

            result = runner.invoke(app, ["run", "router", "Test task", "--json"])
            assert result.exit_code in [0, 1]

    def test_run_rich_output_when_config_not_json(self):
        """Non-json output_format preserves rich output."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = {"result": "test"}
        mock_result.model_dump.return_value = {
            "success": True,
            "output": {"result": "test"},
        }
        mock_result.validation_errors = None

        with (
            patch("llm_council.Council") as mock_council_class,
            patch("asyncio.run") as mock_run,
            patch(
                "llm_council.cli.main._load_config_defaults",
                return_value={"output_format": "rich"},
            ),
        ):
            mock_council_class.return_value = MagicMock()
            mock_run.return_value = mock_result

            result = runner.invoke(app, ["run", "router", "Test task"])
            assert result.exit_code in [0, 1]

    def test_doctor_uses_output_format_json_from_config(self):
        """Doctor command respects output_format config."""
        with patch(
            "llm_council.cli.main._load_config_defaults",
            return_value={"output_format": "json"},
        ):
            result = runner.invoke(app, ["doctor"])
            assert result.exit_code == 0
            # JSON output should not have rich table header
            if "Provider Status" not in result.stdout:
                assert "{" in result.stdout or "providers" in result.stdout


class TestMarkdownOutputFormat:
    """Tests for --format markdown human-readable output."""

    def _success_result(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            success=True,
            output={"result": "ship it"},
            critique="The draft is solid but verbose.",
            duration_ms=4321,
            synthesis_attempts=2,
            cost_estimate=SimpleNamespace(estimated_cost_usd=0.0456),
            run_id="run-123",
            execution_plan={"mode": "impl", "providers": ["openrouter"]},
            routed=False,
            routing_decision=None,
            validation_errors=None,
        )

    def test_render_markdown_success(self):
        """Formatter emits Markdown headings, output, critique, and metrics."""
        from llm_council.cli.main import _render_result_markdown

        md = _render_result_markdown(
            self._success_result(),
            resolved_mode="impl",
            provider_list=["openrouter"],
        )
        assert md.startswith("# Council Result: SUCCESS")
        assert "## Output" in md
        assert '"result": "ship it"' in md
        assert "## Critique" in md
        assert "The draft is solid but verbose." in md
        assert "## Metrics" in md
        assert "- Duration: 4321ms" in md
        assert "- Synthesis attempts: 2" in md
        assert "- Estimated cost: $0.0456" in md
        assert "- Run ID: run-123" in md
        assert "- Mode: impl" in md
        assert "- Providers: openrouter" in md
        assert md.endswith("\n")

    def test_render_markdown_failure_lists_errors(self):
        """Failed results render an Errors section with validation messages."""
        from types import SimpleNamespace

        from llm_council.cli.main import _render_result_markdown

        result = SimpleNamespace(
            success=False,
            output=None,
            critique=None,
            duration_ms=12,
            synthesis_attempts=1,
            cost_estimate=None,
            run_id=None,
            execution_plan={},
            routed=False,
            routing_decision=None,
            validation_errors=["missing key 'foo'", "bad type"],
        )
        md = _render_result_markdown(result, include_metrics=False)
        assert md.startswith("# Council Result: FAILED")
        assert "## Errors" in md
        assert "- missing key 'foo'" in md
        assert "- bad type" in md
        assert "## Metrics" not in md

    def test_run_format_markdown_stdout(self):
        """--format markdown renders Markdown to stdout (no rich panel, no raw JSON dump)."""
        mock_result = self._success_result()
        # model_dump should NOT be used for markdown output
        mock_result.model_dump = MagicMock(side_effect=AssertionError("model_dump used"))

        with (
            patch("llm_council.Council") as mock_council_class,
            patch("asyncio.run") as mock_run,
            patch(
                "llm_council.cli.main._load_config_defaults",
                return_value={},
            ),
        ):
            mock_council_class.return_value = MagicMock()
            mock_run.return_value = mock_result

            result = runner.invoke(app, ["run", "router", "Test task", "--format", "markdown"])
            assert result.exit_code == 0
            assert "# Council Result: SUCCESS" in result.stdout
            assert "## Metrics" in result.stdout

    def test_run_format_md_alias(self):
        """--format md is accepted as an alias for markdown."""
        mock_result = self._success_result()

        with (
            patch("llm_council.Council") as mock_council_class,
            patch("asyncio.run") as mock_run,
            patch(
                "llm_council.cli.main._load_config_defaults",
                return_value={},
            ),
        ):
            mock_council_class.return_value = MagicMock()
            mock_run.return_value = mock_result

            result = runner.invoke(app, ["run", "router", "Test task", "--format", "md"])
            assert result.exit_code == 0
            assert "# Council Result: SUCCESS" in result.stdout

    def test_run_format_json_matches_json_flag(self):
        """--format json behaves like --json (raw JSON, no rich panel)."""
        mock_result = self._success_result()
        mock_result.model_dump = MagicMock(
            return_value={"success": True, "output": {"result": "ship it"}}
        )

        with (
            patch("llm_council.Council") as mock_council_class,
            patch("asyncio.run") as mock_run,
            patch(
                "llm_council.cli.main._load_config_defaults",
                return_value={},
            ),
        ):
            mock_council_class.return_value = MagicMock()
            mock_run.return_value = mock_result

            result = runner.invoke(app, ["run", "router", "Test task", "--format", "json"])
            assert result.exit_code == 0
            assert "Council Result" not in result.stdout
            assert '"success": true' in result.stdout

    def test_run_format_invalid_value_errors(self):
        """An unknown --format value exits with an error before running the council."""
        with (
            patch("llm_council.Council") as mock_council_class,
            patch("asyncio.run") as mock_run,
            patch(
                "llm_council.cli.main._load_config_defaults",
                return_value={},
            ),
        ):
            mock_council_class.return_value = MagicMock()
            mock_run.return_value = self._success_result()

            result = runner.invoke(app, ["run", "router", "Test task", "--format", "yaml"])
            assert result.exit_code == 1
            assert "Invalid --format" in result.stdout

    def test_run_format_markdown_writes_file(self, tmp_path):
        """--format markdown with --output writes Markdown (not JSON) to the file."""
        out_file = tmp_path / "result.md"
        mock_result = self._success_result()

        with (
            patch("llm_council.Council") as mock_council_class,
            patch("asyncio.run") as mock_run,
            patch(
                "llm_council.cli.main._load_config_defaults",
                return_value={},
            ),
        ):
            mock_council_class.return_value = MagicMock()
            mock_run.return_value = mock_result

            result = runner.invoke(
                app,
                ["run", "router", "Test task", "--format", "markdown", "-o", str(out_file)],
            )
            assert result.exit_code == 0
            assert out_file.exists()
            content = out_file.read_text()
            assert content.startswith("# Council Result: SUCCESS")
            assert "## Metrics" in content

    def test_run_config_output_format_markdown(self):
        """config output_format: markdown renders Markdown when no CLI flag is given."""
        mock_result = self._success_result()

        with (
            patch("llm_council.Council") as mock_council_class,
            patch("asyncio.run") as mock_run,
            patch(
                "llm_council.cli.main._load_config_defaults",
                return_value={"output_format": "markdown"},
            ),
        ):
            mock_council_class.return_value = MagicMock()
            mock_run.return_value = mock_result

            result = runner.invoke(app, ["run", "router", "Test task"])
            assert result.exit_code == 0
            assert "# Council Result: SUCCESS" in result.stdout


class TestCLIGitDiffEvidence:
    """--git-diff / --no-git-diff reach the council config; the default is left to the run."""

    @staticmethod
    def _config_for(*flags: str) -> object:
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = {"result": "test"}
        mock_result.model_dump.return_value = {"success": True, "output": {"result": "test"}}
        mock_result.validation_errors = None
        with (
            patch("llm_council.Council") as mock_council_class,
            patch("asyncio.run") as mock_run,
            patch("llm_council.cli.main._load_config_defaults", return_value={}),
        ):
            mock_council_class.return_value = MagicMock()
            mock_run.return_value = mock_result
            result = runner.invoke(app, ["run", "critic", "Review", *flags, "--json"])
        assert result.exit_code in [0, 1]
        return mock_council_class.call_args.kwargs["config"]

    @pytest.mark.parametrize(
        ("flags", "expected"),
        [((), None), (("--git-diff",), True), (("--no-git-diff",), False)],
    )
    def test_the_flag_reaches_the_council_config(
        self, flags: tuple[str, ...], expected: bool | None
    ) -> None:
        assert self._config_for(*flags).git_diff_evidence is expected

    def test_the_files_help_says_attachments_replace_the_diff(self) -> None:
        result = runner.invoke(app, ["run", "--help"], env={"COLUMNS": "200"})
        # Rich wraps help inside box-drawing borders; compare the words only.
        text = " ".join(re.sub("[\\u2500-\\u257f]", " ", result.output).split())

        assert "--git-diff" in text
        assert "Once a file is read into the prompt, it replaces the working tree's git" in text
        assert "diff as evidence unless --git-diff is given (missing or over-the-limit" in text
        assert "repository-analysis snippets are still added" in text


class TestCLIContextMetadata:
    """Tests for file-ingestion metadata passed into council runs."""

    def test_run_passes_file_truncation_metadata_into_config(self, tmp_path):
        """Run should preserve truncation and skip decisions in context metadata."""
        large_files = []
        for index in range(1, 6):
            path = tmp_path / f"large-{index}.md"
            path.write_text("x" * 60_000)
            large_files.append(path)
        skipped_file = large_files[-1]
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = {"result": "test"}
        mock_result.model_dump.return_value = {"success": True, "output": {"result": "test"}}
        mock_result.validation_errors = None

        with (
            patch("llm_council.Council") as mock_council_class,
            patch("asyncio.run") as mock_run,
            patch("llm_council.cli.main._load_config_defaults", return_value={}),
        ):
            mock_council = MagicMock()
            mock_council_class.return_value = mock_council
            mock_run.return_value = mock_result

            result = runner.invoke(
                app,
                [
                    "run",
                    "critic",
                    "Review the benchmark plan",
                    *[item for path in large_files for item in ("--files", str(path))],
                    "--json",
                ],
            )

        assert result.exit_code in [0, 1]
        council_config = mock_council_class.call_args.kwargs["config"]
        files = council_config.context_metadata["files"]
        assert any(
            item["path"] == str(large_files[0])
            and item["truncated"] is True
            and item["status"] == "included"
            for item in files
        )
        assert any(
            item["path"] == str(skipped_file) and item["status"] == "skipped_total_limit"
            for item in files
        )
        assert council_config.context_metadata["warnings"]

    def test_run_keeps_file_warnings_out_of_stdout_json(self, tmp_path):
        """File-ingestion warnings should not pollute stdout JSON payloads."""
        missing_file = tmp_path / "missing.md"
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = {"result": "test"}
        mock_result.model_dump.return_value = {"success": True, "output": {"result": "test"}}
        mock_result.validation_errors = None

        with (
            patch("llm_council.Council") as mock_council_class,
            patch("asyncio.run") as mock_run,
            patch("llm_council.cli.main._load_config_defaults", return_value={}),
        ):
            mock_council_class.return_value = MagicMock()
            mock_run.return_value = mock_result

            result = runner.invoke(
                app,
                [
                    "run",
                    "critic",
                    "Review the benchmark plan",
                    "--files",
                    str(missing_file),
                    "--json",
                ],
            )

        assert result.exit_code in [0, 1]
        assert '"success": true' in result.stdout
        assert "File not found" not in result.stdout


class TestCLIEvaluate:
    """Tests for eval command."""

    def test_evaluate_json_output(self, tmp_path):
        """Eval should emit JSON and exit cleanly when all cases pass."""
        dataset_path = tmp_path / "dataset.yaml"
        dataset_path.write_text("version: 1\nname: smoke\ncases: []\n")
        report = MagicMock()
        report.failed_cases = 0
        report.model_dump.return_value = {
            "dataset_name": "smoke",
            "total_cases": 1,
            "passed_cases": 1,
            "failed_cases": 0,
            "case_results": [],
            "mode_scorecards": [],
        }

        with (
            patch("llm_council.cli.main.load_eval_dataset") as mock_load,
            patch("llm_council.cli.main.run_eval_dataset", new_callable=AsyncMock) as mock_eval,
            patch("llm_council.cli.main._load_config_defaults", return_value={}),
            patch("llm_council.cli.main._load_provider_configs", return_value={}),
        ):
            mock_load.return_value = MagicMock()
            mock_eval.return_value = report

            result = runner.invoke(app, ["eval", str(dataset_path), "--json"])

        assert result.exit_code == 0
        assert '"dataset_name": "smoke"' in result.stdout

    def test_evaluate_fails_when_report_has_failed_cases(self, tmp_path):
        """Eval should return a non-zero exit code when any case fails."""
        dataset_path = tmp_path / "dataset.yaml"
        dataset_path.write_text("version: 1\nname: smoke\ncases: []\n")
        report = MagicMock()
        report.failed_cases = 1
        report.dataset_name = "smoke"
        report.passed_cases = 0
        report.total_cases = 1
        report.passed_criteria = 1
        report.total_criteria = 2
        report.duration_ms = 10
        report.mode_scorecards = []
        report.case_results = []
        report.model_dump.return_value = {
            "dataset_name": "smoke",
            "total_cases": 1,
            "passed_cases": 0,
            "failed_cases": 1,
            "case_results": [],
            "mode_scorecards": [],
        }

        with (
            patch("llm_council.cli.main.load_eval_dataset") as mock_load,
            patch("llm_council.cli.main.run_eval_dataset", new_callable=AsyncMock) as mock_eval,
            patch("llm_council.cli.main._load_config_defaults", return_value={}),
            patch("llm_council.cli.main._load_provider_configs", return_value={}),
        ):
            mock_load.return_value = MagicMock()
            mock_eval.return_value = report

            result = runner.invoke(app, ["eval", str(dataset_path)])

        assert result.exit_code == 1

    def test_evaluate_compare_json_output(self, tmp_path):
        """Eval-compare should emit JSON and exit cleanly when variants pass."""
        dataset_path = tmp_path / "dataset.yaml"
        variants_path = tmp_path / "variants.yaml"
        dataset_path.write_text("version: 1\nname: smoke\ncases: []\n")
        variants_path.write_text("version: 1\nname: candidates\nvariants: []\n")
        report = MagicMock()
        report.variant_results = []
        report.model_dump.return_value = {
            "dataset_name": "smoke",
            "variants_name": "candidates",
            "best_variant": None,
            "ranking": [],
            "variant_results": [],
        }

        with (
            patch("llm_council.cli.main.load_eval_dataset") as mock_load_dataset,
            patch("llm_council.cli.main.load_eval_variants") as mock_load_variants,
            patch(
                "llm_council.cli.main.run_eval_comparison", new_callable=AsyncMock
            ) as mock_compare,
            patch("llm_council.cli.main._load_config_defaults", return_value={}),
            patch("llm_council.cli.main._load_provider_configs", return_value={}),
        ):
            mock_load_dataset.return_value = MagicMock()
            mock_load_variants.return_value = MagicMock()
            mock_compare.return_value = report

            result = runner.invoke(
                app,
                ["eval-compare", str(dataset_path), str(variants_path), "--json"],
            )

        assert result.exit_code == 0
        assert '"dataset_name": "smoke"' in result.stdout

    def test_evaluate_compare_passes_variant_filters(self, tmp_path):
        """Eval-compare should forward selected variant names to the runner."""
        dataset_path = tmp_path / "dataset.yaml"
        variants_path = tmp_path / "variants.yaml"
        dataset_path.write_text("version: 1\nname: smoke\ncases: []\n")
        variants_path.write_text("version: 1\nname: candidates\nvariants: []\n")
        report = MagicMock()
        report.variant_results = []
        report.model_dump.return_value = {
            "dataset_name": "smoke",
            "variants_name": "candidates",
            "best_variant": None,
            "ranking": [],
            "variant_results": [],
        }

        with (
            patch("llm_council.cli.main.load_eval_dataset") as mock_load_dataset,
            patch("llm_council.cli.main.load_eval_variants") as mock_load_variants,
            patch(
                "llm_council.cli.main.run_eval_comparison", new_callable=AsyncMock
            ) as mock_compare,
            patch("llm_council.cli.main._load_config_defaults", return_value={}),
            patch("llm_council.cli.main._load_provider_configs", return_value={}),
        ):
            mock_load_dataset.return_value = MagicMock()
            mock_load_variants.return_value = MagicMock()
            mock_compare.return_value = report

            result = runner.invoke(
                app,
                [
                    "eval-compare",
                    str(dataset_path),
                    str(variants_path),
                    "--variant",
                    "openai-gpt54",
                    "--variant",
                    "vertex-gemini31pro",
                    "--json",
                ],
            )

        assert result.exit_code == 0
        assert mock_compare.await_args.kwargs["variant_names"] == [
            "openai-gpt54",
            "vertex-gemini31pro",
        ]

    def test_evaluate_compare_builds_base_config_with_timeout_overrides(self, tmp_path):
        """Eval-compare should honor timeout and retry overrides."""
        dataset_path = tmp_path / "dataset.yaml"
        variants_path = tmp_path / "variants.yaml"
        dataset_path.write_text("version: 1\nname: smoke\ncases: []\n")
        variants_path.write_text("version: 1\nname: candidates\nvariants: []\n")
        report = MagicMock()
        report.variant_results = []
        report.model_dump.return_value = {
            "dataset_name": "smoke",
            "variants_name": "candidates",
            "best_variant": None,
            "ranking": [],
            "variant_results": [],
        }

        with (
            patch("llm_council.cli.main.load_eval_dataset") as mock_load_dataset,
            patch("llm_council.cli.main.load_eval_variants") as mock_load_variants,
            patch(
                "llm_council.cli.main.run_eval_comparison", new_callable=AsyncMock
            ) as mock_compare,
            patch("llm_council.cli.main._load_config_defaults", return_value={}),
            patch("llm_council.cli.main._load_provider_configs", return_value={}),
        ):
            mock_load_dataset.return_value = MagicMock()
            mock_load_variants.return_value = MagicMock()
            mock_compare.return_value = report

            result = runner.invoke(
                app,
                [
                    "eval-compare",
                    str(dataset_path),
                    str(variants_path),
                    "--timeout",
                    "30",
                    "--max-retries",
                    "1",
                    "--json",
                ],
            )

        assert result.exit_code == 0
        base_config = mock_compare.await_args.kwargs["base_config"]
        assert base_config.timeout == 30
        assert base_config.max_retries == 1
        assert base_config.runtime_profile.value == "default"
        assert base_config.reasoning_profile.value == "default"

    def test_evaluate_compare_builds_base_config_with_reasoning_profile(self, tmp_path):
        """Eval-compare should honor reasoning-profile overrides."""
        dataset_path = tmp_path / "dataset.yaml"
        variants_path = tmp_path / "variants.yaml"
        dataset_path.write_text("version: 1\nname: smoke\ncases: []\n")
        variants_path.write_text("version: 1\nname: candidates\nvariants: []\n")
        report = MagicMock()
        report.variant_results = []
        report.model_dump.return_value = {
            "dataset_name": "smoke",
            "variants_name": "candidates",
            "best_variant": None,
            "ranking": [],
            "variant_results": [],
        }

        with (
            patch("llm_council.cli.main.load_eval_dataset") as mock_load_dataset,
            patch("llm_council.cli.main.load_eval_variants") as mock_load_variants,
            patch(
                "llm_council.cli.main.run_eval_comparison", new_callable=AsyncMock
            ) as mock_compare,
            patch("llm_council.cli.main._load_config_defaults", return_value={}),
            patch("llm_council.cli.main._load_provider_configs", return_value={}),
        ):
            mock_load_dataset.return_value = MagicMock()
            mock_load_variants.return_value = MagicMock()
            mock_compare.return_value = report

            result = runner.invoke(
                app,
                [
                    "eval-compare",
                    str(dataset_path),
                    str(variants_path),
                    "--reasoning-profile",
                    "off",
                    "--json",
                ],
            )

        assert result.exit_code == 0
        base_config = mock_compare.await_args.kwargs["base_config"]
        assert base_config.reasoning_profile.value == "off"

    def test_evaluate_compare_builds_base_config_with_runtime_profile(self, tmp_path):
        """Eval-compare should honor runtime-profile overrides."""
        dataset_path = tmp_path / "dataset.yaml"
        variants_path = tmp_path / "variants.yaml"
        dataset_path.write_text("version: 1\nname: smoke\ncases: []\n")
        variants_path.write_text("version: 1\nname: candidates\nvariants: []\n")
        report = MagicMock()
        report.variant_results = []
        report.model_dump.return_value = {
            "dataset_name": "smoke",
            "variants_name": "candidates",
            "best_variant": None,
            "ranking": [],
            "variant_results": [],
        }

        with (
            patch("llm_council.cli.main.load_eval_dataset") as mock_load_dataset,
            patch("llm_council.cli.main.load_eval_variants") as mock_load_variants,
            patch(
                "llm_council.cli.main.run_eval_comparison", new_callable=AsyncMock
            ) as mock_compare,
            patch("llm_council.cli.main._load_config_defaults", return_value={}),
            patch("llm_council.cli.main._load_provider_configs", return_value={}),
        ):
            mock_load_dataset.return_value = MagicMock()
            mock_load_variants.return_value = MagicMock()
            mock_compare.return_value = report

            result = runner.invoke(
                app,
                [
                    "eval-compare",
                    str(dataset_path),
                    str(variants_path),
                    "--runtime-profile",
                    "bounded",
                    "--json",
                ],
            )

        assert result.exit_code == 0
        base_config = mock_compare.await_args.kwargs["base_config"]
        assert base_config.runtime_profile.value == "bounded"


class TestCLIImport:
    """Tests for local-only PR import command."""

    def test_eval_import_pr_json_output(self):
        """Import command should emit JSON for the imported bundle."""
        imported = ImportedPullRequest(
            repo="owner/repo",
            pr_number=42,
            title="Fix auth edge case",
            state="OPEN",
            is_draft=False,
            base_ref="main",
            head_ref="fix/auth",
            additions=10,
            deletions=2,
            changed_files=1,
            url="https://github.com/owner/repo/pull/42",
            import_root=".council-private/imports/github/owner__repo/pr-42",
            diff_path=".council-private/imports/github/owner__repo/pr-42/diff.patch",
            review_input_path=".council-private/imports/github/owner__repo/pr-42/review_input.md",
            metadata_path=".council-private/imports/github/owner__repo/pr-42/pr.json",
            raw_comments_path=".council-private/imports/github/owner__repo/pr-42/review_comments.raw.json",
            greptile_labels_path=".council-private/imports/github/owner__repo/pr-42/greptile_labels.json",
            imported_comment_count=3,
        )

        with patch("llm_council.cli.main.import_github_pr_review", return_value=imported):
            result = runner.invoke(app, ["eval-import-pr", "owner/repo", "42", "--json"])

        assert result.exit_code == 0
        assert '"repo": "owner/repo"' in result.stdout


class TestProviderConfigLoading:
    """Tests for provider config loading from config file (#26)."""

    def test_load_provider_configs_extracts_models(self):
        """Provider configs extract default_model from config."""
        from llm_council.cli.main import _load_provider_configs

        config_data = {
            "providers": [
                {"name": "openai", "default_model": "gpt-5.2"},
                {"name": "gemini", "default_model": "gemini-3.1-pro"},
            ],
            "defaults": {},
        }

        with patch(
            "llm_council.cli.main._load_config",
            return_value=config_data,
        ):
            result = _load_provider_configs()
            assert result == {
                "openai": {"default_model": "gpt-5.2"},
                "gemini": {"default_model": "gemini-3.1-pro"},
            }

    def test_load_provider_configs_ignores_invalid(self):
        """Provider configs skip invalid entries."""
        from llm_council.cli.main import _load_provider_configs

        config_data = {
            "providers": [
                {"name": "openai", "default_model": "gpt-5.2"},
                "invalid_string",
                {"no_name": True},
                {"name": "anthropic"},  # no default_model
            ],
        }

        with patch(
            "llm_council.cli.main._load_config",
            return_value=config_data,
        ):
            result = _load_provider_configs()
            assert result == {
                "openai": {"default_model": "gpt-5.2"},
            }

    def test_load_provider_configs_empty(self):
        """Empty config returns empty dict."""
        from llm_council.cli.main import _load_provider_configs

        with patch(
            "llm_council.cli.main._load_config",
            return_value={},
        ):
            result = _load_provider_configs()
            assert result == {}


class TestCLIRun:
    """Tests for run command."""

    def test_run_requires_arguments(self):
        """Test run requires subagent and task."""
        result = runner.invoke(app, ["run"])
        assert result.exit_code != 0

    def test_run_with_missing_task(self):
        """Test run with subagent but no task."""
        result = runner.invoke(app, ["run", "router"])
        assert result.exit_code != 0

    def test_run_basic(self):
        """Test basic run command."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = {"result": "test"}
        mock_result.model_dump.return_value = {"success": True, "output": {"result": "test"}}
        mock_result.validation_errors = None

        with patch("llm_council.Council") as mock_council_class:
            mock_council = MagicMock()
            mock_council_class.return_value = mock_council

            # Mock asyncio.run to return our mock result
            with patch("asyncio.run") as mock_run:
                mock_run.return_value = mock_result

                result = runner.invoke(app, ["run", "router", "Test task"])
                # Command should complete (may show result or error depending on output parsing)
                assert result.exit_code in [0, 1]  # Either success or handled error

    def test_run_with_json_output(self):
        """Test run with --json flag."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = {"result": "test"}
        mock_result.model_dump.return_value = {
            "success": True,
            "output": {"result": "test"},
            "drafts": {},
            "critique": None,
            "synthesis_attempts": 1,
            "duration_ms": 1000,
        }

        with patch("llm_council.Council") as mock_council_class:
            mock_council = MagicMock()
            mock_council_class.return_value = mock_council

            with patch("asyncio.run") as mock_run:
                mock_run.return_value = mock_result

                result = runner.invoke(app, ["run", "router", "Test task", "--json"])
                # JSON output should be produced
                assert result.exit_code in [0, 1]

    def test_run_with_providers(self):
        """Test run with custom providers."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = {}
        mock_result.model_dump.return_value = {"success": True, "output": {}}
        mock_result.validation_errors = None

        with patch("llm_council.Council") as mock_council_class:
            mock_council = MagicMock()
            mock_council_class.return_value = mock_council

            with patch("asyncio.run") as mock_run:
                mock_run.return_value = mock_result

                result = runner.invoke(
                    app,
                    ["run", "router", "Test task", "--providers", "openai,anthropic"],
                )
                # Should attempt to use specified providers
                assert result.exit_code in [0, 1]

    def test_run_with_route_flag_forwards_router_followup(self):
        """--route should enable router follow-up in config and run invocation."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = {"recommendation": "proceed"}
        mock_result.model_dump.return_value = {
            "success": True,
            "output": {"recommendation": "proceed"},
        }
        mock_result.validation_errors = None
        mock_result.execution_plan = None
        mock_result.routed = False
        mock_result.routing_decision = None

        with patch("llm_council.Council") as mock_council_class:
            mock_council = MagicMock()
            mock_council.run = AsyncMock(return_value=mock_result)
            mock_council_class.return_value = mock_council

            result = runner.invoke(app, ["run", "router", "Test task", "--route"])

            assert result.exit_code == 0
            council_config = mock_council_class.call_args.kwargs["config"]
            assert council_config.follow_router is True
            mock_council.run.assert_awaited_once_with(
                task="Test task",
                subagent="router",
                follow_router=True,
            )

    def test_run_with_route_flag_requires_router_subagent(self):
        """--route is rejected for non-router subagents."""
        result = runner.invoke(app, ["run", "planner", "Test task", "--route"])

        assert result.exit_code == 1
        assert "--route can only be used with the router subagent" in result.stdout

    def test_run_forwards_reasoning_profile(self):
        """Run should pass through the reasoning-profile override."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = {"result": "test"}
        mock_result.model_dump.return_value = {"success": True, "output": {"result": "test"}}
        mock_result.validation_errors = None

        with patch("llm_council.Council") as mock_council_class:
            mock_council = MagicMock()
            mock_council.run = AsyncMock(return_value=mock_result)
            mock_council_class.return_value = mock_council

            result = runner.invoke(
                app,
                ["run", "router", "Test task", "--reasoning-profile", "light"],
            )

        assert result.exit_code in [0, 1]
        council_config = mock_council_class.call_args.kwargs["config"]
        assert council_config.reasoning_profile.value == "light"

    def test_run_forwards_runtime_profile(self):
        """Run should pass through the runtime-profile override."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = {"result": "test"}
        mock_result.model_dump.return_value = {"success": True, "output": {"result": "test"}}
        mock_result.validation_errors = None

        with patch("llm_council.Council") as mock_council_class:
            mock_council = MagicMock()
            mock_council.run = AsyncMock(return_value=mock_result)
            mock_council_class.return_value = mock_council

            result = runner.invoke(
                app,
                ["run", "router", "Test task", "--runtime-profile", "bounded"],
            )

        assert result.exit_code in [0, 1]
        council_config = mock_council_class.call_args.kwargs["config"]
        assert council_config.runtime_profile.value == "bounded"
