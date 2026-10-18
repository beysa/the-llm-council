"""Codex CLI adapter (fork-local behaviour on top of upstream 0.8.4): ending the whole process
tree, sharing the ChatGPT sign-in, the implicit reasoning effort, removing the isolated home.

The process-tree tests start REAL processes. They never look up or end a process by pid or by
name: a worker proves it is alive by appending to a heartbeat file that only it writes, and it
exits by itself when a stop file appears (or after a hard time box). A job is only ever asked
about its own members. Every path the tests delete lives under pytest's `tmp_path`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import sys
import tempfile
import textwrap
import threading
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_council.providers.base import GenerateRequest, ReasoningConfig
from llm_council.providers.cli import codex as codex_module
from llm_council.providers.cli.codex import CodexCLIProvider

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Job Objects are Windows-only")


# --------------------------------------------------------------------------- process tree
GRANDCHILD = textwrap.dedent(
    """
    import sys, time, pathlib
    beat, stop = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    held = open(sys.argv[3], "a") if len(sys.argv) > 3 else None   # blocks deletion on Windows
    end = time.monotonic() + 40            # hard time box: never outlive the test run
    while time.monotonic() < end and not stop.exists():
        with beat.open("a") as fh:
            fh.write("x")
        time.sleep(0.05)
    """
)
# The launcher stands in for the `codex.CMD` shim: it starts the real worker and waits for it.
LAUNCHER = textwrap.dedent(
    """
    import subprocess, sys
    subprocess.run([sys.executable, "-c", sys.argv[1], sys.argv[2], sys.argv[3]])
    """
)


def _size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


async def _wait_until_beating(beat: Path, timeout: float = 15.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        first = _size(beat)
        await asyncio.sleep(0.3)
        if _size(beat) > first:
            return
    raise AssertionError("the worker never started beating")


async def _still_beating(beat: Path, settle: float = 1.0, window: float = 1.0) -> bool:
    await asyncio.sleep(settle)
    first = _size(beat)
    await asyncio.sleep(window)
    return _size(beat) > first


async def _eventually(condition: Callable[[], bool], timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        await asyncio.sleep(0.1)
    return condition()


@contextlib.asynccontextmanager
async def _tree(
    tmp_path: Path, *, suspended: bool
) -> AsyncIterator[tuple[asyncio.subprocess.Process, Path, Path]]:
    """A launcher -> worker tree whose cleanup does not depend on the code under test."""

    beat, stop = tmp_path / "beat.txt", tmp_path / "stop.flag"
    options = {"creationflags": codex_module._CREATE_SUSPENDED} if suspended else {}
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        LAUNCHER,
        GRANDCHILD,
        str(beat),
        str(stop),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        **options,
    )
    try:
        yield proc, beat, stop
    finally:
        stop.write_text("stop", encoding="utf-8")  # a running worker exits by itself
        if proc.returncode is None:
            with contextlib.suppress(OSError):
                proc.kill()  # also ends a launcher that is still suspended
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=5)
        assert not await _still_beating(beat, settle=0.5), "a test worker was left running"


def _active_in_job(job: int) -> int:
    """How many processes are alive in a job this test owns (asked of the job itself)."""

    import ctypes
    from ctypes import wintypes

    class _Accounting(ctypes.Structure):
        _fields_ = [
            ("TotalUserTime", ctypes.c_int64),
            ("TotalKernelTime", ctypes.c_int64),
            ("ThisPeriodTotalUserTime", ctypes.c_int64),
            ("ThisPeriodTotalKernelTime", ctypes.c_int64),
            ("TotalPageFaultCount", wintypes.DWORD),
            ("TotalProcesses", wintypes.DWORD),
            ("ActiveProcesses", wintypes.DWORD),
            ("TotalTerminatedProcesses", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.LPVOID,
    ]
    info = _Accounting()
    assert kernel32.QueryInformationJobObject(job, 1, ctypes.byref(info), ctypes.sizeof(info), None)
    return int(info.ActiveProcesses)


@windows_only
class TestWholeTreeIsEnded:
    @pytest.mark.asyncio
    async def test_terminating_the_job_ends_the_grandchild(self, tmp_path: Path) -> None:
        async with _tree(tmp_path, suspended=True) as (proc, beat, _stop):
            job = codex_module._adopt_into_job(proc)
            try:
                assert job, "a real child must end up in a job"
                await _wait_until_beating(beat)  # it was resumed, and the worker runs
                assert _active_in_job(job) >= 2, "launcher and worker are both members"

                codex_module._terminate_job(job)

                await asyncio.wait_for(proc.wait(), timeout=5)
                assert _active_in_job(job) == 0
                assert not await _still_beating(beat), "the worker survived the job"
            finally:
                codex_module._close_job(job)

    @pytest.mark.asyncio
    async def test_closing_the_job_handle_alone_ends_the_tree(self, tmp_path: Path) -> None:
        # What happens when a council process dies with the handle open.
        async with _tree(tmp_path, suspended=True) as (proc, beat, _stop):
            job = codex_module._adopt_into_job(proc)
            assert job
            try:
                await _wait_until_beating(beat)
            finally:
                codex_module._close_job(job)
            await asyncio.wait_for(proc.wait(), timeout=5)
            assert not await _still_beating(beat), "kill-on-close did not end the worker"

    @pytest.mark.asyncio
    async def test_killing_only_the_launcher_leaves_the_grandchild_running(
        self, tmp_path: Path
    ) -> None:
        # The defect being fixed, pinned: this is all `proc.kill()` ever did on Windows.
        async with _tree(tmp_path, suspended=False) as (proc, beat, _stop):
            await _wait_until_beating(beat)
            proc.kill()
            await proc.wait()
            assert await _still_beating(beat), "expected an orphan: the premise of the fix is gone"


@windows_only
class TestAChildIsNeverLeftSuspendedOrLoose:
    """Whatever goes wrong while adopting, the child is ended and the failure is loud."""

    @staticmethod
    async def _assert_ended_without_ever_running(
        proc: asyncio.subprocess.Process, beat: Path
    ) -> None:
        await asyncio.wait_for(proc.wait(), timeout=5)  # a suspended child would hang here
        assert _size(beat) == 0, "the child ran although it could not be confined"

    @pytest.mark.asyncio
    async def test_no_reachable_handle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # What a future Python that renames the private attributes would look like.
        monkeypatch.setattr(codex_module, "_own_process_handle", lambda _proc: None)
        async with _tree(tmp_path, suspended=True) as (proc, beat, _stop):
            with pytest.raises(RuntimeError, match="does not expose the child's process handle"):
                codex_module._adopt_into_job(proc)
            await self._assert_ended_without_ever_running(proc, beat)

    @pytest.mark.asyncio
    async def test_the_handle_lookup_itself_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(_proc: object) -> int:
            raise ValueError("transport changed shape")

        monkeypatch.setattr(codex_module, "_own_process_handle", _boom)
        async with _tree(tmp_path, suspended=True) as (proc, beat, _stop):
            with pytest.raises(RuntimeError, match="could not confine.*transport changed shape"):
                codex_module._adopt_into_job(proc)
            await self._assert_ended_without_ever_running(proc, beat)

    @pytest.mark.asyncio
    async def test_the_job_assignment_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # NULL is never a valid process handle, so AssignProcessToJobObject fails. Running
        # Codex without a job would be the original leak, so this must fail closed.
        monkeypatch.setattr(codex_module, "_own_process_handle", lambda _proc: 0)
        async with _tree(tmp_path, suspended=True) as (proc, beat, _stop):
            with pytest.raises(RuntimeError, match="could not confine"):
                codex_module._adopt_into_job(proc)
            await self._assert_ended_without_ever_running(proc, beat)

    @pytest.mark.asyncio
    async def test_the_resume_fails_after_a_successful_assignment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(codex_module, "_resume_process", lambda _handle: 0xC0000008)
        async with _tree(tmp_path, suspended=True) as (proc, beat, _stop):
            with pytest.raises(RuntimeError, match="NtResumeProcess failed.*0xc0000008"):
                codex_module._adopt_into_job(proc)
            await self._assert_ended_without_ever_running(proc, beat)

    @pytest.mark.asyncio
    async def test_an_interrupt_while_adopting_still_ends_the_child(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _interrupted(_handle: int) -> int:
            raise KeyboardInterrupt

        monkeypatch.setattr(codex_module, "_resume_process", _interrupted)
        async with _tree(tmp_path, suspended=True) as (proc, beat, _stop):
            with pytest.raises(KeyboardInterrupt):
                codex_module._adopt_into_job(proc)
            await self._assert_ended_without_ever_running(proc, beat)

    @pytest.mark.asyncio
    async def test_even_a_failing_import_is_inside_the_ownership_scope(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The very first statement after the platform gate: nothing may run between "the
        # child exists, suspended" and "somebody is answerable for ending it".
        monkeypatch.setitem(sys.modules, "ctypes", None)  # makes `import ctypes` raise
        async with _tree(tmp_path, suspended=True) as (proc, beat, _stop):
            with pytest.raises(RuntimeError, match="could not confine"):
                codex_module._adopt_into_job(proc)
            await self._assert_ended_without_ever_running(proc, beat)


class TestAdoptionRefusesWhatIsNotOurs:
    def test_a_fake_process_is_never_adopted(self) -> None:
        # Unit tests hand the adapter MagicMock processes with made-up pids. Opening a process
        # by such a pid could drag a stranger into a kill-on-close job.
        fake = MagicMock()
        fake.pid = 4
        assert codex_module._own_process_handle(fake) is None
        assert codex_module._adopt_into_job(fake) is None
        fake._transport.get_extra_info.assert_not_called()
        fake.kill.assert_not_called()

    def test_job_helpers_accept_no_job(self) -> None:
        codex_module._terminate_job(None)
        codex_module._close_job(None)


# --------------------------------------------------------------- the native call ends the tree
@windows_only
class TestRunCliEndsTheWholeTree:
    """The adapter's own process runner, not just the helpers: a deadline ends the grandchild."""

    @pytest.mark.asyncio
    async def test_a_deadline_ends_the_launcher_and_its_worker(self, tmp_path: Path) -> None:
        beat, stop = tmp_path / "beat.txt", tmp_path / "stop.flag"
        provider = CodexCLIProvider(cli_path="codex")
        loop = asyncio.get_running_loop()
        try:
            with pytest.raises(RuntimeError, match="timed out"):
                await provider._run_cli(
                    [sys.executable, "-c", LAUNCHER, GRANDCHILD, str(beat), str(stop)],
                    env=dict(os.environ),
                    cwd=str(tmp_path),
                    deadline=loop.time() + 3,
                )
            assert _size(beat) > 0, "the worker never ran, so nothing was proven"
            assert not await _still_beating(beat), "the worker outlived the call"
        finally:
            stop.write_text("stop", encoding="utf-8")


class TestSpawnOptions:
    @pytest.mark.asyncio
    async def test_suspended_only_with_the_native_spawn_on_windows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[dict[str, Any]] = []

        async def _spawn(*_args: object, **kwargs: Any) -> Any:
            seen.append(kwargs)
            proc = AsyncMock()
            proc.returncode = 0
            proc.communicate.return_value = (b"codex-cli 0.160.1\n", b"")
            return proc

        provider = CodexCLIProvider(cli_path="codex")
        monkeypatch.setattr(asyncio, "create_subprocess_exec", _spawn)
        # A substituted spawn may return an object that cannot be resumed: never suspend it.
        await provider._run_cli(["codex"], env={}, cwd=str(tmp_path), deadline=_far())
        monkeypatch.setattr(codex_module, "_NATIVE_SPAWN", _spawn)
        await provider._run_cli(["codex"], env={}, cwd=str(tmp_path), deadline=_far())

        assert "creationflags" not in seen[0]
        if sys.platform == "win32":
            assert seen[1]["creationflags"] == codex_module._CREATE_SUSPENDED
        else:
            assert "creationflags" not in seen[1]


def _far() -> float:
    return asyncio.get_running_loop().time() + 30


# ------------------------------------------------------------------- shared ChatGPT sign-in
CHATGPT_AUTH = {
    "auth_mode": "chatgpt",
    "tokens": {
        "id_token": "synthetic-id",
        "access_token": "synthetic-access",
        "refresh_token": "synthetic-refresh-original",
        "account_id": "synthetic-account",
    },
}


def _auth_text(refresh: str) -> str:
    data = json.loads(json.dumps(CHATGPT_AUTH))
    data["tokens"]["refresh_token"] = refresh
    return json.dumps(data)


@pytest.fixture
def real_sign_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A ChatGPT sign-in in a CODEX_HOME of its own, with no other auth source around."""
    root = tmp_path / "real-codex-home"
    root.mkdir()
    real = root / "auth.json"
    real.write_text(_auth_text("synthetic-refresh-original"), encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(root))
    for key in ("CODEX_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    return real


def _codex_saves(auth_file: Path, content: str) -> None:
    """What codex-rs FileAuthStorage::save does: truncate and rewrite in place."""
    with auth_file.open("w", encoding="utf-8") as fh:
        fh.write(content)


def _kept(real: Path) -> list[Path]:
    return sorted(real.parent.glob("auth.json.council-*"))


class TestTheSignInIsSharedNotCopied:
    """Codex rotates the ChatGPT refresh token on every refresh; a stale copy signs you out.

    Upstream copies auth.json into the isolated home. A refresh Codex saves there would die
    with the home while the real file kept a refresh token that is already spent, so this
    fork hard-links it instead and never writes the real file itself.
    """

    @staticmethod
    def _home(provider: CodexCLIProvider) -> Path:
        return Path(provider._create_isolated_cli_home(provider._resolve_runtime()))

    def test_a_refresh_codex_writes_reaches_the_real_file_at_once(self, real_sign_in: Path) -> None:
        provider = CodexCLIProvider(cli_path="codex")
        home = self._home(provider)
        try:
            _codex_saves(home / ".codex" / "auth.json", _auth_text("refreshed"))
            assert real_sign_in.read_text(encoding="utf-8") == _auth_text("refreshed")
        finally:
            provider._release_home(str(home))
        assert not home.exists()
        assert real_sign_in.read_text(encoding="utf-8") == _auth_text("refreshed")
        assert not _kept(real_sign_in)

    def test_codex_signing_out_in_the_isolated_home_leaves_the_real_file(
        self, real_sign_in: Path
    ) -> None:
        provider = CodexCLIProvider(cli_path="codex")
        home = self._home(provider)
        (home / ".codex" / "auth.json").unlink()
        provider._release_home(str(home))
        assert real_sign_in.read_text(encoding="utf-8") == _auth_text("synthetic-refresh-original")

    def test_a_sign_in_that_cannot_be_shared_is_refused_not_copied(
        self, real_sign_in: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        homes: list[Path] = []
        mkdtemp = tempfile.mkdtemp

        def _recording(*args: Any, **kwargs: Any) -> str:
            path = mkdtemp(*args, **kwargs)
            homes.append(Path(path))
            return path

        def _no_link(*_args: object, **_kwargs: object) -> None:
            raise OSError("cross-device link")

        monkeypatch.setattr(codex_module.tempfile, "mkdtemp", _recording)
        monkeypatch.setattr(codex_module.os, "link", _no_link)
        provider = CodexCLIProvider(cli_path="codex")
        with pytest.raises(RuntimeError, match="Refusing to\\s+copy it"):
            self._home(provider)
        assert homes and not any(home.exists() for home in homes), "the home was left behind"

    def test_a_filesystem_that_cannot_tell_files_apart_is_refused(
        self, real_sign_in: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(codex_module, "_file_identity", lambda _path: None)
        with pytest.raises(RuntimeError, match="cannot confirm"):
            self._home(CodexCLIProvider(cli_path="codex"))

    def test_a_sign_in_cut_short_is_reported_and_left_alone(
        self, real_sign_in: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = CodexCLIProvider(cli_path="codex")
        home = self._home(provider)
        _codex_saves(home / ".codex" / "auth.json", '{"tokens": ')
        with caplog.at_level("WARNING", logger="llm_council.providers.cli.codex"):
            provider._release_home(str(home))
        assert "codex login" in caplog.text
        assert real_sign_in.read_text(encoding="utf-8") == '{"tokens": '
        assert not _kept(real_sign_in)

    def test_a_real_file_replaced_elsewhere_keeps_the_calls_save_aside(
        self, real_sign_in: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = CodexCLIProvider(cli_path="codex")
        home = self._home(provider)
        # The desktop app saves by temp file + rename: the two names stop sharing a file.
        replacement = real_sign_in.with_name("auth.json.tmp")
        replacement.write_text(_auth_text("desktop"), encoding="utf-8")
        os.replace(replacement, real_sign_in)
        _codex_saves(home / ".codex" / "auth.json", _auth_text("call"))
        with caplog.at_level("WARNING", logger="llm_council.providers.cli.codex"):
            provider._release_home(str(home))
        assert real_sign_in.read_text(encoding="utf-8") == _auth_text("desktop")
        kept = _kept(real_sign_in)
        assert len(kept) == 1 and kept[0].read_text(encoding="utf-8") == _auth_text("call")

    def test_a_sign_out_elsewhere_is_never_undone(self, real_sign_in: Path) -> None:
        provider = CodexCLIProvider(cli_path="codex")
        home = self._home(provider)
        real_sign_in.unlink()
        _codex_saves(home / ".codex" / "auth.json", _auth_text("call"))
        provider._release_home(str(home))
        assert not real_sign_in.exists()
        assert not _kept(real_sign_in)

    def test_api_key_auth_is_still_copied(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "apikey-home"
        root.mkdir()
        text = json.dumps({"auth_mode": "apikey", "OPENAI_API_KEY": "synthetic-key"})
        (root / "auth.json").write_text(text, encoding="utf-8")
        monkeypatch.setenv("CODEX_HOME", str(root))
        for key in ("CODEX_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
            monkeypatch.delenv(key, raising=False)
        provider = CodexCLIProvider(cli_path="codex")
        home = self._home(provider)
        try:
            isolated = home / ".codex" / "auth.json"
            assert isolated.read_text(encoding="utf-8") == text
            assert codex_module._file_identity(isolated) != codex_module._file_identity(
                root / "auth.json"
            )
            assert str(home) not in provider._shared_sign_ins
        finally:
            provider._release_home(str(home))

    @pytest.mark.asyncio
    async def test_generate_shares_the_sign_in_and_removes_the_home(
        self, real_sign_in: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        homes: list[str] = []
        provider = _generate_ready(monkeypatch)

        async def _exec(*args: object, **kwargs: Any) -> Any:
            codex_home = Path(kwargs["env"]["CODEX_HOME"])
            homes.append(str(codex_home.parent))
            _codex_saves(codex_home / "auth.json", _auth_text("refreshed-during-call"))
            return _answering_process(args)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", _exec)
        response = await provider.generate(GenerateRequest(prompt="test"))
        assert response.text == "ANSWER"
        assert real_sign_in.read_text(encoding="utf-8") == _auth_text("refreshed-during-call")
        assert homes and not any(Path(home).exists() for home in homes)
        assert not provider._shared_sign_ins


# ------------------------------------------------------------- the implicit reasoning effort
def _generate_ready(monkeypatch: pytest.MonkeyPatch) -> CodexCLIProvider:
    """A provider whose native version/catalog checks are stubbed (they have own contracts)."""
    monkeypatch.setattr(
        CodexCLIProvider, "_verify_native_contract", AsyncMock(return_value="codex-cli 0.160.1")
    )
    monkeypatch.setattr(
        CodexCLIProvider,
        "_generation_catalog",
        AsyncMock(return_value="/synthetic/models.json"),
    )
    monkeypatch.setattr(codex_module, "terminate_process_tree", AsyncMock())
    return CodexCLIProvider(cli_path="codex")


def _answering_process(args: tuple[object, ...]) -> Any:
    output = Path(str(args[list(args).index("-o") + 1]))
    output.write_text("ANSWER", encoding="utf-8")
    proc = AsyncMock()
    proc.returncode = 0
    proc.communicate.return_value = (
        b'{"type":"turn.started"}\n'
        b'{"type":"item.completed","item":{"type":"agent_message","text":"ANSWER"}}\n'
        b'{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n',
        b"",
    )
    return proc


@pytest.fixture
def api_key_auth(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "empty-codex-home"))
    monkeypatch.setenv("CODEX_API_KEY", "synthetic-unit-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)


class TestTheReasoningEffortIsAskedForExplicitly:
    """--ignore-user-config drops config.toml's effort; the catalog default is `low`."""

    @staticmethod
    async def _argv(monkeypatch: pytest.MonkeyPatch, request: GenerateRequest) -> list[str]:
        provider = _generate_ready(monkeypatch)
        seen: list[list[str]] = []

        async def _exec(*args: object, **_kwargs: object) -> Any:
            seen.append([str(a) for a in args])
            return _answering_process(args)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", _exec)
        response = await provider.generate(request)
        assert response.text == "ANSWER"
        return seen[-1]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, "xhigh"),
            ("high", "high"),
            ("  medium  ", "medium"),
            ("", None),
            ("   ", None),
            ("--sandbox danger-full-access", "xhigh"),
            ("a b", "xhigh"),
        ],
    )
    async def test_the_default_and_its_override(
        self,
        api_key_auth: None,
        monkeypatch: pytest.MonkeyPatch,
        raw: str | None,
        expected: str | None,
    ) -> None:
        if raw is None:
            monkeypatch.delenv(codex_module._REASONING_EFFORT_ENV, raising=False)
        else:
            monkeypatch.setenv(codex_module._REASONING_EFFORT_ENV, raw)
        argv = await self._argv(monkeypatch, GenerateRequest(prompt="test"))
        efforts = [a for a in argv if a.startswith("model_reasoning_effort=")]
        assert efforts == ([f'model_reasoning_effort="{expected}"'] if expected else [])

    @pytest.mark.asyncio
    async def test_a_requested_effort_wins_over_the_default(
        self, api_key_auth: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(codex_module._REASONING_EFFORT_ENV, "xhigh")
        request = GenerateRequest(
            prompt="test", reasoning=ReasoningConfig(enabled=True, effort="low")
        )
        argv = await self._argv(monkeypatch, request)
        assert [a for a in argv if a.startswith("model_reasoning_effort=")] == [
            'model_reasoning_effort="low"'
        ]

    @pytest.mark.asyncio
    async def test_an_unsupported_implicit_default_is_dropped_not_sent(
        self, api_key_auth: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = _generate_ready(monkeypatch)
        efforts: list[str | None] = []

        async def _catalog(_self: object, _model: str, effort: str | None, **_kw: object) -> str:
            efforts.append(effort)
            if effort is not None:
                raise ValueError(f"Codex unsupported reasoning effort {effort} for model m")
            return "/synthetic/models.json"

        monkeypatch.setattr(CodexCLIProvider, "_generation_catalog", _catalog)
        seen: list[list[str]] = []

        async def _exec(*args: object, **_kwargs: object) -> Any:
            seen.append([str(a) for a in args])
            return _answering_process(args)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", _exec)
        response = await provider.generate(GenerateRequest(prompt="test"))
        assert efforts == ["xhigh", None]
        assert not [a for a in seen[-1] if a.startswith("model_reasoning_effort=")]
        assert response.raw["reasoning"] == {"effort": None, "status": "uncontrolled"}
        assert response.raw["reasoning_effort_source"] is None

    @pytest.mark.asyncio
    async def test_an_unsupported_requested_effort_still_fails(
        self, api_key_auth: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = _generate_ready(monkeypatch)
        monkeypatch.setattr(
            CodexCLIProvider,
            "_generation_catalog",
            AsyncMock(side_effect=ValueError("Codex unsupported reasoning effort medium for m")),
        )
        request = GenerateRequest(
            prompt="test", reasoning=ReasoningConfig(enabled=True, effort="medium")
        )
        with pytest.raises(ValueError, match="unsupported reasoning effort"):
            await provider.generate(request)


# ----------------------------------------------------------------- stream events and cleanup
class TestOnlyTurnFailedIsTerminal:
    @staticmethod
    def _state(*lines: str) -> Any:
        state = codex_module._LiveCodexState()
        for line in lines:
            codex_module._ingest_codex_stdout_line(line, state)
        return state

    def test_an_error_before_a_completed_turn_is_transient(self) -> None:
        state = self._state(
            '{"type":"turn.started"}',
            '{"type":"error","message":"stream disconnected; retrying"}',
            '{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}',
            '{"type":"turn.completed"}',
        )
        assert state.saw_turn_completed and not state.error_message

    def test_an_error_after_the_turn_completed_still_fails(self) -> None:
        state = self._state('{"type":"turn.completed"}', '{"type":"error"}')
        assert state.error_message

    def test_a_failed_turn_fails_whatever_preceded_it(self) -> None:
        state = self._state('{"type":"error","message":"hiccup"}', '{"type":"turn.failed"}')
        assert state.error_message and not state.saw_turn_completed


class TestTheIsolatedHomeIsRemoved:
    def test_removal_is_retried_while_windows_lets_go_of_the_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        original = shutil.rmtree
        failures = iter([PermissionError("in use"), PermissionError("in use")])

        def _flaky(path: Any, *args: Any, **kwargs: Any) -> None:
            error = next(failures, None)
            if error is not None:
                raise error
            original(path, *args, **kwargs)

        monkeypatch.setattr(codex_module.sys, "platform", "win32")
        monkeypatch.setattr(codex_module.shutil, "rmtree", _flaky)
        monkeypatch.setattr(codex_module, "_HOME_REMOVAL_RETRY_SECONDS", 0)
        codex_module._cleanup_home(home)
        assert not home.exists()

    def test_a_home_that_cannot_be_removed_is_reported_not_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _never(*_args: object, **_kwargs: object) -> None:
            raise PermissionError("in use")

        monkeypatch.setattr(codex_module.sys, "platform", "win32")
        monkeypatch.setattr(codex_module.shutil, "rmtree", _never)
        monkeypatch.setattr(codex_module, "_HOME_REMOVAL_RETRY_SECONDS", 0)
        with pytest.raises(RuntimeError, match="cleanup incomplete"):
            codex_module._cleanup_home(tmp_path / "home")
