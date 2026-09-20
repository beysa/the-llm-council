"""Codex CLI adapter: the stall window, ending the whole process tree, removing the isolated home.

The process-tree tests start REAL processes. They never look up or end a process by pid or by
name: a worker proves it is alive by appending to a heartbeat file that only it writes, and it
exits by itself when a stop file appears (or after a hard time box). A job is only ever asked
about its own members. Every path the tests delete lives under pytest's `tmp_path`.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import sys
import tempfile
import textwrap
import threading
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_council.providers.base import (
    ErrorType,
    GenerateRequest,
    GenerateResponse,
    classify_error,
)
from llm_council.providers.cli import codex as codex_module
from llm_council.providers.cli.codex import CodexCLIProvider

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Job Objects are Windows-only")


# --------------------------------------------------------------------------- stall window
class TestStallWindow:
    def test_defaults_to_the_request_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(codex_module._STALL_SECONDS_ENV, raising=False)
        provider = CodexCLIProvider(cli_path="codex")
        # The old rule was min(45, max(15, timeout * 0.33)): 45 s at the usual 599 s timeout,
        # which ended healthy turns that were still reasoning.
        assert provider._stall_after_turn_started_seconds(599.0) == 599.0
        assert provider._stall_after_turn_started_seconds(30.0) == 30.0

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("45", 45.0),
            (" 120.5 ", 120.5),
            ("900", 599.0),
            ("inf", 599.0),
            ("1e400", 599.0),
            ("0", 599.0),
            ("-5", 599.0),
            ("nan", 599.0),
            ("45s", 599.0),
            ("soon", 599.0),
            ("  ", 599.0),
            ("", 599.0),
        ],
    )
    def test_env_override_is_bounded_and_forgiving(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: float
    ) -> None:
        monkeypatch.setenv(codex_module._STALL_SECONDS_ENV, raw)
        provider = CodexCLIProvider(cli_path="codex")
        assert provider._stall_after_turn_started_seconds(599.0) == expected

    @staticmethod
    async def _silent_turn_error(tmp_path: Path, *, stall: float, timeout: float) -> str:
        """The error `generate()` raises for a turn that starts and then says nothing."""

        class SilentCodex:
            def __init__(self) -> None:
                self.stdout = asyncio.StreamReader()
                self.stderr = asyncio.StreamReader()
                self.returncode: int | None = None

            async def wait(self) -> int:
                while self.returncode is None:
                    await asyncio.sleep(0)
                return self.returncode

        process = SilentCodex()
        process.stdout.feed_data(b'{"type":"turn.started"}\n')

        async def _fake_terminate(_proc: object, grace_seconds: float = 1.0) -> None:
            process.returncode = -9
            process.stdout.feed_eof()
            process.stderr.feed_eof()

        provider = CodexCLIProvider(cli_path="codex")
        with (
            patch.object(provider, "_create_isolated_cli_home", return_value=str(tmp_path / "h")),
            patch.object(provider, "_stall_after_turn_started_seconds", return_value=stall),
            patch("asyncio.create_subprocess_exec", return_value=process),
            patch(
                "llm_council.providers.cli.codex._terminate_live_process",
                side_effect=_fake_terminate,
            ),
            pytest.raises(RuntimeError) as caught,
        ):
            await provider.generate(GenerateRequest(prompt="test", timeout_seconds=timeout))
        return str(caught.value)

    @pytest.mark.asyncio
    async def test_the_deadline_wins_when_both_limits_are_reached_together(
        self, tmp_path: Path
    ) -> None:
        # Stalled from the first moment, and past the deadline by the second pass.
        error = await self._silent_turn_error(tmp_path, stall=0.0, timeout=0.01)
        assert "timed out after" in error

    @pytest.mark.asyncio
    async def test_without_an_explicit_shorter_window_there_is_no_stall_rule(
        self, tmp_path: Path
    ) -> None:
        error = await self._silent_turn_error(tmp_path, stall=0.3, timeout=0.3)
        assert "timed out after" in error

    @pytest.mark.asyncio
    async def test_each_limit_keeps_its_classification(self, tmp_path: Path) -> None:
        # The wording of an error IS its contract: `classify_error` matches substrings such
        # as "timeout", "429" or "500", and the class decides how often the call is retried.
        stalled = await self._silent_turn_error(tmp_path, stall=0.02, timeout=500)
        timed_out = await self._silent_turn_error(tmp_path, stall=0.2, timeout=0.2)

        assert "stalled after turn.started" in stalled
        assert classify_error(stalled, 1) is ErrorType.UNKNOWN  # one retry, as before
        assert not any(ch.isdigit() for ch in stalled), "a number can spell 429, 401 or 500"
        assert classify_error(timed_out, 1) is ErrorType.TIMEOUT


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


# A stand-in for the real CLI, launched the way npm launches it: `codex.cmd` -> interpreter ->
# worker. It reads the prompt, starts a worker that shares its stdout pipe AND keeps a file open
# inside the isolated HOME (what made the real homes undeletable), and reports `turn.started`.
# Modes: "silent" never answers; "answer" answers and then lingers, the way Codex does;
# "answer-exit" answers, takes half a second to shut down, leaves a marker and exits by itself
# after telling its worker to go too; "answer-exit-worker-stays" does the same but leaves the
# worker running, the way a descendant may outlive the launcher.
FAKE_CODEX = textwrap.dedent(
    """
    import json, os, subprocess, sys, time
    beat, stop = os.environ["FAKE_CODEX_BEAT"], os.environ["FAKE_CODEX_STOP"]
    mode = os.environ["FAKE_CODEX_MODE"]
    held = os.path.join(os.environ["HOME"], ".codex", "held-open.log")
    sys.stdin.read()
    subprocess.Popen(
        [sys.executable, "-c", os.environ["FAKE_CODEX_WORKER"], beat, stop, held],
        stdin=subprocess.DEVNULL, close_fds=False,
    )
    print(json.dumps({"type": "turn.started"}), flush=True)
    if mode != "silent":
        while not (os.path.exists(beat) and os.path.getsize(beat)):
            time.sleep(0.05)               # a real turn ends long after its workers are up
        with open(sys.argv[sys.argv.index("-o") + 1], "w", encoding="utf-8") as fh:
            fh.write("READY")
        print(json.dumps({"type": "turn.completed"}), flush=True)
    if mode.startswith("answer-exit"):
        time.sleep(0.5)
        if mode == "answer-exit":
            with open(stop, "w", encoding="utf-8") as fh:
                fh.write("the worker goes too")
        with open(os.environ["FAKE_CODEX_EXITED"], "w", encoding="utf-8") as fh:
            fh.write("shut down by itself")
        sys.exit(0)
    end = time.monotonic() + 40
    while time.monotonic() < end and not os.path.exists(stop):
        time.sleep(0.05)
    """
)


@windows_only
class TestGenerateEndsTheTreeAndItsHome:
    @pytest.fixture
    def fake_codex(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
        home = tmp_path / "home"
        home.mkdir()
        script = tmp_path / "fake_codex.py"
        script.write_text(FAKE_CODEX, encoding="utf-8")
        shim = tmp_path / "codex.cmd"
        shim.write_text(f'@"{sys.executable}" "{script}" %*\n', encoding="utf-8")
        paths = {
            "shim": shim,
            "beat": tmp_path / "beat.txt",
            "stop": tmp_path / "stop.flag",
            "exited": tmp_path / "exited.txt",
            "homes": home / ".codex" / ".tmp",
            "temp": tmp_path / "temp",
        }
        paths["temp"].mkdir()
        # `codex.Path` IS pathlib.Path: this replaces Path.home for the whole interpreter
        # until monkeypatch undoes it. No test below may touch the developer's real home -
        # nor litter the real temp directory: the prompt, schema and last-message files of
        # a call go where the test can see whether they were removed.
        monkeypatch.setattr("llm_council.providers.cli.codex.Path.home", lambda: home)
        monkeypatch.setattr(tempfile, "tempdir", str(paths["temp"]))
        monkeypatch.setenv("FAKE_CODEX_BEAT", str(paths["beat"]))
        monkeypatch.setenv("FAKE_CODEX_STOP", str(paths["stop"]))
        monkeypatch.setenv("FAKE_CODEX_EXITED", str(paths["exited"]))
        monkeypatch.setenv("FAKE_CODEX_WORKER", GRANDCHILD)
        monkeypatch.delenv(codex_module._STALL_SECONDS_ENV, raising=False)
        return paths

    @staticmethod
    async def _assert_nothing_is_left(paths: dict[str, Path]) -> None:
        try:
            assert _size(paths["beat"]) > 0, "the worker never ran, so the test proves nothing"
            assert not await _still_beating(paths["beat"]), "the worker outlived the call"
            assert list(paths["homes"].iterdir()) == [], "the isolated home was left behind"
            # The prompt file is the child's stdin: it can only go once the tree is dead.
            assert list(paths["temp"].iterdir()) == [], "temp files of the call were left behind"
        finally:
            paths["stop"].write_text("stop", encoding="utf-8")

    @pytest.mark.asyncio
    async def test_a_silent_turn_runs_to_the_deadline_and_leaves_nothing(
        self, fake_codex: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_CODEX_MODE", "silent")
        provider = CodexCLIProvider(cli_path=str(fake_codex["shim"]))

        with pytest.raises(RuntimeError, match="timed out after 4"):
            await provider.generate(GenerateRequest(prompt="test", timeout_seconds=4))
        await self._assert_nothing_is_left(fake_codex)

    @pytest.mark.asyncio
    async def test_the_optional_fast_fail_still_ends_the_tree(
        self, fake_codex: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_CODEX_MODE", "silent")
        monkeypatch.setenv(codex_module._STALL_SECONDS_ENV, "2")
        provider = CodexCLIProvider(cli_path=str(fake_codex["shim"]))

        with pytest.raises(RuntimeError, match="stalled after turn.started"):
            await provider.generate(GenerateRequest(prompt="test", timeout_seconds=60))
        await self._assert_nothing_is_left(fake_codex)

    @pytest.mark.asyncio
    async def test_an_answered_turn_that_lingers_is_ended_after_a_bounded_grace(
        self, fake_codex: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_CODEX_MODE", "answer")
        monkeypatch.setattr(codex_module, "_NATURAL_EXIT_GRACE_SECONDS", 1.0)
        provider = CodexCLIProvider(cli_path=str(fake_codex["shim"]))

        started = time.monotonic()
        response = await provider.generate(GenerateRequest(prompt="test", timeout_seconds=60))

        assert isinstance(response, GenerateResponse) and response.text == "READY"
        assert time.monotonic() - started < 15, "the fake lingers for 40 s; the grace must not"
        await self._assert_nothing_is_left(fake_codex)

    @pytest.mark.asyncio
    async def test_the_grace_is_cut_short_by_the_deadline(
        self, fake_codex: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_CODEX_MODE", "answer")
        monkeypatch.setattr(codex_module, "_NATURAL_EXIT_GRACE_SECONDS", 30.0)
        provider = CodexCLIProvider(cli_path=str(fake_codex["shim"]))

        started = time.monotonic()
        response = await provider.generate(GenerateRequest(prompt="test", timeout_seconds=4))

        assert isinstance(response, GenerateResponse) and response.text == "READY"
        assert time.monotonic() - started < 12
        await self._assert_nothing_is_left(fake_codex)

    @pytest.mark.asyncio
    async def test_a_finished_turn_is_allowed_to_shut_down_by_itself(
        self, fake_codex: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # On Windows Codex works in the real ~/.codex, so it must not be cut off mid-shutdown -
        # and once the whole tree has gone by itself there is nothing left to wait for.
        monkeypatch.setenv("FAKE_CODEX_MODE", "answer-exit")
        monkeypatch.setattr(codex_module, "_NATURAL_EXIT_GRACE_SECONDS", 40.0)
        provider = CodexCLIProvider(cli_path=str(fake_codex["shim"]))

        started = time.monotonic()
        response = await provider.generate(GenerateRequest(prompt="test", timeout_seconds=90))

        assert isinstance(response, GenerateResponse) and response.text == "READY"
        assert fake_codex["exited"].exists(), "Codex was ended before it could shut down"
        assert time.monotonic() - started < 25, "it waited although the job was already empty"
        assert list(fake_codex["homes"].iterdir()) == []

    @pytest.mark.asyncio
    async def test_the_grace_waits_for_a_descendant_that_outlives_the_launcher(
        self, fake_codex: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The launcher's exit proves nothing about what it started: the worker may still be
        # writing. The grace is for the whole job, and only then is the job ended.
        monkeypatch.setenv("FAKE_CODEX_MODE", "answer-exit-worker-stays")
        monkeypatch.setattr(codex_module, "_NATURAL_EXIT_GRACE_SECONDS", 4.5)
        provider = CodexCLIProvider(cli_path=str(fake_codex["shim"]))

        response = await provider.generate(GenerateRequest(prompt="test", timeout_seconds=60))

        assert isinstance(response, GenerateResponse) and response.text == "READY"
        launcher_gone = fake_codex["exited"].stat().st_mtime  # 0.5 s after the turn completed
        last_beat = fake_codex["beat"].stat().st_mtime
        assert last_beat - launcher_gone > 1.5, "the worker was ended as soon as the launcher left"
        await self._assert_nothing_is_left(fake_codex)  # ...but it IS ended once the grace is over

    @pytest.mark.asyncio
    async def test_a_cancelled_call_leaves_nothing(
        self, fake_codex: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Nothing stops Codex on this path except the `finally` block: the job has to be closed
        # there, and before the home is removed, or the worker's open file keeps the home alive.
        monkeypatch.setenv("FAKE_CODEX_MODE", "silent")
        provider = CodexCLIProvider(cli_path=str(fake_codex["shim"]))
        call = asyncio.ensure_future(
            provider.generate(GenerateRequest(prompt="test", timeout_seconds=60))
        )
        await _wait_until_beating(fake_codex["beat"])

        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        await self._assert_nothing_is_left(fake_codex)

    @pytest.mark.asyncio
    async def test_a_second_cancellation_during_cleanup_still_leaves_nothing(
        self, fake_codex: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_CODEX_MODE", "silent")
        provider = CodexCLIProvider(cli_path=str(fake_codex["shim"]))
        call = asyncio.ensure_future(
            provider.generate(GenerateRequest(prompt="test", timeout_seconds=60))
        )
        await _wait_until_beating(fake_codex["beat"])

        call.cancel()
        for _ in range(3):  # let it reach the `finally` block, where it waits for the removal
            await asyncio.sleep(0)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call

        # The removal runs in a worker thread, which a cancellation cannot stop half way.
        assert await _eventually(lambda: list(fake_codex["homes"].iterdir()) == [])
        await self._assert_nothing_is_left(fake_codex)


class TestTheGraceCoversBothExits:
    @pytest.mark.asyncio
    async def test_a_launcher_that_is_already_gone_does_not_cancel_the_grace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The end of the turn may be noticed only after the launcher has exited. Its
        # descendants can still be shutting down: they get the grace all the same, and only
        # then does `finally` close the job on them.
        class FinishedAndGone:
            def __init__(self) -> None:
                self.stdout = asyncio.StreamReader()
                self.stderr = asyncio.StreamReader()
                self.returncode: int | None = 0

            async def wait(self) -> int:
                return 0

        process = FinishedAndGone()
        for event in (
            b'{"type":"turn.started"}\n',
            b'{"type":"item.completed","item":{"type":"agent_message","text":"READY"}}\n',
            b'{"type":"turn.completed"}\n',
        ):
            process.stdout.feed_data(event)
        process.stdout.feed_eof()
        process.stderr.feed_eof()

        order: list[str] = []
        still_busy = iter([True, True, True, False])

        def _descendants_busy(_job: int) -> bool:
            order.append("asked the job")
            return next(still_busy)

        monkeypatch.setattr(codex_module, "_adopt_into_job", lambda _proc: 1234)
        monkeypatch.setattr(codex_module, "_job_is_active", _descendants_busy)
        monkeypatch.setattr(codex_module, "_terminate_job", lambda _job: order.append("terminated"))
        monkeypatch.setattr(codex_module, "_close_job", lambda _job: order.append("closed"))
        provider = CodexCLIProvider(cli_path="codex")
        with (
            patch.object(provider, "_create_isolated_cli_home", return_value=str(tmp_path / "h")),
            patch("asyncio.create_subprocess_exec", return_value=process),
        ):
            response = await provider.generate(GenerateRequest(prompt="test", timeout_seconds=30))

        assert isinstance(response, GenerateResponse) and response.text == "READY"
        assert order == ["asked the job"] * 4 + ["closed"]


class TestSpawnOptions:
    @pytest.mark.asyncio
    async def test_the_child_is_born_suspended_on_windows_only(self, tmp_path: Path) -> None:
        provider = CodexCLIProvider(cli_path="codex")
        process = AsyncMock()
        process.communicate.return_value = (b"", b"")
        process.returncode = 0
        seen: dict[str, object] = {}

        def _fake_exec(*_args: object, **kwargs: object) -> AsyncMock:
            seen.update(kwargs)
            return process

        with (
            patch.object(provider, "_create_isolated_cli_home", return_value=str(tmp_path / "h")),
            patch("asyncio.create_subprocess_exec", side_effect=_fake_exec),
        ):
            await provider.generate(GenerateRequest(prompt="test"))

        assert seen["start_new_session"] is True
        if sys.platform == "win32":
            assert seen["creationflags"] == codex_module._CREATE_SUSPENDED
        else:
            assert "creationflags" not in seen


class TestNoSpareCredentialOnWindows:
    def test_the_sign_in_is_copied_only_where_the_isolated_home_is_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # On Windows Codex ignores the isolated HOME and signs in from the real ~/.codex
        # (verified live: it answers with nothing copied), so a copy there is only a second
        # credential on disk - the thing that was found in 20 leaked homes.
        home = tmp_path / "home"
        (home / ".codex").mkdir(parents=True)
        (home / ".codex" / "auth.json").write_text('{"fake": "token"}', encoding="utf-8")
        monkeypatch.setattr("llm_council.providers.cli.codex.Path.home", lambda: home)

        cli_home = Path(CodexCLIProvider(cli_path="codex")._create_isolated_cli_home())

        copied = [p.name for p in (cli_home / ".codex").iterdir()]
        assert copied == ([] if sys.platform == "win32" else ["auth.json"])


class TestTheIsolatedHomeIsRemoved:
    """`generate()` with a mocked process: what happens to the files it created."""

    @pytest.fixture(autouse=True)
    def call_temp_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        # The prompt / schema / last-message files of a call, where the test can see them.
        temp = tmp_path / "temp"
        temp.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(temp))
        return temp

    @pytest.mark.asyncio
    async def test_temp_files_are_retried_too(
        self, tmp_path: Path, call_temp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The prompt file is the child's stdin. Right after the job is closed the dying
        # launcher still has it open, and Windows refuses the delete: 127 such files piled
        # up in two hours while the unlink was attempted only once.
        home = tmp_path / "codex-home"
        home.mkdir()
        refused: list[str] = []
        real_unlink = os.unlink

        def _still_open_twice(path: str, *args: object, **kwargs: object) -> None:
            if "llm-council-codex-prompt-" in str(path) and len(refused) < 2:
                refused.append(str(path))
                raise PermissionError(13, "The process cannot access the file", str(path))
            real_unlink(path, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(codex_module.os, "unlink", _still_open_twice)
        monkeypatch.setattr(codex_module.time, "sleep", lambda _seconds: None)
        await self._generate(CodexCLIProvider(cli_path="codex"), home)

        assert len(refused) == 2
        assert list(call_temp_dir.iterdir()) == []
        assert not home.exists()

    @staticmethod
    async def _generate(provider: CodexCLIProvider, home: Path) -> None:
        process = AsyncMock()
        process.communicate.return_value = (b"", b"")
        process.returncode = 0
        with (
            patch.object(provider, "_create_isolated_cli_home", return_value=str(home)),
            patch("asyncio.create_subprocess_exec", return_value=process),
        ):
            await provider.generate(GenerateRequest(prompt="test"))

    @pytest.mark.asyncio
    async def test_removal_is_retried_while_windows_lets_go_of_the_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "codex-home"
        home.mkdir()
        attempts: list[str] = []
        threads: set[str] = set()
        real = shutil.rmtree

        def _held_twice(path: str, *args: object, **kwargs: object) -> None:
            attempts.append(path)
            threads.add(threading.current_thread().name)
            if len(attempts) > 2:
                real(path, *args, **kwargs)  # type: ignore[arg-type]

        # `codex_module.shutil` and `.time` ARE the shared modules; monkeypatch restores them.
        monkeypatch.setattr(codex_module.shutil, "rmtree", _held_twice)
        monkeypatch.setattr(codex_module.time, "sleep", lambda _seconds: None)
        await self._generate(CodexCLIProvider(cli_path="codex"), home)

        assert attempts == [str(home)] * 3
        assert not home.exists()
        # Off the event loop, and not in the executor, where a queued job can be cancelled.
        assert threads == {"codex-home-removal"}

    @pytest.mark.asyncio
    async def test_a_home_that_cannot_be_removed_is_reported_not_swallowed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        home = tmp_path / "codex-home"
        home.mkdir()
        attempts: list[str] = []
        naps: list[float] = []

        monkeypatch.setattr(
            codex_module.shutil, "rmtree", lambda path, **_kw: attempts.append(path)
        )
        monkeypatch.setattr(codex_module.time, "sleep", naps.append)
        with caplog.at_level("WARNING", logger=codex_module.logger.name):
            await self._generate(CodexCLIProvider(cli_path="codex"), home)

        assert len(attempts) == 10
        assert naps == [0.2] * 9, "no point in waiting after the last attempt"
        assert [r for r in caplog.records if str(home) in r.getMessage()], "the leak was silent"

    @pytest.mark.asyncio
    async def test_a_child_that_cannot_be_confined_fails_the_call_and_cleans_up(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "codex-home"
        home.mkdir()

        def _refuse(_proc: object) -> int:
            raise RuntimeError("Codex CLI: could not confine the child to a job object")

        monkeypatch.setattr(codex_module, "_adopt_into_job", _refuse)
        with pytest.raises(RuntimeError, match="could not confine"):
            await self._generate(CodexCLIProvider(cli_path="codex"), home)
        assert not home.exists()

    @pytest.mark.asyncio
    async def test_the_home_is_still_removed_when_no_worker_thread_can_be_had(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "codex-home"
        home.mkdir()

        class NoNewThreads:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def start(self) -> None:
                raise RuntimeError("can't start new thread")

            def is_alive(self) -> bool:
                return False

        # Only the name `threading` inside the adapter module is replaced.
        monkeypatch.setattr(codex_module, "threading", MagicMock(Thread=NoNewThreads))
        await self._generate(CodexCLIProvider(cli_path="codex"), home)
        assert not home.exists()

    @windows_only
    @pytest.mark.asyncio
    async def test_a_child_interrupted_before_it_was_confined_is_not_left_suspended(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The gap between the spawn and the job: whoever interrupts it must not leave a
        # suspended launcher behind - it could never exit by itself.
        spawned: list[asyncio.subprocess.Process] = []
        real_exec = asyncio.create_subprocess_exec

        async def _spy_exec(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
            proc = await real_exec(*args, **kwargs)  # type: ignore[arg-type]
            spawned.append(proc)
            return proc

        def _interrupted(_proc: object) -> int:
            raise KeyboardInterrupt  # and, unlike the real function, ends nothing

        home = tmp_path / "home"
        home.mkdir()
        shim = tmp_path / "codex.cmd"
        shim.write_text(f'@"{sys.executable}" -c "import time; time.sleep(30)"\n', encoding="utf-8")
        monkeypatch.setattr("llm_council.providers.cli.codex.Path.home", lambda: home)
        monkeypatch.setattr(codex_module, "_adopt_into_job", _interrupted)
        monkeypatch.setattr(codex_module.asyncio, "create_subprocess_exec", _spy_exec)

        started = time.monotonic()
        try:
            with pytest.raises(KeyboardInterrupt):
                await CodexCLIProvider(cli_path=str(shim)).generate(GenerateRequest(prompt="test"))

            assert len(spawned) == 1
            # `generate` ended it itself, at once: had it only waited, this would take 5 s,
            # and a child that stayed suspended would never finish at all.
            await asyncio.wait_for(spawned[0].wait(), timeout=5)
            assert time.monotonic() - started < 4
            assert list((home / ".codex" / ".tmp").iterdir()) == []
        finally:
            for proc in spawned:  # whatever the code under test did: nothing stays suspended
                if proc.returncode is None:
                    with contextlib.suppress(OSError):
                        proc.kill()


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
