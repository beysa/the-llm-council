"""
Codex CLI provider adapter.

Wraps the ``codex`` CLI for non-interactive generation via ``codex exec``.
Useful for agent-to-agent delegation and environments where CLI auth
is available but API keys may not be.

SECURITY NOTE: Uses asyncio.create_subprocess_exec with argument lists,
which is safe from shell injection (equivalent to execFile in Node.js).
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import hashlib
import importlib
import json
import logging
import os
import re
import shlex
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlsplit

from llm_council.logging import redact_account_identity
from llm_council.providers.base import (
    DoctorResult,
    ErrorType,
    GenerateRequest,
    GenerateResponse,
    ProviderAdapter,
    ProviderCapabilities,
    classify_error,
)
from llm_council.providers.cli._compatibility import (
    check_verified_version,
    native_identity,
    selected_cli_path,
)
from llm_council.providers.cli._subprocess import terminate_process_tree

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-6.1-sol"
# Generation only: native tools are separately disabled and catalog-validated.
DEFAULT_FLAGS = "--sandbox read-only --skip-git-repo-check"
# The compiler imports this tested baseline; runtime metadata reports the observed version.
_VERIFIED_VERSION = "codex-cli 0.149.1"
_VERIFIED_VERSIONS = (_VERIFIED_VERSION, "codex-cli 0.153.3", "codex-cli 0.160.1")
_CLEANUP_SECONDS = 1.0

# An isolated CODEX_HOME (and --ignore-user-config) drops the user's config.toml, so a
# request without reasoning controls would run at the model's catalog default effort
# (`low` for gpt-6.1-sol in 0.160.1) instead of the profile's. When the request carries no
# reasoning, ask for this effort explicitly. Set LLM_COUNCIL_CODEX_REASONING_EFFORT to
# override, or to empty to leave the effort to Codex. The value is still checked against
# the model's catalog levels; an unsupported implicit default is dropped, not sent.
_REASONING_EFFORT_ENV = "LLM_COUNCIL_CODEX_REASONING_EFFORT"
_DEFAULT_REASONING_EFFORT = "xhigh"

# Windows: the child is born suspended and confined to a kill-on-close Job Object.
_CREATE_SUSPENDED = 0x00000004
_NATIVE_SPAWN = asyncio.create_subprocess_exec
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
# Windows releases a dead process's file handles a moment after it ends.
_HOME_REMOVAL_ATTEMPTS = 10
_HOME_REMOVAL_RETRY_SECONDS = 0.2
_GENERATION_CONFIG = {
    "features.shell_tool": False,
    "features.view_image": False,
    "features.image_generation": False,
    "features.multi_agent": False,
    "features.apps": False,
    "features.plugins": False,
    "features.hooks": False,
    "web_search": "disabled",
    "tools.update_plan.enabled": False,
    "tools.experimental_request_user_input.enabled": False,
}

# Unsafe modes that require explicit opt-in
_UNSAFE_FLAGS = {"--full-auto", "--sandbox workspace-write", "--approval-mode yolo"}
_UNSAFE_WARNING = (
    "WARNING: Codex CLI is running with permissive flags that allow local file/env access. "
    "This is unsafe with untrusted inputs. Ensure you trust the task source."
)

# Codex CLI panics under an aggressively stripped environment when launched
# from nested council subprocesses. Preserve the ambient runtime and strip only
# telemetry variables that can destabilize or leak outer-session tracing.
_ENV_DENYLIST_PREFIXES = (
    "OTEL_",
    "LANGSMITH_",
    "LANGCHAIN_",
)
_ENV_SESSION_KEYS = {"CODEX_THREAD_ID", "CODEX_SESSION_ID", "CODEX_INTERNAL_ORIGINATOR_OVERRIDE"}
_CREDENTIAL_ENV_KEY = re.compile(
    r"(?:^|[_-])(?:api[_-]?keys?|secrets?|passwords?|credentials?|authorization|cookies?|"
    r"(?:access|refresh|auth|bearer|id|session)[_-]?tokens?|key|token)$|^AUTH_SECRET_\d+$",
    re.IGNORECASE,
)


@dataclass
class _CodexRuntime:
    env: dict[str, str]
    auth: bytes | None
    category: str
    config: dict[str, str]
    # The selected root's auth.json when ChatGPT auth came from it: shared, never copied.
    auth_path: Path | None = None

    def diagnostics_env(self) -> dict[str, str]:
        values = dict(self.env)
        if self.auth:
            auth = json.loads(self.auth)
            secrets = [auth.get("OPENAI_API_KEY"), *(auth.get("tokens") or {}).values()]
            values.update(
                {
                    f"AUTH_SECRET_{i}": value
                    for i, value in enumerate(secrets)
                    if isinstance(value, str)
                }
            )
        return values


def _config_args(config: dict[str, Any]) -> list[str]:
    return [arg for key, value in config.items() for arg in ("-c", f"{key}={json.dumps(value)}")]


def _cleanup_home(home: str | Path) -> None:
    original = sys.exc_info()[1]
    try:
        # Windows frees a dead child's handles (its stdin file, a half-written save)
        # a moment after it exits, so a first rmtree can fail on files about to be
        # released. Retry briefly there; elsewhere one attempt is authoritative.
        attempts = _HOME_REMOVAL_ATTEMPTS if sys.platform == "win32" else 1
        for remaining in range(attempts - 1, -1, -1):
            try:
                shutil.rmtree(home)
                break
            except FileNotFoundError:
                break
            except OSError:
                if not remaining:
                    raise
                time.sleep(_HOME_REMOVAL_RETRY_SECONDS)
    except OSError as exc:
        logger.error("Codex cleanup incomplete: temporary home remains at %s", home)
        if original is not None:
            raise original from exc
        raise RuntimeError(f"Codex cleanup incomplete: temporary home remains at {home}") from exc


def _own_process_handle(proc: asyncio.subprocess.Process) -> int | None:
    """Return the OS handle asyncio already holds for a real child, else None.

    Never OpenProcess(pid): a pid may belong to somebody else by the time it is
    used (and unit tests hand in fakes with made-up pids). The handle inside the
    Popen object can only ever be our own child.
    """

    transport = getattr(proc, "_transport", None)
    if not isinstance(transport, asyncio.SubprocessTransport):
        return None
    handle = getattr(transport.get_extra_info("subprocess"), "_handle", None)
    return int(handle) if isinstance(handle, int) else None


def _resume_process(handle: int) -> int:
    """Windows: let a process that was created suspended run. Returns an NTSTATUS (0 = ok).

    `NtResumeProcess` is undocumented but long stable. The documented route,
    `ResumeThread`, needs the primary thread handle, which Popen closes at once.
    """

    if sys.platform != "win32":
        return 0
    import ctypes
    from ctypes import wintypes

    ntdll = ctypes.WinDLL("ntdll")
    ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
    ntdll.NtResumeProcess.restype = ctypes.c_long
    return int(ntdll.NtResumeProcess(handle))


def _adopt_into_job(proc: asyncio.subprocess.Process) -> int | None:
    """Windows: confine a child that was created SUSPENDED to a kill-on-close Job Object.

    `proc.kill()` ends only the `codex.CMD` shim; node and codex.exe live on, keep
    spending quota and keep the isolated HOME open so it can never be removed. A
    parent-pid walk (`taskkill /T`) is no answer either: after pid reuse it adopts
    unrelated processes. A job has neither problem - whatever a member spawns is
    born inside the job, and because this child has not run yet, nothing can have
    escaped.

    Returns the job handle; None on other platforms and for test doubles, where
    nothing was created suspended. A real child never leaves here suspended, and
    never runs outside a job: if anything at all goes wrong it is ended and the
    failure is raised (as RuntimeError, unless it is an interrupt), because Codex
    outside a job IS the leak this prevents.
    """

    if sys.platform != "win32" or not isinstance(proc, asyncio.subprocess.Process):
        return None

    job: int | None = None
    try:
        import ctypes
        from ctypes import wintypes

        handle = _own_process_handle(proc)
        if handle is None:
            raise RuntimeError("this Python does not expose the child's process handle")

        class _BasicLimits(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BasicLimits),
                ("IoInfo", ctypes.c_uint64 * 6),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            job,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ) or not kernel32.AssignProcessToJobObject(job, handle):
            raise ctypes.WinError(ctypes.get_last_error())

        status = _resume_process(handle)
        if status != 0:
            raise RuntimeError(f"NtResumeProcess failed (NTSTATUS 0x{status & 0xFFFFFFFF:08x})")
        return job
    except BaseException as exc:
        # The child has not run yet. Whatever happened, it must neither stay suspended
        # nor be let loose outside a job, and the job handle must not leak.
        with contextlib.suppress(Exception):
            _close_job(job)  # kill-on-close ends the child if it got as far as the job
        with contextlib.suppress(Exception):
            proc.kill()
        if isinstance(exc, Exception):
            raise RuntimeError(
                f"Codex CLI: could not confine the child to a job object ({exc}); it was "
                "ended rather than run where it could not be stopped."
            ) from exc
        raise


def _terminate_job(job: int | None) -> None:
    """Windows: end every process in the job at once."""

    if sys.platform != "win32" or not job:
        return
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    if not kernel32.TerminateJobObject(job, 1):
        # Not fatal: closing the handle ends the job as well.
        logger.warning("Codex CLI: TerminateJobObject failed (%s)", ctypes.get_last_error())


def _close_job(job: int | None) -> None:
    """Windows: release the job handle.

    Kill-on-close ends whatever is still in the job, and the same happens when a
    council process dies with the handle open, so it cannot leave Codex behind.
    """

    if sys.platform != "win32" or not job:
        return
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    if not kernel32.CloseHandle(job):
        logger.warning("Codex CLI: closing the job handle failed (%s)", ctypes.get_last_error())


# The ChatGPT sign-in Codex keeps in CODEX_HOME. Its refresh token ROTATES: every refresh issues a
# new one and the old one is dead from then on. A council call must therefore SHARE the user's
# file and never copy it - a refresh into a throwaway copy dies with the copy and leaves the real
# file holding a spent token, which signs the user out at their next refresh ("your refresh token
# was already used"). Codex saves by truncating and rewriting the file in place and signs out with
# remove_file (codex-rs FileAuthStorage), so a HARD LINK shares every refresh, while a sign-out in
# the call's home drops only that entry. Not a symlink: a save through a symlink whose target was
# signed out elsewhere would create the real file again and undo that sign-out. This module never
# writes the real file itself: Codex and the desktop app write it without a lock
# (openai/codex#10332), so any write of ours could race theirs and put back a spent token.
_SIGN_IN_FILE = "auth.json"


@dataclass
class _SharedSignIn:
    real: Path
    isolated: Path
    identity: tuple[int, int]  # (st_dev, st_ino) of the file both names shared
    digest: str  # sha256 of its content when it was shared


def _file_identity(path: Path) -> tuple[int, int] | None:
    """(st_dev, st_ino), or None where the filesystem cannot tell files apart (st_ino 0)."""

    st = path.stat()
    return (st.st_dev, st.st_ino) if st.st_ino else None


def _resolved_identity(path: Path) -> tuple[int, int] | None:
    """_file_identity, or None when the name does not resolve to a file that can be stat'ed."""

    try:
        return _file_identity(path)
    except OSError:
        return None


def _name_exists(path: Path) -> bool:
    """Whether the name is there, even as a link that no longer resolves.

    Only a missing name is False: Codex signs out with remove_file, which removes the name.
    Any other failure to look it up propagates rather than passing for a sign-out.
    """

    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


class _SignInOnAnotherVolumeError(RuntimeError):
    """The isolated home is on another volume than the sign-in, and no hard link crosses one."""


def _share_sign_in(source: Path, target: Path) -> tuple[tuple[int, int], str]:
    """Hard-link `target` to the user's real sign-in, or refuse. Never a symlink or a copy."""

    try:
        os.link(source, target)
    except OSError as exc:
        # Windows reports a cross-volume link as WinError 17, which Python maps to EXDEV.
        refusal = _SignInOnAnotherVolumeError if exc.errno == errno.EXDEV else RuntimeError
        raise refusal(
            f"Codex CLI: cannot share your sign-in with the isolated home ({exc}). Refusing to "
            "copy it: Codex rotates its refresh token, and a refresh into a copy would sign "
            "you out of Codex everywhere else."
        ) from exc
    identity = _file_identity(target)
    if identity is None or identity != _file_identity(source):
        target.unlink(missing_ok=True)
        raise RuntimeError(
            "Codex CLI: this filesystem cannot confirm that the isolated sign-in is your real "
            "one; refusing to run Codex on it."
        )
    return identity, hashlib.sha256(target.read_bytes()).hexdigest()


def _is_valid_json(data: bytes) -> bool:
    try:
        json.loads(data)
    except ValueError:
        return False
    return True


def _keep_aside(real: Path, data: bytes) -> Path:
    """Store `data` next to `real` under a new, unique, owner-only name. Never overwrites."""

    fd, name = tempfile.mkstemp(prefix=f"{real.name}.council-", suffix=".json", dir=real.parent)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    return Path(name)


def _check_shared_sign_in(shared: _SharedSignIn) -> None:
    """After the call, report what the two names show - never repair the real file.

    - Both names still share one file: if it no longer parses on two reads, a Codex ended in
      the middle of its in-place save cut it short; by then its refresh token has rotated, so
      the only fix is to sign in again.
    - They no longer share one file (either side was replaced), or that cannot be confirmed
      (the real name no longer resolves, e.g. a dangling link), and the call's side holds a
      save the call made: that sign-in may exist nowhere else, so it is kept aside and
      reported.
    - A name is gone: a sign-out - in the call's home, or elsewhere, which wins; nothing is
      recreated or kept. Only a missing name counts; any other failure is reported.
    """

    real, isolated = shared.real, shared.isolated
    try:
        if not _name_exists(isolated) or not _name_exists(real):
            return
        still_shared = (
            _resolved_identity(isolated) == shared.identity
            and _resolved_identity(real) == shared.identity
        )
        if still_shared:
            if not _is_valid_json(real.read_bytes()) and not _is_valid_json(real.read_bytes()):
                logger.warning(
                    "Codex CLI: a Codex process ended while saving your sign-in and %s is now "
                    "unreadable. Sign in again with `codex login`.",
                    real,
                )
            return
        data = isolated.read_bytes()
        if hashlib.sha256(data).hexdigest() == shared.digest:
            return
        try:
            kept = _keep_aside(real, data)
        except OSError as exc:
            logger.warning(
                "Codex CLI: Codex saved a sign-in during a council call that %s may not hold, "
                "and it could not be kept (%s), so it is lost with the call's home. If Codex "
                "asks you to sign in, run `codex login`.",
                real,
                exc,
            )
            return
        logger.warning(
            "Codex CLI: Codex saved a sign-in during a council call that %s may not hold: the "
            "two no longer share one file, or that could not be confirmed. If Codex asks you "
            "to sign in, run `codex login`. The call's version was kept as %s; delete it once "
            "you no longer need it.",
            real,
            kept,
        )
    except OSError as exc:
        logger.warning(
            "Codex CLI: could not check the sign-in %s shared with a council call (%s). If "
            "Codex asks you to sign in, run `codex login`.",
            real,
            exc,
        )


def _prepare_schema_for_codex(schema: dict[str, Any]) -> dict[str, Any]:
    """Transform a JSON schema for Codex structured output strictness."""

    result: dict[str, Any] = {}

    for key, value in schema.items():
        if key == "$schema":
            continue
        if key == "additionalProperties":
            continue

        if key == "properties" and isinstance(value, dict):
            result[key] = {
                prop_name: _prepare_schema_for_codex(prop_schema)
                if isinstance(prop_schema, dict) and prop_schema.get("type") == "object"
                else (
                    {
                        **prop_schema,
                        "items": _prepare_schema_for_codex(prop_schema["items"]),
                    }
                    if isinstance(prop_schema, dict)
                    and prop_schema.get("type") == "array"
                    and isinstance(prop_schema.get("items"), dict)
                    and prop_schema["items"].get("type") == "object"
                    else prop_schema
                )
                for prop_name, prop_schema in value.items()
            }
            result["required"] = list(value.keys())
            result["additionalProperties"] = False
        elif key == "required":
            continue
        elif isinstance(value, dict) and value.get("type") == "object":
            result[key] = _prepare_schema_for_codex(value)
        else:
            result[key] = value

    if schema.get("type") == "object" and "additionalProperties" not in result:
        result["additionalProperties"] = False

    return result


def _extract_error_message(stdout_text: str) -> str:
    """Extract a Codex error payload from JSONL stdout when stderr is empty.

    Recognizes both ``type: "error"`` events (synchronous client-side errors,
    e.g. an invalid JSON schema) and ``type: "turn.failed"`` events (server-side
    rejections such as an unsupported model). The latter arrives with exit
    code 0, so callers must consult this before assuming success.
    """

    for line in reversed(stdout_text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        event_type = payload.get("type")
        if event_type == "error":
            message = payload.get("message")
            if isinstance(message, str) and message:
                return message
        elif event_type == "turn.failed":
            error_payload = payload.get("error")
            if isinstance(error_payload, dict):
                message = error_payload.get("message")
                if isinstance(message, str) and message:
                    return message
    return ""


@dataclass
class _LiveCodexState:
    """Incremental subprocess state for Codex CLI calls."""

    stdout_parts: list[str] = field(default_factory=list)
    stderr_parts: list[str] = field(default_factory=list)
    agent_message: str = ""
    error_message: str = ""
    usage: dict[str, int] | None = None
    saw_turn_completed: bool = False
    saw_event: bool = False
    protocol_error: str = ""
    # Set by a bare `error` event; a `turn.completed` that follows clears it again.
    transient_error: bool = False


def _ingest_codex_stdout_line(line: str, state: _LiveCodexState) -> None:
    """Update live state from a single Codex JSONL stdout line."""

    state.stdout_parts.append(line)
    stripped = line.strip()
    if not stripped:
        return

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        if state.saw_event or stripped.startswith(("{", "[")):
            state.protocol_error = "malformed or truncated JSONL"
        return

    if not isinstance(payload, dict) or not isinstance(payload.get("type"), str):
        state.protocol_error = "malformed JSONL event"
        return

    state.saw_event = True
    event_type = payload.get("type")
    if event_type == "turn.started":
        state.saw_turn_completed = False
        state.agent_message = ""
    elif event_type == "item.completed":
        if state.saw_turn_completed:
            state.protocol_error = "item after terminal event"
        item = payload.get("item")
        if isinstance(item, dict) and item.get("type") == "agent_message":
            text = item.get("text")
            if isinstance(text, str):
                state.agent_message = text
            else:
                state.protocol_error = "malformed agent message"
    elif event_type == "turn.completed":
        state.saw_turn_completed = True
        if state.transient_error:
            # A bare `error` (e.g. a retried stream) before a completed turn did not
            # stop it; `turn.failed` and an error after completion still fail.
            state.error_message = ""
            state.transient_error = False
        usage = payload.get("usage")
        if isinstance(usage, dict):
            try:
                prompt_tokens = int(usage.get("input_tokens", 0) or 0) + int(
                    usage.get("cached_input_tokens", 0) or 0
                )
                completion_tokens = int(usage.get("output_tokens", 0) or 0)
            except (ValueError, TypeError):
                state.protocol_error = "malformed usage"
                return
            state.usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
    elif event_type == "turn.failed":
        # Codex emits turn.failed with exit code 0 when the server rejects the
        # request -- e.g. a model this CLI version does not support. Capture the
        # message so the post-loop handler surfaces it instead of an empty success.
        state.saw_turn_completed = False
        state.error_message = "Codex turn.failed"
        state.transient_error = False
        error_payload = payload.get("error")
        if isinstance(error_payload, dict):
            message = error_payload.get("message")
            if isinstance(message, str) and message:
                state.error_message = message
    elif event_type == "error":
        message = payload.get("message")
        state.error_message = message if isinstance(message, str) and message else "Codex error"
        state.transient_error = not state.saw_turn_completed


async def _read_codex_stdout(stream: asyncio.StreamReader | None, state: _LiveCodexState) -> None:
    """Consume Codex stdout incrementally."""

    if stream is None:
        return

    pending = b""
    while chunk := await stream.read(65536):
        pending += chunk
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            _ingest_codex_stdout_line(line.decode("utf-8") + "\n", state)
    if pending:
        _ingest_codex_stdout_line(pending.decode("utf-8"), state)


async def _read_codex_stderr(stream: asyncio.StreamReader | None, state: _LiveCodexState) -> None:
    """Consume Codex stderr incrementally."""

    if stream is None:
        return

    while chunk := await stream.read(65536):
        state.stderr_parts.append(chunk.decode("utf-8", errors="replace"))


class CodexCLIProvider(ProviderAdapter):
    """Codex CLI provider adapter."""

    name: ClassVar[str] = "codex"
    capabilities: ClassVar[ProviderCapabilities] = ProviderCapabilities(
        streaming=False,
        tool_use=False,
        structured_output=True,
        multimodal=False,
        max_tokens=4096,
    )

    def __init__(
        self,
        cli_path: str | None = None,
        default_model: str | None = None,
        default_flags: str | None = None,
        timeout: int = 120,
    ) -> None:
        self._cli_path = selected_cli_path(cli_path, "codex")
        self._default_model = default_model or DEFAULT_MODEL
        self._default_flags = default_flags or DEFAULT_FLAGS
        self._timeout = timeout
        # isolated home -> the real sign-in it shares, for the post-call check
        self._shared_sign_ins: dict[str, _SharedSignIn] = {}

    def _build_command(
        self,
        *,
        model: str,
        output_last_message_path: str | None = None,
        output_schema_path: str | None = None,
    ) -> list[str]:
        """Build the CLI command as argument list (safe from injection)."""
        if not self._cli_path:
            raise RuntimeError("Codex CLI not found.")

        cmd = [self._cli_path, "exec"]
        cmd.extend(shlex.split(self._default_flags))
        cmd.extend(["--json", "--color", "never"])
        cmd.extend(["-m", model])
        if output_schema_path:
            cmd.extend(["--output-schema", output_schema_path])
        if output_last_message_path:
            cmd.extend(["-o", output_last_message_path])

        # prompt is fed via stdin in generate(), NOT argv. Windows
        # CreateProcess caps the command line at ~32KB; schema+prompt exceed it.
        return cmd

    def _check_unsafe_flags(self) -> None:
        """Emit warning if using unsafe permissive flags."""
        flags_str = self._default_flags
        for unsafe_flag in _UNSAFE_FLAGS:
            if unsafe_flag in flags_str:
                warnings.warn(_UNSAFE_WARNING, UserWarning, stacklevel=3)
                break

    def _get_subprocess_env(self) -> dict[str, str]:
        """Keep routing/auth identity, not outer session or telemetry identity."""
        return {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(_ENV_DENYLIST_PREFIXES) and key not in _ENV_SESSION_KEYS
        }

    def _resolve_runtime(self) -> _CodexRuntime:
        """Snapshot one supported identity/route; never consult a fallback home."""
        env = self._get_subprocess_env()
        source = Path(env.get("CODEX_HOME") or (Path.home() / ".codex")).resolve()
        config: dict[str, Any] = {}
        config_path = source / "config.toml"
        if config_path.exists():
            try:
                # Python 3.10 has no TOML parser. Do not guess at native configuration.
                parser = importlib.import_module(
                    "tomllib" if sys.version_info >= (3, 11) else "tomli"
                )
            except ImportError as exc:
                raise ValueError(
                    "Codex unsupported auth/route config: TOML parser unavailable"
                ) from exc
            try:
                config = parser.loads(config_path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                raise ValueError("Codex unsupported auth/route configuration") from exc
        providers = config.get("model_providers", {})
        if (
            config.get("model_provider", "openai") != "openai"
            or not isinstance(providers, dict)
            or "openai" in providers
        ):
            raise ValueError("Codex unsupported configured model provider route")
        if "profile" in config:
            raise ValueError("Codex unsupported auth/route profile")
        if "chatgpt_base_url" in config:
            raise ValueError(
                "Codex unsupported auth route: chatgpt_base_url does not route generation"
            )
        if config.get("cli_auth_credentials_store", "file") != "file":
            raise ValueError("Codex unsupported auth storage; only explicit file auth is supported")
        unsupported_env = (
            "CODEX_ACCESS_TOKEN",
            "CODEX_AUTH_TOKEN",
            "OPENAI_ORG_ID",
            "OPENAI_ORGANIZATION",
            "OPENAI_PROJECT_ID",
            "OPENAI_API_BASE",
            "AZURE_OPENAI_ENDPOINT",
        )
        if any(env.get(key) for key in unsupported_env):
            raise ValueError("Codex unsupported ambient auth/route source")
        # Native .credentials.json is an MCP OAuth store, not model auth. Do not read/copy it.
        auth_path = source / "auth.json"
        # Verified 0.149.1 precedence: explicit Codex key, then selected file.
        # OPENAI_API_KEY belongs to other providers and is not Codex CLI auth.
        key = env.get("CODEX_API_KEY")
        auth = None
        category = "missing"
        account = None
        if key:
            if not key.strip():
                raise ValueError("Codex unsupported empty ambient auth key")
            auth = json.dumps({"auth_mode": "apikey", "OPENAI_API_KEY": key}).encode()
            category = "apikey"
        elif auth_path.exists():
            try:
                auth = auth_path.read_bytes()
                data = json.loads(auth)
            except (ValueError, OSError) as exc:
                raise ValueError("Codex unsupported or unreadable auth file") from exc
            if not isinstance(data, dict) or set(data) - {
                "auth_mode",
                "OPENAI_API_KEY",
                "tokens",
                "last_refresh",
            }:
                raise ValueError("Codex unsupported auth file shape")
            key, tokens = data.get("OPENAI_API_KEY"), data.get("tokens")
            if (
                isinstance(key, str)
                and key.strip()
                and tokens is None
                and data.get("auth_mode") in (None, "apikey")
            ):
                category = "apikey"
            elif (
                key is None
                and isinstance(tokens, dict)
                and data.get("auth_mode") in (None, "chatgpt")
            ):
                fields = {"id_token", "access_token", "refresh_token", "account_id"}
                if set(tokens) != fields or not all(
                    isinstance(tokens[x], str) and tokens[x] for x in fields
                ):
                    raise ValueError("Codex unsupported ChatGPT auth token shape")
                category = "chatgpt"
                account = tokens["account_id"]
            else:
                raise ValueError("Codex unsupported or ambiguous auth file identity")
        native = {"cli_auth_credentials_store": "file", "model_provider": "openai"}
        method = config.get("forced_login_method")
        if method is not None:
            if (
                method not in ("api", "chatgpt")
                or category != {"api": "apikey", "chatgpt": "chatgpt"}[method]
            ):
                raise ValueError("Codex auth does not satisfy forced login method")
            native["forced_login_method"] = method
        workspace = config.get("forced_chatgpt_workspace_id")
        if workspace is not None:
            if not isinstance(workspace, str) or category != "chatgpt" or workspace != account:
                raise ValueError("Codex auth does not satisfy forced workspace")
            native["forced_chatgpt_workspace_id"] = workspace
        ambient_url = env.pop("OPENAI_BASE_URL", None)
        if ambient_url and config.get("openai_base_url") not in (None, ambient_url):
            raise ValueError("Codex ambiguous OpenAI route sources")
        value = ambient_url or config.get("openai_base_url")
        if value is not None:
            if not isinstance(value, str):
                raise ValueError("Codex unsupported route URL")
            url = urlsplit(value)
            if (
                url.scheme not in ("http", "https")
                or not url.hostname
                or url.username
                or url.password
                or url.query
                or url.fragment
            ):
                raise ValueError("Codex unsupported route URL")
            if category not in ("apikey", "chatgpt"):
                raise ValueError("Codex route requires a supported auth identity")
            native["openai_base_url"] = value
        shared = auth_path if category == "chatgpt" else None
        for key in ("CODEX_API_KEY", "OPENAI_API_KEY"):
            env.pop(key, None)
        return _CodexRuntime(env, auth, category, native, shared)

    def _create_isolated_cli_home(self, runtime: _CodexRuntime | None = None) -> str:
        runtime = runtime or self._resolve_runtime()
        try:
            return self._new_cli_home(runtime)
        except _SignInOnAnotherVolumeError:
            if runtime.auth_path is None:  # only a shared sign-in is ever linked
                raise
            # TEMP can be on another volume than the shared sign-in (2026-10-21: TEMP on F:,
            # ~/.codex on C:), and a hard link cannot leave its volume. The sign-in's own
            # directory is the one place certain to be on that volume and writable, so the
            # home is made there instead; a second refusal is final, never a copy.
            return self._new_cli_home(runtime, runtime.auth_path.parent)

    def _new_cli_home(self, runtime: _CodexRuntime, parent: Path | None = None) -> str:
        cli_home = Path(tempfile.mkdtemp(prefix="llm-council-codex-home-", dir=parent))
        try:
            codex_dir = cli_home / ".codex"
            codex_dir.mkdir()
            if runtime.auth_path is not None:
                # Shared, not copied: see _SIGN_IN_FILE. Refused rather than copied
                # where a hard link is impossible.
                isolated = codex_dir / _SIGN_IN_FILE
                identity, digest = _share_sign_in(runtime.auth_path, isolated)
                self._shared_sign_ins[str(cli_home)] = _SharedSignIn(
                    runtime.auth_path, isolated, identity, digest
                )
            elif runtime.auth is not None:
                try:
                    auth_path = codex_dir / "auth.json"
                    auth_path.touch(mode=0o600)
                    auth_path.write_bytes(runtime.auth)
                except OSError as exc:
                    raise RuntimeError(
                        "Codex auth copy failed; refusing another auth source"
                    ) from exc
            (cli_home / "work").mkdir()
            return str(cli_home)
        except BaseException:
            self._release_home(str(cli_home))
            raise

    def _release_home(self, cli_home: str) -> None:
        """Check a shared sign-in (only once no Codex runs there), then remove the home."""
        shared = self._shared_sign_ins.pop(cli_home, None)
        if shared is not None:
            try:
                _check_shared_sign_in(shared)
            except Exception:
                logger.exception("Codex CLI: the shared sign-in check failed")
        _cleanup_home(cli_home)

    def _isolated_env(self, cli_home: str, runtime: _CodexRuntime) -> dict[str, str]:
        env = dict(runtime.env)
        env.update(HOME=cli_home, CODEX_HOME=str(Path(cli_home) / ".codex"))
        env["XDG_CONFIG_HOME"] = str(Path(cli_home) / ".config")
        # Set, never inherited: from `info` up Codex prints its telemetry to stderr, every
        # event naming the account (user.email, user.account_id), and error reports keep
        # stderr lines. At `warn` real warnings and errors still come through.
        env["RUST_LOG"] = "warn"
        return env

    def _request_timeout(self, request: GenerateRequest) -> float:
        return (
            float(request.timeout_seconds) if request.timeout_seconds is not None else self._timeout
        )

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise asyncio.TimeoutError
        return remaining

    async def _run_cli(
        self,
        cmd: list[str],
        *,
        env: dict[str, str],
        cwd: str,
        deadline: float,
        stdin: Any = None,
        events: bool = False,
        diagnostics_env: dict[str, str] | None = None,
    ) -> tuple[int, _LiveCodexState]:
        """Own spawn, one set of pipe readers and finite cancellation-safe cleanup."""
        try:
            self._remaining(deadline)
        except asyncio.TimeoutError:
            raise RuntimeError("Codex CLI timed out before subprocess spawn")
        proc = None
        job: int | None = None
        readers: list[asyncio.Future[Any]] = []
        state = _LiveCodexState()
        cancellation: asyncio.CancelledError | None = None
        spawn_options: dict[str, Any] = {"start_new_session": True}
        if sys.platform == "win32" and asyncio.create_subprocess_exec is _NATIVE_SPAWN:
            # Born suspended so that it is inside the job before it can spawn:
            # killing only the codex.CMD launcher leaves node/codex.exe running.
            # Only with the real spawn, whose result _adopt_into_job can resume; a
            # substituted spawn (a test seam or wrapper) may hand back an object
            # without the handle, which would leave the child suspended forever.
            spawn_options["creationflags"] = _CREATE_SUSPENDED
        spawn = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *cmd,
                stdin=stdin,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=cwd,
                **spawn_options,
            )
        )
        try:
            # Python 3.10 wait_for() can swallow cancellation racing completion.
            done, _ = await asyncio.wait({spawn}, timeout=self._remaining(deadline))
            if not done:
                raise asyncio.TimeoutError
            proc = spawn.result()
            job = _adopt_into_job(proc)
            if isinstance(proc.stdout, asyncio.StreamReader) and isinstance(
                proc.stderr, asyncio.StreamReader
            ):
                if events:
                    stdout_reader = _read_codex_stdout(proc.stdout, state)
                else:
                    stream = proc.stdout

                    async def read_probe() -> None:
                        data = await stream.read()
                        state.stdout_parts.append(data.decode("utf-8", errors="replace"))

                    stdout_reader = read_probe()
                readers = [
                    asyncio.create_task(stdout_reader),
                    asyncio.create_task(_read_codex_stderr(proc.stderr, state)),
                    asyncio.create_task(proc.wait()),
                ]
                joined = asyncio.gather(*readers)
                readers.append(joined)
                joined_done, _ = await asyncio.wait({joined}, timeout=self._remaining(deadline))
                if not joined_done:
                    raise asyncio.TimeoutError
                joined.result()
            else:
                # Test doubles and alternate asyncio process implementations.
                communicate = asyncio.create_task(proc.communicate())
                readers.append(communicate)
                communicate_done, _ = await asyncio.wait(
                    {communicate}, timeout=self._remaining(deadline)
                )
                if not communicate_done:
                    raise asyncio.TimeoutError
                stdout, stderr = communicate.result()
                if events:
                    for line in stdout.decode("utf-8").splitlines(keepends=True):
                        _ingest_codex_stdout_line(line, state)
                else:
                    state.stdout_parts.append(stdout.decode("utf-8", errors="replace"))
                state.stderr_parts.append(stderr.decode("utf-8", errors="replace"))
            if proc.returncode is None:
                raise RuntimeError("Codex CLI exited without a return code")
            return proc.returncode, state
        except asyncio.TimeoutError:
            if events and state.error_message:
                self._raise_cli_error(
                    state, proc.returncode if proc else -1, diagnostics_env or env
                )
            raise RuntimeError("Codex CLI timed out within the request deadline")
        except asyncio.CancelledError as exc:
            cancellation = exc
            raise
        finally:

            async def cleanup() -> None:
                nonlocal proc
                try:
                    if proc is None:
                        # A cancelled spawn can already have created an OS process.
                        proc = await asyncio.wait_for(asyncio.shield(spawn), _CLEANUP_SECONDS)
                    # Windows: end the whole job first; the helper below only reaches
                    # the direct child there.
                    _terminate_job(job)
                    await terminate_process_tree(proc, _CLEANUP_SECONDS)
                except asyncio.TimeoutError as exc:
                    spawn.cancel()
                    raise RuntimeError("Codex cleanup incomplete: process did not settle") from exc
                finally:
                    with contextlib.suppress(Exception):
                        _close_job(job)
                    for reader in readers:
                        if not reader.done():
                            reader.cancel()
                    if readers:
                        done, pending = await asyncio.wait(readers, timeout=_CLEANUP_SECONDS)
                        for task in done:
                            if not task.cancelled():
                                task.exception()
                        if pending:
                            raise RuntimeError("Codex cleanup incomplete: readers did not settle")

            cleanup_task = asyncio.create_task(cleanup())
            while not cleanup_task.done():
                try:
                    await asyncio.shield(cleanup_task)
                except asyncio.CancelledError as exc:
                    if cancellation is None:
                        cancellation = exc
                except Exception:
                    break
            if cancellation is not None:
                cleanup_error = None if cleanup_task.cancelled() else cleanup_task.exception()
                if cleanup_error:
                    logger.error("Codex cleanup incomplete during cancellation")
                    raise cancellation from cleanup_error
                raise cancellation
            cleanup_task.result()

    async def _verify_native_contract(
        self,
        *,
        env: dict[str, str],
        cwd: str,
        deadline: float,
        diagnostics_env: dict[str, str] | None = None,
        identity: dict[str, str | None] | None = None,
    ) -> str:
        if not self._cli_path:
            raise RuntimeError("Codex CLI not found")
        code, state = await self._run_cli(
            [self._cli_path, "--version"],
            env=env,
            cwd=cwd,
            deadline=deadline,
        )
        version = "".join(state.stdout_parts).strip()
        if version not in _VERIFIED_VERSIONS:
            # Sanitize before constructing an exception, including its cause/context.
            version = self._redact(version, diagnostics_env if diagnostics_env is not None else env)
        if identity is not None:
            identity["cli_version"] = version
        check_verified_version(version, _VERIFIED_VERSIONS, path=self._cli_path, code=code)
        return version

    async def _login_status_text(
        self,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        deadline: float | None = None,
        native_details: dict[str, str] | None = None,
    ) -> str | None:
        if not self._cli_path:
            return None
        cli_home = None
        native: dict[str, str] = {}
        diagnostics_env = env
        if deadline is None:
            deadline = asyncio.get_running_loop().time() + 5
        try:
            if env is None:
                runtime = self._resolve_runtime()
                cli_home = self._create_isolated_cli_home(runtime)
                env = self._isolated_env(cli_home, runtime)
                cwd = str(Path(cli_home) / "work")
                native = runtime.config
                diagnostics_env = runtime.diagnostics_env()
                version = await self._verify_native_contract(
                    env=env, cwd=cwd, deadline=deadline, diagnostics_env=diagnostics_env
                )
                if native_details is not None:
                    native_details.update(cli_path=self._cli_path, cli_version=version)
            if cwd is None:
                raise ValueError("Codex login status requires an isolated working directory")
            code, state = await self._run_cli(
                [self._cli_path, *_config_args(native), "login", "status"],
                env=env,
                cwd=cwd,
                deadline=deadline,
            )
            text = "".join(state.stdout_parts).strip() or "".join(state.stderr_parts).strip()
            if code != 0 and "logged in" in text.lower() and "not logged in" not in text.lower():
                return f"CLI login status failed (exit {code})"
            return self._redact(text, diagnostics_env or env) or f"CLI returned exit code {code}"
        finally:
            if cli_home:
                self._release_home(cli_home)

    async def _generation_catalog(
        self,
        model: str,
        effort: str | None,
        *,
        version: str,
        env: dict[str, str],
        cwd: str,
        deadline: float,
        root: Path,
    ) -> str:
        if not self._cli_path:
            raise RuntimeError("Codex CLI not found")
        code, state = await self._run_cli(
            [self._cli_path, "debug", "models", "--bundled"],
            env=env,
            cwd=cwd,
            deadline=deadline,
        )
        if code:
            raise ValueError("Codex bundled model catalog discovery failed")
        try:
            catalog = json.loads("".join(state.stdout_parts))
        except ValueError as exc:
            raise ValueError("Codex unsupported bundled model catalog JSON") from exc
        if (
            not isinstance(catalog, dict)
            or set(catalog) != {"models"}
            or not isinstance(catalog["models"], list)
        ):
            raise ValueError("Codex unsupported bundled model catalog shape")
        models = catalog["models"]
        slugs = [entry.get("slug") for entry in models if isinstance(entry, dict)]
        if (
            len(slugs) != len(models)
            or any(not isinstance(slug, str) or not slug for slug in slugs)
            or len(set(slugs)) != len(slugs)
        ):
            raise ValueError("Codex unsupported bundled model catalog entries")
        if model not in slugs:
            raise ValueError(f"Codex unsupported model in bundled catalog: {model}")
        selected = models[slugs.index(model)]
        if version == "codex-cli 0.160.1":
            if (
                selected.get("experimental_supported_tools")
                not in ([], ["send_user_message_async", "clock"])
                or selected.get("tool_mode") not in (None, "direct", "code_mode_only")
                or selected.get("multi_agent_version") not in (None, "disabled", "v2")
            ):
                raise ValueError("Codex unsupported model catalog tool shape")
            # These model-level controls override feature flags in exactly 0.160.1.
            selected["experimental_supported_tools"] = []
            if selected.get("tool_mode") is not None:
                selected["tool_mode"] = "direct"
            if selected.get("multi_agent_version") is not None:
                selected["multi_agent_version"] = "disabled"
        if (
            "apply_patch_tool_type" not in selected
            or selected["apply_patch_tool_type"] not in (None, "freeform", "function")
            or selected.get("experimental_supported_tools") != []
        ):
            raise ValueError("Codex unsupported model catalog tool shape")
        levels = selected.get("supported_reasoning_levels")
        if (
            not isinstance(levels, list)
            or not levels
            or any(
                not isinstance(level, dict) or not isinstance(level.get("effort"), str)
                for level in levels
            )
        ):
            raise ValueError("Codex unsupported model catalog reasoning shape")
        if effort is not None and effort not in {level["effort"] for level in levels}:
            raise ValueError(f"Codex unsupported reasoning effort {effort} for model {model}")
        # Preserve model identity, wire format and reasoning; remove only native tools.
        selected["apply_patch_tool_type"] = None
        path = root / "models.json"
        path.write_text(json.dumps(catalog), encoding="utf-8")
        return str(path)

    def _validate_generation_flags(self) -> None:
        flags = iter(shlex.split(self._default_flags))
        for flag in flags:
            if flag == "--skip-git-repo-check":
                continue
            if flag == "--sandbox" and next(flags, None) == "read-only":
                continue
            if flag in ("-c", "--config"):
                value = next(flags, "")
                if value.startswith("developer_instructions="):
                    continue
            raise ValueError("Codex unsupported flags for generation-only auth/route isolation")

    @staticmethod
    def _request_text(request: GenerateRequest) -> tuple[str, str]:
        systems: list[str] = []
        users: list[str] = []
        if request.messages:
            for message in request.messages:
                if not isinstance(message.content, str):
                    raise ValueError("Codex unsupported non-text message content")
                if message.role == "system" and not users:
                    systems.append(message.content)
                elif message.role == "user":
                    users.append(message.content)
                else:
                    raise ValueError(
                        "Codex unsupported history: only leading system then user messages"
                    )
            prompt = "\n\n".join(users)
        else:
            prompt = request.prompt or ""
        if not prompt.strip():
            raise ValueError("Codex requires non-empty user input")
        return "\n\n".join(systems), prompt

    @staticmethod
    def _reasoning_effort(request: GenerateRequest) -> str | None:
        reasoning = request.reasoning
        if reasoning is None:
            return None
        if reasoning.budget_tokens is not None or reasoning.thinking_level is not None:
            raise ValueError("Codex unsupported reasoning budget or thinking_level")
        if not reasoning.enabled:
            if reasoning.effort not in (None, "none"):
                raise ValueError("Codex contradictory disabled reasoning and effort")
            return "none"
        if reasoning.effort is None:
            raise ValueError("Codex reasoning requires an explicit supported effort")
        return reasoning.effort

    @staticmethod
    def _default_reasoning_effort() -> str | None:
        """Effort to ask for when the request carries none; None leaves it to Codex.

        Codex does not check the value, so anything that is not a bare word falls back
        to the default instead of reaching the CLI.
        """
        raw = os.environ.get(_REASONING_EFFORT_ENV)
        effort = _DEFAULT_REASONING_EFFORT if raw is None else raw.strip()
        if not effort:
            return None
        return effort if effort.replace("_", "").isalnum() else _DEFAULT_REASONING_EFFORT

    @staticmethod
    def _redact(text: str, env: dict[str, str]) -> str:
        # Keep conservative long-secret protection without treating short feature flags as secrets.
        secrets = {
            value
            for key, value in env.items()
            if value
            and (
                _CREDENTIAL_ENV_KEY.search(key)
                or (len(value) >= 4 and re.search(r"KEY|TOKEN|SECRET|PASSWORD", key, re.IGNORECASE))
            )
        }
        for value in sorted(secrets, key=len, reverse=True):
            text = text.replace(value, "[REDACTED]")
        text = re.sub(r"(?i)(bearer\s+)[^\s\"']+", r"\1[REDACTED]", text)
        text = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", text)
        return redact_account_identity(text)

    def _error_details(self, stderr_text: str) -> str:
        lines = [line.strip() for line in stderr_text.splitlines() if line.strip()]
        error_lines = [line for line in lines if line.startswith("ERROR:")]
        return "\n".join(error_lines[-3:] or lines[-8:])

    def _raise_cli_error(
        self, state: _LiveCodexState, code: int | None, env: dict[str, str]
    ) -> None:
        details = self._redact(
            "\n".join(
                filter(
                    None,
                    [
                        self._error_details("".join(state.stderr_parts)),
                        state.error_message,
                    ],
                )
            ),
            env,
        )
        error_type = classify_error(details, code or 0)
        labels = {
            ErrorType.BILLING: "BILLING ERROR",
            ErrorType.AUTH: "AUTH ERROR",
            ErrorType.MODEL_UNAVAILABLE: "MODEL UNAVAILABLE",
            ErrorType.RATE_LIMIT: "RATE LIMIT",
        }
        label = labels.get(error_type, f"CLI failed ({error_type.value})")
        raise RuntimeError(f"{label}: {details or f'exit {code}'}")

    async def generate(
        self, request: GenerateRequest
    ) -> GenerateResponse | AsyncIterator[GenerateResponse]:
        identity = native_identity(self._cli_path)
        try:
            response = await self._generate(request, identity)
        except BaseException as exc:
            # Include validation, spawn, timeout, cancellation and cleanup failures.
            exc.__dict__["native_identity"] = dict(identity)
            raise
        response.raw = {**(response.raw or {}), "native_identity": dict(identity)}
        return response

    async def _generate(
        self, request: GenerateRequest, identity: dict[str, str | None]
    ) -> GenerateResponse:
        if request.stream:
            raise NotImplementedError("Streaming not supported for CLI")
        deadline = asyncio.get_running_loop().time() + self._request_timeout(request)
        instructions, prompt = self._request_text(request)
        effort = self._reasoning_effort(request)
        self._check_unsafe_flags()
        self._validate_generation_flags()
        if not self._cli_path:
            raise RuntimeError("Codex CLI not found")
        runtime = self._resolve_runtime()
        if runtime.category == "missing":
            raise ValueError("Codex has no supported auth in the selected root or CODEX_API_KEY")
        cli_home = self._create_isolated_cli_home(runtime)
        version = "not observed"
        try:
            env = self._isolated_env(cli_home, runtime)
            cwd = str(Path(cli_home) / "work")
            version = await self._verify_native_contract(
                env=env,
                cwd=cwd,
                deadline=deadline,
                diagnostics_env=runtime.diagnostics_env(),
                identity=identity,
            )
            model = request.model or self._default_model
            root = Path(cli_home)
            effort_source = "request" if effort is not None else None
            if effort is None and (effort := self._default_reasoning_effort()) is not None:
                effort_source = _REASONING_EFFORT_ENV
            try:
                catalog_path = await self._generation_catalog(
                    model,
                    effort,
                    version=version,
                    env=env,
                    cwd=cwd,
                    deadline=deadline,
                    root=root,
                )
            except ValueError as exc:
                if effort_source != _REASONING_EFFORT_ENV or "reasoning effort" not in str(exc):
                    raise
                # An implicit default must never fail a call the request did not shape.
                logger.warning("Codex CLI: %s; leaving the effort to Codex", exc)
                effort = effort_source = None
                catalog_path = await self._generation_catalog(
                    model,
                    effort,
                    version=version,
                    env=env,
                    cwd=cwd,
                    deadline=deadline,
                    root=root,
                )
            output_path = root / "last-message.txt"
            schema_path = None
            if request.structured_output:
                schema_path = root / "schema.json"
                schema_path.write_text(
                    json.dumps(
                        _prepare_schema_for_codex(dict(request.structured_output.json_schema))
                    ),
                    encoding="utf-8",
                )
            stdin_path = root / "prompt.txt"
            stdin_path.write_text(prompt, encoding="utf-8")
            cmd = self._build_command(
                model=model,
                output_last_message_path=str(output_path),
                output_schema_path=str(schema_path) if schema_path else None,
            )
            cmd.extend(["--ignore-user-config", "--ignore-rules", "--ephemeral", "--strict-config"])
            cmd.extend(_config_args(runtime.config))
            cmd.extend(_config_args(_GENERATION_CONFIG))
            if version == "codex-cli 0.160.1":
                cmd.extend(["-c", "features.goals=false"])
            cmd.extend(["-c", f"model_catalog_json={json.dumps(catalog_path)}"])
            if instructions:
                instruction_path = root / "instructions.txt"
                instruction_path.write_text(instructions, encoding="utf-8")
                cmd.extend(["-c", f"model_instructions_file={json.dumps(str(instruction_path))}"])
            if effort is not None:
                cmd.extend(["-c", f"model_reasoning_effort={json.dumps(effort)}"])
            with stdin_path.open("rb") as stdin:
                code, state = await self._run_cli(
                    cmd,
                    env=env,
                    cwd=cwd,
                    deadline=deadline,
                    stdin=stdin,
                    events=True,
                    diagnostics_env=runtime.diagnostics_env(),
                )
            if code != 0 or state.error_message:
                self._raise_cli_error(state, code, runtime.diagnostics_env())
            if state.protocol_error:
                raise RuntimeError(f"Codex CLI {state.protocol_error}")
            if not state.saw_turn_completed:
                raise RuntimeError("Codex CLI missing terminal turn.completed")
            output = (
                output_path.read_text(encoding="utf-8")
                if output_path.exists()
                else state.agent_message
            )
            if not output.strip():
                raise RuntimeError("Codex CLI empty final output")
            return GenerateResponse(
                text=output,
                content=output,
                usage=state.usage,
                raw={
                    "stdout": "".join(state.stdout_parts),
                    "native_version": version,
                    "cli_path": self._cli_path,
                    "model": model,
                    "auth_category": runtime.category,
                    "reasoning": {
                        "effort": effort,
                        "status": "mapped" if effort else "uncontrolled",
                    },
                    # "request", the LLM_COUNCIL_CODEX_REASONING_EFFORT default, or None.
                    "reasoning_effort_source": effort_source,
                },
            )
        except (ValueError, RuntimeError) as exc:
            raise type(exc)(
                f"{self._cli_path!r} ({identity['cli_version'] or version}): "
                f"{self._redact(str(exc), runtime.diagnostics_env())}"
            ) from exc
        finally:
            self._release_home(cli_home)

    async def supports(self, capability: str) -> bool:
        return self.supports_capability(capability)

    async def doctor(self) -> DoctorResult:
        if not self._cli_path:
            return DoctorResult(ok=False, message="CLI not found")

        native_details: dict[str, str] = {}
        try:
            status_text = await self._login_status_text(native_details=native_details)
        except asyncio.TimeoutError:
            return DoctorResult(ok=False, message="CLI login status check timed out")
        except Exception as exc:  # pragma: no cover - defensive path
            return DoctorResult(ok=False, message=f"CLI login status check failed: {exc}")

        status_text = status_text or "CLI login status check failed"
        lowered = status_text.lower()
        if native_details:
            status_text = (
                f"{native_details['cli_path']} ({native_details['cli_version']}): {status_text}"
            )

        if "not logged in" in lowered or "logged out" in lowered:
            return DoctorResult(ok=False, message=status_text, details=native_details or None)
        if "logged in" in lowered:
            return DoctorResult(ok=True, message=status_text, details=native_details or None)
        return DoctorResult(ok=False, message=status_text, details=native_details or None)


def _register() -> None:
    from llm_council.providers.registry import get_registry

    with contextlib.suppress(ValueError):
        get_registry().register_provider("codex", CodexCLIProvider)


_register()
