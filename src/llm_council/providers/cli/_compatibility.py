"""Admit only native releases covered by the synthetic contract matrix."""

from __future__ import annotations

import os
import re
import shutil

# The batch file npm writes for a package bin that is a native executable: it finds its
# own directory (%~dp0) and runs the executable with every argument (%*). Line breaks,
# letter case and the spaces before %* may vary; nothing else may.
_NPM_EXECUTABLE_SHIM = re.compile(
    r"@ECHO off\r?\n"
    r"GOTO start\r?\n"
    r":find_dp0\r?\n"
    r"SET dp0=%~dp0\r?\n"
    r"EXIT /b\r?\n"
    r":start\r?\n"
    r"SETLOCAL\r?\n"
    r"CALL :find_dp0\r?\n"
    r'"%dp0%\\(?P<target>[^"%!\r\n]+\.exe)" +%\*(?:\r?\n)?',
    re.IGNORECASE,
)


def selected_cli_path(path: str | None, command: str) -> str | None:
    """Resolve before isolated children change cwd; retain the selected entrypoint."""
    selected = path or shutil.which(command)
    if selected is None:
        return None
    return os.path.abspath(shutil.which(selected) or selected)


def npm_shim_target(path: str) -> str | None:
    """The executable an npm cmd shim runs with all of its arguments, or None.

    Any other batch file (another template, one more command or argument, a variable in
    the path, a file that does not exist) names nothing: what it would run is unknown.
    """
    try:
        with open(path, "rb") as shim_file:
            text = shim_file.read().decode("ascii")
    except (OSError, UnicodeDecodeError):
        return None
    shim = _NPM_EXECUTABLE_SHIM.fullmatch(text)
    if shim is None:
        return None
    target = os.path.normpath(os.path.join(os.path.dirname(path), *shim["target"].split("\\")))
    return target if os.path.isfile(target) else None


def native_identity(path: str | None) -> dict[str, str | None]:
    """Safe, request-local identity, not a dump of subprocess output or environment."""
    return {
        "cli_path": path,
        "cli_realpath": os.path.realpath(path) if path else None,
        "cli_version": None,
    }


def check_verified_version(
    observed: str, verified: tuple[str, ...], *, path: str, code: int | None
) -> None:
    """Version/help compatibility cannot prove tool or instruction isolation."""
    if code != 0 or observed not in verified:
        raise RuntimeError(
            f"unsupported or unverified CLI version at {path!r}: observed {observed!r} "
            f"(exit {code}); native-tested versions: {', '.join(verified)}. "
            "Untested updates remain unsupported and require native contract verification."
        )
