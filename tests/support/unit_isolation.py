"""Process-wide isolation guard for unit tests."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

_PROCESS_EVENTS = frozenset(
    {
        "os.posix_spawn",
        "os.posix_spawnp",
        "os.system",
        "subprocess.Popen",
    },
)

_NETWORK_EVENTS = frozenset(
    {
        "socket.bind",
        "socket.connect",
        "socket.getaddrinfo",
        "socket.gethostbyaddr",
        "socket.gethostbyname",
        "socket.gethostbyname_ex",
        "socket.sendto",
    },
)

_guard_installed = False


class UnitIsolationViolation(RuntimeError):
    """Raised when a unit test attempts network or process access."""


def _enforce_unit_isolation(event: str, _arguments: Sequence[Any]) -> None:
    if event in _NETWORK_EVENTS or event in _PROCESS_EVENTS:
        raise UnitIsolationViolation(
            f"unit tests must not perform external access (audit event: {event})",
        )


def install_unit_isolation_guard() -> None:
    """Install the irreversible process-wide audit hook exactly once."""
    global _guard_installed

    if _guard_installed:
        return

    sys.addaudithook(_enforce_unit_isolation)
    _guard_installed = True
