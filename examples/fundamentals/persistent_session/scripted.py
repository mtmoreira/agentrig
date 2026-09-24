"""Run two opaque requests through one bounded local process."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile

from agentrig.core import CancellationSource, RunContext, RunId
from agentrig.core.clock import SystemClock
from agentrig.core.identity import Uuid4IdGenerator
from agentrig.integrations import PersistentProcessSession, ProcessSpec


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionReport:
    first_request_count: int
    second_request_count: int
    observed_values: tuple[str, str]
    final_state: str

    def to_json(self) -> str:
        return json.dumps({
            "final_state": self.final_state,
            "first_request_count": self.first_request_count,
            "observed_values": list(self.observed_values),
            "second_request_count": self.second_request_count,
        }, indent=2, sort_keys=True)


def _context() -> RunContext:
    clock = SystemClock()
    return RunContext.create_root(
        clock=clock,
        id_generator=Uuid4IdGenerator(RunId),
        cancellation=CancellationSource().token,
    )


async def run_example() -> SessionReport:
    worker = str(Path(__file__).with_name("worker.py").resolve())
    with tempfile.TemporaryDirectory(prefix="agentrig-session-example-") as root:
        session = PersistentProcessSession(ProcessSpec(
            argv=(str(Path(sys.executable).resolve()), "-I", "-S", worker),
            cwd=str(Path(root).resolve()),
            timeout_seconds=2,
            max_output_bytes=4096,
        ))
        context = _context()
        await session.start(context)
        await session.health(context)
        first = await session.request({"value": "alpha"}, context)
        second = await session.request({"value": "beta"}, context)
        await session.shutdown(context)
        if not isinstance(first, Mapping) or not isinstance(second, Mapping):
            raise AssertionError("example worker returned an invalid payload")
        first_received = first["received"]
        second_received = second["received"]
        if (not isinstance(first_received, Mapping)
                or not isinstance(second_received, Mapping)):
            raise AssertionError("example worker lost its request payload")
        first_count = first["request_count"]
        second_count = second["request_count"]
        if type(first_count) is not int or type(second_count) is not int:
            raise AssertionError("example worker returned an invalid request count")
        return SessionReport(
            first_request_count=first_count,
            second_request_count=second_count,
            observed_values=(str(first_received["value"]), str(second_received["value"])),
            final_state=session.state,
        )


def main() -> None:
    print(asyncio.run(run_example()).to_json())


if __name__ == "__main__":
    main()
