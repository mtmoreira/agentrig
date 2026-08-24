from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import patch

from agentrig.capabilities import ToolInvocation
from agentrig.core import CancellationSource, ExecutionStatus, RunContext, RunId
from agentrig.integrations import CommandInput, CommandTool, DetachedCommandTool


@dataclass(frozen=True)
class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 8, 22, 8, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return 100.0


@dataclass
class Ids:
    value: int = 0

    def generate(self) -> RunId:
        self.value += 1
        return RunId(f"run-{self.value}")


def context() -> RunContext:
    return RunContext.create_root(
        clock=FixedClock(),
        id_generator=Ids(),
        cancellation=CancellationSource().token,
    )


class CommandToolTest(unittest.IsolatedAsyncioTestCase):
    def tool(self, *, bound: int = 1000) -> CommandTool:
        return CommandTool(
            tool_id="process.printf",
            version="1",
            purpose="Print deterministic test output.",
            executable="/usr/bin/printf",
            working_directory="/private/tmp",
            max_output_bytes=bound,
        )

    async def test_executes_without_shell_and_returns_bounded_output(self) -> None:
        tool = self.tool(bound=4)
        invocation = ToolInvocation(
            invocation_id="invoke-1",
            contract=tool.contract,
            input=CommandInput(arguments=("abcdef",)),
        )

        process = FakeProcess(stdout=b"abcdef")
        with patch(
            "agentrig.integrations.process.asyncio.create_subprocess_exec",
            return_value=process,
        ) as create_process:
            result = await tool.invoke(invocation, context())

        self.assertEqual(result.status, ExecutionStatus.SUCCEEDED)
        self.assertEqual(result.unwrap().stdout, "abcd")
        self.assertTrue(result.unwrap().truncated)
        self.assertEqual(create_process.call_args.args[:2], ("/usr/bin/printf", "abcdef"))
        self.assertNotIn("shell", create_process.call_args.kwargs)
        self.assertEqual(create_process.call_args.kwargs["env"], {})

    async def test_rejects_environment_outside_explicit_allowlist(self) -> None:
        tool = self.tool()
        invocation = ToolInvocation(
            invocation_id="invoke-2",
            contract=tool.contract,
            input=CommandInput(environment={"PATH": "/usr/bin"}),
        )

        result = await tool.invoke(invocation, context())

        self.assertEqual(result.status, ExecutionStatus.FAILED)
        if result.failure is None:
            raise AssertionError("failed command must include details")
        self.assertEqual(result.failure.code, "command.environment_not_allowed")

    def test_input_detaches_private_arguments_and_environment_from_repr(self) -> None:
        environment = {"PRIVATE_VALUE": "secret"}
        command_input = CommandInput(
            arguments=("private-argument",),
            environment=environment,
        )
        environment["PRIVATE_VALUE"] = "changed"

        if command_input.environment is None:
            raise AssertionError("command environment must be present")
        self.assertEqual(command_input.environment["PRIVATE_VALUE"], "secret")
        self.assertNotIn("private-argument", repr(command_input))
        self.assertNotIn("secret", repr(command_input))

    async def test_observes_cancellation_before_process_creation(self) -> None:
        source = CancellationSource()
        source.cancel("stop")
        run_context = RunContext.create_root(
            clock=FixedClock(),
            id_generator=Ids(),
            cancellation=source.token,
        )
        tool = self.tool()
        invocation = ToolInvocation(
            invocation_id="invoke-3",
            contract=tool.contract,
            input=CommandInput(),
        )

        with self.assertRaises(asyncio.CancelledError):
            await tool.invoke(invocation, run_context)

    async def test_detached_tool_acknowledges_process_without_streams(self) -> None:
        tool = DetachedCommandTool(
            tool_id="process.viewer",
            version="1",
            purpose="Open a local viewer.",
            executable="/usr/bin/true",
            working_directory="/private/tmp",
        )
        invocation = ToolInvocation(
            invocation_id="invoke-4",
            contract=tool.contract,
            input=CommandInput(arguments=("private-wave.vcd",)),
        )

        process = FakeProcess(pid=4321)
        with patch(
            "agentrig.integrations.process.asyncio.create_subprocess_exec",
            return_value=process,
        ) as create_process:
            result = await tool.invoke(invocation, context())

        self.assertEqual(result.status, ExecutionStatus.SUCCEEDED)
        self.assertEqual(result.unwrap().process_id, 4321)
        self.assertEqual(
            create_process.call_args.kwargs["stdout"],
            asyncio.subprocess.DEVNULL,
        )
        self.assertEqual(
            create_process.call_args.kwargs["stderr"],
            asyncio.subprocess.DEVNULL,
        )
        self.assertNotIn("private-wave.vcd", repr(result.unwrap()))


if __name__ == "__main__":
    unittest.main()


@dataclass
class FakeProcess:
    stdout: bytes = b""
    stderr: bytes = b""
    returncode: int | None = 0
    pid: int = 1234

    async def communicate(self) -> tuple[bytes, bytes]:
        return self.stdout, self.stderr

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode if self.returncode is not None else 0
