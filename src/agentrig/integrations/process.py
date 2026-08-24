"""Guarded local-process implementation of AgentRig's typed tool boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityKind,
    DataRetention,
    ToolContract,
    ToolInvocation,
    ToolResult,
    ToolSchema,
)
from agentrig.core import EffectProfile, Failure, FailureKind, JsonValue, RunContext


@dataclass(frozen=True, slots=True, kw_only=True)
class CommandInput:
    """Arguments and allowlisted environment values for one invocation."""

    arguments: tuple[str, ...] = field(default=(), repr=False)
    environment: Mapping[str, str] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        arguments = tuple(self.arguments)
        _validate_strings("command argument", arguments)
        environment = dict(self.environment or {})
        _validate_environment(environment)
        object.__setattr__(self, "arguments", arguments)
        object.__setattr__(
            self,
            "environment",
            MappingProxyType(environment),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CommandOutput:
    """Bounded decoded process output."""

    exit_code: int
    stdout: str = field(repr=False)
    stderr: str = field(repr=False)
    truncated: bool

    def __post_init__(self) -> None:
        if isinstance(self.exit_code, bool) or not isinstance(
            self.exit_code, int
        ):
            raise TypeError("command exit_code must be an integer")
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise TypeError("command output streams must be strings")
        if not isinstance(self.truncated, bool):
            raise TypeError("command truncated must be a bool")


@dataclass(frozen=True, slots=True, kw_only=True)
class DetachedCommandOutput:
    """Acknowledgement that a detached process was created."""

    process_id: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.process_id, bool)
            or not isinstance(self.process_id, int)
            or self.process_id < 1
        ):
            raise ValueError("detached command process_id must be positive")


class CommandTool:
    """Execute one fixed executable without a shell or inherited environment."""

    def __init__(
        self,
        *,
        tool_id: str,
        version: str,
        purpose: str,
        executable: str,
        working_directory: str,
        base_arguments: tuple[str, ...] = (),
        allowed_environment_variables: tuple[str, ...] = (),
        effect_profile: EffectProfile = EffectProfile.IDEMPOTENT,
        timeout_seconds: float = 60.0,
        max_output_bytes: int = 1_000_000,
    ) -> None:
        executable_path = Path(executable)
        working_path = Path(working_directory)
        if not executable_path.is_absolute():
            raise ValueError("command executable must be an absolute path")
        if not working_path.is_absolute():
            raise ValueError("command working directory must be an absolute path")
        if working_path == Path("/") or ".." in working_path.parts:
            raise ValueError("command working directory must be bounded")
        if timeout_seconds <= 0:
            raise ValueError("command timeout must be positive")
        if isinstance(max_output_bytes, bool) or max_output_bytes < 1:
            raise ValueError("command output bound must be a positive integer")
        _validate_strings("command base argument", base_arguments)
        _validate_strings(
            "command environment variable",
            allowed_environment_variables,
        )
        if len(set(allowed_environment_variables)) != len(
            allowed_environment_variables
        ):
            raise ValueError("command environment allowlist contains duplicates")

        self._executable = str(executable_path)
        self._working_directory = str(working_path)
        self._base_arguments = tuple(base_arguments)
        self._allowed_environment = frozenset(allowed_environment_variables)
        self._timeout_seconds = timeout_seconds
        self._max_output_bytes = max_output_bytes
        self._contract = ToolContract[CommandInput, CommandOutput](
            descriptor=CapabilityDescriptor(
                capability_id=tool_id,
                version=version,
                kind=CapabilityKind.TOOL,
                data_retention=DataRetention.NOT_RETAINED,
            ),
            purpose=purpose,
            effect_profile=effect_profile,
            input_schema=_input_schema(tool_id),
            output_schema=_output_schema(tool_id),
        )

    @property
    def contract(self) -> ToolContract[CommandInput, CommandOutput]:
        return self._contract

    async def invoke(
        self,
        invocation: ToolInvocation[CommandInput, CommandOutput],
        context: RunContext,
    ) -> ToolResult[CommandOutput]:
        if invocation.contract != self._contract:
            raise ValueError("command invocation contract does not match tool")
        process: asyncio.subprocess.Process | None = None
        try:
            context.cancellation.raise_if_cancelled()
            if context.deadline is not None:
                context.deadline.raise_if_expired(context.clock)
            arguments = tuple(invocation.input.arguments)
            _validate_strings("command argument", arguments)
            supplied_environment = dict(invocation.input.environment or {})
            if set(supplied_environment) - self._allowed_environment:
                return _failure(
                    invocation,
                    "command environment exceeds its allowlist",
                    "command.environment_not_allowed",
                )
            _validate_environment(supplied_environment)
            environment = {
                name: supplied_environment[name]
                for name in sorted(supplied_environment)
            }
            process = await asyncio.create_subprocess_exec(
                self._executable,
                *self._base_arguments,
                *arguments,
                cwd=self._working_directory,
                env=environment,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            timeout = self._timeout_seconds
            if context.deadline is not None:
                timeout = min(
                    timeout,
                    context.deadline.remaining_seconds(context.clock),
                )
            communicate = asyncio.create_task(process.communicate())
            cancelled = asyncio.create_task(context.cancellation.wait_cancelled())
            try:
                done, _ = await asyncio.wait(
                    {communicate, cancelled},
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if communicate not in done:
                    await _terminate(process)
                    code = (
                        "command.cancelled"
                        if cancelled in done
                        else "command.deadline_exceeded"
                    )
                    kind = (
                        FailureKind.CANCELLED
                        if cancelled in done
                        else FailureKind.DEADLINE_EXCEEDED
                    )
                    return ToolResult.from_failure(
                        invocation=invocation,
                        failure=Failure(
                            kind=kind,
                            message="command execution did not complete",
                            code=code,
                        ),
                    )
                stdout, stderr = communicate.result()
            finally:
                communicate.cancel()
                cancelled.cancel()
                await asyncio.gather(
                    communicate,
                    cancelled,
                    return_exceptions=True,
                )
            output = _bounded_output(
                process.returncode,
                stdout,
                stderr,
                self._max_output_bytes,
            )
            return ToolResult.succeeded(
                invocation=invocation,
                encoded_output=self._contract.output_schema.encode(output),
            )
        except asyncio.CancelledError:
            if process is not None:
                await _terminate(process)
            raise
        except (OSError, ValueError):
            return _failure(
                invocation,
                "command execution failed",
                "command.execution_failed",
            )


class DetachedCommandTool:
    """Start one fixed executable without waiting for process completion."""

    def __init__(
        self,
        *,
        tool_id: str,
        version: str,
        purpose: str,
        executable: str,
        working_directory: str,
        base_arguments: tuple[str, ...] = (),
        allowed_environment_variables: tuple[str, ...] = (),
    ) -> None:
        executable_path = Path(executable)
        working_path = Path(working_directory)
        if not executable_path.is_absolute():
            raise ValueError("command executable must be an absolute path")
        if not working_path.is_absolute():
            raise ValueError("command working directory must be an absolute path")
        if working_path == Path("/") or ".." in working_path.parts:
            raise ValueError("command working directory must be bounded")
        _validate_strings("command base argument", base_arguments)
        _validate_strings(
            "command environment variable",
            allowed_environment_variables,
        )
        if len(set(allowed_environment_variables)) != len(
            allowed_environment_variables
        ):
            raise ValueError("command environment allowlist contains duplicates")

        self._executable = str(executable_path)
        self._working_directory = str(working_path)
        self._base_arguments = tuple(base_arguments)
        self._allowed_environment = frozenset(allowed_environment_variables)
        self._contract = ToolContract[CommandInput, DetachedCommandOutput](
            descriptor=CapabilityDescriptor(
                capability_id=tool_id,
                version=version,
                kind=CapabilityKind.TOOL,
                data_retention=DataRetention.NOT_RETAINED,
            ),
            purpose=purpose,
            effect_profile=EffectProfile.NON_REPEATABLE,
            input_schema=_input_schema(tool_id),
            output_schema=_detached_output_schema(tool_id),
        )

    @property
    def contract(self) -> ToolContract[CommandInput, DetachedCommandOutput]:
        return self._contract

    async def invoke(
        self,
        invocation: ToolInvocation[CommandInput, DetachedCommandOutput],
        context: RunContext,
    ) -> ToolResult[DetachedCommandOutput]:
        if invocation.contract != self._contract:
            raise ValueError("command invocation contract does not match tool")
        try:
            context.cancellation.raise_if_cancelled()
            if context.deadline is not None:
                context.deadline.raise_if_expired(context.clock)
            arguments = tuple(invocation.input.arguments)
            _validate_strings("command argument", arguments)
            supplied_environment = dict(invocation.input.environment or {})
            if set(supplied_environment) - self._allowed_environment:
                return _detached_failure(
                    invocation,
                    "command environment exceeds its allowlist",
                    "command.environment_not_allowed",
                )
            _validate_environment(supplied_environment)
            environment = {
                name: supplied_environment[name]
                for name in sorted(supplied_environment)
            }
            process = await asyncio.create_subprocess_exec(
                self._executable,
                *self._base_arguments,
                *arguments,
                cwd=self._working_directory,
                env=environment,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )
            output = DetachedCommandOutput(process_id=process.pid)
            return ToolResult.succeeded(
                invocation=invocation,
                encoded_output=self._contract.output_schema.encode(output),
            )
        except asyncio.CancelledError:
            raise
        except (OSError, ValueError):
            return _detached_failure(
                invocation,
                "command execution failed",
                "command.execution_failed",
            )


async def _terminate(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        process.terminate()
        await asyncio.wait_for(process.wait(), timeout=1.0)
    except (ProcessLookupError, TimeoutError):
        if process.returncode is None:
            process.kill()
            await process.wait()


def _bounded_output(
    exit_code: int | None,
    stdout: bytes,
    stderr: bytes,
    bound: int,
) -> CommandOutput:
    combined = stdout + stderr
    truncated = len(combined) > bound
    remaining = bound
    bounded_stdout = stdout[:remaining]
    remaining -= len(bounded_stdout)
    bounded_stderr = stderr[:remaining]
    return CommandOutput(
        exit_code=exit_code if exit_code is not None else -1,
        stdout=bounded_stdout.decode("utf-8", errors="replace"),
        stderr=bounded_stderr.decode("utf-8", errors="replace"),
        truncated=truncated,
    )


def _input_schema(tool_id: str) -> ToolSchema[CommandInput]:
    return ToolSchema(
        schema_id=f"{tool_id}.input.v1",
        json_schema={"type": "object", "required": ["arguments"]},
        encoder=_encode_input,
        decoder=_decode_input,
    )


def _output_schema(tool_id: str) -> ToolSchema[CommandOutput]:
    return ToolSchema(
        schema_id=f"{tool_id}.output.v1",
        json_schema={
            "type": "object",
            "required": ["exit_code", "stdout", "stderr", "truncated"],
        },
        encoder=_encode_output,
        decoder=_decode_output,
    )


def _detached_output_schema(tool_id: str) -> ToolSchema[DetachedCommandOutput]:
    return ToolSchema(
        schema_id=f"{tool_id}.output.v1",
        json_schema={"type": "object", "required": ["process_id"]},
        encoder=_encode_detached_output,
        decoder=_decode_detached_output,
    )


def _encode_input(value: CommandInput) -> JsonValue:
    return {
        "arguments": value.arguments,
        "environment": dict(value.environment or {}),
    }


def _decode_input(value: JsonValue) -> CommandInput:
    if not isinstance(value, Mapping):
        raise ValueError("command input must be an object")
    arguments = value.get("arguments")
    environment = value.get("environment", {})
    if not isinstance(arguments, (list, tuple)) or not isinstance(
        environment, Mapping
    ):
        raise ValueError("command input has invalid fields")
    copied_arguments: list[str] = []
    for argument in arguments:
        if not isinstance(argument, str) or "\x00" in argument:
            raise ValueError(
                "command argument must be a string without NUL bytes"
            )
        copied_arguments.append(argument)
    copied_environment: dict[str, str] = {}
    for name, item in environment.items():
        if not isinstance(name, str) or not isinstance(item, str):
            raise ValueError("command environment is invalid")
        copied_environment[name] = item
    _validate_environment(copied_environment)
    return CommandInput(
        arguments=tuple(copied_arguments),
        environment=copied_environment,
    )


def _encode_output(value: CommandOutput) -> JsonValue:
    return {
        "exit_code": value.exit_code,
        "stdout": value.stdout,
        "stderr": value.stderr,
        "truncated": value.truncated,
    }


def _decode_output(value: JsonValue) -> CommandOutput:
    if not isinstance(value, Mapping):
        raise ValueError("command output must be an object")
    exit_code = value.get("exit_code")
    stdout = value.get("stdout")
    stderr = value.get("stderr")
    truncated = value.get("truncated")
    if (
        isinstance(exit_code, bool)
        or not isinstance(exit_code, int)
        or not isinstance(stdout, str)
        or not isinstance(stderr, str)
        or not isinstance(truncated, bool)
    ):
        raise ValueError("command output has invalid fields")
    return CommandOutput(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        truncated=truncated,
    )


def _encode_detached_output(value: DetachedCommandOutput) -> JsonValue:
    return {"process_id": value.process_id}


def _decode_detached_output(value: JsonValue) -> DetachedCommandOutput:
    if not isinstance(value, Mapping):
        raise ValueError("detached command output must be an object")
    process_id = value.get("process_id")
    if isinstance(process_id, bool) or not isinstance(process_id, int):
        raise ValueError("detached command output has invalid fields")
    return DetachedCommandOutput(process_id=process_id)


def _validate_strings(name: str, values: tuple[object, ...]) -> None:
    for value in values:
        if not isinstance(value, str) or "\x00" in value:
            raise ValueError(f"{name} must be a string without NUL bytes")


def _validate_environment(environment: Mapping[str, str]) -> None:
    for name, value in environment.items():
        if (
            not isinstance(name, str)
            or not name
            or "=" in name
            or "\x00" in name
            or not isinstance(value, str)
            or "\x00" in value
        ):
            raise ValueError("command environment is invalid")


def _failure(
    invocation: ToolInvocation[CommandInput, CommandOutput],
    message: str,
    code: str,
) -> ToolResult[CommandOutput]:
    return ToolResult.from_failure(
        invocation=invocation,
        failure=Failure(
            kind=FailureKind.INVALID_INPUT,
            message=message,
            code=code,
        ),
    )


def _detached_failure(
    invocation: ToolInvocation[CommandInput, DetachedCommandOutput],
    message: str,
    code: str,
) -> ToolResult[DetachedCommandOutput]:
    return ToolResult.from_failure(
        invocation=invocation,
        failure=Failure(
            kind=FailureKind.INVALID_INPUT,
            message=message,
            code=code,
        ),
    )
