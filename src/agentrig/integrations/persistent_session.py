"""Bounded, correlated JSON-lines sessions for trusted local processes."""

from __future__ import annotations

import asyncio
import json
import os
import signal
from collections.abc import Awaitable, Callable
from typing import NoReturn, TypeVar, cast

from agentrig.core import (
    DeadlineExceeded,
    JsonValue,
    RunCancelled,
    RunContext,
    redact_private_trace_arguments,
)
from agentrig.core._json import freeze_json_value, thaw_json_value
from agentrig.integrations.bounded_process import ProcessSpec

_T = TypeVar("_T")
_VERSION = "1.0"
_REAP_SECONDS = 2.0
_SAFE_CODES = frozenset({
    "session_not_ready", "session_closed", "session_failed",
    "session_protocol", "session_correlation", "session_eof",
    "session_output_limit", "session_input_limit", "session_timeout",
    "session_cancelled", "session_spawn_failed", "session_cleanup_failed",
})


class SessionFailure(RuntimeError):
    """Content-free terminal failure requiring explicit recovery."""

    def __init__(self, code: str) -> None:
        self.code = code if code in _SAFE_CODES else "session_protocol"
        super().__init__(self.code)


class PersistentProcessSession:
    """Own one no-shell process; never replay a failed request automatically.

    The worker emits ``ready`` first. Requests and responses carry a version,
    kind, and correlation ID; payloads are opaque JSON. ``ping``/``pong`` and
    ``shutdown``/``stopped`` are lifecycle frames. Any protocol or execution
    fault terminates and reaps the owned process before a typed failure escapes.
    A failed session can only restart through ``recover``.
    """

    def __init__(self, spec: ProcessSpec) -> None:
        if not isinstance(spec, ProcessSpec) or not spec.reject_overflow:
            raise ValueError("persistent session requires a rejecting ProcessSpec")
        self._spec = spec
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_overflow = False
        self._state = "new"
        self._next_id = 1
        self._lock = asyncio.Lock()

    @property
    def state(self) -> str:
        """Return only the safe lifecycle state, never process content."""
        return self._state

    async def start(self, context: RunContext) -> None:
        async with self._lock:
            if self._state != "new":
                raise SessionFailure("session_not_ready")
            try:
                self._capture(context, "process.request", {
                    "operation": "start",
                    "argv": redact_private_trace_arguments(self._spec.argv),
                    "cwd": self._spec.cwd,
                    "environment": "not_captured",
                    "shell": False,
                })
                await self._bounded(self._spawn_and_ready, context)
                self._capture(context, "process.response", {
                    "operation": "start", "state": "ready",
                })
                self._state = "ready"
            except BaseException as error:
                await self._fail_and_reap(error, context, "start")

    async def request(self, payload: JsonValue, context: RunContext) -> JsonValue:
        """Send one bounded opaque request; wrong IDs never feed later calls."""
        frozen = freeze_json_value("session request payload", payload)
        async with self._lock:
            self._require_ready()
            request_id = str(self._next_id)
            self._next_id += 1
            try:
                self._capture(context, "process.request", {
                    "operation": "request", "id": request_id, "payload": frozen,
                })
                response = await self._bounded(
                    lambda: self._request(frozen, request_id), context
                )
                self._capture(context, "process.response", {
                    "operation": "request", "id": request_id, "payload": response,
                })
                return response
            except BaseException as error:
                await self._fail_and_reap(error, context, "request")

    async def health(self, context: RunContext) -> None:
        async with self._lock:
            self._require_ready()
            try:
                self._capture(context, "process.request", {"operation": "health"})
                await self._bounded(self._health, context)
                self._capture(context, "process.response", {
                    "operation": "health", "state": "ready",
                })
            except BaseException as error:
                await self._fail_and_reap(error, context, "health")

    async def shutdown(self, context: RunContext) -> None:
        async with self._lock:
            if self._state == "closed":
                return
            if self._state != "ready":
                await self._reap()
                self._state = "closed"
                return
            try:
                self._capture(context, "process.request", {"operation": "shutdown"})
                await self._bounded(self._shutdown, context)
                self._capture(context, "process.response", {
                    "operation": "shutdown", "state": "closed",
                })
                self._state = "closed"
                cancellation = await self._reap_shielded()
                if cancellation is not None:
                    raise cancellation
            except BaseException as error:
                await self._fail_and_reap(error, context, "shutdown")

    async def recover(self, context: RunContext) -> None:
        """Explicitly spawn a fresh process after a terminal failure."""
        async with self._lock:
            if self._state != "failed":
                raise SessionFailure("session_not_ready")
            self._state = "new"
            try:
                self._capture(context, "process.request", {
                    "operation": "recover",
                    "argv": redact_private_trace_arguments(self._spec.argv),
                    "cwd": self._spec.cwd,
                    "environment": "not_captured",
                    "shell": False,
                })
                await self._bounded(self._spawn_and_ready, context)
                self._capture(context, "process.response", {
                    "operation": "recover", "state": "ready",
                })
                self._state = "ready"
            except BaseException as error:
                await self._fail_and_reap(error, context, "recover")

    def _require_ready(self) -> None:
        if self._state == "closed":
            raise SessionFailure("session_closed")
        if self._state == "failed":
            raise SessionFailure("session_failed")
        if self._state != "ready":
            raise SessionFailure("session_not_ready")

    @staticmethod
    def _capture(context: RunContext, kind: str,
                 content: dict[str, JsonValue]) -> None:
        capture = context.private_trace_capture
        if capture is not None and capture.include_process:
            capture.record(context, kind=kind, metadata={}, content=content)

    async def _spawn_and_ready(self) -> None:
        if os.name != "posix":
            raise SessionFailure("session_spawn_failed")
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self._spec.argv,
                cwd=self._spec.cwd,
                env=dict(self._spec.environment),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                limit=self._spec.max_output_bytes + 1,
            )
        except Exception:
            raise SessionFailure("session_spawn_failed") from None
        self._stderr_overflow = False
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        frame = await self._read_frame()
        if frame != {"version": _VERSION, "kind": "ready"}:
            raise SessionFailure("session_protocol")

    async def _request(self, payload: JsonValue, request_id: str) -> JsonValue:
        await self._write_frame({
            "version": _VERSION, "kind": "request", "id": request_id,
            "payload": thaw_json_value(payload),
        })
        frame = await self._read_frame()
        if frame.get("version") != _VERSION or frame.get("kind") != "response":
            raise SessionFailure("session_protocol")
        if frame.get("id") != request_id:
            raise SessionFailure("session_correlation")
        if set(frame) != {"version", "kind", "id", "payload"}:
            raise SessionFailure("session_protocol")
        try:
            return freeze_json_value(
                "session response payload",
                cast(JsonValue, frame["payload"]),
            )
        except (TypeError, ValueError):
            raise SessionFailure("session_protocol") from None

    async def _health(self) -> None:
        await self._write_frame({"version": _VERSION, "kind": "ping"})
        if await self._read_frame() != {"version": _VERSION, "kind": "pong"}:
            raise SessionFailure("session_protocol")

    async def _shutdown(self) -> None:
        await self._write_frame({"version": _VERSION, "kind": "shutdown"})
        if await self._read_frame() != {"version": _VERSION, "kind": "stopped"}:
            raise SessionFailure("session_protocol")
        if self._process is None:
            raise SessionFailure("session_protocol")
        if await self._process.wait() != 0:
            raise SessionFailure("session_protocol")

    async def _write_frame(self, frame: dict[str, object]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise SessionFailure("session_eof")
        encoded = (json.dumps(frame, separators=(",", ":"), ensure_ascii=False)
                   + "\n").encode("utf-8")
        if len(encoded) > self._spec.max_output_bytes:
            raise SessionFailure("session_input_limit")
        try:
            process.stdin.write(encoded)
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            raise SessionFailure("session_eof") from None

    async def _read_frame(self) -> dict[str, object]:
        process = self._process
        if process is None or process.stdout is None:
            raise SessionFailure("session_eof")
        try:
            line = await process.stdout.readline()
        except ValueError:
            raise SessionFailure("session_output_limit") from None
        if not line:
            raise SessionFailure("session_eof")
        if len(line) > self._spec.max_output_bytes:
            raise SessionFailure("session_output_limit")
        if not line.endswith(b"\n"):
            raise SessionFailure("session_eof")
        if self._stderr_overflow:
            raise SessionFailure("session_output_limit")
        try:
            decoded = json.loads(line.decode("utf-8"))
        except (UnicodeError, ValueError):
            raise SessionFailure("session_protocol") from None
        if not isinstance(decoded, dict) or any(not isinstance(key, str)
                                                for key in decoded):
            raise SessionFailure("session_protocol")
        return decoded

    async def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        seen = 0
        while chunk := await process.stderr.read(16384):
            seen = min(self._spec.max_output_bytes + 1, seen + len(chunk))
            if seen > self._spec.max_output_bytes:
                self._stderr_overflow = True

    async def _bounded(self, operation: Callable[[], Awaitable[_T]], context: RunContext) -> _T:
        if not isinstance(context, RunContext):
            raise TypeError("session context is invalid")
        context.cancellation.raise_if_cancelled()
        timeout = self._spec.timeout_seconds
        if context.deadline is not None:
            context.deadline.raise_if_expired(context.clock)
            timeout = min(timeout, context.deadline.remaining_seconds(context.clock))
        task: asyncio.Future[object] = asyncio.ensure_future(operation())
        cancelled: asyncio.Future[object] = asyncio.ensure_future(
            context.cancellation.wait_cancelled()
        )
        try:
            done, _ = await asyncio.wait({task, cancelled}, timeout=timeout,
                                         return_when=asyncio.FIRST_COMPLETED)
            if cancelled in done:
                context.cancellation.raise_if_cancelled()
            if task not in done:
                if context.deadline is not None:
                    context.deadline.raise_if_expired(context.clock)
                raise SessionFailure("session_timeout")
            result = cast(_T, task.result())
            if context.cancellation.is_cancelled:
                context.cancellation.raise_if_cancelled()
            if context.deadline is not None:
                context.deadline.raise_if_expired(context.clock)
            return result
        finally:
            task.cancel()
            cancelled.cancel()
            await asyncio.gather(task, cancelled, return_exceptions=True)

    async def _reap(self) -> None:
        process = self._process
        self._process = None
        stderr_task = self._stderr_task
        self._stderr_task = None
        cleanup_failed = False
        try:
            if process is not None:
                if process.stdin is not None:
                    process.stdin.close()
                if process.returncode is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except OSError:
                        cleanup_failed = True
                try:
                    await asyncio.wait_for(process.wait(), timeout=_REAP_SECONDS)
                except asyncio.TimeoutError:
                    cleanup_failed = True
        finally:
            if stderr_task is not None:
                stderr_task.cancel()
                await asyncio.gather(stderr_task, return_exceptions=True)
        if cleanup_failed:
            raise SessionFailure("session_cleanup_failed")

    async def _reap_shielded(
        self,
        initial_cancellation: asyncio.CancelledError | None = None,
    ) -> asyncio.CancelledError | None:
        cleanup = asyncio.create_task(self._reap())
        cancellation = initial_cancellation
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError as error:
                cancellation = error
            except Exception:
                break
        cleanup.result()
        return cancellation

    async def _fail_and_reap(
        self,
        error: BaseException,
        context: RunContext,
        operation: str,
    ) -> NoReturn:
        self._state = "failed"
        cancellation = await self._reap_shielded(
            error if isinstance(error, asyncio.CancelledError) else None
        )
        self._capture(context, "process.response", {
            "operation": operation,
            "state": "failed",
            "error_code": _failure_code(error),
        })
        if cancellation is not None:
            raise cancellation
        raise error


def _failure_code(error: BaseException) -> str:
    if isinstance(error, SessionFailure):
        return error.code
    if isinstance(error, DeadlineExceeded):
        return "deadline_exceeded"
    if isinstance(error, (RunCancelled, asyncio.CancelledError)):
        return "cancelled"
    return "session_failed"
