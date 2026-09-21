"""Bounded POSIX command execution; no shell, ambient environment or retries."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
import math
import os
from pathlib import PurePosixPath
import selectors
import signal
import subprocess
from threading import Event
import time
from types import MappingProxyType

from agentrig.core import DeadlineExceeded, JsonValue, RunCancelled, RunContext, redact_private_trace_arguments


@dataclass(frozen=True, slots=True, kw_only=True)
class ProcessSpec:
    """Trusted adapter input. Authority and executable identity belong upstream."""

    argv: tuple[str, ...] = field(repr=False)
    cwd: str = field(repr=False)
    environment: Mapping[str, str] = field(default_factory=dict, repr=False)
    timeout_seconds: float = 60.0
    max_output_bytes: int = 1_000_000
    reject_overflow: bool = True

    def __post_init__(self) -> None:
        argv = tuple(self.argv)
        if (not argv or len(argv) > 256 or any(not isinstance(arg, str) or '\0' in arg
                for arg in argv) or sum(len(arg.encode()) for arg in argv) > 131072):
            raise ValueError('process arguments invalid')
        for path in (argv[0], self.cwd):
            if (not isinstance(path, str) or not path.startswith('/') or path.startswith('//')
                    or str(PurePosixPath(path)) != path or '..' in PurePosixPath(path).parts
                    or any(ord(char) < 32 or ord(char) == 127 for char in path)):
                raise ValueError('process path invalid')
        if self.cwd == '/':
            raise ValueError('process working directory invalid')
        environment = dict(self.environment)
        if len(environment) > 64 or any(not isinstance(key, str) or not key or '=' in key
                or '\0' in key or not isinstance(value, str) or '\0' in value
                for key, value in environment.items()):
            raise ValueError('process environment invalid')
        if sum(len(key.encode()) + len(value.encode()) for key, value in environment.items()) > 65536:
            raise ValueError('process environment invalid')
        if (isinstance(self.timeout_seconds, bool) or not math.isfinite(self.timeout_seconds)
                or not 0 < self.timeout_seconds <= 3600):
            raise ValueError('process timeout invalid')
        if type(self.max_output_bytes) is not int or not 1 <= self.max_output_bytes <= 16 * 1024 * 1024:
            raise ValueError('process output bound invalid')
        if type(self.reject_overflow) is not bool:
            raise ValueError('process overflow policy invalid')
        object.__setattr__(self, 'argv', argv)
        object.__setattr__(self, 'environment', MappingProxyType(environment))


@dataclass(frozen=True, slots=True, kw_only=True)
class ProcessOutput:
    exit_code: int
    stdout: bytes = field(repr=False)
    stderr: bytes = field(repr=False)
    truncated: bool

    def __post_init__(self) -> None:
        if (type(self.exit_code) is not int or not isinstance(self.stdout, bytes) or
                not isinstance(self.stderr, bytes) or type(self.truncated) is not bool):
            raise ValueError('process output invalid')


class ProcessFailure(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code if code in {'process_timeout', 'process_output_limit',
            'process_cancelled', 'process_cleanup_failed'} else 'process_failed'
        super().__init__(self.code)


def _collect(spec: ProcessSpec, stop: Event) -> ProcessOutput:
    """Own Popen exclusively until reap, preventing process-group ID reuse.

    Do not poll/wait while pipes are open: even an exited leader remains our
    unreaped child, so killing its group cannot hit a reused process-group ID.
    Descendants keeping pipes open are bounded by the same cancellation/timeout.
    This is process lifecycle containment, not a sandbox for hostile executables.
    OS process creation and uninterruptible kernel I/O cannot be given a strict
    user-space wall-clock guarantee; failed cleanup is reported as uncertainty.
    """
    process: subprocess.Popen[bytes] | None = None
    selector: selectors.BaseSelector | None = None
    buffers = [bytearray(), bytearray()]
    seen = 0
    deadline = time.monotonic() + spec.timeout_seconds
    try:
        selector = selectors.DefaultSelector()
        if stop.is_set():
            raise ProcessFailure('process_cancelled')
        process = subprocess.Popen(spec.argv, cwd=spec.cwd, env=dict(spec.environment),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, close_fds=True)
        assert process.stdout is not None and process.stderr is not None
        for index, stream in enumerate((process.stdout, process.stderr)):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, index)
        while selector.get_map():
            if stop.is_set():
                raise ProcessFailure('process_cancelled')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProcessFailure('process_timeout')
            for key, _ in selector.select(min(remaining, 0.02)):
                try:
                    chunk = os.read(key.fd, 16384)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                # Saturate the count; arbitrarily long output cannot grow it.
                seen = min(spec.max_output_bytes + 1, seen + len(chunk))
                if seen > spec.max_output_bytes and spec.reject_overflow:
                    raise ProcessFailure('process_output_limit')
                buffer = buffers[key.data]
                buffer.extend(chunk[:max(0, spec.max_output_bytes - len(buffer))])
        # Pipes may close before the process exits. wait() only reaps on success;
        # after TimeoutExpired the group remains owned and can be killed safely.
        while True:
            if stop.is_set():
                raise ProcessFailure('process_cancelled')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProcessFailure('process_timeout')
            try:
                code = process.wait(timeout=min(remaining, 0.02))
                break
            except subprocess.TimeoutExpired:
                continue
        stdout = bytes(buffers[0])
        stderr = bytes(memoryview(buffers[1])[:max(0, spec.max_output_bytes - len(stdout))])
        return ProcessOutput(exit_code=code, stdout=stdout, stderr=stderr,
                             truncated=seen > spec.max_output_bytes)
    except ProcessFailure:
        raise
    except Exception:
        raise ProcessFailure('process_failed') from None
    finally:
        if selector is not None:
            selector.close()
        if process is not None:
            cleanup_failed = False
            if process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError:
                    cleanup_failed = True
                finally:
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        cleanup_failed = True
            for remaining_stream in (process.stdout, process.stderr):
                if remaining_stream is not None:
                    remaining_stream.close()
            if cleanup_failed:
                raise ProcessFailure('process_cleanup_failed') from None


async def run_bounded_process(spec: ProcessSpec, context: RunContext) -> ProcessOutput:
    """Cancellation is delivered only after the worker cleans up its own child.

    Readers retain at most two output bounds plus fixed read buffers. Creating
    the result adds at most one bound; returned streams together fit one bound.
    No output is logged.
    Repeated native task cancellation cannot abandon the cleanup worker.
    """
    context.cancellation.raise_if_cancelled()
    if context.deadline is not None:
        context.deadline.raise_if_expired(context.clock)
    if os.name != 'posix':
        raise ProcessFailure('process_failed')
    capture = context.private_trace_capture
    if capture is not None and capture.include_process:
        capture.record(context, kind='process.request', metadata={}, content={
            'argv': _trace_argv(spec.argv), 'cwd': spec.cwd,
            'environment': 'not_captured', 'timeout_seconds': spec.timeout_seconds,
            'max_output_bytes': spec.max_output_bytes, 'shell': False,
        })
    started_at = context.clock.monotonic()
    stop = Event()
    worker = asyncio.create_task(asyncio.to_thread(_collect, spec, stop))
    cancelled = asyncio.create_task(context.cancellation.wait_cancelled())
    native_cancel: asyncio.CancelledError | None = None
    timed_out = False
    try:
        timeout = spec.timeout_seconds
        if context.deadline is not None:
            timeout = min(timeout, context.deadline.remaining_seconds(context.clock))
        try:
            done, _ = await asyncio.wait({worker, cancelled}, timeout=timeout,
                                         return_when=asyncio.FIRST_COMPLETED)
            timed_out = not done
        except asyncio.CancelledError as error:
            native_cancel = error
        if native_cancel is not None or cancelled.done() or timed_out:
            stop.set()
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError as error:
                native_cancel = error
                stop.set()
            except Exception:
                break
        if native_cancel is not None:
            # Retrieve failure to avoid an unobserved task exception on cancellation.
            if not worker.cancelled():
                worker.exception()
            raise native_cancel
        if context.cancellation.is_cancelled:
            worker.exception()
            context.cancellation.raise_if_cancelled()
        if timed_out:
            worker.exception()
            if context.deadline is not None:
                context.deadline.raise_if_expired(context.clock)
            raise ProcessFailure('process_timeout')
        if context.deadline is not None:
            worker.exception()
            context.deadline.raise_if_expired(context.clock)
        result = worker.result()
        if capture is not None and capture.include_process:
            capture.record(context, kind='process.response', metadata={}, content={
                'exit_code': result.exit_code,
                'stdout': result.stdout.decode('utf-8', errors='replace'),
                'stderr': result.stderr.decode('utf-8', errors='replace'),
                'output_truncated': result.truncated,
                'elapsed_ms': max(0, int((context.clock.monotonic() - started_at) * 1000)),
            })
        return result
    except (ProcessFailure, RunCancelled, DeadlineExceeded, asyncio.CancelledError) as error:
        if capture is not None and capture.include_process:
            code = (error.code if isinstance(error, ProcessFailure) else
                    'deadline_exceeded' if isinstance(error, DeadlineExceeded) else 'cancelled')
            capture.record(context, kind='process.response', metadata={}, content={
                'error_code': code, 'exit_code': None, 'stdout': None, 'stderr': None,
                'elapsed_ms': max(0, int((context.clock.monotonic() - started_at) * 1000)),
            })
        raise
    finally:
        cancelled.cancel()
        # No awaited cleanup after this point: a second cancellation must not
        # replace the already completed worker's native cancellation identity.
        await asyncio.gather(cancelled, return_exceptions=True)


def _trace_argv(argv: tuple[str, ...]) -> list[JsonValue]:
    """Redact recognizable credential flags as well as credential-shaped text."""
    return list(redact_private_trace_arguments(argv))
