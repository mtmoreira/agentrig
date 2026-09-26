"""Real local process contract for persistent correlated sessions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
import tempfile
import unittest

from agentrig.core import (
    CancellationSource,
    Deadline,
    DeadlineExceeded,
    PrivateTraceCapture,
    PrivateTraceRecord,
    RunCancelled,
    RunContext,
    RunId,
)
from agentrig.core.clock import SystemClock
from agentrig.core.identity import Uuid4IdGenerator
from agentrig.integrations import PersistentProcessSession, ProcessSpec, SessionFailure


def context(*, source: CancellationSource | None = None,
            timeout_seconds: float | None = None,
            capture: PrivateTraceCapture | None = None) -> RunContext:
    clock = SystemClock()
    return RunContext.create_root(
        clock=clock,
        id_generator=Uuid4IdGenerator(RunId),
        cancellation=(source or CancellationSource()).token,
        deadline=(Deadline.after(timeout_seconds, clock)
                  if timeout_seconds is not None else None),
        private_trace_capture=capture,
    )


@dataclass
class Recorder:
    records: list[PrivateTraceRecord] = field(default_factory=list)

    def record(self, record: PrivateTraceRecord) -> None:
        self.records.append(record)


_WORKER = """
import json, os, sys, time
mode = sys.argv[1]
def send(value):
    sys.stdout.write(json.dumps(value, separators=(',', ':')) + '\\n')
    sys.stdout.flush()
if mode == 'bad_ready':
    send({'version':'1.0','kind':'not_ready'})
else:
    send({'version':'1.0','kind':'ready'})
for line in sys.stdin:
    frame = json.loads(line)
    kind = frame['kind']
    if kind == 'ping':
        send({'version':'1.0','kind':'pong'})
    elif kind == 'shutdown':
        send({'version':'1.0','kind':'wrong'} if mode == 'bad_shutdown' else
             {'version':'1.0','kind':'stopped'})
        break
    elif kind == 'request':
        if mode == 'hang':
            time.sleep(10)
        elif mode == 'eof':
            os._exit(1)
        elif mode == 'noise':
            sys.stdout.write('worker banner\\n')
            sys.stdout.flush()
        elif mode == 'oversize':
            sys.stdout.write('x' * 2048 + '\\n')
            sys.stdout.flush()
        elif mode == 'stderr_oversize':
            os.write(2, b'x' * 2048)
            time.sleep(.05)
            send({'version':'1.0','kind':'response','id':frame['id'],
                  'payload':frame['payload']})
        elif mode == 'late_write':
            time.sleep(.5)
            open('late-write', 'w').write('bad')
        else:
            send({'version':'1.0','kind':'response',
                  'id': 'wrong' if mode == 'wrong_id' else frame['id'],
                  'payload':frame['payload']})
"""


class PersistentSessionIntegrationTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="agentrig-session-")
        self.addCleanup(temporary.cleanup)
        self.cwd = str(Path(temporary.name).resolve())
        self.worker = Path(self.cwd) / "worker.py"
        self.worker.write_text(_WORKER, encoding="utf-8")

    def session(self, mode: str, *, timeout_seconds: float = 2) -> PersistentProcessSession:
        spec = ProcessSpec(
            argv=(str(Path(sys.executable).resolve()), "-I", "-S",
                  str(self.worker), mode),
            cwd=self.cwd,
            timeout_seconds=timeout_seconds,
            max_output_bytes=512,
        )
        return PersistentProcessSession(spec)

    async def test_readiness_health_multiple_requests_and_shutdown(self) -> None:
        session = self.session("normal")
        run = context()
        await session.start(run)
        await session.health(run)
        self.assertEqual(await session.request({"sample": 1}, run), {"sample": 1})
        self.assertEqual(await session.request({"sample": 2}, run), {"sample": 2})
        await session.shutdown(run)
        self.assertEqual(session.state, "closed")
        with self.assertRaisesRegex(SessionFailure, "session_closed"):
            await session.request({"sample": 3}, run)

    async def test_opt_in_private_process_capture_is_redacted(self) -> None:
        recorder = Recorder()
        run = context(capture=PrivateTraceCapture(
            recorder=recorder,
            include_process=True,
        ))
        session = self.session("normal")
        await session.start(run)
        self.assertEqual(
            await session.request({"password": "private-value"}, run),
            {"password": "private-value"},
        )
        await session.shutdown(run)

        self.assertEqual(
            [record.kind for record in recorder.records],
            ["process.request", "process.response"] * 3,
        )
        encoded = [json.loads(record.content) for record in recorder.records]
        self.assertEqual(encoded[2]["payload"]["password"], "[REDACTED]")
        self.assertEqual(encoded[3]["payload"]["password"], "[REDACTED]")
        self.assertNotIn("private-value", repr(recorder.records))

    async def test_invalid_readiness_and_failed_shutdown_are_reaped(self) -> None:
        invalid = self.session("bad_ready")
        with self.assertRaisesRegex(SessionFailure, "session_protocol"):
            await invalid.start(context())
        self.assertEqual(invalid.state, "failed")

        shutdown = self.session("bad_shutdown")
        run = context()
        await shutdown.start(run)
        with self.assertRaisesRegex(SessionFailure, "session_protocol"):
            await shutdown.shutdown(run)
        self.assertEqual(shutdown.state, "failed")

    async def test_corrupt_frames_fail_closed_and_require_explicit_recovery(self) -> None:
        for mode, expected in (
            ("wrong_id", "session_correlation"),
            ("noise", "session_protocol"),
            ("oversize", "session_output_limit"),
            ("stderr_oversize", "session_output_limit"),
            ("eof", "session_eof"),
        ):
            with self.subTest(mode=mode):
                session = self.session(mode)
                run = context()
                await session.start(run)
                with self.assertRaisesRegex(SessionFailure, expected):
                    await session.request({"private": "value"}, run)
                self.assertEqual(session.state, "failed")
                with self.assertRaisesRegex(SessionFailure, "session_failed"):
                    await session.request({"again": True}, run)
                self.assertNotIn("private", repr(session))
                await session.shutdown(run)

    async def test_oversized_input_fails_before_write(self) -> None:
        session = self.session("normal")
        run = context()
        await session.start(run)
        with self.assertRaisesRegex(SessionFailure, "session_input_limit"):
            await session.request({"value": "x" * 1024}, run)
        self.assertEqual(session.state, "failed")
        await session.shutdown(run)

    async def test_whole_request_timeout_cancellation_and_recovery(self) -> None:
        session = self.session("hang", timeout_seconds=0.15)
        run = context()
        await session.start(run)
        with self.assertRaisesRegex(SessionFailure, "session_timeout"):
            await session.request({"value": 1}, run)
        self.assertEqual(session.state, "failed")
        await session.recover(run)
        self.assertEqual(session.state, "ready")
        await session.shutdown(run)

        deadline = self.session("hang")
        await deadline.start(context())
        with self.assertRaises(DeadlineExceeded):
            await deadline.request({"value": 2}, context(timeout_seconds=0.1))
        self.assertEqual(deadline.state, "failed")
        await deadline.shutdown(context())

        source = CancellationSource()
        cancelled = self.session("hang")
        await cancelled.start(context(source=source))
        task = asyncio.create_task(cancelled.request({"value": 2}, context(source=source)))
        await asyncio.sleep(0.05)
        source.cancel("stop")
        with self.assertRaises(RunCancelled):
            await task
        self.assertEqual(cancelled.state, "failed")
        await cancelled.shutdown(context())

        native = self.session("late_write")
        await native.start(context())
        native_task = asyncio.create_task(native.request({"value": 3}, context()))
        await asyncio.sleep(0.05)
        native_task.cancel()
        await asyncio.sleep(0)
        native_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await native_task
        await asyncio.sleep(0.55)
        self.assertEqual(native.state, "failed")
        self.assertFalse((Path(self.cwd) / "late-write").exists())
        await native.shutdown(context())


if __name__ == "__main__":
    unittest.main()
