"""Real local subprocess tests; no runtime daemon, provider or network."""

import asyncio
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from agentrig.core import CancellationSource, Deadline, DeadlineExceeded, RunContext, RunId
from agentrig.core.clock import SystemClock
from agentrig.core.identity import Uuid4IdGenerator
from agentrig.integrations import ProcessFailure, ProcessSpec, run_bounded_process


def context(source=None, seconds=None):
    clock = SystemClock()
    return RunContext.create_root(clock=clock, id_generator=Uuid4IdGenerator(RunId),
        cancellation=(source or CancellationSource()).token,
        deadline=Deadline.after(seconds, clock) if seconds is not None else None)


class BoundedProcessIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='agentrig-process-')
        self.addCleanup(self.temporary.cleanup)
        self.cwd = str(Path(self.temporary.name).resolve())

    def spec(self, program, **changes):
        return ProcessSpec(argv=(str(Path(sys.executable).resolve()), '-I', '-S', '-c', program),
            cwd=self.cwd, timeout_seconds=3, **changes)

    async def test_exact_argv_environment_and_nonzero_exit(self):
        program = 'import os,sys; print(os.getenv("AGENTRIG_PARENT_SENTINEL","absent")); print(sys.argv[1]); sys.exit(7)'
        spec = self.spec(program)
        spec = replace(spec, argv=(*spec.argv, 'literal;$(never-executed)'))
        with patch.dict('os.environ', {'AGENTRIG_PARENT_SENTINEL': 'not-inherited'}):
            result = await run_bounded_process(spec, context())
        self.assertEqual(result.exit_code, 7)
        self.assertEqual(result.stdout, b'absent\nliteral;$(never-executed)\n')
        self.assertFalse(result.truncated)
        self.assertNotIn('literal', repr(result))

    async def test_large_dual_stream_output_is_drained_with_fixed_capture(self):
        spec = self.spec('import os; [(os.write(1,b"a"*8192),os.write(2,b"b"*8192)) for _ in range(512)]',
                         max_output_bytes=4096, reject_overflow=False)
        result = await run_bounded_process(spec, context())
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout, b'a' * 4096)
        self.assertEqual(result.stderr, b'')
        self.assertTrue(result.truncated)

    async def test_rejects_overflow_before_command_can_finish(self):
        spec = self.spec('import os,time; os.write(2,b"x"*8192); time.sleep(10)', max_output_bytes=256)
        with self.assertRaisesRegex(ProcessFailure, '^process_output_limit$'):
            await run_bounded_process(spec, context())

    async def test_timeout_with_closed_pipes_is_still_bounded(self):
        spec = replace(self.spec('import os,time; os.close(1); os.close(2); time.sleep(10)'), timeout_seconds=.15)
        with self.assertRaisesRegex(ProcessFailure, '^process_timeout$'):
            await run_bounded_process(spec, context())

    async def test_exited_parent_cannot_leave_descendant_holding_pipes(self):
        # Popen owns the unreaped leader until cleanup; no PID from disk is killed.
        program = ('import os,time; pid=os.fork(); '
                   'os._exit(0) if pid else None; time.sleep(.5); '
                   'open("descendant-survived","w").write("bad")')
        spec = replace(self.spec(program), timeout_seconds=.15)
        with self.assertRaisesRegex(ProcessFailure, '^process_timeout$'):
            await run_bounded_process(spec, context())
        await asyncio.sleep(.55)
        self.assertFalse((Path(self.cwd) / 'descendant-survived').exists())

    async def test_cooperative_and_repeated_native_cancellation_clean_up(self):
        for native in (False, True):
            with self.subTest(native=native):
                source = CancellationSource()
                spec = self.spec('import time; time.sleep(.5); open("late-write","w").write("bad")')
                task = asyncio.create_task(run_bounded_process(spec, context(source)))
                await asyncio.sleep(.05)
                if native:
                    task.cancel()
                    await asyncio.sleep(0)
                    task.cancel()
                else:
                    source.cancel('cancelled')
                with self.assertRaises(asyncio.CancelledError):
                    await task
                await asyncio.sleep(.55)
                self.assertFalse((Path(self.cwd) / 'late-write').exists())

    async def test_context_deadline_has_native_identity(self):
        with self.assertRaises(DeadlineExceeded):
            await run_bounded_process(self.spec('import time; time.sleep(10)'), context(seconds=.1))

    async def test_missing_executable_error_is_redacted(self):
        spec = replace(self.spec('pass'), argv=(self.cwd + '/private-missing-executable',))
        with self.assertRaisesRegex(ProcessFailure, '^process_failed$') as caught:
            await run_bounded_process(spec, context())
        self.assertNotIn('private-missing', repr(caught.exception))


if __name__ == '__main__':
    unittest.main()
