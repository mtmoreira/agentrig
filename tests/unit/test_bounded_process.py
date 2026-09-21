"""No subprocesses: input and preflight boundaries for the process adapter."""

from dataclasses import replace
import json
import unittest
from unittest.mock import patch

from agentrig.core import CancellationSource, Deadline, DeadlineExceeded, RunCancelled, PrivateTraceCapture
from agentrig.integrations import ProcessSpec, run_bounded_process
from agentrig.integrations.bounded_process import ProcessOutput
from tests.unit.test_process_tool import context


class ProcessSpecTests(unittest.IsolatedAsyncioTestCase):
    def spec(self):
        return ProcessSpec(argv=('/private/fixture/executable', 'private-argument'),
                           cwd='/private/fixture', environment={'VALUE': 'private-value'})

    def test_inputs_are_detached_and_private(self):
        environment = {'VALUE': 'private-value'}
        spec = replace(self.spec(), environment=environment)
        environment['VALUE'] = 'changed'
        self.assertEqual(spec.environment['VALUE'], 'private-value')
        self.assertNotIn('private', repr(spec))
        with self.assertRaises(TypeError):
            spec.environment['VALUE'] = 'changed'

    def test_impossible_limits_paths_and_environment_rejected(self):
        for change in ({'argv': ()}, {'argv': ('relative',)}, {'cwd': '/'}, {'cwd': '/a/../b'},
                       {'cwd': '/a/./b'}, {'timeout_seconds': float('nan')}, {'timeout_seconds': float('inf')},
                       {'timeout_seconds': True}, {'max_output_bytes': True}, {'max_output_bytes': 0},
                       {'reject_overflow': 1}, {'environment': {'X=Y': 'value'}}):
            with self.subTest(change=change), self.assertRaises((ValueError, TypeError)):
                replace(self.spec(), **change)

    async def test_preflight_never_starts_worker(self):
        source = CancellationSource()
        source.cancel('cancelled')
        initial = context()
        expired = replace(initial, deadline=Deadline.after(0, initial.clock))
        with patch('agentrig.integrations.bounded_process.asyncio.to_thread') as worker:
            with self.assertRaises(RunCancelled):
                await run_bounded_process(self.spec(), replace(initial, cancellation=source.token))
            with self.assertRaises(DeadlineExceeded):
                await run_bounded_process(self.spec(), expired)
            worker.assert_not_called()

    async def test_private_process_capture_is_explicit_and_excludes_environment(self):
        records = []

        class Recorder:
            def record(self, record):
                records.append(record)

        async def completed_worker(*args):
            return ProcessOutput(exit_code=3, stdout=b'synthetic stdout', stderr=b'synthetic stderr', truncated=True)

        run = replace(context(), private_trace_capture=PrivateTraceCapture(
            recorder=Recorder(), include_process=True))
        with patch('agentrig.integrations.bounded_process.asyncio.to_thread', completed_worker):
            result = await run_bounded_process(self.spec(), run)
        self.assertEqual(result.exit_code, 3)
        self.assertEqual([record.kind for record in records], ['process.request', 'process.response'])
        self.assertNotIn('private-value', records[0].content)
        self.assertEqual(json.loads(records[1].content)['stdout'], 'synthetic stdout')
        self.assertTrue(json.loads(records[1].content)['output_truncated'])

        records.clear()
        run = replace(run, private_trace_capture=PrivateTraceCapture(recorder=Recorder()))
        with patch('agentrig.integrations.bounded_process.asyncio.to_thread', completed_worker):
            await run_bounded_process(self.spec(), run)
        self.assertEqual(records, [])


if __name__ == '__main__':
    unittest.main()
