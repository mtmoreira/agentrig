"""Scripted Lima command translations only; never real VM qualification."""

from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from agentrig.capabilities import BackendAction, BackendAuthority, BackendFailure, BackendRequest
from agentrig.integrations import LimaLifecycleBackend, ProcessOutput, WorkspaceStagePlan
from agentrig.testing import MemoryBackendJournal
from tests.unit.test_process_tool import context


class LimaLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def test_staging_mapping_is_fixed_and_never_implies_guest_visibility(self):
        plan = self.plan(BackendAction.PREPARE)
        workspace = WorkspaceStagePlan(host_root=Path('/private/fixture/staging'), guest_root='/mnt/agentrig',
            operation_id='e' * 32, files={'fixture.txt': b'synthetic'})
        with patch('agentrig.integrations.lima_lifecycle.stage_workspace') as stage:
            self.backend.stage(plan, workspace, workspace.digest, context())
            stage.assert_called_once()
            changed = replace(workspace, guest_root='/mnt/unrelated')
            with self.assertRaisesRegex(BackendFailure, 'backend_plan_changed'):
                self.backend.stage(plan, changed, changed.digest, context())
            self.assertEqual(stage.call_count, 1)
        self.assertEqual(self.calls, [])

    def setUp(self):
        self.state = 'absent'
        self.calls = []
        self.failure = False
        self.journal = MemoryBackendJournal()
        self.backend = LimaLifecycleBackend(journal=self.journal,
            verify_artifacts=lambda plan: self.state != 'absent', execute=self.execute)

    def plan(self, action, operation='a' * 32):
        return self.backend.plan(BackendRequest(action=action, operation_id=operation,
            configuration={'executable': '/private/fixture/bin/limactl',
                'executable_sha256': 'sha256:' + '1' * 64, 'state_root': '/private/fixture',
                'configuration_file': '/private/fixture/reviewed.yaml',
                'configuration_sha256': 'sha256:' + '2' * 64,
                'artifact_closure_sha256': 'sha256:' + '3' * 64, 'instance_id': 'c' * 32}))

    def grant(self, plan):
        return BackendAuthority(plan_digest=plan.digest, operation_id=plan.request.operation_id,
                                effects=plan.effects)

    async def execute(self, spec, context):
        self.calls.append(spec)
        command = spec.argv[2]
        output = b''
        code = 0
        if command == 'list':
            if self.state == 'absent':
                code = 1
            else:
                name = 'managed-' + 'c' * 32
                output = '|'.join(json.dumps(value) for value in
                    (name, '/private/fixture/lima/' + name, self.state, 'vz', 'aarch64')).encode()
        else:
            self.state = {'create': 'Stopped', 'start': 'Running', 'stop': 'Stopped'}[command]
            if self.failure:
                code = 1
        return ProcessOutput(exit_code=code, stdout=output, stderr=b'', truncated=False)

    async def test_prepare_start_stop_are_exact_non_retrying_commands(self):
        for action, operation, expected in ((BackendAction.PREPARE, 'a' * 32, 'prepared'),
                                            (BackendAction.START, 'b' * 32, 'running'),
                                            (BackendAction.STOP, 'd' * 32, 'stopped')):
            plan = self.plan(action, operation)
            result = await self.backend.apply(plan, self.grant(plan), context())
            self.assertEqual(result.status, expected)
            self.assertFalse(result.binding['workload_transport_qualified'])
        commands = [spec.argv[2:] for spec in self.calls]
        name = 'managed-' + 'c' * 32
        self.assertEqual(commands[1], ('create', '--name', name, '/private/fixture/reviewed.yaml'))
        self.assertEqual(commands[4], ('start', '--timeout=120s', name))
        self.assertEqual(commands[7], ('stop', name))
        self.assertEqual(len(commands), 9)
        self.assertTrue(all(spec.reject_overflow for spec in self.calls))
        self.assertTrue(all(spec.environment['HOME'] == '/private/fixture' for spec in self.calls))

    async def test_failed_start_needs_inspection_not_replay(self):
        self.state = 'Stopped'
        self.failure = True
        plan = self.plan(BackendAction.START)
        with self.assertRaisesRegex(BackendFailure, 'backend_execution_failed'):
            await self.backend.apply(plan, self.grant(plan), context())
        self.assertEqual(self.journal.current().status, 'uncertain')
        count = len(self.calls)
        with self.assertRaisesRegex(BackendFailure, 'backend_reconciliation_required'):
            await self.backend.apply(plan, self.grant(plan), context())
        self.assertEqual(len(self.calls), count)
        self.failure = False
        inspection = self.plan(BackendAction.INSPECT)
        with self.assertRaisesRegex(BackendFailure, 'backend_authority_required'):
            await self.backend.reconcile(inspection, self.grant(plan), context())
        result = await self.backend.reconcile(inspection, self.grant(inspection), context())
        self.assertEqual(result.status, 'running')
        self.assertEqual(self.journal.current().status, 'reconciled')

    async def test_missing_artifact_verifier_blocks_all_processes(self):
        backend = replace(self.backend, verify_artifacts=None)
        plan = backend.plan(self.plan(BackendAction.PREPARE).request)
        with self.assertRaisesRegex(BackendFailure, 'backend_plan_blocked'):
            await backend.apply(plan, self.grant(plan), context())
        self.assertEqual(self.calls, [])
        self.assertIsNone(self.journal.current())

    def test_internal_control_socket_path_is_rejected_during_planning(self):
        request = self.plan(BackendAction.PREPARE).request
        root = '/private/tmp/openrtl-m42b-lima-live-v2/state'
        changed = replace(request, configuration={**request.configuration,
            'state_root': root, 'configuration_file': root + '/reviewed.yaml'})
        with self.assertRaisesRegex(ValueError, 'root invalid'):
            self.backend.plan(changed)
        self.assertEqual(self.calls, [])

    async def test_unknown_or_changed_instance_cannot_be_stopped(self):
        self.state = 'Stopped'
        plan = self.plan(BackendAction.STOP)
        with self.assertRaisesRegex(BackendFailure, 'backend_reconciliation_required'):
            await self.backend.apply(plan, self.grant(plan), context())
        self.assertEqual([spec.argv[2] for spec in self.calls], ['list'])

    async def test_artifact_error_is_redacted_and_no_command_runs(self):
        def failed(plan):
            raise ValueError('private-artifact-details')
        backend = replace(self.backend, verify_artifacts=failed)
        plan = backend.plan(self.plan(BackendAction.START).request)
        with self.assertRaisesRegex(BackendFailure, '^backend_execution_failed$'):
            await backend.apply(plan, self.grant(plan), context())
        self.assertEqual(self.calls, [])
        self.assertIsNone(self.journal.current())

    async def test_unknown_instance_output_stays_uncertain(self):
        async def wrong(spec, context):
            return ProcessOutput(exit_code=0, stdout=b'"other"|"/private/other"|"Running"|"vz"|"aarch64"',
                                 stderr=b'', truncated=False)
        self.state = 'Running'
        backend = replace(self.backend, execute=wrong)
        plan = backend.plan(self.plan(BackendAction.INSPECT).request)
        with self.assertRaisesRegex(BackendFailure, 'backend_invalid_result'):
            await backend.apply(plan, self.grant(plan), context())
        self.assertEqual(self.journal.current().status, 'uncertain')


if __name__ == '__main__':
    unittest.main()
