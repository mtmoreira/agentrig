"""Scripted Lima observations only; no VM, forwarding or guest execution."""

from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import unittest

from agentrig.capabilities import BackendAction, BackendFailure, BackendRequest
from agentrig.core import CancellationSource, Deadline, DeadlineExceeded, RunCancelled
from agentrig.integrations import (
    GuestWorkspaceEndpoint, JournaledGuestServiceRetirement, LimaGuestServiceRetirementAdapter,
    LimaLifecycleBackend, LimaQuiescenceObservation, LimaRetirementObservation,
    bind_lima_guest_service_generation, decode_lima_service_observation,
    guest_retirement_operation_digest, guest_service_retirement_digest,
)
from agentrig.testing import MemoryBackendJournal
from tests.unit.test_guest_operations import context


class LimaRetirementTests(unittest.TestCase):
    def setUp(self):
        self.journal = MemoryBackendJournal()
        self.backend = LimaLifecycleBackend(journal=MemoryBackendJournal(),
            verify_artifacts=lambda plan: True)
        self.request = BackendRequest(action=BackendAction.INSPECT, operation_id='a' * 32,
            configuration={'executable': '/private/fixture/bin/limactl',
                'executable_sha256': 'sha256:' + '1' * 64, 'state_root': '/private/fixture',
                'configuration_file': '/private/fixture/reviewed.json',
                'configuration_sha256': 'sha256:' + '2' * 64,
                'artifact_closure_sha256': 'sha256:' + '3' * 64, 'instance_id': 'c' * 32})
        self.plan = self.backend.plan(self.request)
        self.endpoint = GuestWorkspaceEndpoint(root=Path('/private/fixture/transport'),
            instance_id='c' * 32, device=4, inode=5, uid=501)
        self.generation = bind_lima_guest_service_generation(self.backend, self.plan,
            self.endpoint, 'sha256:' + '4' * 64)
        self.calls = []

    def observation(self, phase):
        value = {'schema': 'agentrig.lima-service-observation.v1', 'phase': phase,
            'binding_digest': self.generation.digest, 'instance': self.generation.instance,
            'generation_digest': self.generation.generation_digest}
        if phase == 'quiescent':
            value.update(vm_state='stopped', forwarding_state='stopped',
                         active_clients=0, generation_fenced=True)
        else:
            value.update(socket_state='absent', service_lock_retained=True,
                         workspace_retained=True)
        return json.dumps(value, sort_keys=True).encode()

    def adapter(self):
        test = self
        @contextmanager
        def fence(generation, ctx):
            test.calls.append(('fence-enter', generation))
            try:
                yield decode_lima_service_observation(test.observation('quiescent'), generation,
                                                      phase='quiescent')
            finally:
                test.calls.append(('fence-exit', generation))
        def retire(endpoint, digest, ctx):
            test.calls.append(('retire', endpoint))
        def observe(generation, ctx):
            test.calls.append(('observe', generation))
            return decode_lima_service_observation(test.observation('retired'), generation,
                                                   phase='retired')
        return LimaGuestServiceRetirementAdapter(generation=self.generation,
            fence_generation=fence, retire_socket=retire, observe_retired=observe)

    def test_generation_binds_reviewed_inspect_plan_endpoint_and_backend(self):
        self.assertEqual(self.generation.instance, 'managed-' + 'c' * 32)
        self.assertNotEqual(self.generation.digest, self.generation.generation_digest)
        for request, endpoint in [
                (replace(self.request, action=BackendAction.STOP), self.endpoint),
                (self.request, replace(self.endpoint, root=Path('/private/other/transport'))),
                (replace(self.request, configuration={**self.request.configuration,
                    'instance_id': 'd' * 32}), self.endpoint)]:
            with self.subTest(action=request.action, endpoint=endpoint.root):
                with self.assertRaises(BackendFailure):
                    bind_lima_guest_service_generation(self.backend, self.backend.plan(request),
                        endpoint, 'sha256:' + '4' * 64)
        changed = bind_lima_guest_service_generation(self.backend, self.plan,
            replace(self.endpoint, inode=6), 'sha256:' + '4' * 64)
        self.assertNotEqual(changed.digest, self.generation.digest)

    def test_exact_bounded_observation_decoder(self):
        quiet = decode_lima_service_observation(self.observation('quiescent'), self.generation,
                                                phase='quiescent')
        retired = decode_lima_service_observation(self.observation('retired'), self.generation,
                                                  phase='retired')
        self.assertEqual(quiet, LimaQuiescenceObservation(binding_digest=self.generation.digest))
        self.assertEqual(retired, LimaRetirementObservation(binding_digest=self.generation.digest))
        bad = [b'', b'{' + b' ' * 4096, self.observation('quiescent') + b'\n{}',
            self.observation('quiescent').replace(b'"active_clients": 0', b'"active_clients": false'),
            self.observation('quiescent').replace(b'"stopped"', b'"running"', 1),
            self.observation('quiescent').replace(b'"binding_digest"',
                b'"binding_digest":"duplicate","binding_digest"', 1)]
        for data in bad:
            with self.subTest(data=data[:40]):
                with self.assertRaisesRegex(BackendFailure, 'backend_invalid_result'):
                    decode_lima_service_observation(data, self.generation, phase='quiescent')

    def test_full_coordinator_uses_exact_generation_fence_and_observer(self):
        operation = 'd' * 32
        result = JournaledGuestServiceRetirement(journal=self.journal,
            adapter=self.adapter()).apply(self.endpoint, operation,
                guest_retirement_operation_digest(self.endpoint, operation, 'retire'), context(),
                journal_write_authorized=True)
        self.assertEqual((result.status, result.observation), ('completed', 'stopped'))
        self.assertEqual([name for name, _ in self.calls],
                         ['fence-enter', 'retire', 'observe', 'fence-exit'])
        self.assertEqual(self.journal.current(), result)

    def test_missing_seams_and_wrong_endpoint_fail_before_callbacks(self):
        with self.assertRaisesRegex(BackendFailure, 'backend_plan_blocked'):
            with LimaGuestServiceRetirementAdapter(generation=self.generation).quiescent(
                    self.endpoint, context()):
                pass
        with self.assertRaisesRegex(BackendFailure, 'backend_plan_blocked'):
            replace(self.adapter(), retire_socket=None).retire(self.endpoint,
                guest_service_retirement_digest(self.endpoint), context())
        with self.assertRaisesRegex(BackendFailure, 'backend_plan_blocked'):
            replace(self.adapter(), observe_retired=None).inspect_retired(self.endpoint, context())
        with self.assertRaisesRegex(BackendFailure, 'backend_plan_changed'):
            with self.adapter().quiescent(replace(self.endpoint, inode=6), context()):
                pass

    def test_wrong_observations_and_private_errors_are_normalized(self):
        @contextmanager
        def wrong_fence(*args):
            yield LimaQuiescenceObservation(binding_digest='sha256:' + '0' * 64)
        with self.assertRaisesRegex(BackendFailure, 'backend_invalid_result'):
            with replace(self.adapter(), fence_generation=wrong_fence).quiescent(self.endpoint, context()):
                pass
        def failed_observer(*args):
            raise OSError('private runtime response')
        with self.assertRaisesRegex(BackendFailure, '^backend_execution_failed$'):
            replace(self.adapter(), observe_retired=failed_observer).inspect_retired(self.endpoint, context())

    def test_cancelled_and_expired_contexts_never_call_backend_seams(self):
        source = CancellationSource()
        source.cancel('synthetic stop')
        cancelled = replace(context(), cancellation=source.token)
        current = context()
        expired = replace(current, deadline=Deadline.after(0, current.clock))
        digest = guest_service_retirement_digest(self.endpoint)
        for ctx, error in [(cancelled, RunCancelled), (expired, DeadlineExceeded)]:
            with self.subTest(error=error.__name__):
                with self.assertRaises(error):
                    self.adapter().retire(self.endpoint, digest, ctx)
        self.assertEqual(self.calls, [])

    def test_context_exit_failure_does_not_complete_journal(self):
        @contextmanager
        def fence(generation, ctx):
            yield LimaQuiescenceObservation(binding_digest=generation.digest)
            raise OSError('private fence release error')
        adapter = replace(self.adapter(), fence_generation=fence)
        operation = 'd' * 32
        with self.assertRaisesRegex(BackendFailure, '^backend_execution_failed$'):
            JournaledGuestServiceRetirement(journal=self.journal, adapter=adapter).apply(
                self.endpoint, operation,
                guest_retirement_operation_digest(self.endpoint, operation, 'retire'), context(),
                journal_write_authorized=True)
        self.assertEqual(self.journal.current().status, 'uncertain')


if __name__ == '__main__':
    unittest.main()
