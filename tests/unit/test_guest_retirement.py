"""Synthetic backend contract checks; no socket, guest or runtime qualification."""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import unittest

from agentrig.capabilities import BackendFailure
from agentrig.core import Cancellation, Deadline, DeadlineExceeded, RunCancelled
from agentrig.integrations import (
    GuestWorkspaceEndpoint, JournaledGuestServiceRetirement,
    guest_retirement_operation_digest, guest_service_retirement_digest,
)
from agentrig.testing.backend_operations import MemoryBackendJournal
from tests.unit.test_guest_operations import context


class RetirementUnitTests(unittest.TestCase):
    def setUp(self):
        self.endpoint = GuestWorkspaceEndpoint(root=Path('/owned/socket'), instance_id='a' * 32,
            device=1, inode=2, uid=1000)
        self.operation = 'b' * 32
        self.journal = MemoryBackendJournal()
        self.calls = []
        self.ctx = context()
        self.fail_at = None
        self.bad_at = None
        self.cancel_at = None
        self.suppress = False
        test = self
        class Adapter:
            def event(self, name):
                test.calls.append(name)
                test.assertEqual(test.journal.current().status, 'uncertain')
                if test.cancel_at == name:
                    raise RunCancelled(Cancellation('cancelled'))
                if test.fail_at == name:
                    raise OSError('private backend detail')
                return True if test.bad_at == name else None

            @contextmanager
            def quiescent(self, endpoint, ctx):
                test.assertEqual(endpoint, test.endpoint)
                binding = self.event('enter')
                try:
                    yield binding
                except Exception:
                    if not test.suppress:
                        raise
                finally:
                    self.event('exit')

            def retire(self, endpoint, digest, ctx):
                test.assertEqual(digest, guest_service_retirement_digest(endpoint))
                return self.event('retire')

            def inspect_retired(self, endpoint, ctx):
                return self.event('inspect')
        self.driver = JournaledGuestServiceRetirement(journal=self.journal, adapter=Adapter())

    def apply(self, action='retire', **options):
        return self.driver.apply(self.endpoint, self.operation,
            guest_retirement_operation_digest(self.endpoint, self.operation, action),
            self.ctx, journal_write_authorized=True, action=action, **options)

    def test_order_intent_fence_effect_observation_release_settlement(self):
        result = self.apply()
        self.assertEqual(self.calls, ['enter', 'retire', 'inspect', 'exit'])
        self.assertEqual((result.status, result.observation), ('completed', 'stopped'))
        self.assertEqual(self.journal.current(), result)
        with self.assertRaisesRegex(BackendFailure, 'backend_operation_reused'):
            self.apply()
        self.assertEqual(len(self.calls), 4)

    def test_lost_reply_never_replays_and_recovery_uses_fresh_fence(self):
        self.fail_at = 'retire'
        with self.assertRaisesRegex(BackendFailure, '^backend_staging_failed$'):
            self.apply()
        with self.assertRaisesRegex(BackendFailure, 'backend_reconciliation_required'):
            self.apply()
        self.fail_at = None
        result = self.apply('inspect')
        self.assertEqual(self.calls, ['enter', 'retire', 'exit', 'enter', 'inspect', 'exit'])
        self.assertEqual((result.status, result.observation), ('reconciled', 'stopped'))

    def test_every_adapter_failure_leaves_uncertainty(self):
        for stage in ['enter', 'retire', 'inspect', 'exit']:
            with self.subTest(stage=stage):
                self.setUp()
                self.fail_at = stage
                with self.assertRaisesRegex(BackendFailure, '^backend_staging_failed$'):
                    self.apply()
                self.assertEqual(self.journal.current().status, 'uncertain')

    def test_invalid_adapter_values_and_suppressed_errors_cannot_settle(self):
        for stage in ['enter', 'retire', 'inspect']:
            with self.subTest(stage=stage):
                self.setUp()
                self.bad_at = stage
                self.suppress = True
                with self.assertRaisesRegex(BackendFailure, 'backend_invalid_result'):
                    self.apply()
                self.assertEqual(self.journal.current().status, 'uncertain')

    def test_callback_cancellation_preserved_without_settlement(self):
        for stage in ['enter', 'retire', 'inspect', 'exit']:
            self.setUp()
            self.cancel_at = stage
            with self.assertRaises(RunCancelled):
                self.apply()
            self.assertEqual(self.journal.current().status, 'uncertain')

    def test_authority_and_missing_adapter_precede_journal(self):
        for writable, digest in [(False, guest_retirement_operation_digest(self.endpoint, self.operation, 'retire')),
                                 (True, guest_service_retirement_digest(self.endpoint))]:
            with self.assertRaisesRegex(BackendFailure, 'backend_authority_required'):
                self.driver.apply(self.endpoint, self.operation, digest, self.ctx,
                    journal_write_authorized=writable)
        with self.assertRaisesRegex(BackendFailure, 'backend_plan_blocked'):
            replace(self.driver, adapter=None).apply(self.endpoint, self.operation,
                guest_retirement_operation_digest(self.endpoint, self.operation, 'retire'),
                self.ctx, journal_write_authorized=True)
        self.assertIsNone(self.journal.current())
        self.assertEqual(self.calls, [])

    def test_inspection_approval_cannot_authorize_retirement(self):
        with self.assertRaisesRegex(BackendFailure, 'backend_authority_required'):
            self.driver.apply(self.endpoint, self.operation,
                guest_retirement_operation_digest(self.endpoint, self.operation, 'inspect'),
                self.ctx, journal_write_authorized=True)
        self.assertIsNone(self.journal.current())

    def test_digest_binds_operation_endpoint_and_action(self):
        value = guest_retirement_operation_digest(self.endpoint, self.operation, 'retire')
        for ep, op, action in [(self.endpoint, 'c' * 32, 'retire'),
                              (replace(self.endpoint, inode=3), self.operation, 'retire'),
                              (self.endpoint, self.operation, 'inspect')]:
            self.assertNotEqual(value, guest_retirement_operation_digest(ep, op, action))
        for op, action in [('invalid', 'retire'), (self.operation, 'unknown')]:
            with self.assertRaises(ValueError):
                guest_retirement_operation_digest(self.endpoint, op, action)

    def test_recovery_rejects_absent_or_different_intent_before_adapter(self):
        with self.assertRaisesRegex(BackendFailure, 'backend_reconciliation_required'):
            self.apply('inspect')
        self.fail_at = 'enter'
        with self.assertRaises(BackendFailure):
            self.apply()
        for ep, op in [(replace(self.endpoint, inode=3), self.operation), (self.endpoint, 'c' * 32)]:
            with self.assertRaisesRegex(BackendFailure, 'backend_reconciliation_required'):
                self.driver.apply(ep, op, guest_retirement_operation_digest(ep, op, 'inspect'),
                    self.ctx, journal_write_authorized=True, action='inspect')
        self.assertEqual(self.calls, ['enter'])

    def test_no_deadline_and_expired_deadline_prevent_journal(self):
        for ctx, error in [(replace(self.ctx, deadline=None), BackendFailure),
                           (replace(self.ctx, deadline=Deadline.after(0, self.ctx.clock)), DeadlineExceeded)]:
            self.ctx = ctx
            with self.assertRaises(error):
                self.apply()
        self.assertIsNone(self.journal.current())

    def test_concurrent_writer_blocks_adapter(self):
        with self.journal.locked():
            with self.assertRaisesRegex(BackendFailure, 'backend_journal_busy'):
                self.apply()
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
