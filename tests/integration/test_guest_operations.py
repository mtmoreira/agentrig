"""Actual journal/files/socketpair recovery with a synthetic authentication adapter.

These tests do not qualify a VM peer, service or forwarded endpoint.
"""

from dataclasses import replace
import json
import os
import unittest
from unittest.mock import patch

from agentrig.capabilities import BackendFailure
from agentrig.core import DeadlineExceeded, RunCancelled
from agentrig.integrations import GuestWorkspaceEndpoint, LocalBackendJournal
from agentrig.integrations.guest_operations import (
    JournaledGuestWorkspaces, guest_operation_digest, inspect_guest_workspace,
)
from tests.integration import test_guest_workspace as guest_fixture


class GuestOperationTests(unittest.TestCase):
    def setUp(self):
        self.f = guest_fixture.GuestWorkspaceTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.plan = self.f.plan
        journal_root = self.f.root / 'journal'
        journal_root.mkdir(mode=0o700)
        self.journal = LocalBackendJournal(journal_root)
        self.endpoint = GuestWorkspaceEndpoint(root=self.f.host, instance_id=self.f.instance,
            device=0, inode=0, uid=os.getuid())
        self.calls = []
        def verify(endpoint, context):
            self.calls.append('verify')
            self.assertEqual(endpoint, self.endpoint)
            self.assertEqual(self.journal.inspect().status, 'uncertain')
        def copy(endpoint, plan, digest, context):
            self.calls.append('copy')
            self.assertEqual(digest, endpoint.transfer_digest(plan))
            return self.f.exchange(plan)
        def inspect(endpoint, plan, context):
            self.calls.append('inspect')
            return inspect_guest_workspace(self.f.guest, plan, endpoint.instance_id, context)
        self.driver = JournaledGuestWorkspaces(journal=self.journal, verify_endpoint=verify,
            copy=copy, inspect=inspect)

    def apply(self, action='copy', driver=None, plan=None):
        plan = self.plan if plan is None else plan
        return (driver or self.driver).apply(self.endpoint, plan,
            guest_operation_digest(self.endpoint, plan, action), self.f.context,
            journal_write_authorized=True, action=action)

    def test_durable_intent_precedes_effect_and_exact_copy_completes(self):
        receipt = self.apply()
        self.assertEqual(self.calls, ['verify', 'copy', 'verify'])
        self.assertFalse(receipt.workload_access_verified)
        self.assertEqual(self.journal.inspect().status, 'completed')
        self.assertEqual(inspect_guest_workspace(self.f.guest, self.plan, self.f.instance, self.f.context), receipt)
        with self.assertRaisesRegex(BackendFailure, 'backend_operation_reused'):
            self.apply()
        self.assertEqual(len(self.calls), 3)

    def lose_reply(self):
        def copy(*args):
            self.f.exchange()
            raise OSError('private transport diagnostic')
        with self.assertRaisesRegex(BackendFailure, '^backend_staging_failed$'):
            self.apply(driver=replace(self.driver, copy=copy))
        self.assertEqual(self.journal.inspect().status, 'uncertain')

    def test_lost_reply_recovers_by_rehashing_without_resending(self):
        self.lose_reply()
        with self.assertRaisesRegex(BackendFailure, 'backend_reconciliation_required'):
            self.apply()
        before = {str(p): p.read_bytes() for p in self.f.guest.rglob('*') if p.is_file()}
        self.apply('inspect')
        self.assertEqual(self.calls, ['verify', 'verify', 'inspect', 'verify'])
        self.assertEqual(self.journal.inspect().status, 'completed')
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.f.guest.rglob('*') if p.is_file()})

    def test_recovery_wrong_authority_or_binding_does_not_inspect(self):
        self.lose_reply()
        with self.assertRaisesRegex(BackendFailure, 'backend_authority_required'):
            self.driver.apply(self.endpoint, self.plan,
                guest_operation_digest(self.endpoint, self.plan, 'copy'), self.f.context,
                journal_write_authorized=True, action='inspect')
        with self.assertRaisesRegex(BackendFailure, 'backend_reconciliation_required'):
            self.apply('inspect', plan=replace(self.plan, files={'input/changed': b'changed'}))
        self.assertEqual(self.calls, ['verify'])

    def test_missing_authentication_blocks_before_journal_write(self):
        with self.assertRaisesRegex(BackendFailure, 'backend_plan_blocked'):
            self.apply(driver=replace(self.driver, verify_endpoint=None))
        self.assertIsNone(self.journal.inspect())
        self.assertEqual(self.calls, [])

    def test_missing_inspector_keeps_uncertainty(self):
        self.lose_reply()
        with self.assertRaisesRegex(BackendFailure, 'backend_plan_blocked'):
            self.apply('inspect', driver=replace(self.driver, inspect=None))
        self.assertEqual(self.journal.inspect().status, 'uncertain')

    def test_failed_authentication_and_bad_receipt_leave_uncertainty(self):
        def verify(*args):
            raise RuntimeError('private authentication failure')
        with self.assertRaisesRegex(BackendFailure, '^backend_staging_failed$'):
            self.apply(driver=replace(self.driver, verify_endpoint=verify))
        self.assertEqual(self.journal.inspect().status, 'uncertain')
        with self.assertRaisesRegex(BackendFailure, 'backend_invalid_result'):
            self.apply('inspect', driver=replace(self.driver, inspect=lambda *args: None))
        self.assertEqual(self.journal.inspect().status, 'uncertain')

    def test_failed_completion_write_preserves_recoverable_intent(self):
        with patch('agentrig.integrations.backend_journal._Writer.settle', side_effect=OSError('private write failure')):
            with self.assertRaisesRegex(BackendFailure, '^backend_journal_invalid$'):
                self.apply()
        self.assertEqual(self.journal.inspect().status, 'uncertain')
        self.apply('inspect')
        self.assertEqual(self.journal.inspect().status, 'completed')

    def test_corrupted_file_never_recovers(self):
        self.lose_reply()
        (self.f.guest / self.plan.operation_id / 'input/design.sv').write_bytes(b'corrupted')
        with self.assertRaisesRegex(BackendFailure, 'backend_staging_failed'):
            self.apply('inspect')
        self.assertEqual(self.journal.inspect().status, 'uncertain')

    def test_partial_receipt_never_recovers_or_creates_files(self):
        self.lose_reply()
        path = self.f.guest / self.plan.operation_id / '.complete.json'
        path.rename(path.with_name('retained-completion'))
        with self.assertRaisesRegex(BackendFailure, 'backend_staging_failed'):
            self.apply('inspect')
        self.assertFalse(path.exists())
        self.assertEqual(self.journal.inspect().status, 'uncertain')

    def test_extra_fifo_rejected_without_reading(self):
        self.f.exchange()
        os.mkfifo(self.f.guest / self.plan.operation_id / 'input/extra', 0o600)
        with self.assertRaisesRegex(BackendFailure, 'backend_staging_failed'):
            inspect_guest_workspace(self.f.guest, self.plan, self.f.instance, self.f.context)

    def test_receipt_identity_and_mode_fail_closed(self):
        self.f.exchange()
        path = self.f.guest / self.plan.operation_id / '.complete.json'
        original = path.read_bytes()
        record = json.loads(original)
        record['inode'] += 1
        path.write_text(json.dumps(record))
        with self.assertRaises(BackendFailure):
            inspect_guest_workspace(self.f.guest, self.plan, self.f.instance, self.f.context)
        path.write_bytes(original)
        path.chmod(0o644)
        with self.assertRaises(BackendFailure):
            inspect_guest_workspace(self.f.guest, self.plan, self.f.instance, self.f.context)

    def test_cancelled_deadline_and_no_deadline_prevent_journal_writes(self):
        for context, error in [(replace(self.f.context, deadline=None), BackendFailure),
                               (self.f.make_context(0), DeadlineExceeded)]:
            with self.assertRaises(error):
                self.driver.apply(self.endpoint, self.plan, guest_operation_digest(self.endpoint, self.plan, 'copy'),
                    context, journal_write_authorized=True)
        self.f.source.cancel('cancelled')
        with self.assertRaises(RunCancelled):
            self.apply()
        self.assertIsNone(self.journal.inspect())

    def test_hardlinked_payload_and_changed_root_fail_inspection(self):
        self.f.exchange()
        operation = self.f.guest / self.plan.operation_id
        target = operation / 'input/design.sv'
        os.link(target, self.f.root / 'outside-link')
        with self.assertRaises(BackendFailure):
            inspect_guest_workspace(self.f.guest, self.plan, self.f.instance, self.f.context)
        other = self.f.root / 'root-link'
        other.symlink_to(self.f.guest, target_is_directory=True)
        with self.assertRaises(BackendFailure):
            inspect_guest_workspace(other, replace(self.plan, guest_root=str(other)), self.f.instance, self.f.context)

    def test_replacement_during_hashing_cannot_complete_inspection(self):
        self.f.exchange()
        from agentrig.integrations.guest_workspace import _verify_copy
        def changed(operation, manifest, check):
            _verify_copy(operation, manifest, check)
            self.f.guest.rename(self.f.root / 'retained-guest')
            self.f.guest.mkdir(mode=0o700)
        with patch('agentrig.integrations.guest_operations._verify_copy', side_effect=changed):
            with self.assertRaises(BackendFailure):
                inspect_guest_workspace(self.f.guest, self.plan, self.f.instance, self.f.context)

    def test_guest_inspection_requires_deadline_before_filesystem_access(self):
        with patch('agentrig.integrations.guest_operations.os.open') as opened:
            with self.assertRaises(BackendFailure):
                inspect_guest_workspace(self.f.guest, self.plan, self.f.instance,
                    replace(self.f.context, deadline=None))
            opened.assert_not_called()


if __name__ == '__main__':
    unittest.main()
