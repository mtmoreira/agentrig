"""Pure authority and journal semantics with synthetic transport observations."""

from dataclasses import replace
from pathlib import Path
import unittest

from agentrig.capabilities import BackendFailure
from agentrig.core import Deadline
from agentrig.integrations import (
    GuestWorkspaceEndpoint, JournaledGuestWorkspaces, WorkspaceStagePlan, guest_operation_digest,
)
from agentrig.integrations.guest_operations import _expected
from agentrig.testing.backend_operations import MemoryBackendJournal
from tests.unit.test_process_tool import context as base_context


def context():
    value = base_context()
    return replace(value, deadline=Deadline.after(5, value.clock))


class GuestOperationUnitTests(unittest.TestCase):
    def setUp(self):
        self.endpoint = GuestWorkspaceEndpoint(root=Path('/owned/socket'), instance_id='a' * 32,
            device=1, inode=2, uid=1000)
        self.plan = WorkspaceStagePlan(host_root=Path('/owned/staging'), guest_root='/guest/staging',
            operation_id='b' * 32, files={'input/design.sv': b'synthetic'})
        self.journal = MemoryBackendJournal()
        self.calls = []
        def copy(endpoint, plan, digest, ctx):
            self.calls.append('copy')
            return _expected(plan, endpoint.instance_id)
        self.driver = JournaledGuestWorkspaces(journal=self.journal,
            verify_endpoint=lambda *args: None, copy=copy)

    def apply(self, driver=None, **options):
        return (driver or self.driver).apply(self.endpoint, self.plan,
            guest_operation_digest(self.endpoint, self.plan, 'copy'), context(),
            journal_write_authorized=True, **options)

    def test_copy_inspection_and_endpoint_changes_have_distinct_grants(self):
        first = guest_operation_digest(self.endpoint, self.plan, 'copy')
        self.assertNotEqual(first, guest_operation_digest(self.endpoint, self.plan, 'inspect'))
        self.assertNotEqual(first, guest_operation_digest(replace(self.endpoint, inode=3), self.plan, 'copy'))
        with self.assertRaises(ValueError):
            guest_operation_digest(self.endpoint, self.plan, 'unknown')

    def test_authority_denied_before_journal_and_copy(self):
        for digest, writable in [(self.plan.digest, True),
                (guest_operation_digest(self.endpoint, self.plan, 'copy'), False)]:
            with self.assertRaisesRegex(BackendFailure, 'backend_authority_required'):
                self.driver.apply(self.endpoint, self.plan, digest, context(), journal_write_authorized=writable)
        self.assertIsNone(self.journal.current())
        self.assertEqual(self.calls, [])

    def test_default_authentication_is_blocked(self):
        with self.assertRaisesRegex(BackendFailure, 'backend_plan_blocked'):
            self.apply(JournaledGuestWorkspaces(journal=self.journal))
        self.assertIsNone(self.journal.current())

    def test_concurrent_writer_blocks_transport(self):
        with self.journal.locked():
            with self.assertRaisesRegex(BackendFailure, 'backend_journal_busy'):
                self.apply()
        self.assertEqual(self.calls, [])

    def test_wrong_receipt_leaves_uncertainty(self):
        def copy(*args):
            return replace(_expected(self.plan, self.endpoint.instance_id), total_bytes=0)
        with self.assertRaisesRegex(BackendFailure, 'backend_invalid_result'):
            self.apply(replace(self.driver, copy=copy))
        self.assertEqual(self.journal.current().status, 'uncertain')

    def test_failed_post_transfer_authentication_does_not_settle(self):
        def verify(*args):
            if self.calls:
                raise ValueError('synthetic private diagnostic')
        with self.assertRaisesRegex(BackendFailure, '^backend_staging_failed$'):
            self.apply(replace(self.driver, verify_endpoint=verify))
        self.assertEqual(self.calls, ['copy'])
        self.assertEqual(self.journal.current().status, 'uncertain')

    def test_invalid_verifier_result_never_calls_copy(self):
        with self.assertRaisesRegex(BackendFailure, 'backend_invalid_result'):
            self.apply(replace(self.driver, verify_endpoint=lambda *args: True))
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
