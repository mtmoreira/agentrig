"""Synthetic lifecycle failures never establish runtime readiness."""

import asyncio
from dataclasses import replace
import unittest

from agentrig.capabilities import (BackendAction, BackendAuthority, BackendDescriptor, BackendFailure,
    BackendOperation, BackendRequest, JournaledLocalBackends, LocalBackendCatalog)
from agentrig.testing import ScriptedLocalBackend
from agentrig.testing.backend_operations import BackendJournalContractSuite, MemoryBackendJournal
from tests.unit.test_process_tool import context


class LifecycleFixture(ScriptedLocalBackend):
    descriptor = BackendDescriptor(backend_id='scripted-lifecycle', version='1',
        capabilities=frozenset({'synthetic-lifecycle.v1'}),
        actions=frozenset({BackendAction.START, BackendAction.INSPECT}))
    fail = False

    async def apply(self, plan, authority, context):
        result = await super().apply(plan, authority, context)
        if self.fail:
            raise asyncio.CancelledError()
        return result


def request(action=BackendAction.START, operation='a' * 32):
    return BackendRequest(action=action, operation_id=operation, configuration={'instance': 'synthetic'})


def grant(plan):
    return BackendAuthority(plan_digest=plan.digest, operation_id=plan.request.operation_id, effects=plan.effects)


class BackendOperationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.backend = LifecycleFixture(['running'] * 4)
        self.journal = MemoryBackendJournal()
        self.driver = JournaledLocalBackends(catalog=LocalBackendCatalog([self.backend]), journal=self.journal)

    def test_shared_journal_contract(self):
        BackendJournalContractSuite().verify(self.journal, BackendOperation.intent(self.backend.plan(request())))

    async def test_success_is_recorded_and_never_replayed(self):
        plan = self.backend.plan(request())
        await self.driver.apply(plan, grant(plan), context(), journal_write_authorized=True)
        self.assertEqual(self.journal.current().status, 'completed')
        with self.assertRaisesRegex(BackendFailure, 'backend_operation_reused'):
            await self.driver.apply(plan, grant(plan), context(), journal_write_authorized=True)
        self.assertEqual(len(self.backend.calls), 1)

    async def test_cancellation_preserves_uncertainty_and_requires_fresh_inspection_grant(self):
        plan = self.backend.plan(request())
        self.backend.fail = True
        with self.assertRaises(asyncio.CancelledError):
            await self.driver.apply(plan, grant(plan), context(), journal_write_authorized=True)
        self.assertEqual(self.journal.current().status, 'uncertain')
        self.backend.fail = False
        other = self.backend.plan(request(operation='b' * 32))
        with self.assertRaisesRegex(BackendFailure, 'backend_reconciliation_required'):
            await self.driver.apply(other, grant(other), context(), journal_write_authorized=True)
        inspection = self.backend.plan(request(BackendAction.INSPECT))
        with self.assertRaisesRegex(BackendFailure, 'backend_authority_required'):
            await self.driver.reconcile(inspection, grant(plan), context(), journal_write_authorized=True)
        await self.driver.reconcile(inspection, grant(inspection), context(), journal_write_authorized=True)
        self.assertEqual(self.journal.current().status, 'reconciled')
        self.assertEqual(self.journal.current().observation, 'running')
        self.assertEqual(len(self.backend.calls), 2)

    async def test_changed_instance_or_operation_never_inspected(self):
        plan = self.backend.plan(request())
        self.journal.create(BackendOperation.intent(plan))
        for invocation in (replace(request(BackendAction.INSPECT), configuration={'instance': 'different'}),
                           request(BackendAction.INSPECT, operation='b' * 32)):
            inspection = self.backend.plan(invocation)
            with self.assertRaisesRegex(BackendFailure, 'backend_plan_changed'):
                await self.driver.reconcile(inspection, grant(inspection), context(), journal_write_authorized=True)
        self.assertEqual(self.backend.calls, ())

    async def test_missing_permissions_create_no_journal_state(self):
        plan = self.backend.plan(request())
        with self.assertRaisesRegex(BackendFailure, 'backend_authority_required'):
            await self.driver.apply(plan, grant(plan), context())
        with self.assertRaisesRegex(BackendFailure, 'backend_authority_required'):
            await self.driver.apply(plan, replace(grant(plan), plan_digest='sha256:' + '0' * 64),
                                    context(), journal_write_authorized=True)
        self.assertIsNone(self.journal.current())
        self.assertEqual(self.backend.calls, ())

    async def test_publication_failure_never_marks_absent_or_retries(self):
        class FailedCompletion(MemoryBackendJournal):
            def settle(self, expected, result):
                raise BackendFailure('backend_journal_invalid')
        journal = FailedCompletion()
        driver = replace(self.driver, journal=journal)
        plan = self.backend.plan(request())
        with self.assertRaisesRegex(BackendFailure, 'backend_journal_invalid'):
            await driver.apply(plan, grant(plan), context(), journal_write_authorized=True)
        self.assertEqual(journal.current().status, 'uncertain')
        self.assertEqual(len(self.backend.calls), 1)


if __name__ == '__main__':
    unittest.main()
