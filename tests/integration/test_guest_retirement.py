"""Real durable journals; backend shutdown/retirement observations are synthetic."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agentrig.capabilities import BackendFailure
from agentrig.core import CancellationSource, Deadline, RunContext, RunId
from agentrig.core.clock import SystemClock
from agentrig.core.identity import Uuid4IdGenerator
from agentrig.integrations import (
    GuestWorkspaceEndpoint, JournaledGuestServiceRetirement, LocalBackendJournal,
    guest_retirement_operation_digest,
)


class RetirementJournalTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='agentrig-retirement-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.root.chmod(0o700)
        self.journal = LocalBackendJournal(self.root)
        self.endpoint = GuestWorkspaceEndpoint(root=self.root / 'synthetic-socket',
            instance_id='a' * 32, device=1, inode=2, uid=os.getuid())
        self.operation = 'b' * 32
        self.calls = []
        self.fail_reply = False
        self.retired = False
        self.inspect_error = False
        test = self
        class Adapter:
            @contextmanager
            def quiescent(self, endpoint, ctx):
                test.assertEqual(test.journal.inspect().status, 'uncertain')
                test.calls.append('enter')
                try:
                    yield None
                finally:
                    test.calls.append('exit')

            def retire(self, endpoint, digest, ctx):
                test.calls.append('retire')
                test.retired = True
                if test.fail_reply:
                    raise OSError('private lost transport reply')

            def inspect_retired(self, endpoint, ctx):
                test.calls.append('inspect')
                if not test.retired or test.inspect_error:
                    raise BackendFailure('backend_staging_failed')
        self.adapter = Adapter()

    def apply(self, action='retire'):
        # Reconstruct the coordinator and journal on each call, like a resumed task.
        clock = SystemClock()
        ctx = RunContext.create_root(clock=clock, id_generator=Uuid4IdGenerator(RunId),
            cancellation=CancellationSource().token, deadline=Deadline.after(5, clock))
        return JournaledGuestServiceRetirement(journal=LocalBackendJournal(self.root),
            adapter=self.adapter).apply(self.endpoint, self.operation,
            guest_retirement_operation_digest(self.endpoint, self.operation, action),
            ctx, journal_write_authorized=True, action=action)

    def test_completion_and_operation_reuse_are_durable(self):
        result = self.apply()
        self.assertEqual(LocalBackendJournal(self.root).inspect(), result)
        self.assertEqual(result.status, 'completed')
        with self.assertRaisesRegex(BackendFailure, 'backend_operation_reused'):
            self.apply()
        self.assertEqual(self.calls, ['enter', 'retire', 'inspect', 'exit'])

    def test_lost_reply_reconciles_without_replaying_effect(self):
        self.fail_reply = True
        with self.assertRaisesRegex(BackendFailure, '^backend_staging_failed$'):
            self.apply()
        original = self.journal.inspect()
        self.assertEqual(original.status, 'uncertain')
        with self.assertRaisesRegex(BackendFailure, 'backend_reconciliation_required'):
            self.apply()
        result = self.apply('inspect')
        self.assertEqual(result.status, 'reconciled')
        self.assertEqual(self.calls.count('retire'), 1)
        self.assertEqual(LocalBackendJournal(self.root).inspect(), result)

    def test_failed_observation_preserves_exact_uncertain_record(self):
        self.fail_reply = True
        with self.assertRaises(BackendFailure):
            self.apply()
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        self.inspect_error = True
        with self.assertRaises(BackendFailure):
            self.apply('inspect')
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})
        self.assertEqual(self.journal.inspect().status, 'uncertain')

    def test_failed_settlement_recovers_after_fresh_observation(self):
        with patch('agentrig.integrations.backend_journal._Writer.settle', side_effect=OSError('private write error')):
            with self.assertRaisesRegex(BackendFailure, '^backend_journal_invalid$'):
                self.apply()
        self.assertEqual(self.journal.inspect().status, 'uncertain')
        self.assertEqual(self.apply('inspect').status, 'reconciled')
        self.assertEqual(self.calls.count('retire'), 1)

    def test_foreign_journal_binding_blocks_backend(self):
        self.fail_reply = True
        with self.assertRaises(BackendFailure):
            self.apply()
        pointer = self.root / 'current.json'
        value = json.loads(pointer.read_text())
        value['binding']['inode'] += 1
        pointer.write_text(json.dumps(value))
        before = list(self.calls)
        with self.assertRaisesRegex(BackendFailure, 'backend_journal_invalid'):
            self.apply('inspect')
        self.assertEqual(self.calls, before)

    def test_pending_storage_and_concurrency_never_call_backend(self):
        with self.journal.locked():
            with self.assertRaisesRegex(BackendFailure, 'backend_journal_busy'):
                self.apply()
        (self.root / '.pending.json').write_bytes(b'crash evidence')
        with self.assertRaisesRegex(BackendFailure, 'backend_journal_invalid'):
            self.apply()
        self.assertEqual(self.calls, [])
        self.assertEqual((self.root / '.pending.json').read_bytes(), b'crash evidence')


if __name__ == '__main__':
    unittest.main()
