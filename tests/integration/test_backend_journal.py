"""Actual private files and locks; no runtime execution."""

from dataclasses import replace
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from agentrig.capabilities import BackendAction, BackendFailure, BackendOperation
from agentrig.integrations import LocalBackendJournal
from agentrig.testing.backend_operations import BackendJournalContractSuite


def intent():
    return BackendOperation(operation_id='a' * 32, backend_id='synthetic', backend_version='1',
        configuration_digest='sha256:' + '1' * 64, plan_digest='sha256:' + '2' * 64,
        action=BackendAction.START)


class BackendJournalIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='agentrig-journal-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.journal = LocalBackendJournal(self.root)

    def test_shared_contract_with_real_durable_files(self):
        BackendJournalContractSuite().verify(self.journal, intent())

    def test_inspection_is_read_only_and_keeps_uncertainty(self):
        self.assertIsNone(self.journal.inspect())
        self.assertEqual(list(self.root.iterdir()), [])
        with self.journal.locked() as writer:
            writer.create(intent())
        before = {path.name: path.read_bytes() for path in self.root.iterdir()}
        self.assertEqual(self.journal.inspect(), intent())
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.root.iterdir()})

    def test_interrupted_intent_survives_a_new_journal_object(self):
        with self.journal.locked() as writer:
            writer.create(intent())
        with LocalBackendJournal(self.root).locked() as writer:
            self.assertEqual(writer.current(), intent())
            with self.assertRaisesRegex(BackendFailure, 'backend_reconciliation_required'):
                writer.create(replace(intent(), operation_id='b' * 32))

    def test_copied_state_is_not_current_instance_authority(self):
        with self.journal.locked() as writer:
            writer.create(intent())
        copied = self.root / 'copy'
        copied.mkdir(mode=0o700)
        for name in ('current.json', 'operation-' + 'a' * 32 + '.json'):
            shutil.copyfile(self.root / name, copied / name)
            (copied / name).chmod(0o600)
        with LocalBackendJournal(copied).locked() as writer:
            with self.assertRaisesRegex(BackendFailure, 'backend_journal_invalid'):
                writer.current()

    def test_symlink_hardlink_and_public_files_rejected(self):
        for kind in ('symlink', 'hardlink', 'public'):
            child = self.root / kind
            child.mkdir(mode=0o700)
            target = child / 'current.json'
            foreign = child / 'fixture.json'
            foreign.write_text('{}')
            foreign.chmod(0o600)
            if kind == 'symlink':
                target.symlink_to(foreign)
            elif kind == 'hardlink':
                os.link(foreign, target)
            else:
                target.write_text('{}')
                target.chmod(0o644)
            with LocalBackendJournal(child).locked() as writer:
                with self.assertRaisesRegex(BackendFailure, 'backend_journal_invalid'):
                    writer.current()

    def test_pending_write_is_preserved_and_blocks_new_effects(self):
        pending = self.root / '.pending.json'
        pending.write_text('interrupted fixture')
        pending.chmod(0o600)
        with self.assertRaisesRegex(BackendFailure, 'backend_journal_invalid'):
            with self.journal.locked() as writer:
                writer.create(intent())
        self.assertEqual(pending.read_text(), 'interrupted fixture')

    def test_writer_cannot_use_a_released_directory_descriptor(self):
        with self.journal.locked() as writer:
            self.assertIsNone(writer.current())
        with self.assertRaisesRegex(BackendFailure, 'backend_journal_invalid'):
            writer.current()


if __name__ == '__main__':
    unittest.main()
