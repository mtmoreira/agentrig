"""Real local staging only; no VM visibility or simulation claim."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from agentrig.capabilities import BackendFailure
from agentrig.core import CancellationSource, RunCancelled, RunContext, RunId
from agentrig.core.clock import SystemClock
from agentrig.core.identity import Uuid4IdGenerator
from agentrig.integrations import WorkspaceStagePlan, stage_workspace


class BackendStagingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='agentrig-stage-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.plan = WorkspaceStagePlan(host_root=self.root, guest_root='/mnt/agentrig',
            operation_id='a' * 32, files={'src/example.txt': b'synthetic input', 'test/check.txt': b'synthetic check'})
        self.source = CancellationSource()
        self.context = RunContext.create_root(clock=SystemClock(), id_generator=Uuid4IdGenerator(RunId),
                                             cancellation=self.source.token)

    def test_exact_files_private_root_and_distinct_guest_path(self):
        result = stage_workspace(self.plan, self.plan.digest, self.context)
        self.assertEqual((result.host_directory / 'src/example.txt').read_bytes(), b'synthetic input')
        self.assertEqual(result.guest_directory, '/mnt/agentrig/' + 'a' * 32)
        self.assertFalse(result.guest_visibility_verified)
        self.assertEqual(result.file_count, 2)
        self.assertTrue((result.host_directory / '.complete.json').is_file())
        self.assertNotIn('synthetic input', repr(result))

    def test_digest_change_and_cancellation_create_nothing(self):
        changed = replace(self.plan, guest_root='/mnt/other')
        with self.assertRaisesRegex(BackendFailure, 'backend_authority_required'):
            stage_workspace(changed, self.plan.digest, self.context)
        self.source.cancel('cancelled')
        with self.assertRaises(RunCancelled):
            stage_workspace(self.plan, self.plan.digest, self.context)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_existing_operation_and_linked_root_are_not_adopted(self):
        directory = self.root / self.plan.operation_id
        directory.mkdir(mode=0o700)
        (directory / 'unrelated.txt').write_text('retain')
        with self.assertRaisesRegex(BackendFailure, 'backend_staging_failed'):
            stage_workspace(self.plan, self.plan.digest, self.context)
        self.assertEqual((directory / 'unrelated.txt').read_text(), 'retain')
        link = self.root / 'link'
        link.symlink_to(directory, target_is_directory=True)
        linked = replace(self.plan, host_root=link)
        with self.assertRaisesRegex(BackendFailure, 'backend_staging_failed'):
            stage_workspace(linked, linked.digest, self.context)

    def test_traversal_file_conflicts_and_oversized_payloads_are_rejected(self):
        for files in ({'../escape': b'x'}, {'a/../b': b'x'}, {'a': b'x', 'a/b': b'y'},
                      {'.env': b'not-a-secret-fixture'}, {'large': b'x' * (1024 * 1024 + 1)}):
            with self.assertRaises(ValueError):
                replace(self.plan, files=files)


if __name__ == '__main__':
    unittest.main()
