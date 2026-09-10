"""Real private synthetic files; no downloaded runtime or dependency qualification."""

from dataclasses import replace
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agentrig.capabilities import BackendArtifact, BackendArtifactManifest, BackendFailure
from agentrig.core import CancellationSource, Deadline, RunCancelled, RunContext, RunId
from agentrig.core.clock import SystemClock
from agentrig.core.identity import Uuid4IdGenerator
from agentrig.integrations import audit_backend_artifacts


class BackendArtifactsIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='agentrig-artifact-audit-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        (self.root / 'bin').mkdir(mode=0o700)
        self.file = self.root / 'bin/tool'
        self.file.write_bytes(b'synthetic never executed')
        self.file.chmod(0o700)
        self.manifest = BackendArtifactManifest(artifacts=(BackendArtifact(artifact_id='tool', path='bin/tool',
            sha256='sha256:' + hashlib.sha256(self.file.read_bytes()).hexdigest(), size_bytes=self.file.stat().st_size,
            executable=True),))
        self.source = CancellationSource()
        clock = SystemClock()
        self.context = RunContext.create_root(clock=clock, id_generator=Uuid4IdGenerator(RunId),
            cancellation=self.source.token, deadline=Deadline.after(10, clock))

    def audit(self):
        return audit_backend_artifacts(self.root, self.manifest, self.context)

    def test_audit_is_read_only_and_cannot_qualify_runtime(self):
        before = {str(p.relative_to(self.root)): p.stat().st_mode for p in self.root.rglob('*')}
        result = self.audit()
        self.assertEqual(result.manifest_digest, self.manifest.digest)
        self.assertEqual(result.file_count, 1)
        self.assertFalse(result.dependency_closure_qualified)
        self.assertFalse(result.execution_authorized)
        self.assertEqual(before, {str(p.relative_to(self.root)): p.stat().st_mode for p in self.root.rglob('*')})

    def test_wrong_hash_size_permissions_and_truncation_are_rejected(self):
        original = self.manifest.artifacts[0]
        for changes in [{'sha256': 'sha256:' + '0' * 64}, {'size_bytes': original.size_bytes + 1}, {'executable': False}]:
            with self.assertRaisesRegex(BackendFailure, '^backend_artifacts_invalid$'):
                audit_backend_artifacts(self.root, BackendArtifactManifest(artifacts=(replace(original, **changes),)), self.context)
        self.file.chmod(0o755)
        with self.assertRaises(BackendFailure):
            self.audit()

    def test_unexpected_file_is_rejected_before_payload_open(self):
        extra = self.root / 'unexpected.txt'
        extra.write_text('synthetic unrelated content')
        original = os.open
        accessed = []
        def observe(path, *args, **kwargs):
            accessed.append(str(path))
            return original(path, *args, **kwargs)
        with patch('agentrig.integrations.backend_artifacts.os.open', side_effect=observe):
            with self.assertRaisesRegex(BackendFailure, '^backend_artifacts_invalid$'):
                self.audit()
        self.assertNotIn('tool', accessed)
        self.assertNotIn('unexpected.txt', accessed)

    def test_symlink_and_hardlink_are_not_followed(self):
        with tempfile.TemporaryDirectory(prefix='agentrig-public-fixture-') as elsewhere:
            foreign = Path(elsewhere) / 'fixture'
            foreign.write_bytes(self.file.read_bytes())
            foreign.chmod(0o700)
            self.file.unlink()
            self.file.symlink_to(foreign)
            with self.assertRaises(BackendFailure):
                self.audit()
            self.file.unlink()
            os.link(foreign, self.file)
            with self.assertRaises(BackendFailure):
                self.audit()

    def test_cancelled_audit_opens_nothing(self):
        self.source.cancel('cancelled')
        with patch('agentrig.integrations.backend_artifacts.os.open') as opened:
            with self.assertRaises(RunCancelled):
                self.audit()
            opened.assert_not_called()

    def test_file_changed_during_hash_is_refused(self):
        original = hashlib.sha256
        class ChangingDigest:
            def __init__(inner):
                inner.digest = original()
            def update(inner, data):
                inner.digest.update(data)
                self.file.write_bytes(b'changed after read')
            def hexdigest(inner):
                return inner.digest.hexdigest()
        with patch('agentrig.integrations.backend_artifacts.hashlib.sha256', side_effect=ChangingDigest):
            with self.assertRaises(BackendFailure):
                self.audit()


if __name__ == '__main__':
    unittest.main()
