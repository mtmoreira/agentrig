"""Real private files containing synthetic image bytes; no Lima execution."""

from dataclasses import replace
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agentrig.capabilities import BackendFailure
from agentrig.core import CancellationSource, Deadline, DeadlineExceeded, RunCancelled, RunContext, RunId
from agentrig.core.clock import SystemClock
from agentrig.core.identity import Uuid4IdGenerator
from agentrig.integrations import LimaConfiguration, audit_lima_configuration
from agentrig.integrations.backend_artifacts import audit_backend_artifacts


class LimaConfigurationAuditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='al-', dir='/tmp')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        data = b'synthetic image bytes'
        self.policy = LimaConfiguration(state_root=self.root,
            guest_image_sha256='sha256:' + hashlib.sha256(data).hexdigest(), guest_image_size=len(data))
        (self.root / 'configuration.json').write_bytes(self.policy.json_bytes())
        (self.root / 'configuration.json').chmod(0o600)
        (self.root / 'artifacts').mkdir(mode=0o700)
        (self.root / 'artifacts/guest.raw').write_bytes(data)
        (self.root / 'artifacts/guest.raw').chmod(0o600)
        self.clock = SystemClock()
        self.source = CancellationSource()
        self.context = RunContext.create_root(clock=self.clock, id_generator=Uuid4IdGenerator(RunId),
            cancellation=self.source.token, deadline=Deadline.after(5, self.clock))

    def test_exact_profile_image_and_no_readiness_claim(self):
        result = audit_lima_configuration(self.policy, self.context)
        self.assertEqual(result.policy_digest, self.policy.digest)
        self.assertFalse(result.dependency_closure_qualified)
        self.assertFalse(result.effective_configuration_verified)
        self.assertFalse(result.execution_authorized)
        self.assertFalse((self.root / 'lima').exists())

    def test_empty_private_global_directory_allowed_but_overrides_not_opened(self):
        (self.root / 'lima').mkdir(mode=0o700)
        (self.root / 'lima/_config').mkdir(mode=0o700)
        audit_lima_configuration(self.policy, self.context)
        # A FIFO would hang if read. Inventory must reject it without opening.
        os.mkfifo(self.root / 'lima/_config/override.yaml', 0o600)
        with self.assertRaisesRegex(BackendFailure, '^backend_artifacts_invalid$'):
            audit_lima_configuration(self.policy, self.context)

    def test_unreviewed_configuration_prevents_image_read(self):
        (self.root / 'configuration.json').write_bytes(b'{"base":["template:docker"]}')
        with patch('agentrig.integrations.lima_configuration.audit_backend_artifacts') as image:
            with self.assertRaises(BackendFailure):
                audit_lima_configuration(self.policy, self.context)
            image.assert_not_called()

    def test_changed_image_and_nonprivate_configuration_fail(self):
        (self.root / 'artifacts/guest.raw').write_bytes(b'changed')
        with self.assertRaises(BackendFailure):
            audit_lima_configuration(self.policy, self.context)
        (self.root / 'configuration.json').chmod(0o644)
        with self.assertRaises(BackendFailure):
            audit_lima_configuration(self.policy, self.context)

    def test_links_and_hardlinks_fail(self):
        configuration = self.root / 'configuration.json'
        os.link(configuration, self.root / 'copy')
        with self.assertRaises(BackendFailure):
            audit_lima_configuration(self.policy, self.context)
        linked = self.root / 'linked'
        linked.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(BackendFailure):
            audit_lima_configuration(replace(self.policy, state_root=linked), self.context)

    def test_cancellation_deadline_and_missing_deadline(self):
        with self.assertRaises(BackendFailure):
            audit_lima_configuration(self.policy, replace(self.context, deadline=None))
        with self.assertRaises(DeadlineExceeded):
            audit_lima_configuration(self.policy, replace(self.context, deadline=Deadline.after(0, self.clock)))
        self.source.cancel('cancelled')
        with self.assertRaises(RunCancelled):
            audit_lima_configuration(self.policy, self.context)

    def test_global_configuration_appearing_during_audit_fails(self):
        def changed(*args):
            result = audit_backend_artifacts(*args)
            (self.root / 'lima').mkdir(mode=0o700)
            return result
        with patch('agentrig.integrations.lima_configuration.audit_backend_artifacts', side_effect=changed):
            with self.assertRaises(BackendFailure):
                audit_lima_configuration(self.policy, self.context)


if __name__ == '__main__':
    unittest.main()
