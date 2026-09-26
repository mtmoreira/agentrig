"""Artifact declarations are immutable inputs, never execution authority."""

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from agentrig.capabilities import BackendArtifact, BackendArtifactManifest, BackendFailure
from agentrig.integrations import audit_backend_artifacts
from tests.unit.test_process_tool import context


def artifact(**kwargs):
    values = dict(artifact_id='guest', path='images/guest.raw', sha256='sha256:' + '1' * 64, size_bytes=3)
    return BackendArtifact(**(values | kwargs))


class BackendArtifactsTests(unittest.TestCase):
    def test_deterministic_immutable_declaration(self):
        first, second = artifact(), artifact(artifact_id='notice', path='LICENSE')
        supplied = [first, second]
        manifest = BackendArtifactManifest(artifacts=supplied)
        supplied.clear()
        self.assertEqual(len(manifest.artifacts), 2)
        self.assertEqual(manifest.digest, BackendArtifactManifest(artifacts=(second, first)).digest)
        self.assertNotIn('guest.raw', repr(manifest))
        self.assertNotIn('guest.raw', repr(first))
        self.assertNotEqual(manifest.digest, replace(manifest, artifacts=(first,)).digest)

    def test_path_secrets_and_impossible_sizes_rejected(self):
        for path in ['../escape', '/absolute', 'a/../b', 'a//b', '.env', 'x/.env.prod',
                     'private.key', 'private.PEM', 'private.p12', 'private.pfx', 'a/./b',
                     'private.crt', 'id_rsa', 'credentials.json']:
            with self.subTest(path=path), self.assertRaises(ValueError):
                artifact(path=path)
        for changes in [{'size_bytes': -1}, {'size_bytes': True}, {'size_bytes': 4 * 1024**3 + 1},
                        {'executable': 1}, {'sha256': 'unbound'}]:
            with self.assertRaises(ValueError):
                artifact(**changes)

    def test_duplicate_and_conflicting_paths_rejected(self):
        for values in [(), (artifact(), artifact()),
                       (artifact(path='a'), artifact(artifact_id='other', path='a/b')),
                       (artifact(), artifact(path='elsewhere'))]:
            with self.assertRaises(ValueError):
                BackendArtifactManifest(artifacts=values)

    def test_missing_deadline_stops_before_any_file_access(self):
        with patch('agentrig.integrations.backend_artifacts.os.open') as opened:
            with self.assertRaisesRegex(BackendFailure, '^backend_artifacts_invalid$'):
                audit_backend_artifacts(Path('/private/fixture'), BackendArtifactManifest(artifacts=(artifact(),)), context())
            opened.assert_not_called()


if __name__ == '__main__':
    unittest.main()
