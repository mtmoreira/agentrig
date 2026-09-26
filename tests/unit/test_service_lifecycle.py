"""Pure service authority tests; no actual guest or filesystem activation."""

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch
from agentrig.capabilities import BackendFailure
from agentrig.integrations import GuestWorkspaceEndpoint, guest_service_retirement_digest, retire_guest_service, serve_guest_requests
from tests.unit.test_guest_operations import context


class ServiceLifecycleUnitTests(unittest.TestCase):
    def setUp(self):
        self.endpoint = GuestWorkspaceEndpoint(root=Path('/owned/socket'), instance_id='a' * 32,
            device=1, inode=2, uid=1000)

    def test_retirement_digest_binds_instance_and_socket_identity(self):
        digest = guest_service_retirement_digest(self.endpoint)
        for field, value in [('root', Path('/owned/other')), ('instance_id', 'b' * 32),
                             ('device', 2), ('inode', 3), ('uid', 1001)]:
            self.assertNotEqual(digest, guest_service_retirement_digest(replace(self.endpoint, **{field: value})))

    def test_retirement_missing_authority_or_verifier_never_opens_state(self):
        with patch('agentrig.integrations.guest_service._service_directory') as open_directory:
            for grant, verifier in [('sha256:' + '0' * 64, lambda *args: None),
                                    (guest_service_retirement_digest(self.endpoint), None)]:
                with self.assertRaises(BackendFailure):
                    retire_guest_service(self.endpoint, grant, context(), verify_quiescent=verifier)
            open_directory.assert_not_called()

    def test_invalid_request_limit_and_missing_authority_never_open_state(self):
        with patch('agentrig.integrations.guest_service._service_directory') as open_directory:
            for limit, allowed in [(0, True), (65, True), (True, True), (1, False)]:
                with self.assertRaises(BackendFailure):
                    serve_guest_requests(Path('/guest'), Path('/socket'), 'a' * 32, context(),
                        request_limit=limit, service_authorized=allowed, verify_peer=lambda *args: None)
            open_directory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
