"""Synthetic kernel metadata; these checks do not qualify a Linux guest."""

import socket
import struct
import unittest
from unittest.mock import Mock, patch

from agentrig.capabilities import BackendFailure
from agentrig.integrations import verify_linux_guest_peer
from tests.unit.test_guest_operations import context


class GuestPeerTests(unittest.TestCase):
    def setUp(self):
        self.connection = Mock(family=socket.AF_UNIX, type=socket.SOCK_STREAM)
        self.connection.getsockopt.return_value = struct.pack('3i', 123, 1000, 1000)
        for target, value in [('sys.platform', 'linux'), ('os.getuid', lambda: 1000),
                              ('os.geteuid', lambda: 1000), ('socket.SO_PEERCRED', 17)]:
            current = patch('agentrig.integrations.guest_service.' + target, value, create=True)
            current.start()
            self.addCleanup(current.stop)

    def test_same_uid_kernel_peer_accepts_without_exposing_metadata(self):
        self.assertIsNone(verify_linux_guest_peer(self.connection, context()))
        self.connection.getsockopt.assert_called_once_with(socket.SOL_SOCKET, 17, 12)

    def test_wrong_uid_invalid_pid_short_metadata_and_kernel_failure_deny(self):
        for payload in [struct.pack('3i', 123, 1001, 1000), struct.pack('3i', 0, 1000, 1000), b'short']:
            self.connection.getsockopt.return_value = payload
            with self.assertRaisesRegex(BackendFailure, '^backend_authority_required$'):
                verify_linux_guest_peer(self.connection, context())
        self.connection.getsockopt.side_effect = OSError('private diagnostic')
        with self.assertRaisesRegex(BackendFailure, '^backend_authority_required$'):
            verify_linux_guest_peer(self.connection, context())

    def test_unsupported_platform_fails_before_kernel_query(self):
        with patch('agentrig.integrations.guest_service.sys.platform', 'darwin'):
            with self.assertRaisesRegex(BackendFailure, '^backend_unavailable$'):
                verify_linux_guest_peer(self.connection, context())
        self.connection.getsockopt.assert_not_called()

    def test_privileged_or_workload_account_and_non_unix_stream_deny(self):
        for uid in [0, 65534]:
            with patch('agentrig.integrations.guest_service.os.getuid', return_value=uid):
                with self.assertRaises(BackendFailure):
                    verify_linux_guest_peer(self.connection, context())
        with patch('agentrig.integrations.guest_service.os.geteuid', return_value=0):
            with self.assertRaises(BackendFailure):
                verify_linux_guest_peer(self.connection, context())
        self.connection.family = socket.AF_INET
        with self.assertRaises(BackendFailure):
            verify_linux_guest_peer(self.connection, context())
        self.connection.getsockopt.assert_not_called()


if __name__ == '__main__':
    unittest.main()
