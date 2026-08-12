from __future__ import annotations

import socket
import subprocess
import unittest

from tests.support.live_test_main import (
    LiveTestOptInRequired,
    require_live_test_opt_in,
)
from tests.support.unit_isolation import (
    UnitIsolationViolation,
    install_unit_isolation_guard,
)


class UnitIsolationTest(unittest.TestCase):
    def test_installation_is_idempotent(self) -> None:
        install_unit_isolation_guard()
        install_unit_isolation_guard()

    def test_socket_access_fails(self) -> None:
        with self.assertRaisesRegex(UnitIsolationViolation, "socket.__new__"):
            socket.socket()

    def test_name_resolution_fails(self) -> None:
        with self.assertRaisesRegex(UnitIsolationViolation, "socket.getaddrinfo"):
            socket.getaddrinfo("example.invalid", 443)

    def test_child_process_access_fails(self) -> None:
        with self.assertRaisesRegex(UnitIsolationViolation, "subprocess.Popen"):
            subprocess.run(["unreachable"], check=False)


class LiveTestPolicyTest(unittest.TestCase):
    def test_live_tests_require_explicit_opt_in(self) -> None:
        with self.assertRaises(LiveTestOptInRequired):
            require_live_test_opt_in({})

    def test_exact_opt_in_value_enables_live_tests(self) -> None:
        require_live_test_opt_in({"AGENTRIG_RUN_LIVE": "1"})


if __name__ == "__main__":
    unittest.main()
