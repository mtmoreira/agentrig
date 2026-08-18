from __future__ import annotations

import asyncio
import socket
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

from tests.support.live_test_main import (
    LiveTestOptInRequired,
    main as live_test_main,
    require_live_test_opt_in,
)
from tests.support.unit_test_main import main as unit_test_main
from tests.support.unit_isolation import (
    UnitIsolationViolation,
    install_unit_isolation_guard,
)


class UnitIsolationTest(unittest.TestCase):
    def test_installation_is_idempotent(self) -> None:
        install_unit_isolation_guard()
        install_unit_isolation_guard()

    def test_socket_connect_fails(self) -> None:
        with socket.socket() as client:
            with self.assertRaisesRegex(UnitIsolationViolation, "socket.connect"):
                client.connect(("127.0.0.1", 9))

    def test_name_resolution_fails(self) -> None:
        with self.assertRaisesRegex(UnitIsolationViolation, "socket.getaddrinfo"):
            socket.getaddrinfo("example.invalid", 443)

    def test_child_process_access_fails(self) -> None:
        with self.assertRaisesRegex(UnitIsolationViolation, "subprocess.Popen"):
            subprocess.run(["unreachable"], check=False)

    def test_asyncio_event_loop_remains_available(self) -> None:
        async def complete() -> str:
            await asyncio.sleep(0)
            return "complete"

        self.assertEqual(asyncio.run(complete()), "complete")


class LiveTestPolicyTest(unittest.TestCase):
    def test_live_tests_require_explicit_opt_in(self) -> None:
        with self.assertRaises(LiveTestOptInRequired):
            require_live_test_opt_in({})

    def test_exact_opt_in_value_enables_live_tests(self) -> None:
        require_live_test_opt_in({"AGENTRIG_RUN_LIVE": "1"})

    def test_live_runner_propagates_buck_failure_status(self) -> None:
        buck_runner = types.ModuleType("__test_main__")
        buck_runner.main = lambda argv: 70  # type: ignore[attr-defined]

        with (
            patch.dict("os.environ", {"AGENTRIG_RUN_LIVE": "1"}, clear=True),
            patch.dict(sys.modules, {"__test_main__": buck_runner}),
            self.assertRaises(SystemExit) as raised,
        ):
            live_test_main()

        self.assertEqual(raised.exception.code, 70)


class TestRunnerExitStatusTest(unittest.TestCase):
    def test_unit_runner_propagates_buck_failure_status(self) -> None:
        buck_runner = types.ModuleType("__test_main__")
        buck_runner.main = lambda argv: 70  # type: ignore[attr-defined]

        with (
            patch.dict(sys.modules, {"__test_main__": buck_runner}),
            self.assertRaises(SystemExit) as raised,
        ):
            unit_test_main()

        self.assertEqual(raised.exception.code, 70)


if __name__ == "__main__":
    unittest.main()
