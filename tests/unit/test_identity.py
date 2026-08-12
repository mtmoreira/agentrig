from __future__ import annotations

import unittest
from uuid import UUID

from agentrig.core import IdGenerator, RunId, Uuid4IdGenerator


def generate_id(generator: IdGenerator[RunId]) -> RunId:
    return generator.generate()


class RunIdTest(unittest.TestCase):
    def test_is_an_opaque_string_value(self) -> None:
        run_id = RunId("run-123")

        self.assertEqual(run_id.value, "run-123")
        self.assertEqual(str(run_id), "run-123")

    def test_rejects_empty_or_padded_values(self) -> None:
        for invalid_value in ("", " run-123", "run-123 "):
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaises(ValueError):
                    RunId(invalid_value)

    def test_uuid_generator_is_deterministically_injectable(self) -> None:
        expected_uuid = UUID("12345678-1234-4234-9234-123456789abc")
        generator = Uuid4IdGenerator(RunId, lambda: expected_uuid)

        self.assertEqual(generate_id(generator), RunId(str(expected_uuid)))


if __name__ == "__main__":
    unittest.main()
