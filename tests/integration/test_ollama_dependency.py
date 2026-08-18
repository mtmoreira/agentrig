from __future__ import annotations

from importlib.metadata import version
import unittest

import ollama


class OllamaDependencyTest(unittest.TestCase):
    def test_buck_packages_the_locked_official_client(self) -> None:
        self.assertEqual(version("ollama"), "0.6.2")
        self.assertTrue(hasattr(ollama, "AsyncClient"))
        self.assertTrue(hasattr(ollama, "ChatResponse"))
        self.assertTrue(hasattr(ollama, "ResponseError"))


if __name__ == "__main__":
    unittest.main()
