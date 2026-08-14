from __future__ import annotations

from importlib.metadata import version
import unittest

import openai_codex


class CodexDependencyTest(unittest.TestCase):
    def test_buck_packages_the_locked_sdk_and_runtime(self) -> None:
        self.assertEqual(version("openai-codex"), "0.144.4")
        self.assertEqual(version("openai-codex-cli-bin"), "0.144.4")
        self.assertTrue(hasattr(openai_codex, "AsyncCodex"))
        self.assertTrue(hasattr(openai_codex, "CodexConfig"))


if __name__ == "__main__":
    unittest.main()
