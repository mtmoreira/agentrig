from __future__ import annotations

from importlib.metadata import version
import unittest

import openai


class OpenAIDependencyTest(unittest.TestCase):
    def test_buck_packages_the_locked_official_sdk(self) -> None:
        self.assertEqual(version("openai"), "2.47.0")
        self.assertTrue(hasattr(openai, "AsyncOpenAI"))


if __name__ == "__main__":
    unittest.main()
