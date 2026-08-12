import unittest
from importlib import resources

import agentrig


class PackageImportTest(unittest.TestCase):
    def test_package_imports_without_initializing_services(self) -> None:
        self.assertEqual(agentrig.__name__, "agentrig")
        self.assertEqual(agentrig.__all__, ())

    def test_package_declares_inline_typing(self) -> None:
        marker = resources.files(agentrig).joinpath("py.typed")

        self.assertTrue(marker.is_file())


if __name__ == "__main__":
    unittest.main()
