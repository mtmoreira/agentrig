from __future__ import annotations

from pathlib import Path
import tempfile
import textwrap
import unittest

from tools.generate_buck_python_deps import (
    DependencyBridgeError,
    generate_locked_dependencies,
    load_dependency_graph,
    normalize_package_name,
    select_locked_wheels,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def write_fixture(contents: str) -> Path:
    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".lock",
        delete=False,
    )
    with temporary:
        temporary.write(textwrap.dedent(contents))
    return Path(temporary.name)


class PythonDependencyBridgeTest(unittest.TestCase):
    def test_checked_in_bridge_matches_optional_dependency_closure(self) -> None:
        generated = generate_locked_dependencies(REPOSITORY_ROOT / "uv.lock")
        checked_in = (
            REPOSITORY_ROOT / "third_party/python/locked_deps.bzl"
        ).read_text()

        self.assertEqual(checked_in, generated)
        self.assertIn('name = "extra-codex"', generated)
        self.assertIn('name = "extra-ollama"', generated)
        self.assertIn('name = "extra-openai"', generated)
        self.assertIn("# openai-codex==0.144.4", generated)
        self.assertIn("# openai-codex-cli-bin==0.144.4", generated)
        self.assertIn("# ollama==0.6.2", generated)
        self.assertIn("# httpx==", generated)
        self.assertIn("# colorama==0.4.6", generated)
        self.assertIn("# openai==2.47.0", generated)
        self.assertIn("# pydantic==", generated)
        self.assertNotIn("# mypy==", generated)

    def test_normalizes_only_safe_distribution_names(self) -> None:
        self.assertEqual(normalize_package_name("Pydantic_Core"), "pydantic-core")
        with self.assertRaisesRegex(
            DependencyBridgeError,
            "cannot form a safe Buck target",
        ):
            normalize_package_name("unsafe/name")

    def test_selects_universal_and_platform_specific_wheels(self) -> None:
        graph = load_dependency_graph(REPOSITORY_ROOT / "uv.lock")

        codex = select_locked_wheels(graph.packages["openai-codex"])
        ollama = select_locked_wheels(graph.packages["ollama"])
        runtime = select_locked_wheels(
            graph.packages["openai-codex-cli-bin"]
        )
        pydantic_core = select_locked_wheels(
            graph.packages["pydantic-core"]
        )

        self.assertEqual(tuple(codex), ("universal",))
        self.assertEqual(tuple(ollama), ("universal",))
        self.assertEqual(
            frozenset(runtime),
            {
                "linux_arm64",
                "linux_x86_64",
                "macos_arm64",
                "macos_x86_64",
                "windows_arm64",
                "windows_x86_64",
            },
        )
        self.assertEqual(frozenset(pydantic_core), frozenset(runtime))
        self.assertIn("cp313-cp313", pydantic_core["macos_arm64"].filename)
        self.assertIn("manylinux", runtime["linux_x86_64"].filename)

    def test_rejects_missing_locked_transitive_dependency(self) -> None:
        fixture = write_fixture(
            """
            version = 1

            [[package]]
            name = "agentrig"
            version = "0.1.0"

            [package.optional-dependencies]
            demo = [{ name = "missing" }]
            """
        )
        self.addCleanup(fixture.unlink)

        with self.assertRaisesRegex(
            DependencyBridgeError,
            "locked dependency 'missing' is missing",
        ):
            load_dependency_graph(fixture)

    def test_excludes_dependency_with_false_buck_python_marker(self) -> None:
        fixture = write_fixture(
            """
            version = 1

            [[package]]
            name = "agentrig"
            version = "0.1.0"

            [package.optional-dependencies]
            demo = [{ name = "dependency" }]

            [[package]]
            name = "dependency"
            version = "1.0.0"
            source = { registry = "https://pypi.org/simple" }
            dependencies = [
                { name = "conditional", marker = "python_version < '3.13'" },
            ]
            wheels = [
                { url = "https://files.pythonhosted.org/dependency-1.0.0-py3-none-any.whl", hash = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" },
            ]
            """
        )
        self.addCleanup(fixture.unlink)

        with self.assertRaisesRegex(
            DependencyBridgeError,
            "marker is unsupported",
        ):
            load_dependency_graph(fixture)

        fixture.write_text(
            fixture.read_text().replace(
                "python_version < '3.13'",
                "python_full_version < '3.13'",
            )
        )
        graph = load_dependency_graph(fixture)
        self.assertEqual(graph.packages["dependency"].dependencies, ())

    def test_includes_dependency_with_true_buck_python_marker(self) -> None:
        fixture = write_fixture(
            """
            version = 1

            [[package]]
            name = "agentrig"
            version = "0.1.0"

            [package.optional-dependencies]
            demo = [{ name = "dependency" }]

            [[package]]
            name = "conditional"
            version = "1.0.0"
            source = { registry = "https://pypi.org/simple" }
            wheels = [
                { url = "https://files.pythonhosted.org/conditional-1.0.0-py3-none-any.whl", hash = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" },
            ]

            [[package]]
            name = "dependency"
            version = "1.0.0"
            source = { registry = "https://pypi.org/simple" }
            dependencies = [
                { name = "conditional", marker = "python_full_version >= '3.13'" },
            ]
            wheels = [
                { url = "https://files.pythonhosted.org/dependency-1.0.0-py3-none-any.whl", hash = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" },
            ]
            """
        )
        self.addCleanup(fixture.unlink)

        graph = load_dependency_graph(fixture)

        self.assertEqual(
            graph.packages["dependency"].dependencies,
            ("conditional",),
        )
        self.assertIn("conditional", graph.packages)

    def test_includes_dependency_used_by_a_supported_host(self) -> None:
        fixture = write_fixture(
            """
            version = 1

            [[package]]
            name = "agentrig"
            version = "0.1.0"

            [package.optional-dependencies]
            demo = [{ name = "dependency" }]

            [[package]]
            name = "conditional"
            version = "1.0.0"
            source = { registry = "https://pypi.org/simple" }
            wheels = [
                { url = "https://files.pythonhosted.org/conditional-1.0.0-py3-none-any.whl", hash = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" },
            ]

            [[package]]
            name = "dependency"
            version = "1.0.0"
            source = { registry = "https://pypi.org/simple" }
            dependencies = [
                { name = "conditional", marker = "sys_platform == 'win32'" },
            ]
            wheels = [
                { url = "https://files.pythonhosted.org/dependency-1.0.0-py3-none-any.whl", hash = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" },
            ]
            """
        )
        self.addCleanup(fixture.unlink)

        graph = load_dependency_graph(fixture)

        self.assertEqual(
            graph.packages["dependency"].dependencies,
            ("conditional",),
        )
        self.assertIn("conditional", graph.packages)


if __name__ == "__main__":
    unittest.main()
