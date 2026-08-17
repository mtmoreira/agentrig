from __future__ import annotations

from pathlib import Path
import tempfile
import textwrap
import unittest

from tools.validate_agent_context import (
    AgentContextError,
    GUIDANCE_PATHS,
    validate_repository,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def write_text(root: Path, relative_path: str | Path, contents: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(contents).lstrip())


def create_repository_fixture(root: Path) -> None:
    for guidance_path in GUIDANCE_PATHS:
        write_text(root, guidance_path, f"# Guidance for {guidance_path}\n")
    write_text(
        root,
        "pyproject.toml",
        """
        [tool.uv.build-backend]
        source-exclude = ["**/AGENTS.md"]
        wheel-exclude = ["**/AGENTS.md"]
        """,
    )
    write_text(root, "src/example.py", "EXAMPLE = True\n")
    write_text(
        root,
        ".agents/skills/example-skill/SKILL.md",
        """
        ---
        name: example-skill
        description: Validate a small example skill used by unit tests.
        ---

        # Example skill

        Read [details](references/details.md) and inspect `src/example.py`.
        """,
    )
    write_text(
        root,
        ".agents/skills/example-skill/agents/openai.yaml",
        """
        interface:
          display_name: "Example Skill"
          short_description: "Validate an example repository skill"
          default_prompt: "Use $example-skill to validate this fixture."
        """,
    )
    write_text(
        root,
        ".agents/skills/example-skill/references/details.md",
        "# Details\n",
    )


class AgentContextValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        create_repository_fixture(self.root)

    def test_checked_in_repository_context_is_valid(self) -> None:
        summary = validate_repository(REPOSITORY_ROOT)

        self.assertEqual(summary.guidance_files, 5)
        self.assertEqual(summary.skills, 2)
        self.assertEqual(summary.references, 10)

    def test_accepts_complete_repository_context(self) -> None:
        summary = validate_repository(self.root)

        self.assertEqual(summary.guidance_files, 5)
        self.assertEqual(summary.skills, 1)
        self.assertEqual(summary.references, 1)

    def test_rejects_invalid_frontmatter_and_metadata(self) -> None:
        skill_path = self.root / ".agents/skills/example-skill/SKILL.md"
        skill_text = skill_path.read_text()
        skill_path.write_text(skill_text.replace("name: example-skill", "name: wrong"))
        with self.assertRaisesRegex(AgentContextError, "name must match"):
            validate_repository(self.root)

        create_repository_fixture(self.root)
        metadata_path = (
            self.root / ".agents/skills/example-skill/agents/openai.yaml"
        )
        metadata_text = metadata_path.read_text()
        metadata_path.write_text(
            metadata_text.replace("$example-skill", "$wrong-skill")
        )
        with self.assertRaisesRegex(AgentContextError, "must mention"):
            validate_repository(self.root)

    def test_rejects_unlinked_and_missing_references(self) -> None:
        orphan = (
            self.root
            / ".agents/skills/example-skill/references/orphan.md"
        )
        orphan.write_text("# Orphan\n")
        with self.assertRaisesRegex(AgentContextError, "unlinked references"):
            validate_repository(self.root)

        orphan.unlink()
        skill_path = self.root / ".agents/skills/example-skill/SKILL.md"
        skill_path.write_text(
            skill_path.read_text().replace("details.md", "missing.md")
        )
        with self.assertRaisesRegex(AgentContextError, "missing references"):
            validate_repository(self.root)

    def test_rejects_missing_repository_paths(self) -> None:
        skill_path = self.root / ".agents/skills/example-skill/SKILL.md"
        skill_path.write_text(
            skill_path.read_text().replace("src/example.py", "src/missing.py")
        )

        with self.assertRaisesRegex(AgentContextError, "references missing path"):
            validate_repository(self.root)

    def test_rejects_context_hygiene_and_missing_distribution_policy(self) -> None:
        guidance_path = self.root / "AGENTS.md"
        guidance_path.write_text("# Guidance\n\n")
        with self.assertRaisesRegex(AgentContextError, "exactly one newline"):
            validate_repository(self.root)

        create_repository_fixture(self.root)
        write_text(
            self.root,
            "pyproject.toml",
            """
            [tool.uv.build-backend]
            source-exclude = ["**/AGENTS.md"]
            wheel-exclude = []
            """,
        )
        with self.assertRaisesRegex(AgentContextError, "wheel-exclude must exclude"):
            validate_repository(self.root)

        create_repository_fixture(self.root)
        write_text(
            self.root,
            "pyproject.toml",
            """
            [tool.uv.build-backend]
            source-exclude = []
            wheel-exclude = ["**/AGENTS.md"]
            """,
        )
        with self.assertRaisesRegex(AgentContextError, "source-exclude must exclude"):
            validate_repository(self.root)


if __name__ == "__main__":
    unittest.main()
