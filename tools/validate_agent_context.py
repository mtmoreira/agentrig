"""Validate repository-owned guidance and progressively disclosed skills."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
import tomllib
from typing import cast


GUIDANCE_PATHS = (
    Path("AGENTS.md"),
    Path("evals/AGENTS.md"),
    Path("examples/AGENTS.md"),
    Path("src/agentrig/AGENTS.md"),
    Path("tests/AGENTS.md"),
)
SKILLS_PATH = Path(".agents/skills")
_ALLOWED_SKILL_ENTRIES = frozenset(
    {"SKILL.md", "agents", "assets", "references", "scripts"}
)
_SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_REFERENCE_LINK = re.compile(
    r"\[[^\]]+\]\((references/[^)#?]+\.md)\)"
)
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_PLACEHOLDER = re.compile(r"\b(?:PLACEHOLDER|TODO)\b")
_REPOSITORY_PREFIXES = (
    "docs/",
    "evals/",
    "examples/",
    "src/",
    "tests/",
    "third_party/",
    "tools/",
)
_REPOSITORY_FILES = frozenset(
    {"AGENTS.md", "pyproject.toml", "uv.lock"}
)


class AgentContextError(ValueError):
    """Raised when repository agent context violates its contract."""


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    """Counts from one successful repository validation."""

    guidance_files: int
    skills: int
    references: int


def validate_repository(repository_root: Path) -> ValidationSummary:
    """Validate all repository-owned agent context beneath one root."""
    root = repository_root.resolve()
    for relative_path in GUIDANCE_PATHS:
        _validate_text_file(root, relative_path)

    _validate_wheel_policy(root)
    skills_root = root / SKILLS_PATH
    if not skills_root.is_dir():
        raise AgentContextError(f"missing skills directory: {SKILLS_PATH}")

    skill_directories = tuple(
        candidate
        for candidate in sorted(skills_root.iterdir())
        if candidate.is_dir()
    )
    if not skill_directories:
        raise AgentContextError("repository must define at least one skill")

    reference_count = 0
    for skill_directory in skill_directories:
        reference_count += validate_skill(root, skill_directory)

    return ValidationSummary(
        guidance_files=len(GUIDANCE_PATHS),
        skills=len(skill_directories),
        references=reference_count,
    )


def validate_skill(repository_root: Path, skill_directory: Path) -> int:
    """Validate one repository skill and return its reference count."""
    expected_name = skill_directory.name
    if not _SKILL_NAME.fullmatch(expected_name) or len(expected_name) > 64:
        raise AgentContextError(f"invalid skill directory name: {expected_name}")

    unexpected = sorted(
        entry.name
        for entry in skill_directory.iterdir()
        if entry.name not in _ALLOWED_SKILL_ENTRIES
    )
    if unexpected:
        names = ", ".join(unexpected)
        raise AgentContextError(
            f"{expected_name} contains unsupported entries: {names}"
        )

    skill_relative = skill_directory.relative_to(repository_root) / "SKILL.md"
    skill_text = _validate_text_file(repository_root, skill_relative)
    frontmatter, body = _parse_frontmatter(skill_text, skill_relative)
    if set(frontmatter) != {"name", "description"}:
        raise AgentContextError(
            f"{skill_relative} frontmatter must contain only name and description"
        )
    if frontmatter["name"] != expected_name:
        raise AgentContextError(
            f"{skill_relative} name must match directory {expected_name!r}"
        )
    description = frontmatter["description"]
    if not description or len(description) > 1024:
        raise AgentContextError(
            f"{skill_relative} description must contain 1 to 1024 characters"
        )
    if "<" in description or ">" in description:
        raise AgentContextError(
            f"{skill_relative} description cannot contain angle brackets"
        )
    if not body.strip():
        raise AgentContextError(f"{skill_relative} body is empty")

    metadata_relative = (
        skill_directory.relative_to(repository_root) / "agents/openai.yaml"
    )
    metadata = _parse_openai_metadata(
        _validate_text_file(repository_root, metadata_relative),
        metadata_relative,
    )
    short_description = metadata["short_description"]
    if not 25 <= len(short_description) <= 64:
        raise AgentContextError(
            f"{metadata_relative} short_description must contain 25 to 64 characters"
        )
    if f"${expected_name}" not in metadata["default_prompt"]:
        raise AgentContextError(
            f"{metadata_relative} default_prompt must mention ${expected_name}"
        )

    references_directory = skill_directory / "references"
    reference_files: set[Path] = set()
    if references_directory.exists():
        if not references_directory.is_dir():
            raise AgentContextError(
                f"{references_directory.relative_to(repository_root)} must be a directory"
            )
        for candidate in references_directory.rglob("*"):
            if not candidate.is_file():
                continue
            relative_reference = candidate.relative_to(skill_directory)
            if candidate.parent != references_directory or candidate.suffix != ".md":
                raise AgentContextError(
                    f"unsupported skill reference: {candidate.relative_to(repository_root)}"
                )
            _validate_text_file(
                repository_root,
                candidate.relative_to(repository_root),
            )
            reference_files.add(relative_reference)

    linked_references = {
        Path(match) for match in _REFERENCE_LINK.findall(skill_text)
    }
    if linked_references != reference_files:
        missing = sorted(str(path) for path in reference_files - linked_references)
        broken = sorted(str(path) for path in linked_references - reference_files)
        details = []
        if missing:
            details.append(f"unlinked references: {', '.join(missing)}")
        if broken:
            details.append(f"missing references: {', '.join(broken)}")
        raise AgentContextError(f"{expected_name}: {'; '.join(details)}")

    for context_file in (skill_directory / "agents").rglob("*"):
        if context_file.is_file() and context_file.name != "openai.yaml":
            raise AgentContextError(
                f"unsupported agent metadata: {context_file.relative_to(repository_root)}"
            )

    for context_file in (skill_directory / "references").glob("*.md"):
        _validate_repository_paths(repository_root, context_file)
    _validate_repository_paths(repository_root, skill_directory / "SKILL.md")
    return len(reference_files)


def _validate_text_file(root: Path, relative_path: Path) -> str:
    path = root / relative_path
    if not path.is_file():
        raise AgentContextError(f"missing context file: {relative_path}")
    contents = path.read_bytes()
    if not contents:
        raise AgentContextError(f"empty context file: {relative_path}")
    if b"\r" in contents:
        raise AgentContextError(f"context file must use LF newlines: {relative_path}")
    if not contents.endswith(b"\n") or contents.endswith(b"\n\n"):
        raise AgentContextError(
            f"context file must end with exactly one newline: {relative_path}"
        )
    text = contents.decode("utf-8")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.rstrip() != line:
            raise AgentContextError(
                f"trailing whitespace: {relative_path}:{line_number}"
            )
    if _PLACEHOLDER.search(text):
        raise AgentContextError(f"placeholder text remains in {relative_path}")
    return text


def _parse_frontmatter(
    skill_text: str,
    skill_path: Path,
) -> tuple[Mapping[str, str], str]:
    lines = skill_text.splitlines()
    if not lines or lines[0] != "---":
        raise AgentContextError(f"{skill_path} has no YAML frontmatter")
    try:
        closing_index = lines.index("---", 1)
    except ValueError as error:
        raise AgentContextError(
            f"{skill_path} has unterminated YAML frontmatter"
        ) from error

    values: dict[str, str] = {}
    for line in lines[1:closing_index]:
        if ": " not in line:
            raise AgentContextError(f"invalid frontmatter line in {skill_path}")
        key, value = line.split(": ", 1)
        if not key or not value or key in values:
            raise AgentContextError(f"invalid frontmatter key in {skill_path}")
        values[key] = value
    return values, "\n".join(lines[closing_index + 1 :])


def _parse_openai_metadata(text: str, metadata_path: Path) -> Mapping[str, str]:
    lines = text.splitlines()
    if not lines or lines[0] != "interface:":
        raise AgentContextError(f"{metadata_path} must define interface metadata")
    values: dict[str, str] = {}
    for line in lines[1:]:
        match = re.fullmatch(r"  ([a-z_]+): (\".*\")", line)
        if match is None:
            raise AgentContextError(f"invalid metadata line in {metadata_path}")
        key, encoded_value = match.groups()
        if key in values:
            raise AgentContextError(f"duplicate metadata key in {metadata_path}")
        try:
            decoded_value = json.loads(encoded_value)
        except json.JSONDecodeError as error:
            raise AgentContextError(
                f"invalid quoted metadata value in {metadata_path}"
            ) from error
        if not isinstance(decoded_value, str) or not decoded_value:
            raise AgentContextError(f"empty metadata value in {metadata_path}")
        values[key] = decoded_value

    required = {"display_name", "short_description", "default_prompt"}
    if set(values) != required:
        raise AgentContextError(
            f"{metadata_path} must contain exactly {', '.join(sorted(required))}"
        )
    return values


def _validate_repository_paths(root: Path, context_file: Path) -> None:
    text = context_file.read_text()
    for token in _INLINE_CODE.findall(text):
        if token in _REPOSITORY_FILES:
            relative_path = Path(token)
        elif token.startswith(_REPOSITORY_PREFIXES):
            if any(character in token for character in " <>*|"):
                continue
            relative_path = Path(token.rstrip("/"))
        else:
            continue
        if not (root / relative_path).exists():
            raise AgentContextError(
                f"{context_file.relative_to(root)} references missing path: {token}"
            )


def _validate_wheel_policy(root: Path) -> None:
    pyproject_path = root / "pyproject.toml"
    document = cast(object, tomllib.loads(pyproject_path.read_text()))
    project = _require_mapping(document, "pyproject document")
    tool = _require_mapping(project.get("tool"), "tool table")
    uv = _require_mapping(tool.get("uv"), "tool.uv table")
    backend = _require_mapping(
        uv.get("build-backend"),
        "tool.uv.build-backend table",
    )
    raw_exclusions = backend.get("wheel-exclude")
    if not isinstance(raw_exclusions, list) or not all(
        isinstance(value, str) for value in raw_exclusions
    ):
        raise AgentContextError("wheel-exclude must be a list of strings")
    if "**/AGENTS.md" not in raw_exclusions:
        raise AgentContextError("wheel-exclude must exclude **/AGENTS.md")


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AgentContextError(f"{label} must be a table")
    return cast(Mapping[str, object], value)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate repository agent context from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    arguments = parser.parse_args(argv)
    try:
        summary = validate_repository(arguments.root)
    except (AgentContextError, OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        print(f"Agent context validation failed: {error}", file=sys.stderr)
        return 1
    print(
        "Agent context is valid: "
        f"{summary.guidance_files} guidance files, "
        f"{summary.skills} skills, {summary.references} references"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
