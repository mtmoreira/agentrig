"""Run strict typing and prove that negative fixtures are rejected."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
_INCOMPATIBLE_FIXTURE = "tests/typing/incompatible_sequence.py"
_EXPECTED_ERROR_CODE = "[misc]"


def _run_mypy(*paths: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "mypy", *paths],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )


def _write_output(result: subprocess.CompletedProcess[str]) -> None:
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)


def main() -> int:
    positive = _run_mypy()
    _write_output(positive)
    if positive.returncode != 0:
        return positive.returncode

    negative = _run_mypy(_INCOMPATIBLE_FIXTURE)
    _write_output(negative)
    diagnostic = f"{negative.stdout}\n{negative.stderr}"
    if negative.returncode == 0:
        print("typecheck failed: incompatible sequence fixture was accepted")
        return 1
    if _INCOMPATIBLE_FIXTURE not in diagnostic:
        print("typecheck failed: negative diagnostic did not name its fixture")
        return 1
    if _EXPECTED_ERROR_CODE not in diagnostic:
        print(
            "typecheck failed: negative fixture did not produce "
            f"{_EXPECTED_ERROR_CODE}"
        )
        return 1

    print("Success: incompatible adjacent sequence types were rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
