"""Buck2 entry point that isolates unit tests before importing test modules."""

from __future__ import annotations

import sys

from tests.support.unit_isolation import install_unit_isolation_guard


def main() -> None:
    """Install unit isolation, then delegate to Buck2's unittest runner."""
    install_unit_isolation_guard()

    import __test_main__

    __test_main__.main(sys.argv)


if __name__ == "__main__":
    main()
