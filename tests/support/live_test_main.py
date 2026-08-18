"""Buck2 entry point that makes live-test execution explicitly opt-in."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping


class LiveTestOptInRequired(RuntimeError):
    """Raised when a live test process was not explicitly enabled."""


def require_live_test_opt_in(environ: Mapping[str, str] | None = None) -> None:
    """Fail rather than silently skipping a live test process."""
    current_environment = os.environ if environ is None else environ
    if current_environment.get("AGENTRIG_RUN_LIVE") != "1":
        raise LiveTestOptInRequired(
            "live tests require AGENTRIG_RUN_LIVE=1; missing credentials must "
            "also fail inside each provider test",
        )


def main() -> None:
    """Validate opt-in, then delegate to Buck2's standard unittest runner."""
    require_live_test_opt_in()

    import __test_main__

    raise SystemExit(__test_main__.main(sys.argv))


if __name__ == "__main__":
    main()
