"""Launch a local comparison review without constructing provider clients."""

from __future__ import annotations

import argparse
from pathlib import Path

from agentrig.eval_review.server import serve_review_directory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--preferences", type=Path)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--open", action="store_true", dest="open_browser")
    args = parser.parse_args()
    try:
        serve_review_directory(
            args.root, preference_path=args.preferences, port=args.port,
            open_browser=args.open_browser,
        )
    except (ValueError, OSError):
        print("Review could not start: verify the private comparison directory and artifacts.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
