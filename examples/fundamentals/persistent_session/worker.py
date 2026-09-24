"""Trusted JSONL worker used only by the persistent-session example."""

from __future__ import annotations

import json
import sys


def _send(frame: object) -> None:
    print(json.dumps(frame, separators=(",", ":")), flush=True)


def main() -> None:
    request_count = 0
    _send({"version": "1.0", "kind": "ready"})
    for line in sys.stdin:
        frame = json.loads(line)
        kind = frame.get("kind")
        if kind == "ping":
            _send({"version": "1.0", "kind": "pong"})
        elif kind == "shutdown":
            _send({"version": "1.0", "kind": "stopped"})
            return
        elif kind == "request":
            request_count += 1
            _send({
                "version": "1.0",
                "kind": "response",
                "id": frame["id"],
                "payload": {
                    "request_count": request_count,
                    "received": frame["payload"],
                },
            })


if __name__ == "__main__":
    main()
