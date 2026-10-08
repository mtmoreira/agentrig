"""Local, provider-free human review of an explicitly exported comparison.

The application supplies a private directory and a digest-bound manifest. Only
listed artifacts are served; arbitrary journal files and filesystem paths are
never exposed. Importing this module does not start a server or inspect storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import stat
import tempfile
import threading
from typing import Any, cast
from urllib.parse import urlsplit
import webbrowser

from agentrig.evals.comparison import (
    ComparisonManifest,
    HumanRankingBallot,
    RankingLedger,
    ReviewAsset,
    aggregate_rankings,
)
from agentrig.selection import JsonPreferenceStore

_MAX_BODY = 128 * 1024
_MAX_LEDGER = 4 * 1024 * 1024
_MAX_ASSET = 64 * 1024 * 1024
_MAX_TEXT = 512 * 1024
_MEDIA = frozenset({"image/png", "image/jpeg", "image/webp", "text/plain", "application/json", "application/pdf"})
_STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/static/review.css": ("review.css", "text/css; charset=utf-8"),
    "/static/review.js": ("review.js", "text/javascript; charset=utf-8"),
    "/static/ranking.js": ("ranking.js", "text/javascript; charset=utf-8"),
}


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON field")
        value[key] = item
    return value


def _decode(content: bytes) -> dict[str, Any]:
    try:
        value = json.loads(content, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError("invalid review document") from None
    if not isinstance(value, dict):
        raise ValueError("review document must be an object")
    return cast(dict[str, Any], value)


def _encode(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _checked_file(root: Path, relative: str, *, limit: int) -> bytes:
    path = root / relative
    try:
        parts = Path(relative).parts
        if not parts or Path(relative).is_absolute() or any(part in {".", ".."} for part in parts):
            raise ValueError("invalid review asset path")
        current = root
        for part in parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("review assets cannot be symlinks")
        path.resolve(strict=True).relative_to(root)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                raise ValueError("review asset exceeds the supported bounds")
            content = stream.read(limit + 1)
            if len(content) > limit:
                raise ValueError("review asset exceeds the supported bounds")
        return content
    except (OSError, ValueError):
        raise ValueError("review artifact is missing, changed, or outside the review directory") from None


@dataclass(frozen=True, slots=True)
class ReviewResponse:
    status: int
    media_type: str
    content: bytes = field(repr=False)


class ReviewApplication:
    """File-backed review service with a socket-free, independently testable API."""

    def __init__(
        self,
        root: Path,
        manifest: ComparisonManifest,
        *,
        review_token: str,
        preference_store: JsonPreferenceStore | None = None,
    ) -> None:
        if not isinstance(manifest, ComparisonManifest):
            raise TypeError("review requires a comparison manifest")
        if not review_token or len(review_token) < 24:
            raise ValueError("review token must be explicitly supplied")
        if not root.is_dir() or root.is_symlink() or root != root.resolve():
            raise ValueError("review root must be an existing canonical directory")
        if os.name == "posix" and stat.S_IMODE(root.stat().st_mode) & 0o077:
            raise ValueError("review root must be owner-only")
        self.root = root
        self.manifest = manifest
        self._review_token = review_token
        self._preference_store = preference_store
        self._lock = threading.RLock()
        self._assets: dict[str, ReviewAsset] = {}
        for review in [*(case.review for case in manifest.cases), *(cell.review for cell in manifest.cells)]:
            for asset in review.assets:
                if asset.media_type not in _MEDIA:
                    raise ValueError("unsupported review artifact media type")
                if asset.kind == "image" and not asset.media_type.startswith("image/"):
                    raise ValueError("image review assets require an image media type")
                if asset.kind == "text" and asset.media_type not in {"text/plain", "application/json"}:
                    raise ValueError("text review assets require a text media type")
                token = self._asset_token(asset)
                self._assets[token] = asset
        self._verify_assets()
        self._load_ledger()

    @staticmethod
    def _asset_token(asset: ReviewAsset) -> str:
        return hashlib.sha256(_encode([asset.path, asset.sha256, asset.media_type, asset.kind])).hexdigest()

    def _asset_bytes(self, asset: ReviewAsset) -> bytes:
        limit = _MAX_TEXT if asset.kind == "text" else _MAX_ASSET
        content = _checked_file(self.root, asset.path, limit=limit)
        if hashlib.sha256(content).hexdigest() != asset.sha256:
            raise ValueError("review artifact digest mismatch")
        return content

    def _verify_assets(self) -> None:
        for asset in self._assets.values():
            self._asset_bytes(asset)

    def _load_ledger(self) -> RankingLedger:
        path = self.root / "rankings.json"
        if not path.exists() and not path.is_symlink():
            return RankingLedger(manifest_digest=self.manifest.digest)
        ledger = RankingLedger.from_data(_decode(_checked_file(self.root, "rankings.json", limit=_MAX_LEDGER)))
        aggregate_rankings(self.manifest, ledger)
        return ledger

    def _save_ledger(self, ledger: RankingLedger) -> None:
        """Publish once atomically; identical resubmission is idempotent."""
        content = _encode(ledger.to_data()) + b"\n"
        if len(content) > _MAX_LEDGER:
            raise ValueError("ranking ledger exceeds the supported size")
        target = self.root / "rankings.json"
        if target.exists() or target.is_symlink():
            existing = self._load_ledger()
            if existing.to_data() != ledger.to_data():
                raise ValueError("rankings are already finalized for this comparison")
            return
        descriptor, name = tempfile.mkstemp(prefix=".rankings-", dir=self.root)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                if self._load_ledger().to_data() != ledger.to_data():
                    raise ValueError("rankings changed concurrently") from None
        finally:
            temporary.unlink(missing_ok=True)

    def view(self) -> dict[str, Any]:
        with self._lock:
            self._verify_assets()
            data = cast(dict[str, Any], self.manifest.to_data())
            for item in [*data["cases"], *data["cells"]]:
                for raw in item["review"]["assets"]:
                    token = hashlib.sha256(_encode([raw["path"], raw["sha256"], raw["media_type"], raw["kind"]])).hexdigest()
                    raw["url"] = "/asset/" + token
                    if raw["kind"] == "text":
                        raw["text"] = self._asset_bytes(self._assets[token]).decode("utf-8")
                    del raw["path"]
            ledger = self._load_ledger()
            summary = aggregate_rankings(self.manifest, ledger)
            data.update({
                "manifest_digest": self.manifest.digest,
                "review_token": self._review_token,
                "preference_enabled": self._preference_store is not None,
                "saved_ballots": [ballot.to_data() for ballot in ledger.ballots],
                "summary": summary.to_data() if ledger.ballots else None,
            })
            return data

    def finalize(self, payload: dict[str, Any]) -> dict[str, Any]:
        if set(payload) != {"manifest_digest", "ballots", "activate_preference"}:
            raise ValueError("invalid ranking submission")
        if payload["manifest_digest"] != self.manifest.digest or type(payload["activate_preference"]) is not bool:
            raise ValueError("ranking submission differs from the comparison")
        raw_ballots = payload["ballots"]
        if not isinstance(raw_ballots, list) or len(raw_ballots) != len(self.manifest.cases):
            raise ValueError("rank every scenario before submitting")
        with self._lock:
            self._verify_assets()
            ledger = RankingLedger(manifest_digest=self.manifest.digest)
            candidate_ids = tuple(candidate.candidate_id for candidate in self.manifest.candidates)
            case_by_id = {case.case_id: case for case in self.manifest.cases}
            groups_by_case: dict[str, list[list[object]]] = {}
            for raw in raw_ballots:
                if not isinstance(raw, dict) or set(raw) != {"case_id", "ranks"}:
                    raise ValueError("invalid scenario ranking")
                case_id = raw["case_id"]
                if not isinstance(case_id, str) or case_id not in case_by_id or case_id in groups_by_case:
                    raise ValueError("duplicate or unknown scenario ranking")
                groups = raw["ranks"]
                if not isinstance(groups, list) or any(not isinstance(group, list) for group in groups):
                    raise ValueError("invalid ranking groups")
                if any(not isinstance(candidate, str) for group in groups for candidate in group):
                    raise ValueError("invalid ranked candidate")
                groups_by_case[case_id] = groups
            for case in self.manifest.cases:
                groups = groups_by_case[case.case_id]
                ballot = HumanRankingBallot(
                    ballot_id="canonicalizing", manifest_digest=self.manifest.digest,
                    case_id=case.case_id, case_version=case.version,
                    candidate_ids=candidate_ids,
                    ranks=tuple(tuple(cast(str, candidate) for candidate in group) for group in groups),
                )
                ballot_id = hashlib.sha256(_encode([self.manifest.digest, case.case_id, ballot.ranks])).hexdigest()
                ballot = replace(ballot, ballot_id=ballot_id)
                ledger = ledger.record(self.manifest, ballot)
            summary = aggregate_rankings(self.manifest, ledger)
            if summary.cases_ranked != len(self.manifest.cases):
                raise ValueError("comparison is incomplete")
            if any(cell.status != "succeeded" for cell in self.manifest.cells) or len(self.manifest.cells) != len(case_by_id) * len(candidate_ids):
                raise ValueError("generation results are incomplete")
            self._save_ledger(ledger)
            preference_status = "unchanged"
            if payload["activate_preference"]:
                if summary.winner_candidate_id is None:
                    preference_status = "tied"
                elif self._preference_store is None:
                    preference_status = "unavailable"
                else:
                    try:
                        self._preference_store.write(summary.to_preference(self.manifest))
                        preference_status = "updated"
                    except (ValueError, OSError):
                        preference_status = "failed"
            return {"summary": summary.to_data(), "preference_status": preference_status}

    def dispatch(self, method: str, path: str, *, body: bytes = b"", review_token: str = "") -> ReviewResponse:
        """Handle bounded review requests without socket or provider dependencies."""
        if len(body) > _MAX_BODY:
            return ReviewResponse(413, "application/json", _encode({"error": "Submission is too large."}))
        try:
            if method == "GET" and path == "/api/comparison":
                return ReviewResponse(200, "application/json", _encode(self.view()))
            if method == "GET" and path.startswith("/asset/"):
                token = path.removeprefix("/asset/")
                asset = self._assets.get(token)
                if asset is None:
                    return ReviewResponse(404, "text/plain", b"Artifact not found.")
                return ReviewResponse(200, asset.media_type, self._asset_bytes(asset))
            if method == "GET" and path in _STATIC:
                filename, media_type = _STATIC[path]
                return ReviewResponse(200, media_type, Path(__file__).with_name("static").joinpath(filename).read_bytes())
            if method == "POST" and path == "/api/finalize":
                if not secrets.compare_digest(review_token, self._review_token):
                    return ReviewResponse(403, "application/json", _encode({"error": "Invalid review session."}))
                return ReviewResponse(200, "application/json", _encode(self.finalize(_decode(body))))
        except (ValueError, TypeError, OSError, UnicodeError):
            return ReviewResponse(409, "application/json", _encode({"error": "Review could not be saved or loaded. Verify complete rankings and unchanged artifacts."}))
        return ReviewResponse(404, "text/plain", b"Not found.")


def serve_review_directory(
    directory: Path,
    *,
    preference_path: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = False,
) -> None:
    """Serve one private comparison on loopback until interrupted."""
    if host != "127.0.0.1" or type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("review must use a valid loopback address and port")
    root = directory.expanduser().absolute()
    manifest = ComparisonManifest.from_data(_decode(_checked_file(root, "comparison.json", limit=4 * 1024 * 1024)))
    app = ReviewApplication(
        root, manifest, review_token=secrets.token_urlsafe(32),
        preference_store=JsonPreferenceStore(path=preference_path) if preference_path is not None else None,
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _handle(self) -> None:
            expected_host = f"127.0.0.1:{self.server.server_port}"  # type: ignore[attr-defined]
            if self.headers.get("Host") != expected_host:
                self.send_error(403, "Invalid review origin")
                return
            origin = self.headers.get("Origin")
            if origin is not None and origin != f"http://{expected_host}":
                self.send_error(403, "Invalid review origin")
                return
            if self.headers.get("Transfer-Encoding"):
                self.send_error(400, "Unsupported request encoding")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_error(400, "Invalid request size")
                return
            if not 0 <= length <= _MAX_BODY:
                self.send_error(413, "Request too large")
                return
            if self.command == "POST" and self.headers.get_content_type() != "application/json":
                self.send_error(415, "JSON required")
                return
            body = self.rfile.read(length) if length else b""
            reply = app.dispatch(self.command, urlsplit(self.path).path, body=body, review_token=self.headers.get("X-Review-Token", ""))
            self.send_response(reply.status)
            self.send_header("Content-Type", reply.media_type)
            self.send_header("Content-Length", str(len(reply.content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(reply.content)

        def do_GET(self) -> None:
            self._handle()

        def do_POST(self) -> None:
            self._handle()

    with ThreadingHTTPServer((host, port), Handler) as server:
        url = f"http://{host}:{server.server_port}/"
        print(f"Review page: {url}", flush=True)
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever(poll_interval=0.5)
        except KeyboardInterrupt:
            pass
