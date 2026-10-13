"""Bounded standard-library HTTP bridge for a caller-owned ComfyUI server."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import uuid

from agentrig.core._json import JsonValue, freeze_json_object, thaw_json_value
from agentrig.core._validation import require_trimmed_string
from agentrig.integrations.comfyui.images import (
    ComfyUIImageClient,
    ComfyUIImageOperation,
    ComfyUIImageRequest,
    ComfyUIImageResult,
)

_MANIFEST_SCHEMA = "agentrig.comfyui-workflows.v1"
_KEY = re.compile(r"(generate|reference_generate|edit):([0-9]|1[0-6])")
_NODE = re.compile(r"[A-Za-z0-9_.:-]{1,128}")
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_IMAGE_BYTES = 40 * 1024 * 1024


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> None:
        return None


_OPENER = build_opener(_NoRedirects())


@dataclass(frozen=True, slots=True, kw_only=True)
class ComfyUIWorkflowTemplate:
    """One API-format workflow and its output node."""

    operation: ComfyUIImageOperation
    source_count: int
    workflow: Mapping[str, JsonValue] = field(repr=False)
    output_node_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.operation, ComfyUIImageOperation):
            raise TypeError("ComfyUI workflow operation is invalid")
        if type(self.source_count) is not int or not 0 <= self.source_count <= 16:
            raise ValueError("ComfyUI workflow source count is invalid")
        if self.operation is ComfyUIImageOperation.GENERATE and self.source_count != 0:
            raise ValueError("ComfyUI generate workflow cannot declare sources")
        if self.operation is not ComfyUIImageOperation.GENERATE and self.source_count == 0:
            raise ValueError("ComfyUI conditioned workflow requires sources")
        frozen = freeze_json_object("ComfyUI workflow", self.workflow)
        if not frozen:
            raise ValueError("ComfyUI workflow must not be empty")
        placeholders = _placeholders(frozen)
        required = {"$agentrig.prompt", "$agentrig.model", "$agentrig.width",
            "$agentrig.height", "$agentrig.seed", "$agentrig.steps"}
        required.update(f"$agentrig.reference.{index}"
                        for index in range(1, self.source_count + 1))
        if not required <= placeholders:
            raise ValueError("ComfyUI workflow is missing required AgentRig placeholders")
        allowed = required | {"$agentrig.reference_strength"}
        if any(value.startswith("$agentrig.") and value not in allowed
               for value in placeholders):
            raise ValueError("ComfyUI workflow contains an unsupported AgentRig placeholder")
        object.__setattr__(self, "workflow", frozen)
        if _NODE.fullmatch(self.output_node_id) is None:
            raise ValueError("ComfyUI output node ID is invalid")


@dataclass(frozen=True, slots=True, kw_only=True)
class ComfyUIWorkflowCatalog:
    """Immutable operation/source-count workflow selection."""

    templates: Mapping[str, ComfyUIWorkflowTemplate]
    quality_steps: Mapping[str, int] = field(
        default_factory=lambda: {"auto": 8, "low": 4, "medium": 8, "high": 16}
    )

    def __post_init__(self) -> None:
        copied = dict(self.templates)
        if not copied:
            raise ValueError("ComfyUI workflow catalog requires templates")
        for key, value in copied.items():
            if _KEY.fullmatch(key) is None or not isinstance(value, ComfyUIWorkflowTemplate):
                raise ValueError("ComfyUI workflow catalog entry is invalid")
            expected = f"{value.operation.value}:{value.source_count}"
            if key != expected:
                raise ValueError("ComfyUI workflow catalog key differs")
        steps = dict(self.quality_steps)
        if set(steps) != {"auto", "low", "medium", "high"} or any(
            type(value) is not int or not 1 <= value <= 100 for value in steps.values()
        ):
            raise ValueError("ComfyUI quality step mapping is invalid")
        object.__setattr__(self, "templates", MappingProxyType(dict(sorted(copied.items()))))
        object.__setattr__(self, "quality_steps", MappingProxyType(dict(sorted(steps.items()))))

    def select(self, request: ComfyUIImageRequest) -> ComfyUIWorkflowTemplate:
        key = f"{request.operation.value}:{len(request.sources)}"
        try:
            return self.templates[key]
        except KeyError:
            raise ValueError("ComfyUI workflow is unavailable for this operation and input count") from None

    @classmethod
    def from_directory(cls, directory: Path) -> "ComfyUIWorkflowCatalog":
        """Read a caller-selected private manifest and API-format workflows."""
        expanded = directory.expanduser()
        root = expanded.resolve(strict=True)
        if root != expanded.absolute() or not root.is_dir():
            raise ValueError("ComfyUI workflow root must be a canonical directory")
        manifest_path = root / "manifest.json"
        raw = _read_json(manifest_path)
        if not isinstance(raw, dict) or raw.get("schema_id") != _MANIFEST_SCHEMA:
            raise ValueError("ComfyUI workflow manifest is invalid")
        entries = raw.get("workflows")
        if not isinstance(entries, list) or not entries:
            raise ValueError("ComfyUI workflow manifest requires workflows")
        templates: dict[str, ComfyUIWorkflowTemplate] = {}
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {
                "operation", "source_count", "file", "output_node_id"
            }:
                raise ValueError("ComfyUI workflow manifest entry is invalid")
            filename = entry["file"]
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError("ComfyUI workflow filename is invalid")
            workflow = _read_json(root / filename)
            template = ComfyUIWorkflowTemplate(
                operation=ComfyUIImageOperation(entry["operation"]),
                source_count=entry["source_count"],
                workflow=workflow,
                output_node_id=entry["output_node_id"],
            )
            key = f"{template.operation.value}:{template.source_count}"
            if key in templates:
                raise ValueError("ComfyUI workflow manifest contains duplicates")
            templates[key] = template
        quality_steps = raw.get("quality_steps", {"auto": 8, "low": 4, "medium": 8, "high": 16})
        if not isinstance(quality_steps, dict):
            raise ValueError("ComfyUI workflow quality steps are invalid")
        return cls(templates=templates, quality_steps=quality_steps)


@dataclass(frozen=True, slots=True, kw_only=True)
class ComfyUIHttpClientFactory:
    """Create short-lived clients for one loopback ComfyUI server."""

    catalog: ComfyUIWorkflowCatalog
    host: str = "http://127.0.0.1:8188"
    request_timeout_seconds: float = 30.0
    generation_timeout_seconds: float = 900.0
    poll_interval_seconds: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.catalog, ComfyUIWorkflowCatalog):
            raise TypeError("ComfyUI workflow catalog is invalid")
        object.__setattr__(self, "host", _loopback_host(self.host))
        for name, value, minimum in (
            ("request timeout", self.request_timeout_seconds, 0.1),
            ("generation timeout", self.generation_timeout_seconds, 1.0),
            ("poll interval", self.poll_interval_seconds, 0.05),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < minimum:
                raise ValueError(f"ComfyUI {name} is invalid")

    def create(self) -> ComfyUIImageClient:
        return _ComfyUIHttpClient(
            catalog=self.catalog,
            host=self.host,
            request_timeout=float(self.request_timeout_seconds),
            generation_timeout=float(self.generation_timeout_seconds),
            poll_interval=float(self.poll_interval_seconds),
        )


class _ComfyUIHttpClient:
    def __init__(self, *, catalog: ComfyUIWorkflowCatalog, host: str,
                 request_timeout: float, generation_timeout: float,
                 poll_interval: float) -> None:
        self._catalog = catalog
        self._host = host.rstrip("/")
        self._request_timeout = request_timeout
        self._generation_timeout = generation_timeout
        self._poll_interval = poll_interval
        self._closed = False

    async def create(self, request: ComfyUIImageRequest) -> ComfyUIImageResult:
        if self._closed:
            raise RuntimeError("ComfyUI client is closed")
        template = self._catalog.select(request)
        names: list[str] = []
        for index, source in enumerate(request.sources, 1):
            extension = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}.get(
                source.media_type.partition(";")[0]
            )
            if extension is None:
                raise ValueError("ComfyUI source media type is unsupported")
            name = f"agentrig-{sha256(source.content).hexdigest()}-{index}{extension}"
            await asyncio.to_thread(self._upload, name, source.media_type, source.content)
            names.append(name)
        seed_source = request.idempotency_key or sha256(request.prompt.encode()).hexdigest()
        seed = int.from_bytes(sha256(seed_source.encode()).digest()[:8], "big") % (2**63 - 1)
        bindings: dict[str, JsonValue] = {
            "$agentrig.prompt": request.prompt,
            "$agentrig.model": request.model,
            "$agentrig.width": request.width,
            "$agentrig.height": request.height,
            "$agentrig.seed": seed,
            "$agentrig.steps": self._catalog.quality_steps[request.quality.value],
            "$agentrig.reference_strength": (
                0.85 if request.input_fidelity is not None and request.input_fidelity.value == "high"
                else 0.55
            ),
        }
        for index, name in enumerate(names, 1):
            bindings[f"$agentrig.reference.{index}"] = name
        workflow = _bind(thaw_json_value(template.workflow), bindings)
        if _placeholders(workflow):
            raise ValueError("ComfyUI workflow contains unbound AgentRig placeholders")
        prompt_id = await asyncio.to_thread(self._queue, workflow)
        image_record = await self._wait(prompt_id, template.output_node_id)
        content = await asyncio.to_thread(self._view, image_record)
        return ComfyUIImageResult(
            content=content,
            media_type="image/png",
            model=request.model,
            workflow_id=prompt_id,
        )

    async def close(self) -> None:
        self._closed = True

    def _upload(self, filename: str, media_type: str, content: bytes) -> None:
        boundary = "agentrig-" + uuid.uuid4().hex
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            f"Content-Type: {media_type}\r\n\r\n"
        ).encode() + content + (
            f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue"
            f"\r\n--{boundary}--\r\n"
        ).encode()
        self._request_json("/upload/image", method="POST", content=body,
                           content_type=f"multipart/form-data; boundary={boundary}")

    def _queue(self, workflow: JsonValue) -> str:
        response = self._request_json("/prompt", method="POST", content=json.dumps({
            "prompt": workflow, "client_id": "agentrig-" + uuid.uuid4().hex,
        }, separators=(",", ":")).encode(), content_type="application/json")
        prompt_id = response.get("prompt_id") if isinstance(response, dict) else None
        if (not isinstance(response, dict)
            or not isinstance(prompt_id, str)
            or _NODE.fullmatch(prompt_id) is None):
            raise RuntimeError("ComfyUI queue response is invalid")
        return prompt_id

    async def _wait(self, prompt_id: str, output_node_id: str) -> dict[str, str]:
        deadline = asyncio.get_running_loop().time() + self._generation_timeout
        while True:
            history = await asyncio.to_thread(self._request_json, "/history/" + prompt_id)
            record = history.get(prompt_id) if isinstance(history, dict) else None
            if isinstance(record, dict):
                status = record.get("status")
                if isinstance(status, dict) and status.get("status_str") == "error":
                    raise RuntimeError("ComfyUI workflow execution failed")
                outputs = record.get("outputs")
                output = outputs.get(output_node_id) if isinstance(outputs, dict) else None
                images = output.get("images") if isinstance(output, dict) else None
                if isinstance(images, list) and images:
                    image = images[0]
                    if isinstance(image, dict) and all(
                        isinstance(image.get(name), str) for name in ("filename", "subfolder", "type")
                    ):
                        return {name: image[name] for name in ("filename", "subfolder", "type")}
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("ComfyUI generation timed out")
            await asyncio.sleep(self._poll_interval)

    def _view(self, image: dict[str, str]) -> bytes:
        query = urlencode(image)
        content = self._request_bytes("/view?" + query)
        if not content.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError("ComfyUI output is not PNG")
        return content

    def _request_json(self, path: str, *, method: str = "GET", content: bytes | None = None,
                      content_type: str | None = None) -> Any:
        raw = self._request_bytes(path, method=method, content=content, content_type=content_type,
                                  maximum=_MAX_JSON_BYTES)
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            raise RuntimeError("ComfyUI returned invalid JSON") from None

    def _request_bytes(self, path: str, *, method: str = "GET", content: bytes | None = None,
                       content_type: str | None = None, maximum: int = _MAX_IMAGE_BYTES) -> bytes:
        headers = {"Accept": "application/json, image/png"}
        if content_type is not None:
            headers["Content-Type"] = content_type
        request = Request(self._host + path, data=content, headers=headers, method=method)
        try:
            with _OPENER.open(request, timeout=self._request_timeout) as response:
                raw = bytes(response.read(maximum + 1))
        except (HTTPError, URLError, TimeoutError) as error:
            raise RuntimeError("ComfyUI HTTP request failed") from error
        if len(raw) > maximum:
            raise RuntimeError("ComfyUI response exceeds the size limit")
        return raw


def _bind(value: Any, bindings: Mapping[str, JsonValue]) -> JsonValue:
    if isinstance(value, str):
        return bindings.get(value, value)
    if isinstance(value, list):
        return [_bind(item, bindings) for item in value]
    if isinstance(value, dict):
        return {str(key): _bind(item, bindings) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise ValueError("ComfyUI workflow contains an unsupported value")


def _placeholders(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value} if value.startswith("$agentrig.") else set()
    if isinstance(value, (list, tuple)):
        result: set[str] = set()
        for item in value:
            result.update(_placeholders(item))
        return result
    if isinstance(value, Mapping):
        result = set()
        for item in value.values():
            result.update(_placeholders(item))
        return result
    return set()


def _loopback_host(value: str) -> str:
    require_trimmed_string("ComfyUI host", value)
    parsed = urlsplit(value)
    if (parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.path not in {"", "/"}):
        raise ValueError("ComfyUI host must use loopback HTTP")
    if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise ValueError("ComfyUI host must not contain credentials, query, or fragment")
    try:
        parsed.port
    except ValueError:
        raise ValueError("ComfyUI host port is invalid") from None
    return value.rstrip("/")


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_JSON_BYTES:
        raise ValueError("ComfyUI workflow file is invalid")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError):
        raise ValueError("ComfyUI workflow file is invalid") from None
    if not isinstance(value, dict):
        raise ValueError("ComfyUI workflow document must be an object")
    return value
