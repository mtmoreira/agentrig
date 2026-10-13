from __future__ import annotations

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from threading import Thread
from typing import Any, cast
import unittest

from agentrig.capabilities import ImageInputRole
from agentrig.integrations.comfyui import (
    ComfyUIImageOperation,
    ComfyUIImageRequest,
    ComfyUIImageSource,
    ComfyUIHttpClientFactory,
    ComfyUIWorkflowCatalog,
    ComfyUIWorkflowTemplate,
)


_PNG = b"\x89PNG\r\n\x1a\nsynthetic"


class _Handler(BaseHTTPRequestHandler):
    uploaded = b""
    queued: dict[str, Any] = {}
    redirect_upload = False

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_POST(self) -> None:
        size = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(size)
        if self.path == "/upload/image":
            if type(self).redirect_upload:
                self.send_response(307)
                self.send_header("Location", "https://example.invalid/private-upload")
                self.end_headers()
                return
            type(self).uploaded = body
            self._json({"name": "uploaded.png", "subfolder": "", "type": "input"})
            return
        if self.path == "/prompt":
            type(self).queued = json.loads(body)
            self._json({"prompt_id": "prompt-1"})
            return
        self.send_error(404)

    def do_GET(self) -> None:
        if self.path == "/history/prompt-1":
            self._json({"prompt-1": {"outputs": {"9": {"images": [{
                "filename": "result.png", "subfolder": "", "type": "output",
            }]}}}})
            return
        if self.path.startswith("/view?"):
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(_PNG)))
            self.end_headers()
            self.wfile.write(_PNG)
            return
        self.send_error(404)

    def _json(self, value: object) -> None:
        content = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


class ComfyUIHttpBridgeTests(unittest.TestCase):
    def test_conditioned_generation_uses_the_standard_comfyui_http_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "reference.json").write_text(json.dumps({
                "1": {"class_type": "LoadImage", "inputs": {
                    "image": "$agentrig.reference.1"}},
                "2": {"class_type": "Sampler", "inputs": {
                    "prompt": "$agentrig.prompt", "model": "$agentrig.model",
                    "width": "$agentrig.width", "height": "$agentrig.height",
                    "seed": "$agentrig.seed", "steps": "$agentrig.steps"}},
                "9": {"class_type": "SaveImage", "inputs": {
                    "filename_prefix": "agentrig"}},
            }))
            (root / "manifest.json").write_text(json.dumps({
                "schema_id": "agentrig.comfyui-workflows.v1",
                "workflows": [{"operation": "reference_generate", "source_count": 1,
                    "file": "reference.json", "output_node_id": "9"}],
            }))
            server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            client = ComfyUIHttpClientFactory(
                catalog=ComfyUIWorkflowCatalog.from_directory(root),
                host=f"http://127.0.0.1:{server.server_port}",
                poll_interval_seconds=0.05,
            ).create()
            try:
                result = asyncio.run(client.create(ComfyUIImageRequest(
                    operation=ComfyUIImageOperation.REFERENCE_GENERATE,
                    model="flux-2-klein-4b-fp8.safetensors",
                    prompt="A private synthetic test scene",
                    width=768,
                    height=1024,
                    output_media_type="image/png",
                    sources=(ComfyUIImageSource(
                        role=ImageInputRole.IDENTITY_REFERENCE,
                        content=b"synthetic-reference",
                        media_type="image/png",
                    ),),
                    idempotency_key="stable-book-cover",
                )))
            finally:
                asyncio.run(client.close())
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

        self.assertEqual(result.content, _PNG)
        self.assertEqual(result.workflow_id, "prompt-1")
        self.assertIn(b"synthetic-reference", _Handler.uploaded)
        prompt = cast(dict[str, Any], _Handler.queued["prompt"])
        self.assertEqual(prompt["1"]["inputs"]["image"],
                         "agentrig-" +
                         sha256(b"synthetic-reference").hexdigest() +
                         "-1.png")
        self.assertEqual(prompt["2"]["inputs"]["model"],
                         "flux-2-klein-4b-fp8.safetensors")
        self.assertEqual(prompt["2"]["inputs"]["steps"], 8)

    def test_loopback_bridge_rejects_redirects_before_sending_images_elsewhere(self) -> None:
        catalog = ComfyUIWorkflowCatalog(templates={
            "reference_generate:1": ComfyUIWorkflowTemplate(
                operation=ComfyUIImageOperation.REFERENCE_GENERATE,
                source_count=1,
                workflow={"1": {"class_type": "Synthetic", "inputs": {
                    "prompt": "$agentrig.prompt", "model": "$agentrig.model",
                    "width": "$agentrig.width", "height": "$agentrig.height",
                    "seed": "$agentrig.seed", "steps": "$agentrig.steps",
                    "reference": "$agentrig.reference.1"}},
                    "9": {"class_type": "SaveImage", "inputs": {}}},
                output_node_id="9",
            ),
        })
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        _Handler.redirect_upload = True
        client = ComfyUIHttpClientFactory(catalog=catalog,
            host=f"http://127.0.0.1:{server.server_port}").create()
        try:
            with self.assertRaisesRegex(RuntimeError, "HTTP request failed"):
                asyncio.run(client.create(ComfyUIImageRequest(
                    operation=ComfyUIImageOperation.REFERENCE_GENERATE,
                    model="synthetic-model", prompt="Synthetic prompt",
                    width=64, height=64, output_media_type="image/png",
                    sources=(ComfyUIImageSource(
                        role=ImageInputRole.IDENTITY_REFERENCE,
                        content=b"private-image", media_type="image/png"),),
                )))
        finally:
            _Handler.redirect_upload = False
            asyncio.run(client.close())
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
