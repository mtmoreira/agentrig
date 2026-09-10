"""Explicit existing endpoint and review-only Lima backend adapters.

No discovery, optional SDK import, installer, shell or runtime contact occurs
when importing this module or constructing a plan. Lima execution deliberately
fails closed until its artifact closure and lifecycle driver are qualified.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
import re

from agentrig.capabilities.local_backend import (
    BackendAction, BackendAuthority, BackendDescriptor, BackendEffect,
    BackendFailure, BackendPlan, BackendRequest, BackendResult, check_backend_call,
)
from agentrig.core import JsonValue, RunContext
from agentrig.core.deadline import DeadlineExceeded


def _path(value: JsonValue) -> str:
    if (not isinstance(value, str) or len(value) > 1024 or not value.startswith("/")
            or value.startswith("//") or str(PurePosixPath(value)) != value
            or any(part == ".." for part in PurePosixPath(value).parts)
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise ValueError("backend path invalid")
    return value


def _pin(value: JsonValue) -> str:
    if not isinstance(value, str) or re.fullmatch(r"sha256:[a-f0-9]{64}", value) is None:
        raise ValueError("backend artifact pin invalid")
    return value


def _integer(value: JsonValue, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("backend resource invalid")
    return value


EndpointInspector = Callable[[BackendPlan, RunContext], Awaitable[BackendResult]]


@dataclass(frozen=True, slots=True, kw_only=True)
class ExistingRootlessDockerBackend:
    """Inspect exactly one supplied endpoint using the caller's verified adapter.

    The inspector must verify executable bytes, socket ownership, rootless
    daemon identity and architecture before returning. This class is a portable
    dispatch boundary, not an alternative to those transport-specific checks.
    It advertises no VM creation, image acquisition or workload execution.
    """

    inspector: EndpointInspector | None = field(default=None, repr=False, compare=False)
    descriptor: BackendDescriptor = field(default=BackendDescriptor(
        backend_id="existing-rootless-docker", version="1",
        capabilities=frozenset({"docker-unix-endpoint.v1", "current-user-rootless.v1"}),
        actions=frozenset({BackendAction.INSPECT})), init=False)

    def plan(self, request: BackendRequest) -> BackendPlan:
        config = request.configuration
        if set(config) != {"executable", "executable_sha256", "socket", "architecture"}:
            raise ValueError("backend configuration invalid")
        if config["architecture"] not in ("arm64", "amd64"):
            raise ValueError("backend architecture invalid")
        material: dict[str, JsonValue] = {
            "executable": _path(config["executable"]),
            "executable_sha256": _pin(config["executable_sha256"]),
            "socket": _path(config["socket"]), "architecture": config["architecture"],
            "discovery": False, "installation": False, "image_pull": False,
        }
        return BackendPlan(descriptor=self.descriptor, request=request,
            effects=frozenset({BackendEffect.RUNTIME_CONTACT}), material=material,
            blockers=() if self.inspector is not None else ("endpoint_inspector_required",))

    async def apply(self, plan: BackendPlan, authority: BackendAuthority,
                    context: RunContext) -> BackendResult:
        check_backend_call(self, plan, authority, context)
        if self.inspector is None:
            raise BackendFailure("backend_unavailable")
        try:
            result = await self.inspector(plan, context)
        except (BackendFailure, DeadlineExceeded):
            raise
        except Exception:
            raise BackendFailure("backend_execution_failed") from None
        if (not isinstance(result, BackendResult) or result.plan_digest != plan.digest
                or result.operation_id != plan.request.operation_id
                or result.status not in {"running", "stopped", "absent"}):
            raise BackendFailure("backend_invalid_result")
        return result


@dataclass(frozen=True, slots=True)
class LimaBackend:
    """Lima 2.2.0 VZ preparation review; it cannot create or start a VM yet.

    Pins are proposed inputs, never observations or authorization. The fixed
    qualification blockers cannot be cleared by config flags or saved receipts.
    Replacing this adapter does not change the portable catalog or its consumer.
    """

    descriptor: BackendDescriptor = field(default=BackendDescriptor(
        backend_id="lima-vz", version="2.2.0-plan.1",
        capabilities=frozenset({"local-vm-plan.v1", "darwin-arm64.v1"}),
        actions=frozenset({BackendAction.PREPARE})), init=False)

    def plan(self, request: BackendRequest) -> BackendPlan:
        config = request.configuration
        required = {"host", "architecture", "state_root", "cpus", "memory_mib", "disk_gib"}
        optional = {"host_archive_sha256", "guest_image", "guest_image_sha256"}
        if not required <= set(config) or set(config) - required - optional:
            raise ValueError("backend configuration invalid")
        if config["host"] != "darwin" or config["architecture"] != "arm64":
            raise ValueError("backend host unsupported")
        root = _path(config["state_root"])
        if root == "/":
            raise ValueError("backend state root invalid")
        cpus = _integer(config["cpus"], 1, 8)
        memory = _integer(config["memory_mib"], 1024, 8192)
        disk = _integer(config["disk_gib"], 8, 128)
        blockers = ["guest_bundle_review_required", "lifecycle_driver_unqualified",
                    "vm_transport_unqualified"]
        # Only this reviewed host release is an eligible proposal. Verification
        # of archive bytes, members and bundled executables is still pending.
        host_pin = "sha256:bbdef91774885a0d05f7b048c4eb89ae2bcf3a0c252ae7ca7934e63df76d93c3"
        if "host_archive_sha256" not in config:
            blockers.append("host_archive_pin_required")
        elif _pin(config["host_archive_sha256"]) != host_pin:
            raise ValueError("backend host archive unsupported")
        images: list[JsonValue] = []
        if {"guest_image", "guest_image_sha256"} <= set(config):
            images.append({"location": _path(config["guest_image"]), "arch": "aarch64",
                           "digest": _pin(config["guest_image_sha256"])})
        else:
            if "guest_image" in config:
                _path(config["guest_image"])
            if "guest_image_sha256" in config:
                _pin(config["guest_image_sha256"])
            blockers.append("guest_image_pin_required")
        material: dict[str, JsonValue] = {
            "lima_version": "2.2.0", "lima_home": root + "/lima",
            "instance": "managed-" + request.operation_id,
            "artifact_policy": "reviewed-local-preprovisioned-image-only",
            "execution_qualified": False,
            "configuration_proposal": {
                "vmType": "vz", "arch": "aarch64", "cpus": cpus,
                "memory": str(memory) + "MiB", "disk": str(disk) + "GiB",
                "images": images, "mounts": [],
                "containerd": {"system": False, "user": False},
                "ssh": {"loadDotSSHPubKeys": False, "forwardAgent": False,
                        "forwardX11": False, "forwardX11Trusted": False,
                        "overVsock": True},
                "propagateProxyEnv": False, "upgradePackages": False,
                "provision": [], "probes": [],
                "portForwards": [{"guestIP": "0.0.0.0", "guestPortRange": [1, 65535],
                                  "proto": "any", "ignore": True}],
            },
            "pending": ["guest-package-and-license-inventory", "disable-default-guest-installation",
                "rootless-engine-and-cgroup-delegation", "owned-unix-forward-and-staging",
                "bounded-process-and-operation-reconciliation", "real-isolation-selftest"],
        }
        return BackendPlan(descriptor=self.descriptor, request=request,
            effects=frozenset({BackendEffect.LOCAL_WRITE}), material=material,
            blockers=tuple(blockers))

    async def apply(self, plan: BackendPlan, authority: BackendAuthority,
                    context: RunContext) -> BackendResult:
        check_backend_call(self, plan, authority, context)
        raise BackendFailure("backend_plan_blocked")
