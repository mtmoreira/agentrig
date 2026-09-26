"""Portable backend behavior; all observations below are synthetic."""

import asyncio
from dataclasses import replace
import unittest

from agentrig.capabilities import (
    BackendAction, BackendAuthority, BackendDescriptor, BackendEffect, BackendFailure,
    BackendPlan, BackendRequest, BackendResult, LocalBackendCatalog,
)
from agentrig.core import Deadline, DeadlineExceeded, RunCancelled
from agentrig.integrations import ExistingRootlessDockerBackend, LimaBackend
from agentrig.testing import LocalBackendContractSuite, ScriptedLocalBackend
from tests.unit.test_scripted_tool import create_context, cancelled_context, FixedClock


OPERATION = "a" * 32
PIN = "sha256:" + "1" * 64


def request():
    return BackendRequest(action=BackendAction.INSPECT, operation_id=OPERATION,
        configuration={"executable": "/private/fixture/docker", "executable_sha256": PIN,
                       "socket": "/private/fixture/docker.sock", "architecture": "arm64"})


def lima_request(**changes):
    configuration = {"host": "darwin", "architecture": "arm64", "state_root": "/private/fixture",
                     "cpus": 2, "memory_mib": 2048, "disk_gib": 20}
    configuration.update(changes)
    return BackendRequest(action=BackendAction.PREPARE, operation_id=OPERATION,
                          configuration=configuration)


def authority(plan):
    return BackendAuthority(plan_digest=plan.digest, operation_id=plan.request.operation_id,
                            effects=plan.effects)


class LocalBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_conformance_suite_for_three_backends(self):
        calls = []

        async def inspect(plan, context):
            calls.append(plan.request.operation_id)
            return BackendResult(plan_digest=plan.digest, operation_id=plan.request.operation_id,
                                  status="running", binding={"synthetic": True})

        scripted = ScriptedLocalBackend(["running"])
        fixtures = [(scripted, request(), lambda: len(scripted.calls)),
                    (ExistingRootlessDockerBackend(inspector=inspect), request(), lambda: len(calls)),
                    (LimaBackend(), lima_request(), lambda: 0)]
        for backend, invocation, count in fixtures:
            with self.subTest(backend=backend.descriptor.backend_id):
                result = await LocalBackendContractSuite(backend=backend, request=invocation,
                    context=create_context(), cancelled_context=cancelled_context(),
                    expired_context=create_context(deadline=Deadline.after(0, FixedClock())),
                    invocation_count=count).verify()
                if backend.descriptor.backend_id == "lima-vz":
                    self.assertIsNone(result)
                else:
                    self.assertEqual(result.status, "running")

    async def test_deadline_and_cancellation_do_not_consume_script(self):
        backend = ScriptedLocalBackend(["running"])
        plan = backend.plan(request())
        catalog = LocalBackendCatalog([backend])
        with self.assertRaises(DeadlineExceeded):
            await catalog.apply(plan, authority(plan),
                                create_context(deadline=Deadline.after(0, FixedClock())))
        with self.assertRaises(RunCancelled):
            await backend.apply(plan, authority(plan), cancelled_context())
        self.assertEqual(backend.calls, ())

    async def test_extra_authority_and_other_operation_rejected(self):
        backend = ScriptedLocalBackend(["running"])
        plan = backend.plan(request())
        for grant in (replace(authority(plan), effects=plan.effects | {BackendEffect.RUNTIME_START}),
                      replace(authority(plan), operation_id="b" * 32)):
            with self.assertRaisesRegex(BackendFailure, "backend_authority_required"):
                await backend.apply(plan, grant, create_context())
        self.assertEqual(backend.calls, ())

    async def test_raw_errors_are_not_exposed(self):
        async def broken(plan, context):
            raise RuntimeError("private-transport-content")

        backend = ExistingRootlessDockerBackend(inspector=broken)
        plan = backend.plan(request())
        for operation in (backend.apply, LocalBackendCatalog([backend]).apply):
            with self.assertRaises(BackendFailure) as caught:
                await operation(plan, authority(plan), create_context())
            self.assertEqual(str(caught.exception), "backend_execution_failed")
            self.assertNotIn("private", repr(caught.exception))

    async def test_adapter_cancellation_preserved(self):
        async def interrupted(plan, context):
            raise asyncio.CancelledError()

        backend = ExistingRootlessDockerBackend(inspector=interrupted)
        plan = backend.plan(request())
        with self.assertRaises(asyncio.CancelledError):
            await LocalBackendCatalog([backend]).apply(plan, authority(plan), create_context())

    async def test_invalid_adapter_results_rejected(self):
        for change in ({"operation_id": "b" * 32}, {"plan_digest": "sha256:" + "2" * 64},
                       {"status": "prepared"}):
            async def invalid(plan, context):
                result = BackendResult(plan_digest=plan.digest, operation_id=plan.request.operation_id,
                                       status="running")
                return replace(result, **change)

            backend = ExistingRootlessDockerBackend(inspector=invalid)
            plan = backend.plan(request())
            with self.assertRaisesRegex(BackendFailure, "backend_invalid_result"):
                await LocalBackendCatalog([backend]).apply(plan, authority(plan), create_context())

    async def test_lima_cannot_be_enabled_by_pins_or_grant(self):
        backend = LimaBackend()
        plan = backend.plan(lima_request(guest_image="/private/fixture/guest.img",
            guest_image_sha256=PIN,
            host_archive_sha256="sha256:bbdef91774885a0d05f7b048c4eb89ae2bcf3a0c252ae7ca7934e63df76d93c3"))
        self.assertEqual(plan.blockers, ("guest_bundle_review_required", "lifecycle_driver_unqualified",
                                        "vm_transport_unqualified"))
        with self.assertRaisesRegex(BackendFailure, "backend_plan_blocked"):
            await backend.apply(plan, authority(plan), create_context())

    async def test_plan_recomputation_errors_are_normalized(self):
        class Changed(ScriptedLocalBackend):
            broken = False

            def plan(self, invocation):
                if self.broken:
                    raise RuntimeError("private-new-config")
                return super().plan(invocation)

        backend = Changed(["running"])
        plan = backend.plan(request())
        backend.broken = True
        with self.assertRaisesRegex(BackendFailure, "backend_plan_changed"):
            await LocalBackendCatalog([backend]).apply(plan, authority(plan), create_context())
        self.assertEqual(backend.calls, ())

    def test_catalog_has_no_fallback_and_pins_descriptors(self):
        backend = ScriptedLocalBackend(["running"])
        catalog = LocalBackendCatalog([backend, LimaBackend()])
        with self.assertRaisesRegex(BackendFailure, "backend_unavailable"):
            catalog.resolve("unknown")
        with self.assertRaisesRegex(BackendFailure, "backend_incompatible"):
            catalog.resolve("lima-vz", frozenset({"docker-unix-endpoint.v1"}))
        self.assertIs(catalog.resolve("scripted-local", frozenset({"docker-unix-endpoint.v1"})), backend)
        backend.descriptor = replace(backend.descriptor, version="2")
        with self.assertRaisesRegex(BackendFailure, "backend_plan_changed"):
            catalog.resolve("scripted-local")
        self.assertEqual(catalog.descriptors[0].version, "1")

    def test_duplicate_registration_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            LocalBackendCatalog([LimaBackend(), LimaBackend()])

    def test_plan_snapshots_and_private_reprs(self):
        private = {"nested": ["private-config-value"]}
        invocation = BackendRequest(action=BackendAction.INSPECT, operation_id=OPERATION,
                                    configuration=private)
        backend = ScriptedLocalBackend(["running"])
        plan = backend.plan(invocation)
        before = plan.digest
        private["nested"].append("changed")
        self.assertEqual(plan.digest, before)
        exported = plan.to_json()
        exported["configuration"]["nested"].append("changed-export")
        self.assertEqual(plan.digest, before)
        self.assertNotIn("private-config", repr(plan))
        self.assertNotIn("private-config", repr(invocation))
        self.assertNotIn("private-config", repr(BackendResult(plan_digest=before, operation_id=OPERATION,
                          status="running", binding=private)))
        with self.assertRaises(TypeError):
            invocation.configuration["change"] = True

    def test_lima_plan_contains_no_ambient_mount_or_installer(self):
        plan = LimaBackend().plan(lima_request())
        config = plan.material["configuration_proposal"]
        self.assertEqual(config["mounts"], ())
        self.assertEqual(config["images"], ())
        self.assertEqual(config["provision"], ())
        self.assertFalse(config["containerd"]["user"])
        self.assertFalse(config["propagateProxyEnv"])
        self.assertFalse(config["ssh"]["loadDotSSHPubKeys"])
        self.assertTrue(config["portForwards"][0]["ignore"])
        self.assertFalse(plan.material["execution_qualified"])

    def test_lima_rejects_unsafe_or_unknown_inputs(self):
        for changes in ({"state_root": "/"}, {"state_root": "/private/../home"},
                        {"state_root": "~/home"}, {"state_root": "/private/./x"},
                        {"state_root": "/private/x\n"}, {"cpus": True}, {"cpus": 9},
                        {"memory_mib": 8193}, {"disk_gib": 129}, {"host": "linux"},
                        {"architecture": "amd64"}, {"template": "docker"},
                        {"qualified": True}, {"host_archive_sha256": PIN},
                        {"guest_image": "https://example.invalid/image"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                LimaBackend().plan(lima_request(**changes))

    def test_existing_endpoint_has_no_lifecycle_or_discovery(self):
        backend = ExistingRootlessDockerBackend()
        plan = backend.plan(request())
        self.assertEqual(plan.blockers, ("endpoint_inspector_required",))
        self.assertFalse(plan.material["discovery"])
        with self.assertRaises(ValueError):
            backend.plan(replace(request(), action=BackendAction.START))

    def test_impossible_values_rejected(self):
        for operation in ("", "../private", "A" * 32):
            with self.assertRaises(ValueError):
                replace(request(), operation_id=operation)
        with self.assertRaises(ValueError):
            BackendDescriptor(backend_id="bad private value", version="1",
                capabilities=frozenset({"ok"}), actions=frozenset({BackendAction.INSPECT}))
        with self.assertRaises(ValueError):
            replace(request(), configuration={"large": "x" * 65536})
        with self.assertRaises(ValueError):
            replace(request(), configuration={"nan": float("nan")})


if __name__ == "__main__":
    unittest.main()
