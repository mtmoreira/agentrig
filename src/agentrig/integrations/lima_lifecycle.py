"""Lima lifecycle candidate; activation requires an application artifact verifier.

The default catalog does not register this driver. It cannot establish workload
isolation or Docker readiness. Its verifier must check the complete approved
local artifact closure and owned instance/configuration before every command.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import json
from pathlib import Path

from agentrig.capabilities.backend_operations import BackendOperationJournal, JournaledLocalBackends
from agentrig.capabilities.local_backend import (
    BackendAction, BackendAuthority, BackendDescriptor, BackendEffect, BackendFailure,
    BackendPlan, BackendRequest, BackendResult, LocalBackendCatalog, check_backend_call,
)
from agentrig.core import JsonValue, RunContext
from agentrig.integrations.bounded_process import ProcessOutput, ProcessSpec, run_bounded_process
from agentrig.integrations.local_backends import _path, _pin
from agentrig.integrations.backend_staging import StagedWorkspace, WorkspaceStagePlan, stage_workspace


# True means this exact owned instance exists; False means verified local absence.
# This is a trusted application seam, never a field decoded from a saved receipt.
LimaArtifactVerifier = Callable[[BackendPlan], bool]
LimaProcess = Callable[[ProcessSpec, RunContext], Awaitable[ProcessOutput]]
_FORMAT = '{{json .Name}}|{{json .Dir}}|{{json .Status}}|{{json .VMType}}|{{json .Arch}}'


@dataclass(frozen=True, slots=True, kw_only=True)
class LimaLifecycleBackend:
    journal: BackendOperationJournal = field(repr=False)
    verify_artifacts: LimaArtifactVerifier | None = field(default=None, repr=False)
    execute: LimaProcess = field(default=run_bounded_process, repr=False)
    descriptor: BackendDescriptor = field(default=BackendDescriptor(
        backend_id='lima-vz-lifecycle-candidate', version='2.2.0-driver.1',
        capabilities=frozenset({'local-vm-lifecycle-candidate.v1', 'darwin-arm64.v1'}),
        actions=frozenset(BackendAction)), init=False)

    def plan(self, request: BackendRequest) -> BackendPlan:
        config = request.configuration
        if set(config) != {'executable', 'executable_sha256', 'state_root', 'configuration_file',
                          'configuration_sha256', 'artifact_closure_sha256', 'instance_id'}:
            raise ValueError('lima lifecycle configuration invalid')
        executable = _path(config['executable'])
        root = _path(config['state_root'])
        configuration = _path(config['configuration_file'])
        instance_id = config['instance_id']
        if (root == '/' or not configuration.startswith(root + '/') or
                not isinstance(instance_id, str) or len(instance_id) != 32 or
                any(char not in '0123456789abcdef' for char in instance_id)):
            raise ValueError('lima lifecycle scope invalid')
        for key in ('executable_sha256', 'configuration_sha256', 'artifact_closure_sha256'):
            _pin(config[key])
        effects = {BackendEffect.RUNTIME_CONTACT, BackendEffect.LOCAL_WRITE}
        if request.action in (BackendAction.PREPARE, BackendAction.START):
            effects.add(BackendEffect.PRIVATE_KEY_CREATION)
        if request.action is BackendAction.START:
            effects.add(BackendEffect.RUNTIME_START)
        return BackendPlan(descriptor=self.descriptor, request=request,
            effects=frozenset(effects), blockers=() if self.verify_artifacts is not None else
            ('approved_local_artifact_verifier_required',), material={
                'executable': executable, 'lima_home': root + '/lima',
                'instance': 'managed-' + instance_id, 'configuration_file': configuration,
                'host_staging_root': root + '/staging', 'guest_staging_root': '/mnt/agentrig',
                'automatic_retry': False, 'runtime_qualified': False,
                'downloads_permitted': False, 'application_ports_or_home_mounts_permitted': False})

    def _driver(self) -> JournaledLocalBackends:
        return JournaledLocalBackends(catalog=LocalBackendCatalog([_Execution(self)]), journal=self.journal)

    def stage(self, plan: BackendPlan, workspace: WorkspaceStagePlan, approved_digest: str,
              context: RunContext) -> StagedWorkspace:
        """Stage approved bytes locally; guest mount visibility remains unverified."""
        context.cancellation.raise_if_cancelled()
        if context.deadline is not None:
            context.deadline.raise_if_expired(context.clock)
        if (plan.descriptor != self.descriptor or self.plan(plan.request) != plan or
                workspace.host_root != Path(str(plan.material['host_staging_root'])) or
                workspace.guest_root != plan.material['guest_staging_root'] or approved_digest != workspace.digest):
            raise BackendFailure('backend_plan_changed')
        self._verify(plan)
        with self.journal.locked() as writer:
            current = writer.current()
            if current is not None and current.status == 'uncertain':
                raise BackendFailure('backend_reconciliation_required')
            return stage_workspace(workspace, approved_digest, context)

    async def apply(self, plan: BackendPlan, authority: BackendAuthority, context: RunContext) -> BackendResult:
        check_backend_call(self, plan, authority, context)
        self._verify(plan)
        return await self._driver().apply(plan, authority, context, journal_write_authorized=True)

    async def reconcile(self, inspection: BackendPlan, authority: BackendAuthority,
                        context: RunContext) -> BackendResult:
        check_backend_call(self, inspection, authority, context)
        self._verify(inspection)
        return await self._driver().reconcile(inspection, authority, context, journal_write_authorized=True)

    def _verify(self, plan: BackendPlan) -> bool:
        if self.verify_artifacts is None:
            raise BackendFailure('backend_plan_blocked')
        try:
            observed = self.verify_artifacts(plan)
        except BackendFailure:
            raise
        except Exception:
            raise BackendFailure('backend_execution_failed') from None
        if type(observed) is not bool:
            raise BackendFailure('backend_invalid_result')
        return observed

    async def _command(self, plan: BackendPlan, arguments: tuple[str, ...],
                       context: RunContext) -> ProcessOutput:
        context.cancellation.raise_if_cancelled()
        if context.deadline is not None:
            context.deadline.raise_if_expired(context.clock)
        self._verify(plan)
        root = plan.request.configuration['state_root']
        executable = plan.material['executable']
        lima_home = plan.material['lima_home']
        assert isinstance(root, str) and isinstance(executable, str) and isinstance(lima_home, str)
        spec = ProcessSpec(argv=(executable, '--tty=false', *arguments), cwd=root,
            environment={'LIMA_HOME': lima_home, 'HOME': root, 'XDG_CONFIG_HOME': root + '/config',
                         'XDG_CACHE_HOME': root + '/cache', 'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'},
            timeout_seconds=180 if plan.request.action is BackendAction.START else 60,
            max_output_bytes=65536, reject_overflow=True)
        result = await self.execute(spec, context)
        self._verify(plan)
        if (not isinstance(result, ProcessOutput) or type(result.exit_code) is not int or
                result.truncated or len(result.stdout) + len(result.stderr) > spec.max_output_bytes):
            raise BackendFailure('backend_invalid_result')
        return result

    async def _inspect(self, plan: BackendPlan, context: RunContext) -> str:
        instance = plan.material['instance']
        assert isinstance(instance, str)
        existed_before = self._verify(plan)
        result = await self._command(plan, ('list', '--format', _FORMAT, instance), context)
        exists_after = self._verify(plan)
        if not existed_before and not exists_after and result.exit_code == 1 and not result.stdout.strip():
            return 'absent'
        if not existed_before or not exists_after or result.exit_code != 0:
            raise BackendFailure('backend_execution_failed')
        try:
            fields = [json.loads(value) for value in result.stdout.decode().strip().split('|')]
        except (ValueError, UnicodeError):
            raise BackendFailure('backend_invalid_result') from None
        expected_dir = str(plan.material['lima_home']) + '/' + instance
        if (len(fields) != 5 or fields[0] != instance or fields[1] != expected_dir or
                fields[2] not in ('Running', 'Stopped') or fields[3:] != ['vz', 'aarch64']):
            raise BackendFailure('backend_invalid_result')
        return {'Running': 'running', 'Stopped': 'stopped'}[fields[2]]

    async def _execute(self, plan: BackendPlan, authority: BackendAuthority, context: RunContext) -> BackendResult:
        check_backend_call(self, plan, authority, context)
        before = await self._inspect(plan, context)
        action = plan.request.action
        instance = plan.material['instance']
        configuration = plan.material['configuration_file']
        assert isinstance(instance, str) and isinstance(configuration, str)
        if action is BackendAction.INSPECT:
            observed = before
        else:
            allowed = {BackendAction.PREPARE: 'absent', BackendAction.START: 'stopped', BackendAction.STOP: 'running'}
            if before != allowed[action]:
                raise BackendFailure('backend_reconciliation_required')
            commands = {BackendAction.PREPARE: ('create', '--name', instance, configuration),
                        BackendAction.START: ('start', '--timeout=120s', instance),
                        BackendAction.STOP: ('stop', instance)}
            result = await self._command(plan, commands[action], context)
            if result.exit_code != 0:
                raise BackendFailure('backend_execution_failed')
            observed = await self._inspect(plan, context)
            expected = 'running' if action is BackendAction.START else 'stopped'
            if observed != expected:
                raise BackendFailure('backend_reconciliation_required')
            if action is BackendAction.PREPARE:
                observed = 'prepared'
        binding: dict[str, JsonValue] = {'instance': instance, 'vm_type': 'vz',
                                      'workload_transport_qualified': False}
        return BackendResult(plan_digest=plan.digest, operation_id=plan.request.operation_id,
                             status=observed, binding=binding)


@dataclass(frozen=True, slots=True)
class _Execution:
    backend: LimaLifecycleBackend

    @property
    def descriptor(self) -> BackendDescriptor:
        return self.backend.descriptor

    def plan(self, request: BackendRequest) -> BackendPlan:
        return self.backend.plan(request)

    async def apply(self, plan: BackendPlan, authority: BackendAuthority, context: RunContext) -> BackendResult:
        return await self.backend._execute(plan, authority, context)
