"""Provider-neutral runtime-backed coding capability composition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from agentrig.agents import (
    Agent,
    AgentContract,
    AgentLimits,
    AgentRuntime,
    ConfiguredAgent,
)
from agentrig.capabilities import (
    CapabilityDescriptor,
    CapabilityFeature,
    CapabilityKind,
    CapabilityLimit,
    ChangedFileEvidence,
    CodingAgent,
    CodingChangeKind,
    CodingResult,
    CodingTask,
    CodingValidationStatus,
    DataRetention,
    ValidationEvidence,
)
from agentrig.core import AgentRigError, EffectProfile, Failure, FailureKind
from agentrig.core._json import JsonValue
from agentrig.core.context import RunContext

TASK_SCHEMA = "example.bounded-coding-task.v1"
REPORT_SCHEMA = "example.bounded-coding-report.v1"

REPORT_JSON_SCHEMA: Mapping[str, JsonValue] = {
    "type": "object",
    "properties": {
        "changed_files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "change_kind": {
                        "type": "string",
                        "enum": ["added", "modified", "deleted"],
                    },
                },
                "required": ["path", "change_kind"],
                "additionalProperties": False,
            },
            "maxItems": 1,
        },
        "validations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "validation_id": {"type": "string"},
                    "summary": {"type": "string"},
                    "exit_code": {"type": "integer", "minimum": 0},
                },
                "required": ["validation_id", "summary", "exit_code"],
                "additionalProperties": False,
            },
            "minItems": 1,
        },
    },
    "required": ["changed_files", "validations"],
    "additionalProperties": False,
}

INSTRUCTIONS = (
    "Complete exactly the encoded coding task inside its workspace. Change only "
    "authorized relative paths, do not use the network, run the requested "
    "validation, and return only the structured report. Report a validation "
    "only after observing its zero exit code."
)


@dataclass(frozen=True, slots=True, kw_only=True)
class CodingReport:
    """Strict provider report before task-relative authorization checks."""

    changed_files: tuple[ChangedFileEvidence, ...]
    validations: tuple[ValidationEvidence, ...]


@dataclass(frozen=True, slots=True)
class CodingTaskCodec:
    schema_id: str = TASK_SCHEMA

    def encode(self, value: CodingTask) -> JsonValue:
        if not isinstance(value, CodingTask):
            raise ValueError("coding task required")
        return {
            "task_id": value.task_id,
            "workspace_id": value.workspace.workspace_id,
            "workspace_root": value.workspace.root_path,
            "writable_roots": value.workspace.writable_roots,
            "objective": value.objective,
            "acceptance_criteria": value.acceptance_criteria,
            "max_changed_files": value.max_changed_files,
        }


@dataclass(frozen=True, slots=True)
class CodingReportCodec:
    schema_id: str = REPORT_SCHEMA

    def decode(self, value: JsonValue) -> CodingReport:
        if not isinstance(value, Mapping) or set(value) != {
            "changed_files",
            "validations",
        }:
            raise ValueError("coding report object required")
        encoded_changes = value["changed_files"]
        encoded_validations = value["validations"]
        if not isinstance(encoded_changes, (list, tuple)):
            raise ValueError("coding report changes must be an array")
        if len(encoded_changes) > 1:
            raise ValueError("coding report exceeds its changed-file bound")
        changes = tuple(_decode_change(item) for item in encoded_changes)
        if not isinstance(encoded_validations, (list, tuple)):
            raise ValueError("coding report validations must be an array")
        if not encoded_validations:
            raise ValueError("coding report requires validation evidence")
        validations = tuple(
            _decode_validation(item) for item in encoded_validations
        )
        return CodingReport(
            changed_files=changes,
            validations=validations,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeCodingAgent:
    """Expose one structured autonomous runtime as a bounded CodingAgent."""

    descriptor: CapabilityDescriptor
    _agent: Agent[CodingTask, CodingReport] = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, CapabilityDescriptor):
            raise TypeError("coding descriptor must be a CapabilityDescriptor")
        if self.descriptor.kind is not CapabilityKind.CODING:
            raise ValueError("coding descriptor must use the coding kind")
        if not isinstance(self._agent, Agent):
            raise TypeError("runtime coding agent requires an Agent")

    async def execute(
        self,
        task: CodingTask,
        context: RunContext,
    ) -> CodingResult:
        if not isinstance(task, CodingTask):
            raise TypeError("runtime coding task must be a CodingTask")
        if not isinstance(context, RunContext):
            raise TypeError("runtime coding context must be a RunContext")
        task.require_supported_by(self.descriptor)
        report = (await self._agent.run(task, context)).unwrap()
        try:
            return CodingResult.succeeded(
                task=task,
                changed_files=report.changed_files,
                validations=report.validations,
            )
        except (TypeError, ValueError) as error:
            raise AgentRigError(
                Failure(
                    kind=FailureKind.INVALID_INPUT,
                    message="coding report violated the authorized task contract",
                    code="example.coding_report_invalid",
                )
            ) from error


def configure_runtime_coding_agent(
    runtime: AgentRuntime,
    *,
    runtime_capability_id: str,
    tool_id: str,
    coding_capability_id: str,
    coding_capability_version: str,
    data_retention: DataRetention,
) -> CodingAgent:
    """Bind a portable coding capability to an injected autonomous runtime."""
    for label, value in (
        ("runtime capability ID", runtime_capability_id),
        ("tool ID", tool_id),
        ("coding capability ID", coding_capability_id),
        ("coding capability version", coding_capability_version),
    ):
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"{label} must be nonempty and trimmed")
    if not isinstance(data_retention, DataRetention):
        raise TypeError("coding data retention must be a DataRetention")
    descriptor = CapabilityDescriptor(
        capability_id=coding_capability_id,
        version=coding_capability_version,
        kind=CapabilityKind.CODING,
        features=frozenset({CapabilityFeature.TOOL_USE}),
        limits={CapabilityLimit.MAX_CHANGED_FILES: 1},
        data_retention=data_retention,
    )
    contract = AgentContract[CodingTask, CodingReport](
        agent_id="bounded-coder",
        version="1",
        purpose="Complete one bounded coding task and return safe evidence",
        input_schema=TASK_SCHEMA,
        output_schema=REPORT_SCHEMA,
        prompt_version="bounded-coder-prompt-1",
        effect_profile=EffectProfile.NON_REPEATABLE,
        limits=AgentLimits(max_turns=1, max_tool_calls=8),
        stopping_policy="validated_coding_report",
        allowed_tools=(tool_id,),
        allowed_capabilities=(runtime_capability_id,),
        permissions={"workspace": "read_write", "network": "denied"},
    )
    return RuntimeCodingAgent(
        descriptor=descriptor,
        _agent=ConfiguredAgent(
            runtime=runtime,
            contract=contract,
            instructions=INSTRUCTIONS,
            input_codec=CodingTaskCodec(),
            output_codec=CodingReportCodec(),
        ),
    )


def _decode_change(value: JsonValue) -> ChangedFileEvidence:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "change_kind",
    }:
        raise ValueError("changed-file evidence object required")
    path = value["path"]
    change_kind = value["change_kind"]
    if not isinstance(path, str):
        raise ValueError("changed-file path must be a string")
    if not isinstance(change_kind, str):
        raise ValueError("changed-file kind must be a string")
    try:
        decoded_kind = CodingChangeKind(change_kind)
    except ValueError as error:
        raise ValueError("changed-file kind is invalid") from error
    return ChangedFileEvidence(path=path, change_kind=decoded_kind)


def _decode_validation(value: JsonValue) -> ValidationEvidence:
    if not isinstance(value, Mapping) or set(value) != {
        "validation_id",
        "summary",
        "exit_code",
    }:
        raise ValueError("validation evidence object required")
    validation_id = value["validation_id"]
    summary = value["summary"]
    exit_code = value["exit_code"]
    if not isinstance(validation_id, str):
        raise ValueError("validation ID must be a string")
    if not isinstance(summary, str):
        raise ValueError("validation summary must be a string")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        raise ValueError("validation exit code must be an integer")
    return ValidationEvidence(
        validation_id=validation_id,
        status=CodingValidationStatus.PASSED,
        summary=summary,
        exit_code=exit_code,
    )
