"""Provider-neutral MCP server bindings with explicit tool authority."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from agentrig.capabilities.base import DataRetention
from agentrig.core._validation import require_trimmed_string


class McpTransport(StrEnum):
    """MCP transports supported by portable agent-runtime adapters."""

    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable_http"


@dataclass(frozen=True, slots=True, kw_only=True)
class McpServerBinding:
    """One bounded MCP server and the exact tools it may expose."""

    server_id: str
    transport: McpTransport
    allowed_tools: tuple[str, ...]
    command: tuple[str, ...] = field(default=(), repr=False)
    url: str | None = field(default=None, repr=False)
    environment_variables: tuple[str, ...] = ()
    bearer_token_environment_variable: str | None = None
    data_retention: DataRetention = DataRetention.NOT_RETAINED

    def __post_init__(self) -> None:
        require_trimmed_string("MCP server ID", self.server_id)
        if not isinstance(self.transport, McpTransport):
            raise TypeError("MCP transport must be an McpTransport")
        tools = _copy_names("MCP allowed tool", self.allowed_tools)
        if not tools:
            raise ValueError("MCP bindings require at least one allowed tool")
        command = _copy_names("MCP command argument", self.command)
        variables = _copy_names(
            "MCP environment variable",
            self.environment_variables,
        )
        if not isinstance(self.data_retention, DataRetention):
            raise TypeError("MCP data_retention must be a DataRetention")

        if self.transport is McpTransport.STDIO:
            if not command:
                raise ValueError("stdio MCP bindings require a command")
            if self.url is not None:
                raise ValueError("stdio MCP bindings cannot define a URL")
            if self.bearer_token_environment_variable is not None:
                raise ValueError(
                    "stdio MCP bindings cannot define a bearer token"
                )
        else:
            if command:
                raise ValueError(
                    "streamable HTTP MCP bindings cannot define a command"
                )
            if variables:
                raise ValueError(
                    "streamable HTTP MCP bindings cannot inherit environment"
                )
            if self.url is None:
                raise ValueError(
                    "streamable HTTP MCP bindings require a URL"
                )
            require_trimmed_string("MCP server URL", self.url)
            if not self.url.startswith(("https://", "http://localhost")):
                raise ValueError(
                    "MCP server URL must use HTTPS or localhost HTTP"
                )
            if self.bearer_token_environment_variable is not None:
                _require_environment_name(
                    self.bearer_token_environment_variable
                )

        for variable in variables:
            _require_environment_name(variable)
        object.__setattr__(self, "allowed_tools", tools)
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "environment_variables", variables)

    @property
    def tool_ids(self) -> tuple[str, ...]:
        """Return stable agent-contract IDs for the exposed tools."""
        return tuple(
            mcp_tool_id(self.server_id, tool) for tool in self.allowed_tools
        )

    def selected(self, tool_ids: tuple[str, ...]) -> McpServerBinding | None:
        """Return a copy exposing only selected stable tool IDs."""
        selected = tuple(
            tool
            for tool in self.allowed_tools
            if mcp_tool_id(self.server_id, tool) in tool_ids
        )
        if not selected:
            return None
        return McpServerBinding(
            server_id=self.server_id,
            transport=self.transport,
            allowed_tools=selected,
            command=self.command,
            url=self.url,
            environment_variables=self.environment_variables,
            bearer_token_environment_variable=(
                self.bearer_token_environment_variable
            ),
            data_retention=self.data_retention,
        )


def mcp_tool_id(server_id: str, tool_name: str) -> str:
    """Create the stable tool identity shared by contracts and events."""
    require_trimmed_string("MCP server ID", server_id)
    require_trimmed_string("MCP tool name", tool_name)
    return f"mcp.{server_id}.{tool_name}"


def _copy_names(field_name: str, values: tuple[str, ...]) -> tuple[str, ...]:
    copied = tuple(values)
    for value in copied:
        require_trimmed_string(field_name, value)
    if len(copied) != len(set(copied)):
        raise ValueError(f"{field_name}s must not contain duplicates")
    return copied


def _require_environment_name(value: str) -> None:
    require_trimmed_string("MCP environment variable", value)
    if not value.replace("_", "A").isalnum() or not value[0].isalpha():
        raise ValueError("MCP environment variable name is invalid")
