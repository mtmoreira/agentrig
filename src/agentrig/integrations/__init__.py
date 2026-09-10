"""Optional provider and executable integrations."""

from agentrig.integrations.local_backends import ExistingRootlessDockerBackend, LimaBackend

from agentrig.integrations.process import (
    CommandInput,
    CommandOutput,
    CommandTool,
    DetachedCommandOutput,
    DetachedCommandTool,
)

__all__ = (
    "ExistingRootlessDockerBackend",
    "LimaBackend",
    "CommandInput",
    "CommandOutput",
    "CommandTool",
    "DetachedCommandOutput",
    "DetachedCommandTool",
)
