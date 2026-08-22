from __future__ import annotations

import unittest

from agentrig.capabilities import (
    DataRetention,
    McpServerBinding,
    McpTransport,
    mcp_tool_id,
)
from agentrig.integrations.openai import (
    CodexApprovalMode,
    CodexSandboxMode,
    CodexSandboxPolicy,
    CodexThreadRequest,
)


class McpServerBindingTest(unittest.TestCase):
    def test_stdio_binding_has_stable_tool_ids_and_selection(self) -> None:
        binding = McpServerBinding(
            server_id="eda",
            transport=McpTransport.STDIO,
            command=("/usr/bin/eda-mcp", "--stdio"),
            allowed_tools=("lint", "simulate"),
            environment_variables=("EDA_LICENSE",),
            data_retention=DataRetention.NOT_RETAINED,
        )

        self.assertEqual(
            binding.tool_ids,
            ("mcp.eda.lint", "mcp.eda.simulate"),
        )
        selected = binding.selected((mcp_tool_id("eda", "simulate"),))
        if selected is None:
            raise AssertionError("selected MCP binding must be present")
        self.assertEqual(selected.allowed_tools, ("simulate",))
        self.assertIsNone(binding.selected(("mcp.other.tool",)))

    def test_rejects_ambiguous_or_insecure_transport_configuration(self) -> None:
        with self.assertRaises(ValueError):
            McpServerBinding(
                server_id="eda",
                transport=McpTransport.STDIO,
                allowed_tools=("simulate",),
            )
        with self.assertRaises(ValueError):
            McpServerBinding(
                server_id="eda",
                transport=McpTransport.STREAMABLE_HTTP,
                url="http://example.com/mcp",
                allowed_tools=("simulate",),
            )

    def test_codex_thread_accepts_only_tools_from_bound_servers(self) -> None:
        binding = McpServerBinding(
            server_id="eda",
            transport=McpTransport.STDIO,
            command=("/usr/bin/eda-mcp",),
            allowed_tools=("simulate",),
        )
        request = CodexThreadRequest(
            model="gpt-test",
            instructions="Use only the selected simulator.",
            sandbox=CodexSandboxPolicy(
                mode=CodexSandboxMode.READ_ONLY,
                cwd="/workspace",
            ),
            approval_mode=CodexApprovalMode.DENY_ALL,
            allowed_tools=("mcp.eda.simulate",),
            mcp_servers=(binding,),
        )
        self.assertEqual(request.mcp_servers, (binding,))
        with self.assertRaises(ValueError):
            CodexThreadRequest(
                model="gpt-test",
                instructions="Request an unavailable tool.",
                sandbox=request.sandbox,
                approval_mode=request.approval_mode,
                allowed_tools=("mcp.eda.lint",),
                mcp_servers=(binding,),
            )


if __name__ == "__main__":
    unittest.main()
