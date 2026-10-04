# OpenAI image model compatibility

AgentRig keeps image input fidelity as a portable capability. The OpenAI adapter
translates that capability according to the selected model.

| Model family | Portable high fidelity | Provider `input_fidelity` field |
| --- | --- | --- |
| GPT Image 1 and 1.5 | Supported | Sent when requested |
| GPT Image 2 | Supported implicitly | Omitted |
| GPT Image 2.5 Flare and Sunburst | Supported implicitly | Omitted |

Both undated GPT Image 2.5 IDs and the `2026-09-08` snapshots use the implicit
translation. Unit coverage verifies that the caller's portable intent remains
valid while the translated provider request omits the legacy field.
