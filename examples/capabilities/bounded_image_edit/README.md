# Bounded image edit

This offline example edits a wholly synthetic Nia-and-Tomas scene through
AgentRig's provider-neutral image contracts. Four inputs have explicit roles:
one edit base, two identity references, and one edit mask. The output artifact
retains every input artifact ID in order.

The executor selects the `primary` route by ID, retries it once after a declared
transient failure, and never invokes the configured `unselected` route. Usage
cost stays `null` because the scripted implementation did not report it. The
printed report contains IDs and classifications, not prompts or image bytes.

```sh
uv run python -m examples.capabilities.bounded_image_edit.scripted
uv run python -m unittest -v \
  examples.capabilities.bounded_image_edit.test_scripted
./buck2 test //examples/capabilities/bounded_image_edit:test
```

No live provider lane is part of this example. An application may compose the
same request and executor with an optional integration only after separately
owning credentials, storage resolution, output publication, and live-call
authorization.
