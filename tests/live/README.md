# Live provider validation

Live tests are isolated from AgentRig's default offline lanes. They require an
explicit opt-in and fail when the selected provider is not authenticated.

The Codex runtime test uses the official SDK's existing authentication, starts
an ephemeral thread in a read-only sandbox with network access disabled, and
requests a small strict JSON result without tools. It verifies the normalized
result, provider metadata, usage, and safe event projection without recording
the prompt or output in events.

Run only this test with:

```sh
./buck2 test //tests/live:codex_runtime -- \
  --env AGENTRIG_RUN_LIVE=1
```

The default model is `gpt-5.6-terra`. Override it explicitly when validating a
different supported Codex model:

```sh
./buck2 test //tests/live:codex_runtime -- \
  --env AGENTRIG_RUN_LIVE=1 \
  --env AGENTRIG_CODEX_LIVE_MODEL=gpt-5.6-terra
```

Authenticate Codex before running the test using a supported ChatGPT login or
API-key login described in the official
[Codex authentication guide](https://developers.openai.com/codex/auth). Never
pass credentials in an AgentRig request, test fixture, or command committed to
the repository.
