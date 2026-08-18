# Live provider validation

Live tests are isolated from AgentRig's default offline lanes. They require an
explicit opt-in and fail when the selected provider is not authenticated.

The Codex runtime test requires an application-scoped API key through
`AGENTRIG_CODEX_LIVE_API_KEY`, maps it into the provider process only when the
client is created, starts an ephemeral thread in a read-only sandbox with
network access disabled, and requests a small strict JSON result without tools.
It verifies the normalized result, provider metadata, usage, and safe event
projection without recording the prompt or output in events.

Run only this test with:

```sh
./buck2 test //tests/live:codex_runtime -- \
  --env AGENTRIG_RUN_LIVE=1 \
  --env AGENTRIG_CODEX_LIVE_API_KEY
```

The default model is `gpt-5.6-terra`. Override it explicitly when validating a
different supported Codex model:

```sh
./buck2 test //tests/live:codex_runtime -- \
  --env AGENTRIG_RUN_LIVE=1 \
  --env AGENTRIG_CODEX_LIVE_API_KEY \
  --env AGENTRIG_CODEX_LIVE_MODEL=gpt-5.6-terra
```

Set `AGENTRIG_CODEX_LIVE_API_KEY` from an application-owned secret source. The
test maps it to the Codex provider process without putting it in an AgentRig
request, fixture, event, failure, or representation. See the official
[Codex authentication guide](https://developers.openai.com/codex/auth), and
never commit the value or include it directly in a shell command.
