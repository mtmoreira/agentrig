# Live provider validation

Live tests are isolated from AgentRig's default offline lanes. They require an
explicit opt-in and fail when required provider configuration is unavailable.

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

## Ollama

The Ollama runtime test requires an exact application-selected host and an
already-installed model through `AGENTRIG_OLLAMA_LIVE_HOST` and
`AGENTRIG_OLLAMA_LIVE_MODEL`. The test does not discover models, pull a model,
or start a local service. It makes one structured-output call with denied
workspace authority, no tools, a 120-second deadline, and a 64-token output
bound. Thinking is explicitly disabled so that the bounded generation produces
the requested structured content rather than an unobserved reasoning payload.
It verifies the normalized result, exact model identity, usage, and safe event
projection without recording the prompt, thinking, or output in events.

For an unauthenticated local service, run:

```sh
: "${AGENTRIG_OLLAMA_LIVE_HOST:?export the exact Ollama host}"
: "${AGENTRIG_OLLAMA_LIVE_MODEL:?export an already-installed model}"
./buck2 test //tests/live:ollama_runtime -- \
  --env AGENTRIG_RUN_LIVE=1 \
  --env "AGENTRIG_OLLAMA_LIVE_HOST=${AGENTRIG_OLLAMA_LIVE_HOST}" \
  --env "AGENTRIG_OLLAMA_LIVE_MODEL=${AGENTRIG_OLLAMA_LIVE_MODEL}"
```

The live test also supports `AGENTRIG_OLLAMA_LIVE_API_KEY` through its
application-owned authentication source. Do not pass that secret as a Buck
`--env name=value` argument because process arguments are not an approved
credential channel. An authenticated live invocation requires a separately
reviewed secret-safe test environment; the local-service command above does not
use or forward ambient Ollama credentials.
