# Live provider validation

Live tests are isolated from AgentRig's default offline lanes. They require an
explicit opt-in and fail when required provider configuration is unavailable.

The Codex runtime test requires an explicit application-selected authentication
mode through `AGENTRIG_CODEX_LIVE_AUTH_MODE`. In `api_key` mode, it resolves
`AGENTRIG_CODEX_LIVE_API_KEY` only when the client is created and maps it into
the provider process. In `ambient` mode, the application deliberately leaves
authentication to the existing Codex environment. The test starts an ephemeral
thread in a read-only sandbox with network access disabled and requests a small
strict JSON result without tools. It verifies the normalized result, provider
metadata, usage, and safe event projection without recording the prompt or
output in events.

Run only this test with:

```sh
: "${AGENTRIG_CODEX_LIVE_API_KEY:?export the application-owned API key}"
AGENTRIG_RUN_LIVE=1 \
AGENTRIG_CODEX_LIVE_AUTH_MODE=api_key \
uv run python -m unittest \
  tests.live.test_codex_runtime_live -v
```

To use an existing Codex login explicitly instead:

```sh
AGENTRIG_RUN_LIVE=1 \
AGENTRIG_CODEX_LIVE_AUTH_MODE=ambient \
uv run python -m unittest tests.live.test_codex_runtime_live -v
```

The default model is `gpt-5.6-terra`. Override it explicitly when validating a
different supported Codex model:

```sh
: "${AGENTRIG_CODEX_LIVE_API_KEY:?export the application-owned API key}"
AGENTRIG_RUN_LIVE=1 \
AGENTRIG_CODEX_LIVE_AUTH_MODE=api_key \
AGENTRIG_CODEX_LIVE_MODEL=gpt-5.6-terra \
uv run python -m unittest tests.live.test_codex_runtime_live -v
```

Set `AGENTRIG_CODEX_LIVE_API_KEY` from an application-owned secret source. The
test maps it to the Codex provider process without putting it in an AgentRig
request, fixture, event, failure, or representation. See the official
[Codex authentication guide](https://developers.openai.com/codex/auth), and
never commit the value or include it directly in a shell command. Ambient mode
does not read, copy, print, or persist the existing authentication material; it
only selects the SDK factory's already-supported ambient behavior.

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
AGENTRIG_RUN_LIVE=1 \
uv run python -m unittest tests.live.test_ollama_runtime_live -v
```

The live test also supports `AGENTRIG_OLLAMA_LIVE_API_KEY` through its
application-owned authentication source. Do not pass that secret as a Buck
`--env name=value` argument because process arguments are not an approved
credential channel. An authenticated live invocation requires a separately
reviewed secret-safe test environment; the local-service command above does not
use or forward ambient Ollama credentials.
