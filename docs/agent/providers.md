# LLM providers

`orbitopt.agent.providers` is the one interface the E14 agent runtime and
compiler use to talk to a language model. It imports only the standard
library and `jsonschema`. With the mock and replay providers, the whole test
suite runs without an API key, a network connection or a model.

## Interface

Every provider implements `LLMProvider`:

```python
from orbitopt.agent.providers import Message, ToolSpec

messages = [Message("system", "You plan orbital transfers."),
            Message("user", "Propose a burn as JSON.")]

reply = provider.generate(messages, max_tokens=256)
reply = provider.generate_structured(messages, schema, max_tokens=256)
reply = provider.call_tools(messages, [ToolSpec("propagate", "...", params_schema)],
                            max_tokens=256)
```

- `schema` and `params_schema` are plain JSON Schema dicts.
- `generate_structured` validates the reply with `jsonschema`. If the reply
  is invalid, the provider sends the validation errors back to the model and
  tries once more. If that reply is also invalid, it raises
  `StructuredOutputError` (with `.raw_text`, `.errors` and `.usage`, the
  tokens of every reply the failure cost). It never returns data that failed
  validation.
- `call_tools` checks every tool call against the parameters schema of the
  tool it names. A call to a tool that was not offered, or with invalid
  arguments, raises `ToolCallError`, a subclass of `StructuredOutputError`,
  which carries that reply's `.usage` too.
- Every reply is a `ProviderResponse` carrying a `Usage` (prompt, completion
  and total tokens, plus `cost_usd` when the server reports one).
- Failures are typed. `ProviderError` means retrying will not help.
  `ProviderUnavailableError` means the provider was unreachable or still
  failing after bounded retries. `BudgetExceededError` means a usage limit
  was hit.

To meter usage, pass `meter=UsageMeter(max_tokens=..., max_calls=...)` to
any provider. The meter refuses a call that would go past `max_calls` before
the call is made. It raises after the call whose tokens went past
`max_tokens`; the error reports the overrun and keeps that call's response.
Repair retries count as calls.

## OpenAI-compatible endpoints

```python
from orbitopt.agent.providers import OpenAICompatProvider

provider = OpenAICompatProvider("https://api.openai.com/v1", "gpt-4o-mini",
                                api_key_env="OPENAI_API_KEY", timeout_s=60)
```

The provider sends `POST {base_url}/chat/completions` through `urllib`. It
reads the key from the named environment variable at call time and never
stores it; the key does not appear in `repr`, and any response text quoted in
an error has it redacted. HTTP 429, HTTP 5xx, timeouts and connection
failures are retried up to `max_retries` times (default 3) with capped
exponential backoff. Other HTTP errors fail at once. Function names on the
wire must match `^[a-zA-Z0-9_-]{1,64}$`, which rejects the dots in tool ids
such as `math.erf.v1`. `wire_name` sends each `.` as `-` (tool ids never
contain `-`), and the provider maps a returned name back to the tool offered.
Two tools that would share a wire name, or a name longer than 64 characters,
raise `ProviderError` before anything is sent. Pass
`supports_json_schema=False` for a server that rejects `response_format`
`json_schema`; the provider then asks for JSON in a system instruction and
validates the reply the same way.

## Local servers

```python
from orbitopt.agent.providers import LocalProvider

provider = LocalProvider("qwen2.5:7b")                       # Ollama, http://localhost:11434/v1
provider = LocalProvider("my-model", "http://127.0.0.1:8080/v1")  # llama.cpp or vLLM
```

`LocalProvider` is the same client with local defaults: no API key, a
300-second timeout, and JSON requested through an instruction.

## Configuration

`make_provider` builds a provider from a dict, for the CLI and config files:

```python
make_provider({"kind": "local", "model": "qwen2.5:7b"})
make_provider({"kind": "openai_compat", "base_url": "...", "model": "...",
               "record": "runs/plan-001.jsonl"})
make_provider({"kind": "replay", "cassette": "runs/plan-001.jsonl"})
make_provider({"kind": "mock", "script": ['{"dv_mps": 12.5}']})
```

## Record and replay

Wrap a provider in `RecordingProvider(inner, "run.jsonl")` to append every
model call to a cassette. `ReplayProvider("run.jsonl")` serves the recorded
calls back without calling a model, so an agent run can be re-executed and
audited later. It goes through the same validation, repair and tool checks
as the recorded run, and recorded failures (outages, malformed tool
arguments) are raised again. Both wrappers report the recorded provider's
`name` and `model` (the replay reads them from the cassette header), so a
run replayed from a cassette names the provider and model that answered it,
as the [runtime trace](runtime.md#the-trace) records them.

Replay matches each request by a SHA-256 digest of its canonical JSON
(sorted keys). Identical requests are served in recorded order. A request
that was not recorded raises `ReplayMismatchError`. `assert_exhausted()`
raises it too if any recorded call was never requested. `ReplayMismatchError`
is deliberately not a `ProviderError`, so code that handles provider
failures cannot swallow a divergence.

The cassette is JSON Lines, format `orbitopt.llm-cassette` version 1:

```text
{"format":"orbitopt.llm-cassette","model":"gpt-4o-mini","provider":"openai_compat","version":1}
{"digest":"sha256:...","request":{...},"response":{...}}
{"digest":"sha256:...","error":{"message":"...","type":"ProviderUnavailableError"},"request":{...}}
```

A request holds the messages, `max_tokens`, `temperature`, `seed`, the
response schema and the tools. It never holds the endpoint, the API key or
its variable name, so a cassette contains no credentials. Message text is
recorded verbatim, so keep secrets out of prompts.

## Mock provider

`MockProvider(script)` replies from a list, or from a function of the
`ChatRequest`. An item can be a string, a `MockReply(text, tool_calls)`, a
full `ProviderResponse`, or an exception instance, which is raised. Failures
are injected as items: `ProviderUnavailableError("timeout")`, a reply that is
not valid JSON, or a `MockReply` calling an unknown tool. Tokens are counted
as whitespace-separated words. Prompt tokens are the words of all message
contents. Completion tokens are the words of the reply text, plus the words
of each tool call's name and `json.dumps(arguments, sort_keys=True)`.
`mock.requests` lists every request the mock received.
