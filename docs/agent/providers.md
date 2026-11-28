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
                                api_key_env="LLM_API_KEY", timeout_s=60)
```

The provider sends `POST {base_url}/chat/completions` through `urllib`. It
reads the key from the named environment variable at call time and never
stores it; the key does not appear in `repr`, and any response text quoted in
an error has it redacted. HTTP 429, HTTP 5xx, timeouts, connection failures
and a reply with `finish_reason` `insufficient_system_resource` are retried
up to `max_retries` times (default 3). Each wait is the capped exponential
backoff or the server's `Retry-After`, whichever is longer, but never more
than 60 seconds. A failed attempt reports no usage, so it costs no tokens;
`ProviderResponse.attempts` says how many attempts a call took, and a
cassette records it. HTTP 401, 402 and 403 raise `ProviderAccountError`
(no later call from this account can succeed); other HTTP errors raise
`ProviderError`. Neither is retried.

`ProviderResponse.model` is the model the server says answered, which can
differ from the one requested: an alias answers under its real name.
`Usage` adds `cache_hit_tokens`, `cache_miss_tokens` and `reasoning_tokens`
when the server reports them, at the top level of `usage` or under
`prompt_tokens_details` and `completion_tokens_details`. Function names on the
wire must match `^[a-zA-Z0-9_-]{1,64}$`, which rejects the dots in tool ids
such as `math.erf.v1`. `wire_name` sends each `.` as `-` (tool ids never
contain `-`), and the provider maps a returned name back to the tool offered.
Two tools that would share a wire name, or a name longer than 64 characters,
raise `ProviderError` before anything is sent.

`structured_output` says how a JSON reply is asked for: `json_schema` (the
default, `response_format` `json_schema`), `json_object` (the server's JSON
mode, with the schema in a system instruction) or `instruction` (the
instruction alone). The reply is validated locally in every mode, with the
same repair round. `send_seed=False` keeps the request's seed off the wire
for a server that has no seed parameter; the seed stays in the canonical
request and its digest. `extra_body` adds server-specific fields and may not
replace one the client sets.

## DeepSeek

```python
from orbitopt.agent.providers import DeepSeekProvider

provider = DeepSeekProvider("deepseek-flash")   # https://api.deepseek.com, key from LLM_API_KEY
```

or `LLM_KIND=deepseek LLM_MODEL=deepseek-flash` with the key in `LLM_API_KEY`.
The preset follows DeepSeek's API documentation (read 2026-11-28):

| DeepSeek | the preset |
|---|---|
| `response_format` is `text` or `json_object`; no `json_schema` | `structured_output="json_object"`; the schema goes in the system instruction and the reply is validated locally |
| JSON mode can return blank content | blank content fails validation and gets the repair round, whose tokens count |
| thinking is on by default, and `temperature` has no effect while it is | sends `thinking: {"type": "disabled"}`, so the caller's temperature applies |
| thinking with tools needs every earlier `reasoning_content` sent back | not supported here; thinking stays off |
| no documented `seed` | `send_seed=False` |
| `usage.prompt_cache_hit_tokens`, `prompt_cache_miss_tokens` | read into `Usage` |
| 429 rate or concurrency limit; 500, 503, 504 | retried with backoff and `Retry-After` |
| 401 bad key, 402 no balance | `ProviderAccountError`, not retried |
| keep-alive blank lines before a non-streaming body | skipped by the JSON parser |
| the alias `deepseek-chat` answers as `deepseek-flash` | `ProviderResponse.model` records `deepseek-flash` |

Tool calls use the OpenAI shape. DeepSeek's strict tool mode needs the
`/beta` base URL and a restricted schema dialect; the preset does not use it,
and tool arguments are validated locally as for every provider. Outputs can
still vary between identical calls at temperature 0, which is why a cassette,
not a re-run, is the reproduction artifact. The offline tests run against a
local server that answers in DeepSeek's shapes (`tests/fake_deepseek.py`).

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
make_provider({"kind": "deepseek", "model": "deepseek-flash"})
make_provider({"kind": "openai_compat", "base_url": "...", "model": "...",
               "record": "runs/plan-001.jsonl"})
make_provider({"kind": "replay", "cassette": "runs/plan-001.jsonl"})
make_provider({"kind": "mock", "script": ['{"dv_mps": 12.5}']})
```

## Configuration from the environment

The E14 campaign, the CLI's `--provider env` and Mission Studio all read one
set of variables, through `orbitopt.llm_env`:

| Variable | Meaning |
|---|---|
| `LLM_KIND` | A `make_provider` kind: `openai_compat` (default), `deepseek` or `local`. |
| `LLM_MODEL` | The model id. Without it no provider is configured. |
| `LLM_BASE_URL` | The endpoint. Required for `openai_compat`; `deepseek` defaults to `https://api.deepseek.com`, `local` to Ollama. |
| `LLM_API_KEY` | The key, for keyed kinds. Read at call time, never stored or recorded. |
| `LLM_TIMEOUT_S` | Optional request timeout in seconds. |

`provider_spec()` returns the `make_provider` dict, which names the key
variable but never holds the key, and `provider_from_env()` builds the
provider (None when a keyed kind has no key). Sampling settings are not
environment variables: each caller fixes them, and the campaign freezes them.

`load_env_file(path)` reads `KEY=VALUE` lines into the environment. It skips
blank lines and `#` comments and strips one pair of quotes. It expands and
executes nothing: `export`, `$VAR` and backticks stay literal or are refused.
A variable that is already set wins. An error names the line number, never
the line, because the file may hold a key. `orbitopt --env-file FILE ...`
and the campaign's `--env-file FILE` call it. In a shell,
`set -a; . ./.env; set +a` does the same for a file of plain assignments.

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
