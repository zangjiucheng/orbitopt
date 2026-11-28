# LLM 供应商

`orbitopt.agent.providers`是E14代理运行时间和编译器用来与语言模型交谈的一个接口. 它只进口标准库和`jsonschema`. 与模拟和重播提供者,整个测试套件运行时没有API键,网络连接或模型.

## 接口

每个提供者执行 `LLMProvider`:

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
- `generate_structured`用`jsonschema`验证答复. 如果回复
  无效, 提供者将验证错误发回 模型 和
  再来一次 如果答复无效,则提出
  `StructuredOutputError` (with `.raw_text`, `.errors` and `.usage`, the
  每一份答复都注明失败费用)。 它永远不会返回失败的数据
  审定。
- `call_tools` 检查每个工具调用 与参数设置
  工具名称。 呼叫未提供或无效的工具
  参数,将 `ToolCallError`,子类为 `StructuredOutputError`,
  上面还有`.usage`
- 每一份答复都是`ProviderResponse`带`Usage`(即时,完成).
  和总令牌,在服务器报告一个时加上`cost_usd`).
- 失败者会被打字。 `ProviderError`表示重试无益.
  `ProviderUnavailableError`表示提供商无法或仍然无法接触
  失败 。 `BudgetExceededError`表示使用限制
  被击中。

To meter usage, pass `meter=UsageMeter(max_tokens=..., max_calls=...)` to
any provider. The meter refuses a call that would go past `max_calls` before
the call is made. It raises after the call whose tokens went past
`max_tokens`; the error reports the overrun and keeps that call's response.
Repair retries count as calls.

## OpenAI 兼容端点

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

`ProviderResponse.model`是服务器所回答的模型,这与所请求的有区别:一个在真名下的别名答案. `Usage`在服务器上报时会加上`cache_hit_tokens`,`cache_miss_tokens`和`reasoning_tokens`,最高水平为`usage`或`prompt_tokens_details`和`completion_tokens_details`. 电线上的函数名必须匹配 `^[a-zA-Z0-9_-]{1,64}$`,它拒绝`math.erf.v1`等工具代号中的点. `wire_name`将每个`.`作为`-`(工具代号从不包含`-`),提供者将一个返回的名称映射回所提供的工具. 有两个工具可以共享一个线名,或者一个比64字符长的名字,在发送任何东西之前提高`ProviderError`.

`structured_output` says how a JSON reply is asked for: `json_schema` (the
default, `response_format` `json_schema`), `json_object` (the server's JSON
mode, with the schema in a system instruction) or `instruction` (the
instruction alone). The reply is validated locally in every mode, with the
same repair round. `send_seed=False` keeps the request's seed off the wire
for a server that has no seed parameter; the seed stays in the canonical
request and its digest. `extra_body` adds server-specific fields and may not
replace one the client sets.

## 深层搜索

```python
from orbitopt.agent.providers import DeepSeekProvider

provider = DeepSeekProvider("deepseek-flash")   # https://api.deepseek.com, key from LLM_API_KEY
```

or `LLM_KIND=deepseek LLM_MODEL=deepseek-flash` with the key in `LLM_API_KEY`.
The preset follows DeepSeek's API documentation (read 2026-11-28):

| 深层搜索 | 预设 |
|---|---|
| `response_format`为`text`或`json_object`;无`json_schema`. | `structured_output="json_object"`; 计划进入系统指令中, 答复在当地验证 |
| JSON模式可以返回空白内容 | 空白内容失败验证并获得修复回合, 其符号计数 |
| `temperature`没有效果 | 发送`thinking: {"type": "disabled"}`,所以呼叫者温度适用 |
| 用工具思考 需要所有早期的 `reasoning_content` 寄回 | 此处不支持; 思维停止 |
| no documented `seed` | `send_seed=False` |
| `usage.prompt_cache_hit_tokens`, `prompt_cache_miss_tokens` | 读入 `Usage` |
| 429率或货币限额;500,503,504 | 复试后退和`Retry-After` |
| 401坏键,402无余额 | `ProviderAccountError`,没有重审 |
| 在非流体前保持空白线条 | 被 JSON 解析器跳过 |
| 化名 `deepseek-chat` 答案为 `deepseek-flash` | `ProviderResponse.model`记录 `deepseek-flash` |

工具调用使用OpenAI形状. DeepSeek严格的工具模式需要`/beta`基座URL和限制的schema方言;预设不使用它,工具参数像每个提供者一样在当地验证. 在温度0时, 输出仍然可以因相同的呼叫而变化, 这就是为什么磁带,而不是重播, 是复制的文物。 离线测试针对一个以DeepSeek形状(`tests/fake_deepseek.py`)回答的本地服务器进行.

## 本地服务器

```python
from orbitopt.agent.providers import LocalProvider

provider = LocalProvider("qwen2.5:7b")                       # Ollama, http://localhost:11434/v1
provider = LocalProvider("my-model", "http://127.0.0.1:8080/v1")  # llama.cpp or vLLM
```

`LocalProvider`是本地默认的同一客户端:没有API密钥,一个300 - 第二次超时,通过指令请求JSON.

## 配置

`make_provider`为 CLI 和配置文件从 dict 建立提供者 :

```python
make_provider({"kind": "local", "model": "qwen2.5:7b"})
make_provider({"kind": "deepseek", "model": "deepseek-flash"})
make_provider({"kind": "openai_compat", "base_url": "...", "model": "...",
               "record": "runs/plan-001.jsonl"})
make_provider({"kind": "replay", "cassette": "runs/plan-001.jsonl"})
make_provider({"kind": "mock", "script": ['{"dv_mps": 12.5}']})
```

## 来自环境的配置

E14运动,CLI的`--provider env`和Mission Studio都读了一套变量,通过`orbitopt.llm_env`:

| 变量 | 含义 |
|---|---|
| `LLM_KIND` | `make_provider`类型:`openai_compat`(默认),`deepseek`或`local`. |
| `LLM_MODEL` | 模型 id. 没有它, 无法配置提供者 。 |
| `LLM_BASE_URL` | 端点. `openai_compat`; `deepseek`默认为`https://api.deepseek.com`, `local`默认为Ollama. |
| `LLM_API_KEY` | 钥匙,用于键类。 在呼叫时阅读,从未存储或记录. |
| `LLM_TIMEOUT_S` | 可选请求超时数秒 。 |

`provider_spec()`返回 `make_provider` dict,它命名了密钥变量但从未持有密钥,`provider_from_env()`建立提供者(当一个密钥类型没有密钥时无密钥). 采样设置不是环境变量:每个呼叫者都会修正,运动会冻结.

`load_env_file(path)`读`KEY=VALUE`线进入环境. 它跳过空白行和`#`评论,并剥去一对引文. `export`, `$VAR`和背影保持文字或被拒绝。 已经设置的变量获胜 。 一个错误给行号命名,从不给行命名,因为文件可能持有密钥. `orbitopt --env-file FILE ...`与竞选的`--env-file FILE`称之. 在外壳中,`set -a; . ./.env; set +a`对普通任务的文件也做同样的事情.

## 记录和重播

在 `RecordingProvider(inner, "run.jsonl")` 中包裹一个提供者, 将 模型 调用到磁带中 。 `ReplayProvider("run.jsonl")`服务于有记录的回电,而不拨打模型,因此代理运行可以在稍后被重新执行和审计. 它经过与记录运行相同的验证,修复和工具检查,记录失败(退出,错误的工具参数)再次被提出. 两个包装器都报告录制的提供商的`name`和`model`(重播从磁带头读取),因此从提供商的磁带名和回答它的模型重播,如[runtime trace](https://github.com/zangjiucheng/orbitopt/blob/main/docs/agent/runtime.md#the-trace)记录的那样.

重复播放每个请求, 用一个 SHA-256 文摘, 来描述它的语句 JSON( 各种密钥) 。 同文申请均按记录顺序提出。 未录制的要求引起`ReplayMismatchError`. `assert_exhausted()`,如果从未要求录制任何电话,也引起注意. `ReplayMismatchError`故意不是`ProviderError`,所以处理提供者故障的代码不能吞噬差异.

The cassette is JSON Lines, format `orbitopt.llm-cassette` version 1:

```text
{"format":"orbitopt.llm-cassette","model":"gpt-4o-mini","provider":"openai_compat","version":1}
{"digest":"sha256:...","request":{...},"response":{...}}
{"digest":"sha256:...","error":{"message":"...","type":"ProviderUnavailableError"},"request":{...}}
```

一个请求持有信件, `max_tokens`, `temperature`, `seed`, 响应计划及工具. 它从未持有端点,API密钥或其可变名称,所以磁带里没有证书. 信件文本是逐字录制的,所以不要让机密出现提示.

## 模拟提供者

`MockProvider(script)`从列表或`ChatRequest`的函数中回复. 一个项目可以是字符串, `MockReply(text, tool_calls)`, 完整的 `ProviderResponse`, 或者一个例外例, 被提出来. 失败被注入为项目:`ProviderUnavailableError("timeout")`,这个答复无效JSON,或者`MockReply`调用一个未知的工具. 托肯语被计算为白空分隔的单词. 提示符是所有信件内容的词 。 完成符是回复文本的单词,加上每个工具调用名称的单词和`json.dumps(arguments, sort_keys=True)`. `mock.requests`列出收到的每个请求模拟.
