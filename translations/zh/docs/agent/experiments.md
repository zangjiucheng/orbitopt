# 实验内存

`orbitopt.agent.experiments`保存着保存的[runtime](runtime.md)运行的数据库. 它记录了每个运行的配置,它的策划者和提供者,它的种子和解析器步骤,最佳目标,最大的约束违反,四个验证状态,以及评估,墙钟和信物的运行成本. 数据库可以恢复,查询,缓存和比较运行. 它是一个SQLite索引,旁边有一个内容地址为JSON商店,它使用标准库,IR和运行时间. 它没有进口LLM SDK.

```python
from orbitopt.agent.experiments import ExperimentDB, compare, lookup, query, record_run, resume_plan
from orbitopt.agent.runtime import RuleBasedPlanner, save_run, solve

digest = save_run(solve(problem, planner=RuleBasedPlanner(), seed=0), "runs/a")
with ExperimentDB("experiments") as db:
    record = record_run(db, "runs/a", expected_manifest_digest=digest)
    record.status, record.objective, record.statuses
    query(db, problem_digest=problem.digest(), verified=True).to_text()
    step = record.best_step
    found = lookup(db, problem, solver_id=step.solver_id, options=step.options, seed=step.seed, x0=step.x0,
                   evaluations=step.evaluations)
    found.hit, found.evaluations          # True, 0
```

## 数据库

`ExperimentDB(root)`打开或创建目录 `root`.

| 路径 | 内容 |
|---|---|
| `index.sqlite` | 每个记录一行,它的解题步骤和缓存密钥. |
| `artifacts/sha256/<2 hex>/<64 hex>.json` | Every run file, stored under the SHA-256 of its exact bytes. |

该指数以WAL模式运行,使用30繁忙超时. 每个写法都是一个`BEGIN IMMEDIATE`交易,所以两个同时写法的过程是互相等待的. 每个语句都将它的值作为参数. 排序列来自固定列表。

`ArtifactStore.put` writes a blob to a temporary file in its directory and
renames it into place. A reader never sees half a blob, and two writers of
the same bytes end with one file. `get` hashes what it reads again and
raises `ArtifactError` when the bytes changed. The store accepts UTF-8 JSON
without NaN or infinity, and it parses with `json.loads` only. It never
unpickles, evaluates or imports stored data.

每个进程或线程使用一个 `ExperimentDB`.

## 记录

`ExperimentRecord.to_dict()`对`agent-experiment-1.0.json`进行验证. `from_dict`先检查计划并提升`MalformedRecordError`.

| 外地 | 含义 |
|---|---|
| `run_id` | `sha256:`摘要 运行的`manifest.json`,值`save_run`返回. |
| `problem_digest`, `fingerprint`, `trace_digest`, `result_digest` | 问题在于IR的文摘和硬约束指纹,痕量文摘和`result_digest`的最好结果. |
| `sealed_digest` | The manifest digest of the sealed run (`orbitopt.optimization.run/2`). |
| `problem_id`, `family` | The family is the `family` argument, else `problem.metadata["family"]`, else the problem id. |
| `planner`, `provider`, `model` | From the trace: the planner name, and the chat provider and model behind an LLM planner (`None` for the rule-based planner and for a 1.0 trace). |
| `seed`, `steps` | 跑种子和每个解子步骤:每个计划员决定(`iteration-0003`)和每个最终-验证候补(`verify-1-scipy.slsqp`),其解子,选项,种子,起始点和评价允许. |
| `best`, `objective`, `sense`, `max_violation` | 最佳结果的标签,它宣称意义上的第一个目标,以及它最大的硬约束违反(0,当问题没有. |
| `statuses` | The final report's four statuses, or the best candidate's screen when no final verification ran. |
| `status`, `stop_reason`, `limits`, `usage` | 如是迹中. `usage`持有迭代,评价,信物和墙刻秒. |
| `budget` | 追踪的`outcome.budget`:每个地方的限值都会缩短运行时间(舞台,资源,效果,标签,备抵). 在字段存在之前存储的记录读作空列表 。 |
| `created_at` | When the run was recorded, UTC, ISO 8601 with microseconds. |
| `orbitopt_version` | 运行运行的版本,从跟踪;`0+unknown`为1.0的跟踪,没有表示. 缓存密钥使用. |
| `integrity` | `checked`(默认摄入)或`replayed`. 见下文。 |
| `files` | 运行目录中的每个运行文件按其路径运行,到存储中的其摘要. |
| `parent_id`, `resumed_from` | 链接到记录运行, 或 `None` 。 |

## 记录运行

`record_run(db, run_dir, *, expected_manifest_digest=None, verify="check", tools=None, registry=None, family=None, parent_id=None, resumed_from=None)`在`run_dir`没有信任:

1. 它读`manifest.json`. 它的文摘是运行的ID. 与
   `expected_manifest_digest`,不同文摘被拒绝.
2. 已经记录的运行返回了它的第一个记录,没有记录。
3. 每个文件 清单名称 和每个文件 封存运行,必须
   在记录的文摘上写上安全的名字和大麻 一个名字与 `..`, 一个
   绝对路径, 反弹或 `[A-Za-z0-9_.-]` 之外的任何字符
   在读完之前先把`UnsafePathError`加起来. 检查是
   运行时间的`safe_relative`,运行时间的`replay`使用.
4. `verify="check"` 默认,运行时间
   [`check_run`](https://github.com/zangjiucheng/orbitopt/blob/main/docs/agent/runtime.md#run-directories-and-replay)运行,被钉在
   显示字节按步读 1 。 它检查追踪和它的文摘
   对照节省的结果,检查密封的结果是
   追踪是最好的 并且让警卫重新等级 密封运行与
   候补人员的记录结果。 无解者运行. 任何发现在
   `CheckReport.integrity`拒绝与`RunIntegrityError`的运行. 那个
   唱片的`integrity`是`checked`.
5. With `verify="replay"`, the runtime's `replay` runs instead. It does the
   与候补重运行的相同检查, 然后重新执行记录
   建议。 任何发现和区别都拒绝逃跑 一个伪造者
   重写任何结果 并且每个摘要都在这里失败 因为解析器
   重算结果 。 纪录的`integrity`是`replayed`.
6. 记录的号码来自检查的追踪,商店得到
   精确的字节 。 档案放进店里 然后
   记录在一次交易中输入索引。 两人的碰撞
   留下没有参考的blobs,这是下一个摄入重用。

使用默认值来便宜记录运行. 重放一个您想要从缓存中服务的步骤的运行, 因为只有 `replayed` 记录有缓存项( 见下文) 。

被取消的运行像其他任何的重播,因为被取消的重复播放的微量记录会在那里取消. 1.0追踪没有取消点,所以被取消的1.0运行不重复播放等值:用默认检查记录. 壁时钟结束的跑步被回放拒绝,除非重放准确复制.

沃尔钟字段不是微量文摘的一部分,重播不对它们进行比较. 没有`expected_manifest_digest`,谁可以重写目录也可以重写`usage.wall_s`.

`export_run(db, run_id, dest)`写着一个记录的跑回商店,每个文件通过它的文摘检查. `replay`然后运行在`dest`上.

## 缓存

缓存条目是产生经核实的,完全重播运行的最佳结果的步骤. A `checked` 记录没有条目: 它的键会说一个有这些选项的解析器和这个种子产生这个结果,只有重新执行才能证明这一点. 它的钥匙是`cache_key(problem_digest, solver_id, options, seed, x0, evaluations, *, version, policy)`:

- 选项和起始点是正常的。 密钥已排序, `None` 值
  被丢弃,每个数字都是浮点,空绘图等于 `None` 。
  IR文摘的12显著数字计数.
- `evaluations`是该级津贴.
- `VersionPolicy`读轨道opt版本:`exact`(默认),`minor`
  (`major.minor`),或`any`. 记录被输入到所有三个,所以
  查找选择策略 。

`lookup(db, problem, *, solver_id, options=None, seed=0, x0=None, evaluations, version_policy=VersionPolicy.EXACT, version=None, tools=None)`在键下行走条目,最先更新,返回一个`CacheLookup`. 只有在所有这些物品都持有时,才提供入境服务:

- Its record still parses and is verified and `replayed`.
- 每一个被封存的档案 仍然被卷进它的文摘
- 它通过记录和存储的问题来回答`problem`.
- 警卫对`problem`的 封存跑道进行重级
  `verify_run`, 绑在记录的明细摘要。 判决通过
  需要`constraints_satisfied`,所以目标或剩余值不
  更长的重评价,域漏洞或 stale 缓存失败。

每个其他条目在`CacheLookup.ignored`中命名为`corrupt`(丢失或更改的文物,或不可读的记录)或`stale`(插入,但不再验证这个问题,例如因为现在一个工具回答不同). 它永远不会归还。

击中两次 `CacheLookup.hit`属实,结果`Provenance.cache_key`为问题文摘,由警卫检查`stale_cache`. `CacheLookup.evaluations`总是0。 这跟运行时间的计时器一样 记录的缓存点击是免费的 `recorded_evaluations`是原运行花费的. 警卫的重级再次评价存储点,`check_evaluations`计算. 关于测试的问题,即每个输入检查一次评估,由独立柜台证实每个评估都打电话。

## 续会

`resume_plan(db, run_id)` returns a `ResumePlan` with the problem (read from
the store), the run seed, its policy, the recorded proposals in order, the
limits, what was used, and `remaining`, which is `limits` minus `used` and
never below zero. `resumable` is true when a limit, a cancellation or a
failed planner ended the run. The runtime is not changed. To pick a run up,
serve `plan.proposals` with a `RecordedPlanner` under `plan.limits`,
`plan.policy` and `plan.seed`, which reproduces the completed iterations.
Then hand over to a live planner with the remaining limits.

## 查询、比较、汇总

`query(db, *, problem_digest=None, solver_id=None, status=None, verified=None, family=None, seed=None, since=None, until=None, order_by="created_at", descending=True, limit=50, offset=0)` 返回带有记录的`Page`和`total`,匹配次数. `solver_id`匹配一个运行在任何步骤的解析器的运行. `seed`是奔跑的种子. `since`和`until`将`created_at`包罗万象,天真的日期是UTC. `order_by`是`ORDER_BY`(`created_at`,`objective`,`max_violation`,`evaluations`,`wall_s`,`tokens`)之一. 丢失的值排序为最后,运行的ID断开链接. `limit`为1至1000.

`compare(db, run_ids)` 并列运行:目标,最佳目标,`gap`(比目标意义上的最好多),`relative_gap`(`gap / max(1, |best|)`,卫士的客观尺度),评价,墙上秒,符和四个状态. 最好的是每份问题摘要,在`constraints_satisfied`通过的运行中。 因此,目标较好的不可行的运行会出现负差,永远不会成为最佳。

`aggregate(db)`给每个家庭的运行次数,`verified`和`unverified_best`运行次数,其余部分,以及中位评价,壁秒和符号.

`Page`,`Comparison`,`Aggregate`和`ResumePlan`各有`to_dict()`(JSON-准备)和`to_text()`(平表).

## 移徙

`schema_version`持有一个整数. `MIGRATIONS[i]`将索引从`i`版本到`i + 1`版本. 0版本开始一个新的文件,所以创建表格是迁移0. 打开一个数据库会运行一个交易中缺失的步数,所以一个步骤会提高版本的位置。 `ExperimentDB(root, migrations=...)`是后一步的钩子. 一个数据库比代码所知道的步数更新,或者一个不是实验数据库的SQLite文件,会提高`SchemaVersionError`.

| 步骤 | 版本 | 它会做什么 |
|---|---|---|
| 0 | 1 | 创建表格。 |
| 1 | 2 | 重写 `integrity` `iterations_replayed`,取消的运行在重播可以取消前获得,重写到 `checked`. 这样的运行通过每个字节和重级的检查并重播其迭代,但结果却并非如此. |

## 错误

每个错误都是`ExperimentError`,除了`UnsafePathError`,运行时间为两个软件包定义和提升. 分会课.

| 错误 | 何时 |
|---|---|
| `RunIntegrityError` | 一个运行目录被更改,伪造,不重播,或者不是运行. `problems`列出调查结果. |
| `UnsafePathError` | 文件名会留下目录 。 |
| `InvalidIdError` | id不是`sha256:`和64小写十六进制数字. |
| `UnknownRunError` | 没有记录有ID,也没有链接名称没有记录运行. |
| `ArtifactError` | 一个存储的blob 丢失, 更改, 或者不是 JSON 。 |
| `MalformedRecordError` | 记录与它的计划不符。 |
| `SchemaVersionError` | 数据库比代码更新,或者不是一个实验数据库. |

## 计量

The rule-based planner ran each instance with seeds 0 to 4, and each run
was recorded, then looked up with its best step (a hit) and with another
seed (a miss). The table gives medians over the five seeds. It was measured
on one machine (Apple M1 Pro, macOS, Python 3.12, the `orbitopt-py312`
environment). Every run was verified, and every hit was served.

| 实例 | 运行评价 | `solve` | 仅`replay` | 点击 |  ̅ | 每一次评价 |
|---|---|---|---|---|---|---|
| cqp-worked-d 2 | 807 | 53 ms | 107 ms | 5.2 ms | 0.14 ms | 1 |
| lp-wyndor-d2 | 419 | 37 ms | 73 ms | 5.7 ms | 0.17 ms | 1 |
| 拉斯特里金-d 5 | 7848 | 343 ms | 444 ms | 7.4 ms | 0.18 ms | 1 |

`record_run` was timed again in each mode: five seeds, each run ingested
into a fresh database three times, 15 timings per cell, on the same
machine. The table gives the median with the range.

| 实例 | 运行评价 | `solve` | `verify="check"` | `verify="replay"` | 检查摄入量的评价 |
|---|---|---|---|---|---|
| cqp-worked-d 2 | 807 | 52 ms | 8.6 ms(7.8到11.6). | 111 ms(95到115). | 3 |
| lp-wyndor-d2 | 419 | 35 ms | 10.0 ms(9.1到10.4). | 75 ms(71到80). | 3 |
| 拉斯特里金-d 5 | 7848 | 305 ms | 13.9 ms(13.0到16.5). | 401 ms(382到455). | 3 |

限制这些数字的是什么:

- 重复播放的摄入器再次运行每个解答,然后是最终解答
  验证的副手第二次,当它重新等级封印时
  运行。 它比跑步本身花费更多 并且随它而成长 它花了112,
  默认更改前的77和397 ms,分布范围相同.
- 检查的摄入量并不取决于运行的评价. 它评价
  问题三次: 封点和两个候补
  点。 Rastrigin 摄入量的简介 将它的一半时间放在
  `jsonschema`(追踪,问题两次,结果和记录),
  第八张13文件用`fsync`各写到店里,
  警卫队第十名 Rastrigin的踪迹有五个迭代 还有其他的
  二,即各行之别.
- 命中并不取决于跑步的大小. 它写了四个封存的文件
  到一个临时目录 并运行一次警卫。 大部分资料都显示
  that time in `jsonschema`, which checks the stored problem twice and the
  记录一次。 警卫评估存储点一次.
- 错失是问题摘要(约0.09 ms),
  在安装的版本缓存之前, 从软件包读取
  元数据将 1.7 ms 的 1.9 ms 成本误差 。
- 每一份记录的查询费大约是0.25 ms。
  对从索引中读取的每个记录进行计划检查。

## 测试

`tests/test_e14_07_experiment_memory.py` drives the package through its
public API on real runs: the rule-based, recorded and mock-LLM planners on
`cqp-worked-d2`, `lp-wyndor-d2` and `rastrigin-d5` from the E14-10
catalogue, and a tool-evaluated problem whose tool counts every evaluation.
It covers the round trip, idempotent ingest, six tampered files, a forger
who rewrites the manifest, a manifest that names a file outside the run,
cache hits, misses, version policies, corrupt and stale entries and the
zero-evaluation accounting, resume plans after a cancellation and after the
evaluation and token limits, gaps checked against values worked out from
the runs, the migration hook on a version 0 database, processes that create
and write one database at once, hostile ids, filter values and file names,
and the record schema.

```bash
PYTHONPATH=src python -m pytest -q tests/test_e14_07_experiment_memory.py
```
