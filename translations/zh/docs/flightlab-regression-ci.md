# Flight Lab可复制性和回归CI(FL-601).

每一次Flight Lab运行都归因于精确的工具链,OpenTsiolkovsky驱动的Gate B输油管有一个数字回归基线. 本页为业务摘要;许可证在[`THIRD_PARTY.md`](https://github.com/zangjiucheng/orbitopt/blob/main/THIRD_PARTY.md)。

## 固定的工具链

| 工具 | 键 | 何处 |
|---|---|---|
| Python/康达堆栈 | Python 3.10, tudatpy 1.0, NumPy 1.26.4, … | `envs/{cpu,gpu}.yml` + `envs/*.conda-lock.yml` (CI job `lockfile-drift` 故障光谱/锁键) |
| 天体(框架桥) | `astropy=6.1.7`,准确 | `envs/{cpu,gpu}.yml` |
| IERS 地球导向数据 | 锁定文件的`astropy-iers-data`(E8-04快照绑定),任何引用回归中的**无下载** | `orbitopt.reference_eop`通过`scripts/{fl601,e204}_reference.py`(`_pin_earth_orientation`). |
| gfold( FL-303 登陆解析器) | `gfold==0.3.2`, exact (wheels need Python >= 3.11) | `envs/flightlab-landing.txt`; CI job `flightlab-landing` runs it on Python 3.12 via `envs/flightlab-landing-ci.txt` |
| jsbsim(E4 6 - DOF 尖锐) | `jsbsim==1.3.3`,散装 | `envs/sixdof-requirements.txt` |
| OpenTsiolkovsky( 外部二进制) | v 1.0.0,每个平台SHA-256 | `scripts/fetch_opentsiolkovsky.py`( 拒绝其他字节; `--check` 验证安装的内容) |

`orbitopt.flightlab.contract.write_run`所写的每份飞行结果表都载有`manifest.toolchain`下的全部内容(解释器,轨道opt,每个工具分发器的安装版本或`"absent"`,OpenTsiolkovsky的绑定版本和实际在磁盘上找到的SHA-256). `tests/test_flightlab_contract.py`称记录完成.

### 为什么回归永远不下载 IERS 数据

框架桥将OpenTsiolkovsky的地球固定状态与天体旋转为GCRS,这需要地球导向数据(UT1-UTC,极地运动). 星座默认下载了最新的IERS-A表,该表每天随着预测成为测量结果而变化:同一航班的GCRS位置在天际间移动了十分之十,远远超出了微米耐受度(回归的首个CI运行在7,670样本上失败). `fl601_reference.py run` / `update`因此只使用与绑定的`astropy-iers-data`捆绑的桌子. E8-04将其硬化为`orbitopt.reference_eop`的共享机制: 参考目录载有`earth_orientation.json`(模版`pinned-bundled`),记录了录音堆的准确版本,它使用的默认离线表(IERS-B),以及参考文献历元时有效的UT1-UTC/极动; 每一次运行都会验证安装的堆栈复制全部,并用`ToolchainMismatch`拒绝,而不是默默地飞行不同的地球方向(Astropy 6 vs 8不同意离线的~28 m). E2-04保持其四行快照的`override`模式——一种机制,每模式政策. 锁文件中的弹出`astropy-iers-data`使快照失效; 与`update`(被审计)和`[fl601-reference-update]`标签重录. 在锁定的环境(CI)中录制,而非本地环境. 容忍政策、乐队和变革历史在[reference-tolerance-audit.md](reference-tolerance-audit.md)直播。

## CI布局 (. GitHub / 工作流程/ci.yml).

只使用文档的更改跳过CI;用作测试输入的基准JSON仍然被封门. 工作分工为:


* **`test`( linux -64, 锁定内嵌)** – Flight Lab 单位测试运行于该机上.
  仅录制固定装置: OT适配测试 重播录制 v 1.0.0
  通过 CLI 输出, 因此此处不需要 OT 二进制( 既不存在) 。
  这工作还经营着`tests/test_flightlab_reference.py`,
  篡改已承诺的回归引用。
* **`flightlab-gateb`(慢, Linux -64, 锁定的内嵌)**-拿下钉子
  OpenTsiolkovsky二进制(SHA-256经下载核实,版本制作),
  飞行参考门B任务
  (`leo_two_stage`目标+`synthetic_two_stage`车,上升段 -只,
  `refly=False`,`run_recovery=False`)与真实的二进制比较
  每个 轨迹 样本,事件和对照
  `tests/data/flightlab/reference/` within the tolerances declared in
  `tests/data/flightlab/reference/tolerances.json`. 两者也一样
  G5卫士:`verify`(参考文件与其出处摘要相符)和
  `tag-check`(任何触摸`tests/data/flightlab/reference/`的需要)
  `[fl601-reference-update]`在push/PR范围的承诺信息中.
* **`flightlab-landing`** — Python 3.12加gfold 0.3.2:恢复试验,
  E2-04和E5-05重播/校对警卫.
* **`cross-env`** — locked TudatPy Python 3.10 plus a Python 3.12 landing
  环境; 验证密封交换机和E5-05加入任务.
* **`test-macos`** — compute tests on the locked arm64/x86_64 environments.
  Flight Lab导航测试使用抓取文件夹/设置而不是
  退出运行或用户偏好 。 没有B门航班运行在macOS CI.

## 参考数据集

```
tests/data/flightlab/reference/
  gateb-leo-two-stage/     the recorded Gate B run dir (manifest, trajectories,
                           events, verification + raw opentsiolkovsky/ outputs)
  tolerances.json          the declared per-quantity tolerances
  PROVENANCE.json          sha256 of every file above, written by the script
```

更新( 例如, 重新固定 OpenTsiolkovsky 时, 或当预定的物理改变土地 ) :

```bash
python scripts/fetch_opentsiolkovsky.py --force   # new pin already in the script
python scripts/fl601_reference.py update          # refly the run, rewrite PROVENANCE.json
git add tests/data/flightlab/reference
git commit -m "test(flightlab): re-record Gate B reference for OT vX.Y.Z [fl601-reference-update]"
```

CI拒绝了其他任何内容:手版的参考文件失败了`verify`(在快速任务中),未标记的参考修改失败了`tag-check`,一个意外的物理变化失败了数字`compare`.

## 手动 macOS- arm 64 运行

CI的门B工作是Linux的(OT macOS二进制和获取方式相同,但我们保留了一个犬形的CI平台). 在 OT- 针或物理敏感变化后, 在 Apple Silicon Mac 上这样做 :

```bash
# 1. locked env (osx-arm64 section) -- see envs/README.md
micromamba create -n orbitopt -f envs/cpu.conda-lock.yml
micromamba activate orbitopt
pip install -e . --no-deps

# 2. the pinned OT binary (macos-arm64 asset, SHA-256 verified)
python scripts/fetch_opentsiolkovsky.py          # installs + --check

# 3. the regression, exactly as CI runs it
python scripts/fl601_reference.py verify
python scripts/fl601_reference.py run --out /tmp/gateb
python scripts/fl601_reference.py compare /tmp/gateb
```

期望`compare`通过;在真正的平台三角洲上,只有在容忍度太紧的情况下,它才能在宣布的容忍度范围内分裂出来 -- -- 即`tolerances.json`更新,它通过与参考物本身相同的更新标注+标记-承诺路线。

RAAN敏感度和固定历元优化测试与同一地球导向表下的未优化插入相比. 他们同时在IERS-A和IERS-B下线; 一个绝对的RAAN 与另一个表记录 并不是他们的先知。 RAAN的容积不变.

## E2-04: the frozen dual-branch reference mission (task #257)

E2-04使用**说明 上升段固定**与实际模拟助推器返回进行任务整合测试(由E5-01作为快速测试例外-见下文E5-01部分). `scripts/e204_reference.py`从常数中构造出其两样上升段弧:分离时有35 t的助推器推进剂,并且规定在400 km/51.6 °处插入. 上升段优化或连续上升段 轨道传播在此回归中运行.

The scenario is `src/orbitopt/data/mission_two_stage_leo_recovery.yaml`.
It replays the frozen ascent, checks the separation fork and mass bookkeeping,
and flies the pinned recovery plan with gfold 0.3.2. Parking propagation is
explicitly `not_checked` and non-applicable to its full-mission gate; the
separate M-07 regression covers that leg. The shipped CI environments separate
landing and TudatPy, rather than proving these packages can never coexist.

`tests/data/mission/reference/e2-04-frozen-mission/`包含参考,容积和`PROVENANCE.json`(输入/文件散列和记录工具链). CI 工作检查来源,参考更新标签,重建输入和数字重运行:

```bash
python scripts/e204_reference.py verify
python scripts/e204_reference.py run --out /tmp/e204
python scripts/e204_reference.py compare /tmp/e204
```

参考跑者瞄准四行的`earth_orientation.json`快照显示其帧转换,并在之后恢复呼叫者表. 这避免了Astropy 6 / 8 选择不同的IERS表下线. 嵌入式数字事件数组使用宣布的容积;布尔、密钥和检查结果仍然准确。

| 终止原因 | 绝对比较预算 |
|---|---|
| 重建 上升段 位置/速度 | `1e-6 m` / `1e-6 m/s` |
| 流动位置/速度 | `0.01 m` / `1e-4 m/s`(不变) |
| 着陆启动计划燃料/持续时间 | `0.001 kg` / `0.01 s` |

The planning budgets apply only to advisory `landing_start` fields: 1 g is
50 times tighter than the solver's 0.05 kg stopping budget, and 10 ms is a
tenth of the closed-loop integrator step. See the [acceptance results](acceptance-e1-e4.md).
The portability fix preserves every recorded flight output. Its metadata-only
update used `python scripts/e204_reference.py update --tolerances-only`, which
verifies existing hashes before adding the frame snapshot and comparison
policy, while retaining the original recording toolchain.

若要独立说明理由的参考更改,请使用`python scripts/e204_reference.py update`,审查由此而来的出处和数字diff,并与`[e204-reference-update]`. CI的`verify`和`tag-check`的警卫一起执行.

## E5-01: the full mission chain flies a computed ascent (task #268)

E5-01 closes the E2-04 evidence gap for the ascent leg: the packaged M-07
scenario (`src/orbitopt/data/mission_two_stage_leo.yaml`, `ascent.mode:
optimize`) is flown now, and `tests/test_mission_e5_real_ascent.py` grades
WHERE the chain's evidence came from -- station times equal the engine's
event log (including solver-determined `max_q`/`impact` times the chain never
asks for), the parking handoff equals the computed track sample at the
engine's `seco` event, staging states equal flown track samples, and the raw
OpenTsiolkovsky outputs sit beside the run directory. Run it like the M-07
step (same `flightlab-gateb` job, same stack):

```bash
python scripts/fetch_opentsiolkovsky.py
pytest tests/test_mission_e5_real_ascent.py -m slow -v
```

没有tudatpy,M-05 轨道传播的腿被记录为`not_checked`(从未通过),恰好是E2-04的情景所宣布的; 快速护卫到处跑 记录的飞行合同检查不需要计算堆栈。

E5-01例外,刻意保留:E2-04冷冻任务保留其合成双样上升段,预设分离/SECO常数作为FAST集成测试. 假想中附加了例外,即建造者、其测试模块和`PROVENANCE.json`(`opentsiolkovsky: not used ...`),以及`tests/test_mission_e5_real_ascent.py`旅行,如果出现任何其他消费者,或者如果假想表面长出一个把手来注射上升段/SECO状态.

## E5-05: the reproducible full-physical reference mission (task #272)

**Recorded:** commit `64f64a6`, from code `74f30c8`, seals the computed-ascent
RTLS mission under `tests/data/mission/reference/e5-05-reference-mission/`.
The vehicle decision is resolved: `synthetic_two_stage_rtls` preserves
stage-1 propellant and restarts its upper stage. The committed trajectory
reaches **265.5 × 278.1 km at 51.0°** from Kourou. It supersedes the blocked
400 km/CMA-ES proposal; recording now flies a fixed OT program, not a new
pygmo search. See the [vehicle decision](adr/flightlab-0008-rtls-capable-vehicle-redesign.md).

| 参考文物 | 含义 |
|---|---|
| `mission-optimized/` | 记录的历史目录名称:计算 OT 上升段,原始后端输出,目标,恢复和合并报告. 没有证据表明这段录音经过了优化。 |
| `mission/` | 重播基线:记录的上升段+新模拟回收. |
| `earth_orientation.json` | 从录制数据包的四行IERS-A,在帧转换时设定范围. |
| `PROVENANCE.json` | 30文件散列,记录代码/工具链,场景/车辆文摘和犬类分离手动身份. |
| `tolerances.json` | 比较预算;修改预算是经过审计的参考资料更新。 |

恢复使用E5- 05 c固定配方(`search: keep`):IIP增速减速减速减速减速,减速减速减速降速预报,0.1最后集成和3.6 g升降法盖. 触地得分按其10 T预备队和速度限制分级. 承诺的单一环境报告通过适用阶段,但其**物理门仍为`not_checked`**:停放轨道传播没有运行。 失踪的轨道传播不算为通行证.

### 重播时

使用带有 gfold 0.3.2 的登陆堆栈; 此重播不需要 OT 执行器或 pygmo :

```bash
python scripts/e505_reference.py verify
python scripts/e505_reference.py run --out runs/e505
python scripts/e505_reference.py compare runs/e505
```

`verify`检查承诺的散列. `run`通过消化和从它的犬类分离状态中重新飞回来消耗所记录的上升段;`compare`输出的等级. [computed mission guide](computed-mission.md)增加了场景导出和当地观看.

`python scripts/e505_reference.py update`将固定程序上升段与OT v 1.0.0重新发布, 在承诺信息中使用 `[e505-reference-update]` 。 `tag-check --base SHA`只守护着这个参考目录;Gate B和E2-04的参考文献有自己的标记. 不要再产生结果来掩盖无法解释的漂移。

### 容忍来源

上升段重播为文摘-同. 回收带是飞行状态的`0.01 m` / Z `1e-4 m/s`,钟表/质量记账方式的`1e-6`. 咨询性起落架燃料/时间诊断使用`0.001 kg` /`0.01 s` ,与溶解器的趋同和整形解析有正当理由. 检查身份和判决都准确无误。 脚本和参考容忍文件文件每个比较预算;[tolerance audit](reference-tolerance-audit.md)涵盖更广泛的参考政策.

### 两个环境(交叉工作)

`crossenv_e505_plan.yaml` replays the computed ascent and **runs parking
propagation** in the locked TudatPy environment. A separate Python 3.12
environment re-flies gfold recovery from the sealed handoff. Both stages
set `options.earth_orientation_snapshot` to the reference snapshot so their
GCRS/ECEF conversions use the recording table, rather than interpreter-specific
IERS defaults; each stage report records the snapshot digest. The
[one-command example](https://github.com/zangjiucheng/orbitopt/blob/main/docs/computed-mission.md#add-parking-propagation-across-environments)
supplies both interpreters; then run:

```bash
python scripts/e505_reference.py crossenv-check runs/crossenv-e505
```

检查员验证了两个交换信封,交接身份,结合物理门证据和对照参考的回收数字. 所有五个物理检查都必须通过;计划或绿色的单层记录不足。 E2-04仍然是快速合成的固定器.
