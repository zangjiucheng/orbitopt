# Flight Lab可复制性和回归CI(FL-601).

每一次Flight Lab运行可归因于精确的工具链,OpenTsiolkovsky驱动的Gate B管道有数字回归基线. 本页为业务摘要;许可证在[`THIRD_PARTY.md`](https://github.com/zangjiucheng/orbitopt/blob/main/THIRD_PARTY.md)。

## 固定的工具链

| 工具 | 键 | 何处 |
|---|---|---|
| Python/康达堆栈 | Python 3.10, tudatpy 1.0, NumPy 1.26.4, … | `envs/{cpu,gpu}.yml` + `envs/*.conda-lock.yml` (CI job `lockfile-drift` 失败光谱/锁键) |
| 天体(框架桥) | `astropy=6.1.7`,准确 | `envs/{cpu,gpu}.yml` |
| IERS 地球导向数据 | 锁文件的`astropy-iers-data`,门B回归的**无下载** | `scripts/fl601_reference.py` (`_pin_earth_orientation`) |
| gfold( FL-303 着陆解析器) | `gfold==0.3.2`,准确 | `envs/flightlab-landing.txt` |
| jsbsim(E4 6 - DOF 尖锐) | `jsbsim==1.3.3`,散装 | `envs/sixdof-requirements.txt` |
| OpenTsiolkovsky( 外部二进制) | v 1.0.0,每个平台SHA-256 | `scripts/fetch_opentsiolkovsky.py`( 拒绝其它字节; `--check` 验证安装的内容) |

`orbitopt.flightlab.contract.write_run`所写的每份飞行结果清单都载有`manifest.toolchain`下的全部内容(解释器,轨道opt,每个工具分发器的安装版本或`"absent"`,OpenTsiolkovsky的绑定版本和实际在磁盘上找到的SHA-256). `tests/test_flightlab_contract.py`称记录完成.

### 为什么回归从未下载 IERS 数据

The frame bridge rotates OpenTsiolkovsky's Earth-fixed states into GCRS
with astropy, which needs Earth-orientation data (UT1-UTC, polar motion).
astropy downloads the latest IERS-A table by default, and that table changes
every day as predictions become measurements: the same flight's GCRS
positions moved by tenths of a metre between days, far beyond the
micrometre tolerances (the regression's first CI run failed on 7,670
samples). `fl601_reference.py run` / `update` therefore use only the table
bundled with the pinned `astropy-iers-data`. It ends before the reference
epoch, so astropy takes UT1 = UTC and zero polar motion there: a frame a few
tens of metres off the true one, but the same on every run with the same
lockfile. Bumping `astropy-iers-data` in the lockfile changes the reference;
re-record it with `update` and the `[fl601-reference-update]` tag. Record
it in the locked environment (CI's), not a local one.

## CI布局 (. GitHub/工作流程/ci.yml).

* **`test`( linux -64, 锁定的内嵌)** – Flight Lab 单位测试运行于
  仅录制固定装置: OT适配测试 重播录制 v 1.0.0
  通过 CLI 输出, 因此此处不需要 OT 二进制( 不存在) 。
  这份工作还经营着`tests/test_flightlab_reference.py`,
  篡改已承诺的回归引用。
* **`flightlab-gateb`(慢, Linux -64, 锁定的内嵌)**-拿下钉子
  OpenTsiolkovsky二进制(SHA-256经下载核实,版本制作),
  飞行的参考门B任务
  (`leo_two_stage`目标+`synthetic_two_stage`车,上升段 -仅限,).
  `refly=False`,`run_recovery=False`)与真实的二进制比较
  每个 轨迹 样本,事件和对照
  `tests/data/flightlab/reference/` within the tolerances declared in
  `tests/data/flightlab/reference/tolerances.json`. 两者也一样
  G5 警卫:`verify`(参考文件与其出处摘要相符)和
  `tag-check`(任何触摸`tests/data/flightlab/reference/`需要的潜水器)
  `[fl601-reference-update]`在push/PR范围的承诺信息中.
* **`test-macos`** — the full suite on the locked env, arm64 + x86_64 (no
  CI的macOSB门工作;

## 参考数据集

```
tests/data/flightlab/reference/
  gateb-leo-two-stage/     the recorded Gate B run dir (manifest, trajectories,
                           events, verification + raw opentsiolkovsky/ outputs)
  tolerances.json          the declared per-quantity tolerances
  PROVENANCE.json          sha256 of every file above, written by the script
```

更新( 例如, 重新固定 OpenTsiolkovsky 时, 或当预定的物理改变土地时 ) :

```bash
python scripts/fetch_opentsiolkovsky.py --force   # new pin already in the script
python scripts/fl601_reference.py update          # refly the run, rewrite PROVENANCE.json
git add tests/data/flightlab/reference
git commit -m "test(flightlab): re-record Gate B reference for OT vX.Y.Z [fl601-reference-update]"
```

CI拒绝了其他任何内容:手编辑的参考文件失败了`verify`(在快速任务中),未标记的参考修改失败了`tag-check`,一个意外的物理变化失败了数字`compare`.

## 手动 macOS- arm 64 运行

CI的门B工作是Linux唯一的(OT macOS二进制是固定的,并按同样的方式获取,但我们保留了一个犬形的CI平台). 在 OT- 针或物理敏化变化后, 在 Apple Silicon Mac 上这样做 :

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

期望`compare`通过;在真正的三角洲上,只有在容忍度太紧的情况下,它才能在宣布的容忍度范围内分裂出来 -- -- 即`tolerances.json`更新,它通过与参考本身相同的更新标注+标记-承诺路径。
