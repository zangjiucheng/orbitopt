# Flight Lab再生与回归 CI (FL-601)

Flight Lab跑可以归因于精确的工具链,由OpenTsiolkovsky驱动的Gate B输油管有一个数字回归基线. 本页为业务摘要;许可证在[`THIRD_PARTY.md`](https://github.com/zangjiucheng/orbitopt/blob/main/THIRD_PARTY.md).

## 被钉住的工具链

| 工具 | 键 | 何处 |
|---|---|---|
| Python / conda 堆栈 | Python 3.10, tudatpy 1.0, NumPy 1.26.4, … | `envs/{cpu,gpu}.yml` + `envs/*.conda-lock.yml` (CI活 → `lockfile-drift` 失败 spec/lock skew) |
| 天体(框架桥) | `astropy=6.1.7`,准确 | `envs/{cpu,gpu}.yml` |
| IERS 地球导向数据 | 锁上文件的`astropy-iers-data`, B出门回放的**无下载** | `scripts/fl601_reference.py` (`_pin_earth_orientation`) |
| gfold (FL-303 登陆解析器) | `gfold==0.3.2`, exact (wheels need Python >= 3.11) | `envs/flightlab-landing.txt`; CI job `flightlab-landing` runs it on Python 3.12 via `envs/flightlab-landing-ci.txt` |
| jsbsim (E4 6-DOF spike) | `jsbsim==1.3.3`,散装 | `envs/sixdof-requirements.txt` |
| OpenTsiolkovsky (外部二进制) | v 1.0.0,每个平台SHA-256 | `scripts/fetch_opentsiolkovsky.py`( 拒绝其它字节; `--check` 验证安装的内容) |

`orbitopt.flightlab.contract.write_run`所写的每份飞行结果清单都载有`manifest.toolchain`下的所有这些内容(解释器,轨道opt,每个工具分发器的已安装版本或`"absent"`,OpenTsiolkovsky's plaed版本和实际在磁盘上找到的SHA-256). `tests/test_flightlab_contract.py`称记录完成.

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

## CI布局 (. GitHub /活活流/ci.yml).

* **`test` (linux -64, 被锁入内地)**-Flight Lab单位测试运行于该站.
  仅录入的固定装置: OT适配器测试回放录出 v 1.0.0
  通过 CLI 输出, 因此这里不需要 OT 二进制( 不存在) 。
  这工作还经营着`tests/test_flightlab_reference.py`,
  篡改已承诺的回归引用。
* **`flightlab-gateb` (慢, linux -64, 有锁入)**-取取被钉.
  OpenTsiolkovsky 二进制 (SHA-256经下载验证,版本出质),
  飞出B出入口
  (`leo_two_stage`目标+`synthetic_two_stage`车,上升段-只,.
  `refly=False`,`run_recovery=False`)与正二进制并作比较.
  每个 轨迹 样本、事件和检查
  `tests/data/flightlab/reference/` 所声明的容忍度
  `tests/data/flightlab/reference/tolerances.json`. 两者也一样
  G5卫:`verify`(参考文件与出处摘要相匹配)和
  `tag-check` (任何触摸`tests/data/flightlab/reference/`需要的diff)
  `[fl601-reference-update]` 在一个承诺消息中的push/PR范围.
* **`test-macos`** — the full suite on the locked env, arm64 + x86_64 (no
  在CI的macOS上,

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

CI拒绝了其他任何内容:手编辑的参考文件失败了`verify`(在快活中),未标记的参考修改失败了`tag-check`,一个意外的物理变化失败了数字`compare`.

## 手动 macOS- arm 64 运行

CI's Gate B 的工作是 Linux 唯一的(OT macOS 二进制被钉住并取回同样的方式,但我们保留了一条可口的CI平台). 在 OT- 接针或物理敏感变化后, 在 Apple Silicon Mac 上这样做 :

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

期望`compare`通过;在一个真正的三角地上,它只有在被宣布的容忍度过紧的情况下才能分出 -- -- 即`tolerances.json`更新,它通过与参考本身相同的更新标注+标记-承诺路径.
