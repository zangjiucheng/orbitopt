# 快速启动

打开预算的 LEO-RDV-001 运行,然后自己计算.

## 打开预计算运行——没有计算堆栈

`examples/runs/leo-rdv-001`是LEO-RDV-001的执着的Mission Studio重播,由舟子密封计划(`examples/runs/leo-rdv-001-sealed-plan.json`)建造. 额外观众就是你们需要的:

```bash
pip install -e '.[viewer]'
orbitopt view examples/runs/leo-rdv-001-scene.json
```

现场携带地球,目标与追逐者轨迹,守望中心和守望点作为标记,以及时间线上的操纵事件.

要装入 Mission Control(`orbitopt-app`)内部的运行,请在 **运行**下加入目录 `examples/runs/leo-rdv-001`. Rendezvous面板出现是因为运行目录持有`rendezvous.json`侧车;它读取了现场出口的同样密封的计划,它的`plan_digest`与`leo-rdv-001-sealed-plan.json`匹配.

## 打开 E13 闭路重播

```bash
orbitopt gnc studio --leo-rdv-001 -o runs/e13-leo-rdv-001
```

Mission Control,选择**运行 – 刷新 – e 13 - Leo-rdv -001**. **机构**条让你遵循追逐者的真相或估计. **遥测**显示导航错误和不确定性,**GNC**列出重播和验证结果. 默认工作室样本使用自己宣布的导航过程噪声,并检查所记录的重力,拖动和操纵. 独立飞行需要tudatpy.

## 计算计划并核实

由 Python 调用 Conda 计算堆栈(参见 [quickstart](quickstart.md) ):

```python
from orbitopt.rendezvous import load_reference_config, canonical_plan, verify_plan
cfg = load_reference_config()
plan = canonical_plan(cfg)          # sealed two-impulse plan, J2-targeted
result = verify_plan(cfg, plan)     # re-propagates; result.passed == all checks
print(plan.total_delta_v_m_s, result.position_error_m)
```

## 以 Flight Lab 重放

```bash
orbitopt rendezvous studio --config src/orbitopt/data/leo_rdv_001.yaml \
    --sealed-plan examples/runs/leo-rdv-001-sealed-plan.json \
    --refly-report docs/benchmarks/e12-05-refly-run/rendezvous-refly-report.json \
    -o runs/leo-rdv-001-studio
```

这写了一个新的运行而不取代承诺的示例 。 重播对照任务配置验证密封计划,并在显示其物理判断之前检查独立报告是否与计划相符. 打开 Mission Control或用 `orbitopt flight to-scene runs/leo-rdv-001-studio -o scene.json`导出.

对于新的配置,请省略`--sealed-plan`,使用`--write-sealed-plan PATH`来保存工作室使用的计划. 一旦有独立证据,即提供匹配的`--refly-report`.

## 重新运行独立复飞

tudatpy重飞需要跨环境堆栈,而不是这个:

```bash
orbitopt crossenv run --plan src/orbitopt/data/crossenv_e12_05_plan.yaml
```

参见[verification report](rendezvous-verification-report.md),其中对每行证据等级进行消化,[ADR rendezvous-0003](adr/rendezvous-0003-independent-refly-tudatpy.md)为复飞设计.
