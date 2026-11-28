# E11R-02 – E11-06 进入峰值-q的差值(任务 # 415 ).

Epic # 409. E11-06门记录存档独立点-质量双重飞密封连续燃烧剧本,主模型上方的进入峰动态压力为**+18.2 %**,与不变的**12 %** `reflight.entry_peak_q`波段对齐,因此一次检查按设计是红色的,而另外八次一致(`speed +4.6 %`,`tilt 2.2°`,`peak_g 0.84 g`):`twin.json`在[`tests/data/mission/reference/e11-06-starbase-continuous-burn-reports/`](https://github.com/zangjiucheng/orbitopt/blob/main/tests/data/mission/reference/e11-06-starbase-continuous-burn-reports/README.md)下. 档案中命名了候选者——大气模型,网格-fin拖曳执行,攻击角度和态度,集成步骤,事件提取——不给其中任何一个定价. 这项任务对他们进行定价,而不再容忍。 机读为[`e11r_02_peak_q_error_budget.json`](https://github.com/zangjiucheng/orbitopt/blob/main/docs/benchmarks/e11r_02_peak_q_error_budget.json);与`E11_ENGINE_STAGES=1 E11_FINS_FIXED=1 python benchmarks/bench_e11r_02_peak_q_error_budget.py --out …`(1 min,CPU只)复制,剧本没有闸门的两个环境变量就拒绝运行),并与`PYTHONPATH=src pytest tests/test_e11r_02_peak_q_error_budget.py`核对.

## 方法

E10-07预算协议,一个公约一次改变,每行一个量度的双飞重飞,而不是手计算,用**主要 模型 峰值q的点数**(主峰q 100 pts)表示. 关闭是精确的建筑, 所以没有什么隐藏在剩余:

```
delta = (twin − twin-on-main-profile) + (twin-on-main-profile − main)
      =    density-profile form line  +   everything else
```

重新运行此堆的封存候选人 复制存档:**+17.81 %**比对**+18.19 %** 存档(0.4 pt of 漂移,主峰135.57 kPa在此对136.67 kPa存档),计划可行,走廊速度241.0 m/s,倾斜11.9 °. 门还读红.

## 预算

| 线条 | 点 | 改变了什么 |
|---|---|---|
| 密度图示窗体 | **+21.10** | 双流读取主 模型 密度剖面 |
| … the 0–25 km segment alone | +20.96 | 同一互换,仅在25公里以下 |
| ...在25公里以上 | +0.17 | 同一互换,仅在25公里以上 |
| 其它东西 | −3.29 | 两人共享一个配置后留下的空白 |
| … integration step 0.10 → 0.05 s | +0.53 | 双步数减半 |
| ...J2掉落 | +0.27 | 点质量重力双胞胎 |
| ...地理潜力 ~几何高度 | −0.34 | 双胞胎自己的US76在不进行层表转换的情况下读取 |
| ...进入窗口定义 | 0.00 | 主 模型 自己的音轨重新进入双子窗口 |
| 测量三角洲 | **+17.81** | (关闭错误 0, 到机器精度). |

One convention swap takes the gap from 17.8 pts to 3.3 pts, and it overshoots:
the rest of the twin pulls the *other* way, so the profile line is not merely
the biggest contributor — it is larger than the discrepancy itself, and no
other single difference can be worth 18 pts. The archived candidates are
priced out by that closure: the step is 0.5 pt, gravity 0.3 pt, the height
convention 0.3 pt, and the event-extraction difference is *exactly* zero (the
main model's peak q is the same number whether it is read over its own
ignition→crossing window or over the twin's cutoff→crossing window).

Grid-fin建模不能以同样的方式被否定,因为鳍承载着如此多的轨迹: 具有自身鳍力的双子复流在617/s的走廊上关闭了潜水,其进入顶峰是52 pts *高点*. 因此,该线不是横跨模型线——据报道它显示杠杆大小. 鳍候选人关闭的原因与其他候选人相同:只有密度配置共享,剩下的一切都是3.3 pts.

## 配置文件分割之处

这两个执行都忠实于其公布的来源——这是模型-*窗体*的区别,而不是错误. 双胞胎结合了US76层方程与自身地质潜力转化(.`orbitopt.verify.
full_recovery_reflight._twin_us76`, layer continuity across the 11/20/32/47/51/
71 km bases is asserted in the E10-07 test). The main model flies the published
piecewise-exponential table (`orbitopt.entry._ATMOSPHERE`), whose base heights
below 100 km are 0, 25, 30, … km: **no base between the ground and 25 km**, so
the entire troposphere and lower stratosphere is one exponential with scale
height 7.249 km, while US76 bends at 11 and 20 km.

| 小时(公里) | 主要单位(公斤/立方米) | twin US76 ρ (kg/m³) | 主/双 |
|---|---|---|---|
| 0 | 1.225 | 1.225 | 1.000 |
| 5 | 0.6146 | 0.7364 | 0.835 |
| 8 | 0.4063 | 0.5258 | 0.773 |
| 11 | 0.2686 | 0.3648 | **0.736** |
| 15 | 0.1547 | 0.1948 | 0.794 |
| 20 | 0.07761 | 0.08891 | 0.873 |
| 25 | 0.03899 | 0.04008 | 0.973 |
| 40 | 0.003972 | 0.003995 | 0.994 |
| 100 | 5.297e−7 | 5.788e−7 | 0.915 |

这两个剖面在海平面上完全一致,在0 – 25 km段(最靠近11 km tropopause)内有高达26%的分歧,在它的上方最多只有8.5%,密度不再重要到q峰. 该峰位于该段内部的模型:**主要9.96公里,双11.34公里**(该峰的高度和时间现在由双胞胎`entry_peak_q_height_m`/`entry_peak_q_t_s`记录). 因此两个模型没有不同意*时间*的峰值为18%:双子将密度更大的空气融合在整个俯冲上,空气沿着它的路径快速积聚,它的峰值同时到达更高和21的q点高于主模型.

## 判罚范围:12%不变,大门保持红色

差异是**genuine model-form difference in the density profile, localised to the single 0–25 km segment of the published piecewise-exponential table the main model flies**,该段携带的剖面线的99%和精确的关闭. 这个波段内部没有执行差异来固定,所以接受分支2适用:该差异被保留,记录,现在解释,12%的波段未受影响——拓宽它会隐藏一个26%的剖面差异,在下一次车辆修订时的表格修订可以移除. `tests/test_e11r_02_peak_q_error_budget.py`锁定了乐队,关闭,剖面线的主导地位和直播剖面比较,因此未来对任一大气的改变模型也必须移动归属.

One real implementation difference was found while auditing the twin, and it
does **not** touch this band: the twin's thrust model reads `cmd.throttle` but
ignores `cmd.min_throttle_scale`, the 13 → 5 → 3 flag the gated main model
flies. That flag is set only by the landing law (`landing.py:1867`), i.e. after
the corridor crossing, so it cannot move the entry peak q — but it does bias the
*impact* lines of the same gate, and it is recorded in the report for the
follow-up that grades those.
