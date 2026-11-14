# GOES 发射至GEO任务计划设计

这个GOES-16-像参考模型是转移轨道和宇宙飞船远地点的升空运动. 它是一个规划模型,而不是重建未公布的燃烧式遥测.

{: . 警告}( .
> 内置的GOES场景载荷**未检查**. 运行高纯度
> 验证 从验证标签,或启用`verification.enabled`
> 在`geo-raising`配置。 平滑动画不是验证结果.

## 1. 任务概览

GOES-R / 16 , GOES-S / 17 和 GOES-T / 18 在阿特拉斯V 541 发射; GOES-U / 19 使用猎鹰重型. 下文的参考简介采用了GOES-16 Atlas V案,其中有一个约5,192千克的航天器. 目标是一个圆形赤道GEO轨道,高度约为35,786公里(42,164公里半径).

半人马座的三次烧伤使飞船进入了高架轨道-近地点转移轨道. 然后,该航天器将近地点升起,并将剩余的轨道倾角通过远地点烧伤去除,然后漂移到它的槽中.

## 2 任务阶段

| 阶段 | 参考或假设 |
|---|---|
| 半人马座燃烧 1 | 泊车轨道:约167 × 541公里,28.15 °;期间约91.6 min. |
| 燃烧 2 | 转移轨道:约187 × 32,716 km,25.68 °. |
| 烧3和分离 | 大约8,108 × 35,286 km,10.6 °;发射后约3.5 h的分离,轨道周期约13 h. |
| 航天器的饲养 | Approximately five apogee burns inferred for GOES-16; intermediate states and exact schedule are illustrative. |
| GEO和漂移 | 将35,786公里的高度通知,然后向目标经度漂移。 |

The 35,286 km injection apogee is about 500 km **below** GEO altitude, not
supersynchronous. GOES-U's exact Falcon Heavy separation elements are not
established in the reference data; do not reuse GOES-16 elements as its flight
trajectory.

## 3. Delta-v预算

对于GOES-16-像注射,一种理想的单阴远地点操作给出了计算的理想估计,航天器份额约为**0.98公里/秒**. 包括多处烧伤和漂流,可以得出大约**1.0 – 1.3 公里/秒**的低信心规划估计,而不是公布的Atlas V飞行预算. 最初的参考研究报告566 m/s,在猎鹰重力注射987 m/s任务要求后,GOES-U;其注射元素仍然未知. 在使用前,应对照特派团特定来源对推进剂和操纵性假设进行检查。

## 4. 关键参数和信心旗

- GOES-16参考文献使用**10.6°**注射轨道倾角;一个常用引用的参考文献.
  9.52 °值适用于GOES-17的参考简介,而不是GOES-16.
- The approximate five-burn count is inferred from GOES-16's campaign, which
  包括一个热异常 额定燃烧计数和中间状态
  此处没有建立。
- 近地点和准确的GOES-U的注入参数是:
  假设或未知数。 模型在赤道节点附近假定远地点.
- 源研究区分了 15 - 年份设计寿命与更长
  推进剂寿命估计。 对模型的验证也不成立。

## 5. 建模优化 在轨道上

`GeoRaisingProblem`屏幕冲动远地点 - 燃烧时刻表. 近地点起火,轨道倾角起火, 场景由 `viz/geo_raising.py` 导出; 以 :

```bash
python examples/10_gto_geo_orbit_raising.py
orbitopt view goes
```

`verify/geo_insertion.py`与地球J2 / J22和太阳/月点质量扰动重新发布时间表,精炼最终燃烧,并报告短期漂移和验证检查. 太阳辐射压力和限时烧伤没有传播。 模型的有限燃烧可行性估计不能代替有限燃烧的轨道传播.

## 6. 开放式问答

确认注射方向、每个燃烧状态和时间表,如果有特定任务来源的话。 GOES-U分离元素,阿特拉斯V号航天器Δv预算以及精确的远地点-燃烧限值仍然不确定. 优化器的烧裂是一个模型化的结果,不是测量飞行遥测.
