# E11-03 robust recovery co-design (task #310)

_generated 2026-11-23T02:14:29Z, git ``_

E10-03闭合阴性:DC8分离状态在可行的信封外,没有脚本在框内拉入(0 / 168). E11-03联合设计了分离STATE本身(重流上升段时间线/态度/燃烧重排)与回收脚本,使用E11-02信封屏幕(带有E10-03扰动画)作为联合设计目标,并将结果与密封的基线相对照在新鲜的,配对的搁置上.

## Hold-out campaign (paired, fresh seeds 501-508 x 21)

| 手臂 | 着陆/ 无 | 费率 | CI95组 |
|---|---|---|---|
| `codesign` | 0 / 168 | 0.0% | [0.000, 0.000] |
| `dc7_pinned` | 44 / 168 | 26.2% | [0.191, 0.351] |
| `starbase_pinned` | 11 / 168 | 6.6% | [0.048, 0.089] |

## 比较(2,Holm校正)

| 比较 | 维夫 | 配对 CI95 | 麦克尼玛准确p | p (霍尔姆语) | 显著 |
|---|---|---|---|---|---|
| 密码符对 dc 7 _ 绑定 | -0.262 | [-0.351, -0.191] | 1.137e-13 | 2.274e-13 | 没错 |
| 代号对星座 _ 绑定 | -0.066 | [-0.089, -0.048] | 0.0009766 | 0.0009766 | 没错 |

## 失败类别

| 手臂 | 规划_不可行 | _ 失败 | 准确性 | 燃料 |
|---|---|---|---|---|
| `codesign` | 163 | 0 | 0 | 5 |
| `dc7_pinned` | 123 | 0 | 0 | 1 |
| `starbase_pinned` | 154 | 0 | 0 | 3 |

理由组合(每臂顶部代码):

- `codesign` : CORRIDOR_MISSED × 157, JOINT_CHANCE_VIOLATED × 6, INSUFFICIENT_PROPELLANT_RESERVE × 5
- `dc7_pinned`:CORRIDOR_MISSED× 123,INSUFFICIENT_PROPELLANT_RESERVE×1.
- `starbase_pinned`:CORRIDOR_MISSED× 154,INSUFFICIENT_PROPELLANT_RESERVE×3.

## 培训协议(筛选判决与飞行结果)

协调0.0(None/None screen-ON原计划与流的样本;假ON是计算在它的屏幕上).

## 高级插入幅度(已报告,接受项目)

密封的DC8 船舶边距 103548 千克 - > 代号103548 千克(+0 千克,舰只燃烧 +0 s) 边距是SECO 干重以上级的-2(舰)推进剂,来自复流的OT 轨道;正燃烧延伸购买SPENDING 船舶推进剂的插入重排,因此三角洲的诚实方向随着dburn增高而下降 -- 据报道,从未隐藏.

## E5-06 联合物理门(代号名义)

`landing`:失败 | `mass_conservation`:通过 | `epoch_frame_continuity`:通过 | `orbit_insertion`:通过 | `parking_propagation`:没有 _ 证明 _ 单 _环境


限制没有放松:是的(上升段检查所有被选中的上升段;在武器之间等级相同的储备)。 密码计数器在0千克软度+目标硬度100000千克(SAME目标检查分数的一套已接受的超群计划;密封的复式武器保留自己的食谱底座 -- -- E10-03/E10-06语义,任何地方都不能放松)
