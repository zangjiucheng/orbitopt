# E11-02 — recovery feasibility envelope (task #309)

E11-02对照一个回收可行性信封(orualopt.flightlab.fificity)筛选分离状态. BEFORE 计划员搜索:一个参考推开加一个分析燃料底座决定IN / OUT 并用零搜索畸形预测飞行原因. 信封条件:燃料底座(分离速度上的火箭方程),走廊到达/误差/tilt/到达速度,闭路着陆预报,输入q和g,着陆推进剂储备边——每个都用计划员自己的物理测量. 大纪元:DC8阻截武器是E10-03时代(DC7密封POWERED配方在DC8车辆上飞行,无网鳍);适用于DC8的写作信封(E11-01网鳍任务)在A/B/E部分分级.

## 1. 参考判决(A部分)

| 钻机 | 光谱 | 信封内 | 联合国 | 预测原因 | 未筛选原因 | 匹配 |
|---|---|---|---|---|---|---|
| `dc7` | `recipe` | IN | 无 | `None` | `-` | 对 |
| `starbase` | `recipe` | IN | 无 | `None` | `-` | 对 |
| `dc8_fins` | `dc8_fins_a60` | OUT | _ 倾斜 | `CORRIDOR_MISSED` | `CORRIDOR_MISSED` | 对 |
| `dc8` | `recipe` | OUT | _预报 | `CORRIDOR_MISSED` | `CORRIDOR_MISSED` | 对 |

## 2 边界地图(B部分)

- `dc7` : 18 / 49 内部; 重度场和治理图在`docs/benchmarks/figures/e11_02_boundary_dc7.png`; 网格 = 下程 ± 9 km x 速度 ± 60 /s 在热静交接处.
- `dc8_fins` : 0 / 49 内部; 重度场和治理图在`docs/benchmarks/figures/e11_02_boundary_dc8_fins.png`; 网格 = 下程 ± 9 km x 速度 ± 60 /s 在热静交接处.

## 3 E10-03搁置协议(C部分)

每个样本都未经筛选(真相,E10-03工人逐字记录)和筛选(预测)。 A *错误拒绝*——屏幕OUT但飞行着陆——是屏幕不允许作出的唯一错误.

| 手臂 | 无 | 屏幕进入( L) | 未登陆( N) | 未降落( N) | 出来但着陆( 假拒绝) | 协议 | 拒绝的样本代码匹配 | 事实与犯罪 |
|---|---|---|---|---|---|---|---|---|
| `dc7_pinned` | 168 | 44 | 0 | 124 | **0** | 1.0 | 124/124 | 44 / 44(匹配) |
| `starbase_pinned` | 168 | 11 | 0 | 157 | **0** | 1.0 | 157/157 | 11 / 11(匹配) |
| `dc8_robust` | 168 | 0 | 0 | 168 | **0** | 1.0 | 168/168 | 0 / 0(匹配) |
| `dc8_binary` | 168 | 0 | 3 | 165 | **0** | 0.9821 | 165/165 | 0 / 0(匹配) |

完全错误拒绝(屏幕OUT,飞行着陆):**0**——屏幕唯一不可原谅的错误是零.

## 4. 双边界复制(D部分)

E10-07的独立双胞胎在自己的大气线,风,和鳍模型上从同一个分离状态上重新显示密封的脚本;波段在基线运行上校准,然后应用到边界对C选择.

| 运行 | 边界边 | 协议 | q (千帕) (千帕) | 负载(g) | 速度( m/s) | 倾斜 (deg) | 错过(m) | 推进剂(吨) |
|---|---|---|---|---|---|---|---|---|
| `D|base|dc7` | — | 无 | 147(单位:± 20%) ✗ | 7.6 (in, ±2g) ✓ | 277(单位:± 40%) | 0.0712(单位:± 3) ✓ | 122 (in, ±1500m) ✓ | 166(单位:± 3%) ✗ |
| `D|base|dc8_fins` | — | 无 | 131(单位:± 20%) | 7.09 (in, ±2g) ✓ | 279(单位:± 40%) ✗ | 26.1(出,± 3 deg) ✓ | 823 (in, ±1500m) ✓ | 301(单位:± 3%) |
| `D|pair|in|dc7_pinned|502|1|cal` | IN | 对 | 160(单位:± 40%) | 7.47 (in, ±2g) ✓ | 270(单位:± 40%) | 0.0444(单位:± 3) ✓ | 65.8 (in, ±1500m) ✓ | 154(单位:± 6%) |
| `D|pair|out|dc7_pinned|501|3|cal` | OUT | 对 | 166(单位:± 40%) | 7.32 (in, ±2g) ✓ | 256(单位:± 40%) | 0.0326(单位:± 3) ✓ | 315 (in, ±1500m) ✓ | 162(单位:± 6%) |

Baselines carry the band calibration (1.5x the measured twin-vs-planner delta, floored at the E11-01 FIN_TOL values); twin disagreement is reported, never hidden. The twin's arrival-speed channel is the known atmosphere-line quantity (E11-01 measured 24%): when that quantity alone disagrees, the boundary verdict rests on q/g/tilt/miss/prop.

## 5. 搜索预算(E部分)

| 状态 | 全面搜索:墙壁/畸形/判决 | 筛选:墙壁/畸形/原因 |
|---|---|---|
| `E|dc7|in-nominal` | 59.8 s / 132 羽毛 / `CORRIDOR_MISSED` | 61.0 s / 132 羽毛/接受 |
| `E|dc7|out-corner` | 59.9 s / 0 羽毛 / `CORRIDOR_MISSED` | 1.5 s / 0 羽毛 / `CORRIDOR_MISSED` |
| `E|dc8_fins|out` | 40.1 s / 0 羽毛 / `CORRIDOR_MISSED` | 0.8 s / 0 羽毛 / `CORRIDOR_MISSED` |

被筛选的呼叫在OUT状态上花费了零搜索的fevals,并返回了预测的飞行原因;全搜索花费了预算(通过规划者自己的搜索钩计算,与默认路径相同的选项). 在推进剂地板下,规划者自己的警卫也跳过搜索,所以节省的是墙钟——一个廉价的参考推开,以对抗全定分级的整合(40 x在角细胞)——而地板内屏幕则将整个全预算通行证退休. DC8-时代臂的真正预算是300+飞行ES训练(见承诺的E10-03报告),屏幕为信封外的州退役.
