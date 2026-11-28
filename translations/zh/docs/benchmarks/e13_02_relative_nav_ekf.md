# E13-02 — relative-navigation EKF: RMSE, consistency, observability, mismatch (task #419)

Epic # 416. 此任务添加了相对态的扩展卡尔曼滤镜(`orbitopt.gnc.ekf`,见[ADR gnc-0002](../adr/gnc-0002-relative-state-ekf-on-the-contract-seam.md)):一个6 - vector LVLH station `(ρ, ρ̇)`,一个可插件过程模型,再利用E12-02 CW和无线配对的轨道传播逐字记录,测量模型为感应家族提供E13-03,以及一个发布路径到E13-01 `Estimate`合同. 接受是统计性的,所以机器读数是[`e13_02_relative_nav_ekf.json`](https://github.com/zangjiucheng/orbitopt/blob/main/docs/benchmarks/e13_02_relative_nav_ekf.json)——70细胞×5种子;复制时使用. `PYTHONPATH=src python
benchmarks/bench_e13_02_relative_nav_ekf.py -o …` (仅限1 min,CPU).

## 实验

Chief at 700 km / 51.6°; deputy at 300 m, either pure radial or pure
along-track, at rest in LVLH. Truth is always the nonlinear two-spacecraft
`propagate_pair` (two-body, J2, or J2 + impulsive differential drag,
`BC_dep = 0.1` m²/kg at ρ₀ = 1.3·10⁻¹¹ kg/m³, H = 50 km); the CW truth family
is `propagate_cw` stepped along the same chief. The filter is CW or
nonlinear(+J2), started at the true state + N(0, diag(30, 30, 30, 0.1, 0.1,
0.1)²) with Q = 10⁻⁹ m²/s³, updated every 60 s for 1 200 s. Five sensor
families: Cartesian position; position+velocity; range/range-rate;
range/azimuth/elevation (0.5 m, 0.5°, 50 µrad); Cartesian+angle stack.
Metrics after a 600 s settle: position/velocity RMSE, per-epoch **NEES**
against the declared χ²(6) 90 % band [1.64, 12.59] and **NIS** per sensor dof,
3σ per-axis coverage (nominal 0.997), and a 20-epoch R-weighted stacked
observability sweep of `W·H_k·Φ(t_k,t₀)`. Clusters report mean/median/q05/q95
over the 5 seeds; 62 declared gates all pass.

## 数字怎么说的?

* **Consistency holds where the model is the truth.** Matched cells with a
  笛卡尔更新:RMSE 0.6 – 0.9 m 与 30 m 初始误差,
  velocity RMSE 1.2–1.9 mm/s, NEES pass 0.79–0.92 of the χ²(6) band, NIS pass
  0.87–0.93, 3σ coverage 1.000 — and dropping below the declared 0.70/0.98
  门故障了套房,所以这是一个测试的权利要求,而不是一种气氛。
* **与模型完全一致。** CW 滤波器在J2 真理在
  300 m is barely distinguishable (RMSE 0.82 m, NEES 0.83) — the differential
  J₂ there is O(10⁻⁶) m/s² and 60 s updates absorb it. Differential drag is
  the sharp case: RMSE 5.1 m (6×), NEES pass **0.14**, NIS pass 0.43, 3σ
  0.51 / 0.36——滤波器报告摄氏度时的共变性
  真相留下模型,这正是NEES/NIS存在的失败.
  抓起来。 非线性(+ J2)进程 模型执行 **没有**救援(0.12):
  不匹配是未建模的力量,而不是动态近似.
* **Native sensors trade RMSE for linearisation bias.** Range/angle pins
  position to 0.52 m RMSE, but its NEES pass is 0.33 vs 0.92 for the same
  笛卡尔单元格 — 带有 30 mi 先前错误 雅各比安的角度 :
  过滤器会继承谎言, 即使真理是匹配的 。 斜体/ UK
  更新就是回路(E13-04);这里的数字就是动机.
* **The observability sweep explains the weak cells.** Range/rate over a
  20-minute arc is **rank 4/6 with an exact null space** (σ_min = 0): the
  along-track/cross-track pair stays unobserved, and the cell pays 56 m RMSE.
  Range+angle is full rank only through arc curvature (σ_min/σ_max 1.5·10⁻⁴,
  along-track geometry 3.7·10⁻⁵) — it pins position and misses velocity, so
  NEES悄悄死亡. 笛卡尔或笛卡尔+角堆是坚固的组合.
* **几何在此对称,种子紧凑.** 放射性和沿轨
  单元格同意在每行匹配的 < 5%(J2不对称在20 min),以及
  per-seed RMSE spread is q05/q95 = 0.51/0.87 m — the estimator, not the
  种子,设定比例。

## 洞穴动物

ephemeris酋长精确于每历元(E13-03的感应误差,纬度和下降是任务范围,而不是这个绳索');u = 0每间隔;NEES门只被宣布为其更新完全观察状态的细胞(Cartesian)——角细胞的数通过扫荡而不是加闸来报告和解释,因为没有它们就是发现,而不是回归.
