# ADR 相对 -0001 :LVLH 帧,CW 作为仅规划的模型,非线性对验收(E12-02).

- **Status:** Accepted
- **Date:** 2026-11-24
- **Deciders:** E12 Orbital Rendezvous & Mission Operations (Epic #389), E12-02 (#392)
- **Code:** `src/orbitopt/physics_io.py`, `src/orbitopt/relative/`
- **Evidence:** `docs/benchmarks/e12_02_cw_validity.json` (+ `.md`),
  `tests/test_e12_02_relative_dynamics.py`, `tests/test_e12_02_cw_validity.py`
- **以 :** `physics_io`单位/框架合同(地心波段)
  请检查date=中的日期值 (帮助) ADR故意不碰, `orbitopt.core`常数集

## 背景和问题

`physics_io.py`拒绝LVLH和VVLH为"一个固定的轨道框架,而不是一个以地球为中心的框架"——只要每个接口都是以太阳为中心或以地球为中心就是正确的. E12汇合工作需要相反:相对国以*与局长旋转*为中心框架. 风险不是增加框架,而是一种急切的泛化侵蚀:地理中心波段检查就是这个项目的单位系统绊线,CW——明显的快模型——正是模型在不适用的地方被默化使用.

## 决定

1. **LVLH / RTN作为新`"orbital"`家族加入帧注册.**,不是作为
   惯性/旋转二聚体的变异: x 射线出(R), y
   along-track (T), z orbit-normal (N); RTN ≡ LVLH; VVLH stays a rejected
   以提示方式拼写。 `check_frame_family`再学习一个家庭标签;
   `EARTH_CENTRED_METRES`光谱,它的波段和每个现有的地心
   检查是字节相同。 新的Specs `RELATIVE_STATE`(以主为中心,
   0...月球距离)和`RELATIVE_PAIR`(两个绝对态,GCRS)
   或 EME2000 - J2需要赤道框) 搭乘相同.
   `PhysicsIOSpec`机与所有其他机体一样.
2. **`orbitopt.relative` is numpy-only and three modules**: `frames`
   (dcm-based ECI ↔ LVLH with the ω×ρ̇ velocity rule, round-trip exact to
   64),`cw`(闭式状态过渡,批量),`propagation`
   (`propagate_pair`:非线性双航天器双体+可选J2,
   在堆叠的12-状态,以首席为中心的LVLH输出上的固定步骤RK4).
3. **CW is a planning model, and the boundary is measured, not asserted.**
   配套基准扫射10米...100公里,持续时间最长
   5轨道,主偏心0...0.2,有和无J2. 结果:
   线性化错误随着分离而四倍增长——绝对
   error ∝ ρ², ≈ 10⁻⁴·ρ² per orbit: 0.01 m at 10 m, 1.05 m at 100 m,
   105 m at 1 km, so the *relative* error is ~1 % per orbit only at
   ρ ≈ 100 m and scales with ρ (no "1 % at any scale") — grows ~t²,
   and is dwarfed by any chief eccentricity ≥ 0.01
   (129 % of a 100 m separation at one orbit). The envelope gate
   (`test_e12_02_cw_validity.py`) 标注索赔: 通知主管,1公里,
   1轨道,误差 < 25% 分离——验收运行
   `propagate_pair`.

## 后果

- 以`"orbital"`框架为中心的国家像其他任何国家一样进行验证
  光谱状态,所以每个现存的边界(光谱检查 − 提升)都保持;但
  相对的波段检查*离职*, 和一个分离的跨越
  带表示一个单位系统错误,而不是相距遥远的对-对光谱
  仍然检查两个绝对状态 与完整的地心带。
- CW输出被刻意检查**没有**波段(CW漂移已经).
  已知的错误在一般t; 维持它在那里隐藏真正的错误;
  宣传员的绝对状态*已经*.
- 超过近圈长或地平线的任何东西都经过信封(J2)
  许多轨道上的世俗漂移,偏心的CW,Hill/LP)已超出范围.
  对于这一层,需要自己的承认 相同的基准
  纪律。
