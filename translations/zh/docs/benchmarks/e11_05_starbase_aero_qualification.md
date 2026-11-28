# E11-05 — Starbase aero model qualification (task #312)

Epic #307, base commit `add2cae`. E5-08 recorded a 12.80 % Starbase stage-1
debris impact-speed difference against the 12 % independent-model band
(`docs/e5-08-new-vehicle-report.md`), and E10-04 confirmed it as a real drag
model difference: replaying the recorded separation state through the twin
debris integrator with the declared CA polar gives +12.7 %
(`tests/data/mission/reference/e5-08-new-vehicle-starbase-reports/
debris-model-difference.json`) (中文(简体) ). 这项任务使助推器型电力起飞拖动模型与这一差异相适应——高纯度相邻的积聚加公共类数据以及所记录的碎片飞行本身,包括E11-01型电网式带/部署状态——和报告差距所在。 机器读数为[`e11_05_starbase_aero_qualification.json`](https://github.com/zangjiucheng/orbitopt/blob/main/docs/benchmarks/e11_05_starbase_aero_qualification.json);与`PYTHONPATH=src python benchmarks/bench_e11_05_starbase_aero_qualification.py --out …`(5 – 10 min,CPU只)复制,并与`PYTHONPATH=src pytest tests/test_booster_aero_e11_05.py`核对.

## 合格模型

`orbitopt.flightlab.booster_aero` builds the power-off Cd(Mach) from the
repo's own semi-empirical primitives (`nose_aero`: Taylor–Maccoll wave drag,
Rayleigh pitot Cp, PG-corrected turbulent Cf, DATCOM base term): forward
nose+shoulder, wetted skin friction, base drag, a stowed-protuberance
increment, and — engines-first — a full-Newtonian aft face over
`aft_face_fraction` of the body cross-section. Three qualification sources
anchor it:

* the E11-01 `grid_fins` block of the DC8 vehicle (deployed-state Cd,
  `public_estimate`,来源于
  [e11_01_grid_fin_authority.md](e11_01_grid_fin_authority.md));
* 两次记录的碎片飞行本身(Starbase和DC7):
  机内发电 Cd 反转是最强的“ 公共数据” 这个 repo
  持有;
* the DC7 booster stage's declared ballistic coefficient
  (BC = 3600 kg/m², dry 275 t, d = 9 m ⇒ implied Cd ≈ 1.20), the F9-class
  车辆表是建立在已公布的图样上。

DESIGN ASSUMPTIONS(在JSON中): 带鳍 t/c=0.12(±50%,线性为带带增量) ; 增生干扰 K = 1.9(1.5 – 2.3 ) ; 左侧面部开口分数0.85(±10%); 平面承载式前向基座(ERRS high subsoonic by ~ 0.05 ).

## A部分——组件表与已宣布的极地

Engines-first stowed build-up: 0.87 @ M0 → 0.754 @ M1 → 1.059 @ M1.1 →
1.408 @ M2 → 1.558–1.61 @ M4–M12. Against the declared DC8 booster table
(0.8 / 0.9 / 1.3 / 1.25 / 1.2 / 1.15): within the declared ±30 %
`public_estimate` band at M0–M0.8 (×1.09, ×0.88), 18–40 % **above** it at
M1.1–M8 — the declared table reads low against a blunt aft-face upper bound,
consistent with real aft cavities being partly vented (hot bleed, open
engine bays). The DC7 BC anchor (Cd ≈ 1.2) sits inside the declared band and
below the full-Newtonian build-up, the same direction. Axial, stowed: the
protuberance increment adds 0.064 to the vehicle's own power-off table
(Starbase v1 has no fins or legs: its qualified axial polar **is** the
declared one, bit-for-bit — see `axial_stacked_stowed_starbase_v1`).

## B部分——记录飞行倒置(决定性证据)

两个碎片CSV都是倒置的:中央偏移的_拖动记录的ECEF速度与双子本身的重力(J2)+Coriolis+离心力包被移除,投射在−v上,除以记录的q·S,排有50 kPa(Mach 4.5 – 6稠带). 星座反转Cd _ eff 0.334 – 0.348 – 0.355 – 0.377(宾比为0.914 / 0.957 / 0.980,意为0.960 ;DC7 – 0.321 – 0.338 – 0.350 – 0.376(意为0.934)). **有记录的航班运行4 – 7% BELOW 已宣布为极地.**. 每一个物理支持的电源脱落修正,这种资格产生——增生(+0.064),部署的鳍(+1.3 – 2.3 at α = 0),钝背面(+)——将重播的*进一步*移出波段. 没有任何可辩驳的校正可以弥补E10-04的缺口.

## C部分——同态重播(E10-04协议,不变)

| 极点 | 双重影响 | 记录 | Δ % |
|---|---|---|---|
| 已申报,星基(控制) | 1741.2 m/s | 1545.0 m/s | **+12.70** |
| 合格轴,Starbase v 1(=已申报) | 1741.2 m/s | 1545.0 m/s | **+12.70** |
| + E11-01 已部署(约束) | 796.3 m/s | 1545.0 m/s | −48.5 |
| declared, DC7 (control) | 1744.1 m/s | 1587.4 m/s | +9.87 |

The control reproduces the committed E10-04 numbers to 0.03 pt (+12.70 vs
+12.68; DC7 exact) — same separation state (raw ECI columns at t_sep, rotated
with the verified `_ecef_to_eci` convention), same recorded mass, twin's own
integrator. The fin-deployed bound shows the Cd lever is large — but points
the wrong way (v1 debris flew with no fins, and the lever only adds drag).

## D部分——网格鳍装配/部署的Cd状态

Stowed increment +0.064 (two-side fairing friction + frontal blockage,
K = 1.9). Deployed-minus-stowed at zero incidence, referred to the body:
+1.59 @ M0, +2.29 @ M1.0 peaking, falling to +1.29 @ M12 (E11-01
`cd_add_mach` plateau; the α > 0 flying state adds the normal-force and
induced terms on top, E11-01 Part A). At the next vehicle revision these
become the two axial polars: `ca_stacked_stowed(declared, DC8_STOWED, S)`
and that plus `fin_state_delta_cd(gf)`.

## E部分 – 主 – 模型 / 独立 – 模型 同步在合格表格上

一个模块生成表格;模型两个都消耗相同的数字. `booster_aero.interp_cd`复制`BoosterModel.cd()`(主模型,np.interp夹)位元(`interp_agreement_abs = 0.0`比9探测器),双胞胎的`_debris_to_impact`直接取同一张桌子. 从密封的E10-07分离交接处与合格DC8发动机-第一台:主1059.0/s对双子988.2/s-7.17%,在同一座12%模型-协议带E10-07使用(它自己的复飞三角洲是8.7%). 该同步索赔以`tests/test_booster_aero_e11_05.py`至0.1 pt.

## 判数 - 大门 12% 不变,星基地保持红色

The qualified power-off model leaves the Starbase debris replay at **+12.70 %
> 12 %**, so the check stays recorded as a genuine aero model difference
(接受分支2). 新的证据将其定位: 记录的航班
倒置 *下级* 宣布的极地, 所以残留不是 助推器 Cd
错误——是OpenTsiolkovsky与双胞胎之间的空气/大气层线.
在密度高的空气中, 正是E10-04放的地方。 DC7案留在里面
乐队(+9.87%). 没有碰过密封文物:YAML是
文摘封存 合格表格在这里作为模块生成
在下一次车辆修订时采用一个命令
(`booster_aero.cd_engines_first_table()` / `ca_stacked_stowed()` /
`fin_state_delta_cd()`,出处标记`qualified (E11-05)`.
