"""Plain-language reason-code catalog for Flight Lab verdicts (FL-503).

Every failing :class:`~orbitopt.flightlab.contract.Check` carries a
machine-readable ``reason_code`` (``PERIAPSIS_IN_ATMOSPHERE``,
``max_q_exceeded``, ``INSUFFICIENT_PROPELLANT_BOOSTBACK``, ...). This module
maps each code to a short title, a plain-language explanation and a suggested
fix, in English and Chinese, so the replay and compare views can say *why* a
run is red and *what to try next* instead of printing an identifier.

Lookup is case-insensitive (the OpenTsiolkovsky adapter emits lower-case
codes, the checks module upper-case ones): :func:`normalize_code` folds both
to upper snake case. ``SAMPLE_ERROR:<Exception>`` (FL-304 Monte Carlo) is
matched by its prefix. Unknown codes never raise: :func:`explain` returns a
generic entry flagged ``known=False`` so the UI can still show the raw code.

``tests/test_flightlab_reasons.py`` scans every module under
``orbitopt/flightlab`` and fails when a reason code is emitted that has no
entry here -- add the entry in the same change that adds the code.

Qt-free, stdlib-only: importable on a contract-only install.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

__all__ = [
    "CATALOG",
    "ReasonEntry",
    "explain",
    "explain_all",
    "is_known",
    "lookup",
    "normalize_code",
]


@dataclass(frozen=True)
class ReasonEntry:
    """One catalogued reason code. ``category`` groups codes for display
    (``insertion``, ``loads``, ``propellant``, ``recovery``, ``landing``,
    ``physics``, ``data``, ``optimizer``, ``montecarlo``); ``severity`` is
    ``fail`` for a violated constraint, ``missing`` for a check that could not
    be assessed, ``info`` for a non-failure outcome."""

    code: str
    category: str
    title_en: str
    title_zh: str
    explanation_en: str
    explanation_zh: str
    fix_en: str
    fix_zh: str
    severity: str = "fail"
    known: bool = True

    def title(self, lang: str = "en") -> str:
        return self.title_zh if lang.startswith("zh") else self.title_en

    def explanation(self, lang: str = "en") -> str:
        return self.explanation_zh if lang.startswith("zh") else self.explanation_en

    def fix(self, lang: str = "en") -> str:
        return self.fix_zh if lang.startswith("zh") else self.fix_en

    def text(self, lang: str = "en") -> str:
        """``title -- explanation  Fix: ...`` in one language."""
        fix_label = "建议" if lang.startswith("zh") else "Fix"
        return f"{self.title(lang)} — {self.explanation(lang)} {fix_label}: {self.fix(lang)}"

    def bilingual(self) -> str:
        """Multi-line EN + 中文 block (tooltips, the explanation panes)."""
        return (f"{self.title_en}\n{self.explanation_en}\nFix: {self.fix_en}\n\n"
                f"{self.title_zh}\n{self.explanation_zh}\n建议：{self.fix_zh}")


def _e(code, category, title_en, title_zh, expl_en, expl_zh, fix_en, fix_zh, severity="fail"):
    return ReasonEntry(code, category, title_en, title_zh, expl_en, expl_zh, fix_en, fix_zh, severity)


_ENTRIES: tuple[ReasonEntry, ...] = (
    # ------------------------------------------------------------ insertion orbit
    _e("PERIAPSIS_IN_ATMOSPHERE", "insertion",
       "Periapsis inside the atmosphere", "近地点位于大气层内",
       "The orbit at engine cutoff dips below 120 km; drag would bring the upper stage and payload "
       "back down within hours. This is a failed mission regardless of the target's own floor.",
       "关机时轨道近地点低于 120 km，大气阻力会在数小时内使上面级和载荷再入，无论目标下限如何都判定任务失败。",
       "Raise the insertion velocity: loft the trajectory (shallower pitch-over), lengthen the upper-stage "
       "burn, or reduce payload / dry mass.",
       "提高入轨速度：放缓程序转弯以抬高弹道、延长上面级燃烧时间，或减少载荷/干质量。"),
    _e("INSERTION_PERIAPSIS_BELOW_MINIMUM", "insertion",
       "Periapsis below the mission minimum", "近地点低于任务最低要求",
       "The orbit clears the atmosphere but its lowest point is under the minimum insertion periapsis "
       "this target declares.",
       "轨道已离开大气层，但近地点低于目标规定的最低入轨近地点。",
       "Add upper-stage delta-v (more propellant or Isp, less payload) or shape the ascent so the cutoff "
       "happens closer to apoapsis.",
       "增加上面级速度增量（更多推进剂、更高比冲或更少载荷），或调整上升弹道使关机点更接近远地点。"),
    _e("INSERTION_PERIAPSIS_OUT_OF_TOLERANCE", "insertion",
       "Periapsis off target", "近地点偏离目标",
       "The achieved periapsis altitude is outside the allowed band around the target periapsis.",
       "实际近地点高度超出目标近地点的允许偏差范围。",
       "Adjust the pitch program and upper-stage cutoff time, or run the ascent optimizer against this target.",
       "调整俯仰程序和上面级关机时间，或针对该目标运行上升段优化器。"),
    _e("INSERTION_APOAPSIS_OUT_OF_TOLERANCE", "insertion",
       "Apoapsis off target", "远地点偏离目标",
       "The achieved apoapsis altitude is outside the allowed band around the target apoapsis — usually "
       "an over- or under-burn of the upper stage.",
       "实际远地点高度超出目标远地点的允许偏差范围，通常是上面级燃烧过多或不足。",
       "Trim the upper-stage burn end time / propellant load, or flatten the late pitch program.",
       "调整上面级关机时间或推进剂加注量，或压平后段俯仰程序。"),
    _e("INSERTION_INCLINATION_OUT_OF_TOLERANCE", "insertion",
       "Inclination off target", "轨道倾角偏离目标",
       "The orbital plane is tilted differently from the target inclination by more than the tolerance.",
       "轨道面倾角与目标倾角之差超过允许偏差。",
       "Correct the launch azimuth (attitude program azimuth) for the site latitude and Earth rotation.",
       "根据发射场纬度和地球自转修正发射方位角（姿态程序方位角）。"),
    _e("INSERTION_RAAN_OUT_OF_TOLERANCE", "insertion",
       "Orbital plane orientation (RAAN) off target", "升交点赤经偏离目标",
       "The orbit's ascending node is in the wrong place: the vehicle launched at the wrong time for the "
       "requested plane.",
       "升交点位置不对：发射时刻与所需轨道面不匹配。",
       "Move the liftoff epoch into the launch window for this RAAN (use the window search), or relax the "
       "RAAN requirement.",
       "将起飞时刻移入该升交点赤经对应的发射窗口（使用窗口搜索），或放宽升交点要求。"),

    # ---------------------------------------------------------------- flight loads
    _e("MAX_Q_EXCEEDED", "loads",
       "Max-Q limit exceeded", "最大动压超限",
       "Peak dynamic pressure during ascent is above the structural limit — the vehicle flies too fast "
       "too low in the atmosphere.",
       "上升段峰值动压超过结构限制——飞行器在低空飞得太快。",
       "Throttle down through max-Q, pitch over later / more gently, or lower liftoff thrust-to-weight.",
       "在最大动压段节流，推迟或放缓程序转弯，或降低起飞推重比。"),
    _e("MAX_ACCEL_EXCEEDED", "loads",
       "Acceleration (g-load) limit exceeded", "过载超限",
       "Thrust acceleration exceeds the g limit, typically near burnout when the stage is light.",
       "推力加速度超过过载限制，通常发生在级燃尽前质量最轻时。",
       "Add a throttle-down before MECO/SECO, or reduce thrust on the light end of the burn.",
       "在一级/二级关机前增加节流段，或在燃烧末段降低推力。"),
    _e("MAX_Q_DATA_MISSING", "data",
       "Max-Q not measured", "未测得最大动压",
       "The target declares a dynamic-pressure limit but the run has no max-Q value or dynamic-pressure "
       "column to check it against.",
       "目标规定了动压限制，但该次运行没有最大动压数值或动压数据列可供检查。",
       "Re-fly with a backend that reports dynamic pressure (OpenTsiolkovsky does), or drop the limit "
       "from the target.",
       "使用能输出动压的后端（如 OpenTsiolkovsky）重新飞行，或从目标中删除该限制。", "missing"),
    _e("MAX_ACCEL_DATA_MISSING", "data",
       "Acceleration not measured", "未测得过载",
       "The target declares a g limit but the run reports no peak acceleration.",
       "目标规定了过载限制，但该次运行未报告峰值加速度。",
       "Re-fly with a backend that reports acceleration, or drop the limit from the target.",
       "使用能输出加速度的后端重新飞行，或从目标中删除该限制。", "missing"),
    _e("LIFTOFF_THRUST_TO_WEIGHT_LOW", "loads",
       "Cannot lift off (T/W < 1)", "无法起飞（推重比 < 1）",
       "Stage-1 thrust at liftoff is below the vehicle's weight; the rocket would sit on the pad.",
       "一级起飞推力小于全箭重量，火箭无法离开发射台。",
       "Add stage-1 thrust (engines) or remove mass (propellant, upper stage, payload).",
       "增加一级推力（发动机）或减轻质量（推进剂、上面级、载荷）。"),

    # ------------------------------------------------------------------ propellant
    _e("PROPELLANT_RESERVE_LOW", "propellant",
       "Propellant reserve below requirement", "剩余推进剂低于要求",
       "Less propellant is left than the target's reserve requires (upper stage at insertion, or the "
       "booster for recovery).",
       "剩余推进剂少于目标要求的储备量（入轨时的上面级，或用于回收的助推级）。",
       "Load more propellant, reduce payload, or improve the trajectory's efficiency (less gravity/drag loss).",
       "增加推进剂加注、减少载荷，或优化弹道以降低重力/阻力损失。"),
    _e("PROPELLANT_RESERVE_DATA_MISSING", "data",
       "Propellant reserve not assessable", "无法评估剩余推进剂",
       "The target declares an upper-stage reserve but the run/vehicle lacks the masses needed to measure it.",
       "目标规定了上面级储备，但运行结果或飞行器文件缺少计算所需的质量数据。",
       "Make sure the vehicle file states each stage's dry mass and the run has an insertion state.",
       "确认飞行器文件给出各级干质量，且运行结果包含入轨状态。", "missing"),
    _e("RECOVERY_PROPELLANT_RESERVE_DATA_MISSING", "data",
       "Recovery reserve not assessable", "无法评估回收储备推进剂",
       "Recovery is planned but the run has no released first-stage track (or the vehicle has no stage-1 "
       "dry mass), so the propellant left for landing cannot be measured.",
       "计划回收，但运行结果没有分离后的一级轨迹（或飞行器缺少一级干质量），无法计算着陆可用推进剂。",
       "Fly with a backend that writes the stage-1 track after separation and give stage 1 a dry mass.",
       "使用能输出分离后一级轨迹的后端飞行，并为一级填写干质量。", "missing"),
    _e("NEGATIVE_MASS", "data",
       "Negative mass in the trajectory", "轨迹中出现负质量",
       "A body's mass went below zero: the backend burned more propellant than the stage carries.",
       "某个物体质量变为负值：后端消耗的推进剂超过了该级的携带量。",
       "Check burn end times against propellant load (mass_initial - mass_dry) and engine flow rate.",
       "核对关机时间与推进剂加注量（初始质量 - 干质量）及发动机流量。"),
    _e("MASS_DISCONTINUITY", "data",
       "Unexplained mass jump", "质量突变",
       "Mass changed abruptly between samples outside a staging/jettison event — the mass bookkeeping is "
       "inconsistent.",
       "在非分离/抛罩事件处质量发生突变——质量记账不一致。",
       "Declare the jettison (fairing, interstage) in the vehicle file or fix the stage masses.",
       "在飞行器文件中声明抛罩/级间段等抛弃物，或修正各级质量。"),

    # -------------------------------------------------------------- physics re-fly
    _e("REFLY_BACKEND_MISSING", "physics",
       "Physics re-fly not available", "无法进行物理复核",
       "The independent orbit re-fly needs tudatpy, which is not installed, so the physics verdict stays "
       "unassessed.",
       "独立的轨道复核需要 tudatpy，但当前环境未安装，因此物理验证结论为“未评估”。",
       "Use the conda environment (envs/cpu.yml) that ships tudatpy, then re-run the simulation.",
       "使用包含 tudatpy 的 conda 环境（envs/cpu.yml）后重新仿真。", "missing"),
    _e("REFLY_NOT_ORBITAL", "physics",
       "Not in orbit — nothing to re-fly", "未入轨，无法复核",
       "The insertion state is suborbital (it would hit the ground within one orbit), so the orbit re-fly "
       "cannot run.",
       "入轨状态为亚轨道（一圈内会坠地），无法进行轨道复核。",
       "Fix the ascent first: see the insertion checks for what is short.",
       "先修正上升段：参见入轨检查中不足之处。"),
    _e("REFLY_BACKEND_ERROR", "physics",
       "Physics re-fly crashed", "物理复核出错",
       "The tudatpy re-fly raised an error, so physics could not be verified.",
       "tudatpy 复核过程中报错，无法完成物理验证。",
       "Open the run assumptions for the error text; re-run, and report it if it persists.",
       "在运行假设中查看错误信息；重新运行，如仍出错请上报。", "missing"),
    _e("REFLY_ELEMENT_DRIFT", "physics",
       "Orbit not self-consistent", "轨道复核偏差过大",
       "Re-integrating the insertion state with an independent propagator drifted beyond the Gate-G5 "
       "limits: the reported orbit is not a consistent orbit.",
       "用独立积分器重新积分入轨状态后，轨道根数漂移超过 G5 门限：报告的轨道自洽性不足。",
       "Check the trajectory frame/epoch (GCRS bridge) and the backend's output step; this is a data "
       "problem, not a tuning one.",
       "检查轨迹坐标系/历元（GCRS 转换）和后端输出步长；这是数据问题而非调参问题。"),

    # ------------------------------------------------- 6-DOF backend (F-01, #226)
    _e("SIXDOF_NON_FINITE_STATE", "physics",
       "6-DOF state went non-finite", "6-DOF 状态出现非法值",
       "The JSBSim 6-DOF integration produced a NaN or infinite state — the flight diverged and no "
       "downstream check means anything.",
       "JSBSim 6-DOF 积分产生 NaN 或无穷大的状态——飞行已发散，后续检查均不可信。",
       "Shorten the integration step (dt_s), lower the throttle, or check the vehicle's mass/propellant "
       "figures in the scenario config.",
       "减小积分步长（dt_s）、降低油门，或检查场景配置中的飞行器质量/推进剂数值。"),
    _e("SIXDOF_HANDOFF_CONTINUITY", "physics",
       "Staging handoff not continuous", "级间分离衔接不连续",
       "At the 6-DOF staging split the child states do not match the frozen stack state within the "
       "declared tolerances (or the mass/momentum bookkeeping closed only loosely) — the handoff lost "
       "or invented state.",
       "6-DOF 级间分离时刻，子级状态与冻结的整箭状态在声明容差内不一致（或质量/动量记账不闭合）——衔接丢失或凭空多出了状态。",
       "Re-check the staging split fractions, the separation impulse and the interstage models; the "
       "handoff must be state-preserving by construction.",
       "重新检查级间分离质量比例、分离冲量与级间段模型；衔接在构造上必须保持状态守恒。"),
    _e("SIXDOF_TRACKING_OUT_OF_BOUNDS", "physics",
       "Attitude tracking out of bounds", "姿态跟踪超出界限",
       "The closed-loop 6-DOF attitude/trajectory deviation exceeded a bound declared in the scenario "
       "config (settled error, gimbal saturation, or MECO-state deviation vs the reference).",
       "闭环 6-DOF 的姿态/轨迹偏差超出场景配置中声明的界限（稳定段误差、伺服饱和率，或 MECO 状态相对基准的偏差）。",
       "Re-tune the PID gains or slow the attitude program down; verify the initial attitude matches the "
       "program's first waypoint.",
       "重新整定 PID 增益或放缓姿态程序；确认初始姿态与程序第一个航点一致。"),
    _e("SIXDOF_MASS_MODEL_MISMATCH", "physics",
       "Mass properties off the declared model", "质量特性偏离声明模型",
       "The integrated mass / centre-of-gravity / inertia trace disagreed with the mass model declared "
       "in the scenario config beyond its tolerances: mass is not conserved against the integrated "
       "flow, or the full/burnout endpoints, or the in-flight interpolation, are off.",
       "积分得到的质量/质心/惯量与场景配置中声明的质量模型在容差范围内不一致：流量积分与质量变化不守恒，"
       "或满载/燃尽端点值、飞行中的插值不符合声明。",
       "Check the scenario's mass_model endpoints against the aircraft XML (tank contents, locations, "
       "dry mass/inertia), the integration step dt_s (the drain shows up one step late), and the "
       "tolerances; a genuine mismatch means the model and the vehicle file disagree.",
       "对照飞行器 XML 检查场景的 mass_model 端点（推进剂装填量与位置、干重/惯量）、积分步长 dt_s"
       "（排流滞后一拍）以及容差；确实不符则说明模型与飞行器文件不一致。"),
    _e("SIXDOF_BURNOUT_NOT_REACHED", "physics",
       "Burnout endpoint not assessed", "燃尽端点未评估",
       "The config declares a mass model with an empty (burnout) endpoint but the run never flew the "
       "propellant down to it, so the burnout CG/inertia endpoint could not be compared.",
       "配置声明了含燃尽（empty）端点的质量模型，但本次运行未把推进剂燃尽到该点，无法比较燃尽质心/惯量端点。",
       "Lengthen duration_s (or drop the ignition time) until the tanks run dry, or leave the burnout "
       "endpoint out of the config if the scenario is not meant to assess it.",
       "延长 duration_s（或提前点火）直到推进剂耗尽；如该场景本就不评估燃尽端点，则从配置中删去该端点。"),

    # --------------------------------------------------------- optimizer (FL-201+)
    _e("INFEASIBLE_PERIAPSIS_OUT_OF_TOLERANCE", "optimizer",
       "Optimizer could not hit periapsis", "优化器未能满足近地点",
       "The best design the optimizer found still misses the target periapsis band.",
       "优化器找到的最优方案仍未达到目标近地点范围。",
       "Give the optimizer more generations/population, widen decision bounds, or add vehicle performance.",
       "增加优化代数/种群规模、放宽决策变量范围，或提升飞行器性能。"),
    _e("INFEASIBLE_APOAPSIS_OUT_OF_TOLERANCE", "optimizer",
       "Optimizer could not hit apoapsis", "优化器未能满足远地点",
       "The best design the optimizer found still misses the target apoapsis band.",
       "优化器找到的最优方案仍未达到目标远地点范围。",
       "Give the optimizer more budget or widen the burn-time bounds.",
       "增加优化预算或放宽燃烧时间范围。"),
    _e("INFEASIBLE_INCLINATION_OUT_OF_TOLERANCE", "optimizer",
       "Optimizer could not hit inclination", "优化器未能满足倾角",
       "No design found reaches the target inclination — often the site latitude makes it unreachable "
       "without a plane change.",
       "未找到达到目标倾角的方案——往往是发射场纬度导致无法直接到达。",
       "Check that the inclination is at least the site latitude, or pick another site.",
       "确认目标倾角不小于发射场纬度，或更换发射场。"),
    _e("INFEASIBLE_RAAN_OUT_OF_TOLERANCE", "optimizer",
       "Optimizer could not hit RAAN", "优化器未能满足升交点赤经",
       "No design reaches the requested orbital plane from this liftoff epoch.",
       "从该起飞时刻出发无法到达要求的轨道面。",
       "Re-derive the liftoff epoch from the launch window for this RAAN.",
       "根据该升交点赤经的发射窗口重新确定起飞时刻。"),
    _e("INFEASIBLE_MAX_Q_EXCEEDED", "optimizer",
       "Optimizer could not respect max-Q", "优化器无法满足最大动压",
       "Every good design breaks the dynamic-pressure limit.",
       "所有可行方案都超过动压限制。",
       "Allow throttling in the decision vector, or relax the max-Q limit if the structure allows it.",
       "在决策变量中允许节流，或在结构允许时放宽最大动压限制。"),
    _e("INFEASIBLE_MAX_ACCEL_EXCEEDED", "optimizer",
       "Optimizer could not respect the g limit", "优化器无法满足过载限制",
       "Every good design breaks the acceleration limit.",
       "所有可行方案都超过过载限制。",
       "Allow end-of-burn throttle-down, or relax the g limit.",
       "允许燃烧末段节流，或放宽过载限制。"),
    _e("INFEASIBLE_PROPELLANT_RESERVE", "optimizer",
       "Optimizer could not keep the reserve", "优化器无法保留储备推进剂",
       "Reaching orbit uses up the propellant the target wants held in reserve.",
       "入轨消耗了目标要求保留的储备推进剂。",
       "Lower payload or the reserve requirement, or add propellant.",
       "降低载荷或储备要求，或增加推进剂。"),
    _e("INFEASIBLE_SIMULATION", "optimizer",
       "No candidate flew", "没有可飞行的候选方案",
       "Every candidate the optimizer tried failed to simulate; there is no flight to report.",
       "优化器尝试的所有候选方案都仿真失败，没有可报告的飞行结果。",
       "Check the base vehicle flies on its own (Simulate) before optimizing; narrow the decision bounds.",
       "优化前先确认基准飞行器可以单独仿真飞行；缩小决策变量范围。"),

    # --------------------------------------------------- recovery: boostback/entry
    _e("INSUFFICIENT_PROPELLANT_BOOSTBACK", "recovery",
       "Not enough propellant for boostback", "回飞点火推进剂不足",
       "The booster runs dry before the boostback burn has turned it around toward the landing site.",
       "助推级在回飞点火完成掉头飞向着陆场之前就已耗尽推进剂。",
       "Hold back more stage-1 propellant (earlier MECO / larger recovery reserve), land downrange instead "
       "of RTLS, or lighten the upper stage.",
       "为一级保留更多推进剂（提前一级关机/加大回收储备），改为海上/下靶场着陆而非返回发射场，或减轻上面级。"),
    _e("INSUFFICIENT_PROPELLANT_ENTRY_BURN", "recovery",
       "Not enough propellant for the entry burn", "再入点火推进剂不足",
       "After boostback the booster cannot afford the entry burn that slows it for re-entry.",
       "回飞后助推级无法负担用于再入减速的再入点火。",
       "Increase the recovery reserve, shorten boostback (downrange landing), or accept a lighter payload.",
       "增加回收储备、缩短回飞（改为下靶场着陆），或接受更轻的载荷。"),
    _e("INSUFFICIENT_PROPELLANT_RESERVE", "recovery",
       "Landing reserve used up", "着陆储备推进剂不足",
       "Boostback and entry burns leave less propellant than the landing burn needs.",
       "回飞和再入点火后剩余推进剂少于着陆点火所需。",
       "Raise the recovery reserve at MECO or trim the boostback/entry burns.",
       "提高一级关机时的回收储备，或缩减回飞/再入点火。"),
    _e("PROPELLANT_EXHAUSTED", "recovery",
       "Propellant exhausted", "推进剂耗尽",
       "A booster burn ended because the tanks ran dry, not because guidance commanded cutoff.",
       "助推级某次点火因推进剂耗尽而结束，而非制导指令关机。",
       "Increase the recovery reserve or shorten the burn that ran dry.",
       "增加回收储备或缩短耗尽推进剂的那次点火。"),
    _e("COMMANDED_CUTOFF", "recovery",
       "Commanded cutoff", "指令关机",
       "A burn ended on the guidance command (normal end of burn).",
       "点火按制导指令正常结束。",
       "Nothing to fix.", "无需处理。", "info"),
    _e("CORRIDOR_MISSED", "recovery",
       "Missed the entry corridor", "未进入再入走廊",
       "The booster reached the entry interface outside the corridor that leads to the landing site.",
       "助推级到达再入界面时偏离通往着陆场的再入走廊。",
       "Re-plan boostback (more burn time / better heading) or move the landing target.",
       "重新规划回飞（更长点火时间/修正航向），或调整着陆目标。"),
    _e("HANDOFF_STATE_DISCONTINUITY", "recovery",
       "Handoff state discontinuity", "交接状态不连续",
       "A recovery segment restarted from a state that is not the one the previous "
       "segment handed over: position, velocity or mass jump across the phase boundary.",
       "某回收段重启时的状态与上一段交接的状态不一致：相位边界处位置、速度或质量出现跳变。",
       "Check the segment boundary against the HandoffState digest; the producer or "
       "the consumer re-derived the initial state instead of using the handed-over one.",
       "对照交接摘要检查段边界：生产者或消费者擅自重新推导了初始状态，而非使用交接状态。"),
    _e("HANDOFF_MASS_NOT_CONSERVED", "recovery",
       "Mass not conserved across handoffs", "交接过程质量不守恒",
       "Mass bookkeeping broke in the recovery chain: mass rose on a flown track, "
       "changed during the ballistic coast between burns, or the chain ended below dry mass.",
       "回收链质量记账出错：飞行轨迹上质量上升、两次点火间滑行段质量变化，或链条终点低于干重。",
       "Inspect the mass column and the propellant flow of the phase the failing check names.",
       "检查失败检查所属相位的质量列与推进剂流量。"),
    _e("CORRIDOR_NOT_REACHED", "recovery",
       "Entry corridor not reached", "未到达再入走廊",
       "The booster's trajectory never reached the entry corridor altitude window.",
       "助推级轨迹始终未到达再入走廊的高度窗口。",
       "Check the separation state and boostback plan; a downrange landing may be the only feasible option.",
       "检查分离状态和回飞方案；可能只能选择下靶场着陆。"),
    _e("ENTRY_Q_LIMIT_EXCEEDED", "recovery",
       "Entry dynamic pressure too high", "再入动压过高",
       "During re-entry the booster sees more dynamic pressure than it is rated for.",
       "再入过程中助推级承受的动压超过其设计能力。",
       "Lengthen the entry burn to slow down before the dense atmosphere.",
       "延长再入点火，使其在进入稠密大气前充分减速。"),
    _e("ENTRY_G_LIMIT_EXCEEDED", "recovery",
       "Entry deceleration too high", "再入过载过高",
       "Aerodynamic deceleration during re-entry exceeds the booster's g limit.",
       "再入时气动减速过载超过助推级的过载限制。",
       "Lengthen the entry burn or steepen/shallow the entry to spread the deceleration.",
       "延长再入点火，或调整再入角以分散减速过载。"),
    _e("RECOVERY_INFEASIBLE", "recovery",
       "Recovery infeasible", "回收不可行",
       "The boostback/entry planner found no plan that brings the booster back.",
       "回飞/再入规划器未找到可使助推级返回的方案。",
       "Increase the recovery reserve or choose a downrange landing.",
       "增加回收储备或选择下靶场着陆。"),
    _e("RECOVERY_LANDING_NOT_ASSESSED", "recovery",
       "Recovery not assessed", "回收未评估",
       "The target asks for a recovery but the booster was never flown back (no recovery backend or an "
       "error), so the landing verdict is missing.",
       "目标要求回收，但助推级未进行回收飞行（缺少回收后端或出错），因此缺少着陆结论。",
       "Install the landing extras (envs/flightlab-landing.txt) and re-simulate, or set recovery to none.",
       "安装着陆相关依赖（envs/flightlab-landing.txt）后重新仿真，或将回收设为“无”。", "missing"),
    _e("RECOVERY_LANDING_NOT_ATTEMPTED", "recovery",
       "Landing burn not attempted", "未执行着陆点火",
       "Boostback/entry succeeded but no landing burn was flown.",
       "回飞/再入成功，但未执行着陆点火。",
       "Check the landing solver dependencies and the landing-burn start conditions.",
       "检查着陆求解器依赖和着陆点火的起始条件。", "missing"),

    # ---------------------------------------------------------- recovery: landing
    _e("LANDING_INFEASIBLE", "landing",
       "No feasible landing burn", "无可行着陆点火方案",
       "The powered-descent solver could not find any landing burn from the state the booster arrives in "
       "(too fast, too far off, or too little propellant).",
       "动力下降求解器无法从助推级到达时的状态找到任何着陆点火方案（速度过快、偏离过远或推进剂过少）。",
       "Arrive slower/closer (longer entry burn, better boostback) or keep more landing propellant.",
       "以更低速度/更近距离到达（延长再入点火、改进回飞），或保留更多着陆推进剂。"),
    _e("LANDING_FAILED", "landing",
       "Landing failed", "着陆失败",
       "The booster flew a landing burn but did not touch down inside the limits.",
       "助推级执行了着陆点火，但未在限制范围内触地。",
       "See the touchdown checks for which limit broke.",
       "查看触地检查，了解哪项限制被突破。"),
    _e("LANDING_NO_TOUCHDOWN", "landing",
       "No touchdown", "未触地",
       "The re-flown landing burn never reached the ground (it hovered out or fell past the horizon).",
       "复核的着陆点火始终未触地（悬停耗尽或越过地平线坠落）。",
       "Start the landing burn later / lower, or give it more thrust margin.",
       "推迟/降低着陆点火起始高度，或提高推力余量。"),
    _e("LANDING_POSITION_OUT_OF_TOLERANCE", "landing",
       "Missed the landing pad", "偏离着陆点",
       "The booster touched down farther from the target than the acceptance radius.",
       "助推级触地点距目标超过允许半径。",
       "Improve boostback targeting or allow more divert in the landing burn.",
       "改进回飞瞄准精度，或允许着陆点火有更大的横向机动。"),
    _e("LANDING_VERTICAL_SPEED_EXCEEDED", "landing",
       "Touchdown too hard (vertical)", "触地垂直速度过大",
       "Vertical speed at touchdown is above the soft-landing limit.",
       "触地时垂直速度超过软着陆限制。",
       "Keep more landing propellant or start the landing burn slightly higher.",
       "保留更多着陆推进剂，或稍微提高着陆点火起始高度。"),
    _e("LANDING_LATERAL_SPEED_EXCEEDED", "landing",
       "Touchdown sliding sideways", "触地横向速度过大",
       "Horizontal speed at touchdown is above the limit; the legs could fail.",
       "触地时水平速度超过限制，着陆腿可能损坏。",
       "Null horizontal velocity earlier in the landing burn (more divert time).",
       "在着陆点火中更早消除水平速度（留出更多机动时间）。"),
    _e("LANDING_TILT_EXCEEDED", "landing",
       "Booster tilted too far", "姿态倾斜过大",
       "During the landing burn the thrust vector tilted further from vertical than allowed.",
       "着陆点火过程中推力方向偏离竖直超过允许值。",
       "Arrive closer to the pad so less divert is needed, or relax the tilt limit if the vehicle allows.",
       "以更接近着陆点的状态到达以减少机动，或在飞行器允许时放宽倾角限制。"),
    _e("LANDING_PROPELLANT_RESERVE_LOW", "landing",
       "Landed with too little propellant", "着陆剩余推进剂不足",
       "Propellant left at touchdown is below the vehicle's landing reserve.",
       "触地时剩余推进剂低于飞行器的着陆储备要求。",
       "Raise the recovery reserve at MECO (costs payload) or trim boostback/entry burns.",
       "提高一级关机时的回收储备（会减少载荷），或缩减回飞/再入点火。"),
    _e("LANDING_MAX_G_EXCEEDED", "landing",
       "Landing burn g-load too high", "着陆点火过载过高",
       "Peak proper acceleration during the landing burn exceeds the vehicle limit.",
       "着陆点火期间峰值过载超过飞行器限制。",
       "Start the landing burn earlier at lower throttle (less aggressive suicide burn).",
       "以更低节流更早开始着陆点火（降低“自杀式点火”的激进程度）。"),
    _e("IMPACT_ELLIPSE_NOT_ASSESSED", "landing",
       "Impact ellipse not assessed", "未评估落点散布椭圆",
       "There were not enough touchdowns/impacts to estimate a dispersion ellipse.",
       "触地/落点样本不足，无法估计散布椭圆。",
       "Run the Monte Carlo with more samples.",
       "增加蒙特卡洛样本数后重新运行。", "missing"),

    # ----------------------------------------------------------- Monte Carlo (FL-304)
    _e("LANDED", "montecarlo",
       "Landed in limits", "在限制范围内着陆",
       "The sample touched down inside every landing limit.",
       "该样本在所有着陆限制范围内触地。",
       "Nothing to fix.", "无需处理。", "info"),
    _e("SAMPLE_ERROR", "montecarlo",
       "Monte Carlo sample crashed", "蒙特卡洛样本出错",
       "One dispersed sample raised an error and produced no verdict; it counts as a failure.",
       "某个散布样本运行出错，未产生结论，计为失败。",
       "Inspect the sample's error text in montecarlo.json; reduce extreme dispersions if they cause it.",
       "查看 montecarlo.json 中该样本的错误信息；如由极端散布引起，可适当缩小散布范围。"),
)

#: code (upper snake case) -> entry.
CATALOG: Mapping[str, ReasonEntry] = {entry.code: entry for entry in _ENTRIES}

#: Codes looked up by prefix (``SAMPLE_ERROR:ValueError`` -> ``SAMPLE_ERROR``).
_PREFIX_CODES = ("SAMPLE_ERROR",)

_NON_WORD = re.compile(r"[^A-Z0-9]+")


def normalize_code(code: str) -> str:
    """Upper snake case: ``max_q_exceeded`` / ``Max-Q exceeded`` ->
    ``MAX_Q_EXCEEDED``; a ``PREFIX:detail`` code keeps only its prefix."""
    text = str(code or "").strip()
    head = text.split(":", 1)[0] if ":" in text else text
    return _NON_WORD.sub("_", head.upper()).strip("_")


def lookup(code: str) -> ReasonEntry | None:
    """The catalog entry for ``code`` (case-insensitive), or None."""
    key = normalize_code(code)
    if not key:
        return None
    entry = CATALOG.get(key)
    if entry is None:
        entry = next((CATALOG[p] for p in _PREFIX_CODES if key.startswith(p)), None)
    return entry


def is_known(code: str) -> bool:
    return lookup(code) is not None


def explain(code: str) -> ReasonEntry:
    """Always an entry: the catalogued one, or a generic ``known=False`` entry
    that still shows the raw code (never raises on an unknown code)."""
    entry = lookup(code)
    if entry is not None:
        return entry
    raw = str(code or "").strip() or "(no reason code)"
    return ReasonEntry(
        code=raw, category="unknown",
        title_en=raw, title_zh=raw,
        explanation_en="This reason code has no plain-language entry yet.",
        explanation_zh="该原因代码尚无通俗解释。",
        fix_en="See the check's value and limit; add the code to orbitopt.flightlab.reasons.",
        fix_zh="请查看该检查的数值与限值；并将该代码补充到 orbitopt.flightlab.reasons。",
        known=False,
    )


def explain_all(codes: Iterable[str]) -> tuple[ReasonEntry, ...]:
    """Entries for ``codes`` in order, duplicates (after normalization) dropped."""
    seen: set[str] = set()
    out: list[ReasonEntry] = []
    for code in codes:
        if not code:
            continue
        entry = explain(code)
        key = entry.code if entry.known else normalize_code(code)
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
    return tuple(out)
