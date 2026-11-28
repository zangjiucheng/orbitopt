# 任务组合

一个会合任务是一个YAML文件,在Schema版本的`1.0`. 参考文献为`src/orbitopt/data/leo_rdv_001.yaml`;其值为任务接受数,固定在任何优化运行之前,而不是运行时发现的金枪鱼。

```bash
orbitopt rendezvous studio --config PATH   # any file this schema
```

## 任务和动态

| 密钥 | 含义 |
|---|---|
| `mission.id` / `title` / `kind` | 身份; `kind: rendezvous-two-impulse`. |
| `dynamics.frame` | 各州惯性框架——`GCRS`. |
| `dynamics.mu` | Earth μ. The loader demands the exact packaged value 3.9860044150×10¹⁴ m³/s²; the Lambert solver's solar-system default can never leak in. |
| `dynamics.j2` | 计划并核实模型-J2上. |

## 轨道和几何

| 密钥 | 含义 |
|---|---|
| `target.*` | 酋长:海拔,轨道倾角,RAAN,纬度论证. |
| `chaser.altitude_m` | 副高度;低于目标,所以漂移沿V-巴前进. |
| `chaser.initial_separation_m` | 分离于共享的历元 t 0(coplana in v 1). |
| `hold_point.rel_lvlh` | LVLH offset of the hold point; arrival means hovering here with zero relative velocity. |

## 预算、容忍度、窗口

| 密钥 | 含义 |
|---|---|
| `budget.delta_v_budget_m_s` | 总QQv上限,优化前固定. |
| `tolerances.max_position_error_m` / `max_relative_speed_m_s` | 停靠点的着陆门(50 m/0.2 m/s for LEO-RDV-001). |
| `windows.drift_max_s` | 出发门必须在一个漂移阶段穿过。 |
| `windows.transfer_tof_min_s` / `max_s` | 飞行时间窗口,优化前固定. |
| `phasing.departure_gate_y_m` | V-的酒吧门 转移离开。 |
| `optimizer.*` | 牛顿针对J2校正的耐受性/穿透帽. |

## 安全区块(可选、添加剂)

一个配置可能携带一个`safety`区块([ADR rendezvous-0002](adr/rendezvous-0002-safety-constraints-whole-trajectory.md));它的存在将舟子规划器从`plan_rendezvous`改为`plan_rendezvous_safe`,目标在限制范围内:

| 密钥 | 含义 |
|---|---|
| `keep_out_radius_m` | 在目标周围保持球体 检查每个样本 |
| `cross_track_max_m` | 走廊被捆绑在越轨外游. |
| `cone_half_angle_deg` / `cone_range_m` | 终端圆锥禁止停靠点附近前行. |
| `approach_speed.range_knots_m` / `cap_m_s` | 范围依赖性相对速度天花板,片面线性. |
| `passive_contingency_s` | No-thrust coast from t0, gate and hold point must stay clear. |
| `check_sample_dt_s` | 强度检查网格; 燃烧时间总是被采样. |

## 强健运动

`src/orbitopt/data/leo_rdv_001_robustness.yaml`是`orbitopt.rendezvous.robustness`消耗的一个单独的运动计划(压力武器,集束,阻断),它宣布[verification report](rendezvous-verification-report.md)强数背后的散射研究,不是任务配置.
