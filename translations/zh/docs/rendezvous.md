# 约会

轨道对流计划 并核实两层轨道对接 在**LEO-RDV-001**参考任务中,追逐者开始于490公里高度,并转移至1公里后面的500公里轨道上的目标的牵引点.

| 页次 | |
|---|---|
| [Quickstart](rendezvous-quickstart.md) | 开启预计算运行,计划并验证任务,重新运行独立复飞. |
| [Theory](rendezvous-theory.md) | 相对运动,CW规划信封,对轨道传播和安全限制. |
| [Mission configuration](rendezvous-mission-config.md) | 集合配置 v 1.0 计划,逐个战区. |
| [Verification report](rendezvous-verification-report.md) | 名义上,独立而有力的证据——单独标注,附有文摘. |

每一层背后的决定是:[relative-0001](adr/relative-0001-lvlh-relative-frames-cw-and-pair-propagation.md)(LVLH帧,CW信封,配对轨道传播),[rendezvous-0001](adr/rendezvous-0001-two-impulse-j2-targeted-lambert.md)(规划者),[rendezvous-0002](adr/rendezvous-0002-safety-constraints-whole-trajectory.md)(安全),[rendezvous-0003](adr/rendezvous-0003-independent-refly-tudatpy.md)(独立复飞),[rendezvous-0004](adr/rendezvous-0004-mission-studio-flightlab-replay.md)(studio replay),[rendezvous-0005](adr/rendezvous-0005-dispersion-campaign-not-a-safety-factor.md)(分散运动). [E12-10 re-flight ADR](adr/rendezvous-0006-independent-robustness-reflight.md)描述了对选定的分散案例的独立检查.

`examples/runs/leo-rdv-001`下有LEO-RDV-001的预计算任务工作室运行;[quickstart](rendezvous-quickstart.md)显示如何在没有计算堆栈的情况下打开.

## 闭路导引(E13).

E13增加相对导航,加长卡尔曼滤波器,反馈引导,限制控制和安全监督. [GNC run contract](gnc-contract.md)定义了共享时钟和真实,观察,估计,命令和执行控制事件的记录.

| 录音研究 | 在其申报条件下的结果 |
|---|---|
| [Feedback guidance](benchmarks/e13_04_closed_loop_guidance.md) | **258/264**搁置案通过反馈,**13/264**通过同样抽取的公开循环. |
| [Constrained control](benchmarks/e13_05_constrained_control.md) | The baseline, LQR and MPC passed **258**, **260** and **255** of 264 cases. The paired comparisons found no statistically significant difference. |
| [Safety supervisor](benchmarks/e13_06_safety_supervisor.md) | 注入的导航故障臂没有错过247的真相侵犯. **155/264**破坏后检查通过. |

这些都是模拟结果 对于宣布的压力病例。 E12封存的参考文献及其独立证据仍载于[verification report](rendezvous-verification-report.md).
