---
title: Engineering
nav_order: 7
has_children: true
description: Implementation notes, numerical benchmarks and regression tests.
---

# 工程学

- [Engineering notes](engineering-notes.md):数字方法和管道行为.
- [Compute worker lifecycle](mission-control-worker-lifecycle.md):工艺保障和测试.
- [CUDA kernel benchmark](benchmarks/fused_kernel_evaluation.md):测量和后端决定.
- [GPU regression testing](gpu-regression/README.md):程序与记录运行.
- [Reference tolerance audit](reference-tolerance-audit.md):EOP披针,比较预算和来源.
- [E1–E4 acceptance review](acceptance-e1-e4.md):经过测试的审校,解决了调查结果和证据限度.

## 工作流程触发器

| 变动 | 自动工作流程 |
|---|---|
| `docs/`的英语Markdown | 页面构建与翻译草稿. |
| README | 翻译稿. |
| `translations/`草案 | 不进行计算测试或重复翻译。 |
| 运行时代码、测试或计算配置 | CPU CI; GPU CI still requires its PR label or a manual run. |

CI跳过仅文档的推和PR. `docs/benchmarks/`下的基准JSON仍在测试中,因为一些套件将其读作回归输入. 页面也会在其来源/API输入值变化时重建;其脚本,模型要求或工作流程变化时翻译会重新运行. 无论路径改变,手动运行仍然可用。
