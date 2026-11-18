---
title: Quickstart
nav_order: 2
description: Install orbitopt, open a mission and run a configuration.
---

# 快速启动

## 安装

克隆寄存器并创建全计算堆栈的锁定环境. 包括央视,pykep,pygmo和tudatpy.

```bash
git clone https://github.com/zangjiucheng/orbitopt.git
cd orbitopt
micromamba create -n orbitopt -f envs/cpu.conda-lock.yml
micromamba activate orbitopt
pip install -e . --no-deps
```

在 Linux 或 Windows 上使用 `envs/gpu.conda-lock.yml`为 CUDA. 在Windows上,`.\scripts\setup_env.ps1`也创造了环境;为CUDA添加`-Gpu`. macOS使用CPU锁. 平台和可选依赖关系请参见[environment details](https://github.com/zangjiucheng/orbitopt/blob/main/envs/README.md).

对于没有计算堆栈的场景文件,从检出处安装`pip install -e .`;使用`pip install -e '.[viewer]'`来添加查看器. Python 3.10 – 3.12得到支持;锁定的计算环境使用Python 3.10.

## 打开任务

```bash
orbitopt                   # Mission Control
orbitopt missions          # list mission IDs
orbitopt view mars         # single-scene viewer
orbitopt export mars -o mars.json
```

内设场景包括`solar-system`,`artemis2`,`goes`,`mars`和`starship-flight-15`. 见[Mission Control guide](guide.md)或[Flight Lab guide](flightlab.md).

## 任务配置

运行前验证 YAML 任务 :

```bash
orbitopt validate examples/mars2020_replay.yaml
orbitopt run examples/mars2020_replay.yaml
orbitopt view examples/mars2020_replay.yaml
```

`validate` checks the schema without running a simulation. `run` computes and
exports the scene and reports; `view` opens a cached or newly computed result.
The supported kinds are `geo-raising`, `mars-transfer`,
`lunar-free-return`, `flight-lab` (a Flight Lab ascent run through the
pipeline; see `src/orbitopt/data/flightlab/mission_leo_two_stage.yaml`) and
`two-stage-leo` (the reference scenario below).
Each has a schema in
[`src/orbitopt/schemas/`](https://github.com/zangjiucheng/orbitopt/tree/main/src/orbitopt/schemas/) and example configurations
in [`examples/`](https://github.com/zangjiucheng/orbitopt/tree/main/examples/). Omitted optional fields use the corresponding
built-in defaults.

要计算每日发射窗口:

```bash
orbitopt launch-window ksc --rla 9.6 --dla 23.4 --epoch 2020-07-30
```

对所有选项使用 `orbitopt --help`或 `orbitopt <command> --help`.

## 发射到轨道

`orbitopt launch`运行一个发射端到尾端:发射窗口从一个地点进入轨道平面,Flight Lab飞行的上升段优化,轨道注入分级对准目标,以验证的一份报告结尾. 它需要绑定的OpenTsiolkovsky二进制(`python scripts/fetch_opentsiolkovsky.py`).

```bash
orbitopt launch --site cape-canaveral --vehicle synthetic_two_stage \
  --orbit 400x400@51.6 --date 2026-11-14 -o launch/
```

输出目录持有目标,上升段运行和`launch.verification.json`. 添加`--raan`以限制轨道平面(飞机内通过后设定升降时间). 简短的搜索(`--generations`,`--population`)可能不会汇合;报告说不是通过,而是通过.

从发射场到中转和轨道插入到铺设的停放轨道的两阶段LEO参考情景将整个链条分为:

```bash
orbitopt run src/orbitopt/data/mission_two_stage_leo.yaml
```

区段及其交接门请参见[Flight Lab](https://github.com/zangjiucheng/orbitopt/blob/main/docs/flightlab.md#mission-pipeline).

计算-上升段 RTLS(需要[landing environment](https://github.com/zangjiucheng/orbitopt/blob/main/envs/README.md)):

```bash
python scripts/e505_reference.py verify
python scripts/e505_reference.py run --out runs/e505
python scripts/e505_reference.py compare runs/e505
```

此重播所记录的计算 上升段 并重播 助推器 恢复 。 参照线在51.0°处达到265.5×278.1公里;在此单环线运行中,停车轨道传播仍为`not_checked`. [computed mission workflow](computed-mission.md)增加了局部取景和跨环境轨道传播指令. E2-04仍为[synthetic integration fixture](https://github.com/zangjiucheng/orbitopt/blob/main/docs/flightlab-regression-ci.md#e2-04-the-frozen-dual-branch-reference-mission-task-257),而非计算参考.

## 验证

一份验证报告记录了动态,集成设置,检查和残留. 一个通过的结果取决于模型和容忍度;它没有确立飞行安全。 GOES是**未检查**在第一个负载:从app的验证标签中运行它的验证,或者在配置中启用它. 当没有候选人通过时,火星运行可能会输出**最佳成绩**结果.

每个检查标注为数值一致性,模型一致性或历史数据验证;`orbitopt benchmarks`对照分析溶液和公布的数据检查物理. 见[verification](verification.md).

关于历史比较和定点检查的区别,见[examples and validation](examples.md);关于报告和管道细节,见[engineering notes](engineering-notes.md)。
