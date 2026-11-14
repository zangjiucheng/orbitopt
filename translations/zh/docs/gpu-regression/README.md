# CUDA 在本地 GPU 服务器上的回归测试

默认的CI (`.github/workflows/ci.yml`) 开发平台没有CUDA硬件. 在这些机器上,`gpu`的标记测试是**跳过**. CUDA路径在工程的本地GPU服务器上进行手取检查,使用以下程序. 每一次运行都会产生[`runs/`](https://github.com/zangjiucheng/orbitopt/tree/main/docs/gpu-regression/runs/)的纪录,这是承诺的.

Acceptance (task #165): a complete record exists for commit `afa0cd9` or a
later one, and every failure in it is attributed.

## 范围

| 区域 | CUDA案件 | CPU参考(同代码,NumPy. |
|---|---|---|
| 兰伯特 | `tests/test_gpu_lambert.py::test_batch_matches_pykep_reference[True]`, `::test_cpu_and_gpu_backends_agree` | rest of `tests/test_gpu_lambert.py` |
| N-体 | `tests/test_nbody_gpu_backend.py`: `test_kepler_batch_*`, `test_moon_and_sun_state_*`, `test_nbody_history_cpu_and_gpu_agree[*]` | `tests/test_nbody_ephemeris_precompute.py` |
| 最接近的方法 | `tests/test_nbody_gpu_backend.py`: `test_closest_approach_cpu_and_gpu_agree[*]`, `test_gpu_closest_approach_matches_gpu_history`, `test_fused_cupy_raw_*`, `test_fused_numba_cuda_*` | `tests/test_nbody_closest_approach.py`, `tests/test_nbody_fused.py` |
| 设备报告 | `tests/test_runner_device.py::test_cupy_backend_reports_gpu` | rest of `tests/test_runner_device.py` |

N-体例和最接近方法的个案比较了`use_gpu=True`与`use_gpu=False`;比较了被保险的`cupy-raw`/`numba-cuda`内核与NumPy生产路径. 宽容度约为1e-9相对. 最接近的阶梯指数必须精确匹配. SPICE种子状态被注入缓存,所以不需要内核.

## 程序

```bash
git fetch && git checkout <commit to test>        # afa0cd9 or later
micromamba create -n orbitopt-gpu -f envs/gpu.conda-lock.yml   # once; see envs/README.md
micromamba activate orbitopt-gpu
python -m pip install -e . --no-deps
nvidia-smi                                         # sanity: driver loaded, GPU visible

python scripts/gpu_regression.py                   # writes docs/gpu-regression/runs/<UTC stamp>-<sha7>.*
```

剧本通过两个 pytest 通过以上范围:

1. `cuda`:`-m gpu`与`ORBITOPT_REQUIRE_GPU=1`. 如果没有可见的设备,
   此通过是一个硬错误( exit 2) , 而不是每个案件跳过的地方 。
2. `cpu-reference`:`-m "not gpu"`,NumPy指同域.

剧本写这些文件:

| 文件 | 内容 |
|---|---|
| `<stem>.md` | **记录**:环境,个案结果和持续时间,故障日志,归属占位符 |
| `<stem>.env.json` | 相同的环境数据,机器可读 |
| `<stem>-cuda.log`, `<stem>-cpu-reference.log` | 完整 pytest 输出 |
| `<stem>-*.junit.xml` | J Unit XML, 个案计时来源 |

The record captures the following, as the task requires:

- GPU 模型(出自`nvidia-smi`和从CuPy设备出自0,并具有计算能力).
- 驱动程序版本,和 CUDA版本:由驱动程序支持的由 `nvidia-smi`版本,外加 CUDA驱动程序 API和由 CuPy所报道的运行时间版本.
- CuPy版本(同为麻木,NumPy和Python.
- 如果工作树是脏的,用一面旗帜
- 每一案件及其结果和持续时间
- 每个失败或出错的大小写的失败日志

附加 pytest 参数在 `--`后,例如 `python scripts/gpu_regression.py -- -x --maxfail=3`.

## 将失败归责

记录中每个失败或出错的案件都有`**Attribution:** _TODO_`行. 对于CUDA案例,该行还显示了匹配的CPU参考测试的结果. 将`_TODO_`替换为其中之一,并说明原因:

| 归属 | 典型证据 |
|---|---|
| **GPU backend divergence** | CUDA 案件失败,CPU 参考证,不匹配程度远远高于记录的容忍度. |
| **Environment / driver** | CuPy导入或`cudaErrorInsufficientDriver`,NVRTC/JIT编译出错,出自内存,或麻木-库达工具包不匹配. 修复主机后重新运行 |
| **Test tolerance** | 不匹配仅略高于耐受性,来自FMA或`libdevice`四舍五入. 显示已观察到的错误 |
| **Not GPU-specific** | CPU 引用失败的方式相同. 追踪为正常错误 |

打开一个问题 每一个分歧和每一个回归 而不是GPU-,并将其与记录联系起来。

## 提交记录

提交 `<stem>.md`和 `<stem>.env.json`, 并在失败时提交 `.log`文件 。 然后在下面的运行索引中添加一行.

## 运行索引

| 日期 (UTC). | 提交 | GPU | CUDA / CuPy | 结果 | 记录 |
|---|---|---|---|---|---|
| 2026-11-14 | `eadd301` | NVIDIA L40S (cc 8.9), driver 610.57.04 | CUDA 13.0 runtime / 13.3 driver API, CuPy 13.6.0 | cuda 14通过; 2跳出; cpu-引用 35通过; 6跳出; 0失败. 跳伞是"麻木"/"麻木-库达"/"JAX"内核(没有被锁入的"GPU env"中)和"无-GPU"倒置(一个"GPU"在场). | [record](runs/20261114-000425-eadd301.md) |

## 自动化( 可选, 低优先级)

`.github/workflows/gpu.yml`在自备跑者上运行相同的脚本,标签为`[self-hosted, linux, x64, gpu]`. 您可以手动启动( Actions → GPU → CI → 运行工作流程) 或将 `gpu-ci` 标签添加到 PR 中 。 记录目录被上传为`gpu-regression-record`文物. 在一名跑者登记之前,请使用上述人工程序。 关于跑道注册,见GitHub's *增加自备跑道员*指南并加入自定义标签`gpu`.
