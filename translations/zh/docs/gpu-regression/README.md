# CUDA 本地 GPU 服务器上的回归测试

默认的CI(`.github/workflows/ci.yml`)和开发平台没有CUDA硬件. 在那些机器上,每个`gpu`的标记测试是**跳过**. CUDA路径在工程的本地GPU服务器上进行人工检查,使用以下程序. 每次运行都会在[`runs/`](https://github.com/zangjiucheng/orbitopt/tree/main/docs/gpu-regression/runs/)产生记录,这是承诺的.

Acceptance (task #165): a complete record exists for commit `afa0cd9` or a
later one, and every failure in it is attributed.

## 范围

| 区域 | CUDA案件 | CPU参考(同代码,NumPy). |
|---|---|---|
| 兰伯特 | `tests/test_gpu_lambert.py::test_batch_matches_pykep_reference[True]`, `::test_cpu_and_gpu_backends_agree` | rest of `tests/test_gpu_lambert.py` |
| N-体 | `tests/test_nbody_gpu_backend.py`: `test_kepler_batch_*`, `test_moon_and_sun_state_*`, `test_nbody_history_cpu_and_gpu_agree[*]` | `tests/test_nbody_ephemeris_precompute.py` |
| 最接近的方法 | `tests/test_nbody_gpu_backend.py`: `test_closest_approach_cpu_and_gpu_agree[*]`, `test_gpu_closest_approach_matches_gpu_history`, `test_fused_cupy_raw_*`, `test_fused_numba_cuda_*` | `tests/test_nbody_closest_approach.py`, `tests/test_nbody_fused.py` |
| 设备报告 | `tests/test_runner_device.py::test_cupy_backend_reports_gpu` | rest of `tests/test_runner_device.py` |

N-体例和最接近方法的个案比较`use_gpu=True`与`use_gpu=False`,比较被保险的`cupy-raw`/`numba-cuda`内核与NumPy生产路径. 宽容度约为1e-9相对. 最接近的阶梯指数必须精确匹配. SPICE种子状态注入缓存,因此不需要内核.

## 程序

```bash
git fetch && git checkout <commit to test>        # afa0cd9 or later
micromamba create -n orbitopt-gpu -f envs/gpu.conda-lock.yml   # once; see envs/README.md
micromamba activate orbitopt-gpu
python -m pip install -e . --no-deps
nvidia-smi                                         # sanity: driver loaded, GPU visible

python scripts/gpu_regression.py                   # writes docs/gpu-regression/runs/<UTC stamp>-<sha7>.*
```

剧本运行两个 pytest 通过以上范围:

1. `cuda`:`-m gpu`与`ORBITOPT_REQUIRE_GPU=1`. 如果没有可见的设备,
   此通过是一个硬错误( exit 2) , 而不是每个案件跳过的地方 。
2. `cpu-reference`:`-m "not gpu"`,NumPy指代相同区域.

剧本写这些文件:

| 文件 | 内容 |
|---|---|
| `<stem>.md` | **记录**:环境,个案结果和持续时间,故障日志,归属占位符 |
| `<stem>.env.json` | 相同的环境数据,机器可读 |
| `<stem>-cuda.log`, `<stem>-cpu-reference.log` | 完整 pyst 输出 |
| `<stem>-*.junit.xml` | J Unit XML, 个案时间的来源 |

记录按任务要求,记录如下:

- GPU model (from `nvidia-smi` and from CuPy device 0, with compute capability)
- 驱动程序版本,以及 CUDA版本:由驱动程序支持的来自 `nvidia-smi`的版本,加上 CUDA驱动程序 API和 CuPy报告的运行时间版本.
- CuPy版本(亦为麻木,NumPy和Python).
- 如果工作树是脏的,用旗子测试
- 每个案件及其结果和持续时间
- 每个失败或错误的大小写的失败日志

额外的 pytest 参数在 `--` 之后, 例如 `python scripts/gpu_regression.py -- -x --maxfail=3` 。

## 将失败归罪于人

记录中每个失败或错误的案件都有`**Attribution:** _TODO_`线. 对于CUDA案例,该行还显示匹配的CPU参考测试的结果. 将`_TODO_`替换为其中之一,并说明原因:

| 归属 | 典型证据 |
|---|---|
| **GPU backend divergence** | CUDA案件失败,CPU参考证,不匹配程度远高于有记录的容忍度. |
| **Environment / driver** | CuPy导入或`cudaErrorInsufficientDriver`,NVRTC /JIT编译错误,出自内存,或麻木-古达工具包不匹配. 修复主机后重运行 |
| **Test tolerance** | 不匹配仅略高于耐受性,来自FMA或`libdevice`四舍五入. 显示已观察到的错误 |
| **Not GPU-specific** | CPU 引用失败 。 追踪为普通错误 |

打开一个问题 对于每一个分歧和每个回归 而不是GPU-的特异性, 并将其与记录联系起来。

## 提交记录

提交 `<stem>.md`和 `<stem>.env.json`, 并提交 `.log` 失败时的文件 。 然后在下面的运行索引中添加一行.

## 运行索引

| 日期(UTC) | 提交 | GPU | CUDA / CuPy | 结果 | 记录 |
|---|---|---|---|---|---|
| _待决:首次在本地GPU服务器_运行. | | | | | |

## 自动化( 可选, 优先级较低)

`.github/workflows/gpu.yml`在自备跑者身上运行相同的脚本,标签为`[self-hosted, linux, x64, gpu]`. 您可以手动启动( Actions → GPU CI → 运行工作流程) 或将 `gpu-ci` 标签添加到 PR 中 。 记录目录作为`gpu-regression-record`文物上传. 在跑者注册之前,请使用上面的人工程序. 关于跑步注册,见GitHub的*增加自备跑车*指南,并加入自定义标签`gpu`.
