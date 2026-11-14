# Fused CUDA kernel evaluation and CPU/GPU benchmark (task #143)

本研究比较了麻风病预估后最接近的筛选后端. 它建议安装一个已引信的Numba CPU后端,并推迟执行已引信的CUDA。 下文的测量是针对所述主机的;本研究没有测量CUDA计时.

## 重现

```bash
pip install numba jax  # optional; absent backends are skipped
PYTHONPATH=src python scripts/bench_nbody_backends.py --n 2000 --steps 300 --json out.json
PYTHONPATH=src python scripts/bench_nbody_backends.py --n 2000 --steps 300 --sun
PYTHONPATH=src python scripts/bench_nbody_backends.py --n 20000 --steps 300
```

输入使用固定的种子和缓存的月球/太阳状态. 计时是热身后中位数;第一个调用包括JIT编译. CUDA 后端仅在设备存在时进行测量. 协议检查见`tests/test_nbody_fused.py`,模拟检查见CUDA-.

## 结果(CPU-只主机)

英特尔 i 5 -8250 U, 4 C/8 T, Python 3.13, NumPy 2.x, Numba 0.68.1和 JAX 0.12; 运行到运行的时间大约为 ± 30%. 原始结果在本目录中为`nbody_backends_cpu_*.json`.

| 候选人x步骤 | 太阳 | NumPy | Numba CPU | JAX CPU |
|---|---|---|---|---|
| 2,000 × 300 | 无 | 0.89 s | 0.044 s | 0.21 s |
| 2,000 × 300 | 对 | 1.50 s | 0.059 s | 0.28 s |
| 20,000 × 300 | 无 | 7.61 s | 0.27 s | 1.47 s |

For 2,000 candidates, Numba spent about 0.022 s in host preparation and
0.019 s in the fused loop. The median relative error against NumPy was about
1e-15; larger differences mostly occurred on trajectories passing through
Earth, close to the singularity.

## 简介:GPU时间往何处走,现在该往何处走.

矢量化的2,000 × 300案在麻风病预计算前发布估计4.5百万阵列操作,之后发布24,762(34,514与太阳). 引信内核使用一次发射加准备和转让。 推算早先的发射费用,对更新后的CuPy路径大约为0.6;**这是估计,不是GPU测量**.

## 后端比较

Numba CUDA共享CPU执行,并支持模拟测试;CuPy RawKernel需要单独CUDA C. JAX在这CPU上比Numba更慢,增加了更大的依赖性. 在选择GPU后端之前,重新测量目标GPU.

## 结论

在量测的批量尺寸上,CPU聚变已经移除了大部分的间接费用和主机准备有限的潜在GPU节约. 如果批量正常超过 100,000 或 2,000 步骤,或者如果麻黄素准备移到设备上,则重新检查 CUDA.

## Production integration (task #164)

`FreeReturnScreeningProblem`支持`auto`,`numpy`,`numba-cpu`和`cupy`. `auto`在明确请求使用GPU时选择CuPy;否则它可以使用Numba进行最接近的批次,至少是64,在适当时会回落到NumPy. Numba的缓存内核避免了后期过程中的重编. `OptimizationResult.backend`和`backend_usage`报告使用的后端.

### 端到端时间

```bash
PYTHONPATH=src python scripts/bench_free_return_backends.py --json out.json
PYTHONPATH=src python scripts/bench_free_return_backends.py --pop 2000 --generations 5 \
  --repeat 1 --backends numpy,numba-cpu
```

Warm runs on an 8-logical-CPU host, 300 steps, six-day coast, no Sun; raw JSON
is in `free_return_backends_cpu_*.json`.

| 人口x几代人 | NumPy PSO | 农巴 CPU PSO | 加速 |
|---|---|---|---|
| 256 × 25 | 3.6 s | 0.21 – 0.23 s | ~16× |
| 2,000 × 5 | 5.2 s | 0.21 s | ~24× |

一个新的过程支付大约1 s一个温暖的JIT缓存,或者9 s一个冷编译.

### 适身协议

上层候选人在这些批次中同意NumPy至0.015公里或更高. 地表以下的轨迹离重力单数很近;这可以改变PSO领带断裂和最终冠军. 后端回归测试使用相对耐受性1e-9;它们不提出地球间切的准确性要求轨迹.
