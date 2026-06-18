# DeepGEMM MXFP4×MXFP4→fp32 dense GEMM — 稳态基准(随机数据)

`bench_mxfp4_stable.py`:对 `../my_test/shape.txt` 的 64 个 shape 测 DeepGEMM 的 MXFP4 dense kernel
(`sm100_fp4_gemm`,经 `deep_gemm.fp8_fp4_gemm_nt`)的**降频稳态**吞吐。落实了 `bench.md` 的结论。

## 口径(为什么这样测)
- **随机真实数据**:`torch.randn` 量化成 MXFP4(`generate_normal`),高熵、有符号 —— 不是低熵合成数据。
  FP4 功耗与数据相关,低熵数据会虚高(见 `bench.md`)。
- **稳态(关键)**:不测前 ~2s,只 run 让 SM 降到稳态,再在稳态下多次取平均。用 `bench_kineto` 实现:
  把 `num_tests` 调到让每一轮 ≈ `TARGET_ROUND_S`(2s)—— warmup 轮(不计时)把 SM 跑到稳态,
  active 轮(计时)整轮都在稳态。等价于"显式 1.5s warmup + bench_kineto"(已验证)。
  对比:16384³ 上 plain bench_kineto(num_tests=30)= 6369 TFLOPS(瞬态虚高),本口径 = ~5800(稳态)。
- **flush_l2**:A+B 操作数 < 50MB 的小 shape 开 flush(小 shape 不降频,flush 避免操作数 L2 常驻虚高);
  ≥ 50MB 的大 shape 关 flush(连续跑保持降频稳态)。
- **C 不需要**:`accumulate=False` → `c=None`。
- **D = fp32**:FP4×FP4 kernel 不支持 bf16 写回。
- **sm_mhz / power_w**:NVML 后台 1ms 采样,取 bench_kineto 窗口内中位数(窗口 ≥2s,稳态占多数 →
  中位数即稳态;窗口 ≥2s 也让 NVML 的 ~1s 滑动平均功率读数稳定下来)。

## 输出
`result/mxfp4_dense_stable.csv`,列:`m,n,k,us,tflops,gbps,sm_mhz,power_w,backend`。
`us` 是单次 kernel 的稳态平均时间;backend=`deepgemm_mxfp4_fp32`。所有 shape 精度 diff=0.0134(< 0.02 通过)。

## scale kernel = 隐藏的 breather(为什么要预打包)
DeepGEMM 每次 `fp8_fp4_gemm_nt` 默认发 1 个 GEMM + 2 个 `transpose_and_pack_fp32_into_ue8m0` scale
kernel。这俩 scale kernel 又小又偏访存/低功耗,**会在连续 GEMM 之间充当"喘息"**:让 SM 时钟恢复、
还把 NVML 功率均值拉低。结果是**降频边界附近的 shape 被测成假的"2062 / 低功耗"**。预打包(把 scale
移出计时区间)后,GEMM 真正背靠背连续跑,这些 shape 才露出真实稳态:

| shape | 旧:cast-inside(有 scale breather) | 新:预打包(纯 GEMM) |
|---|---|---|
| 2048×4096×16384 | 2062 / 233W / 5843 | **1627 / 911W / 5246** |
| 2048×8192×8192 | 2062 / 233W / 5346 | **1732 / 924W / 4894** |
| 8192×2048×8192 | 2062 / 233W / 5310 | **1642 / 851W / 4939** |

所以预打包后的 CSV 更准。`result/mxfp4_dense_stable.castinside.csv` 是旧版本,保留作对比。
(全表中位偏差仅 1.9%,只有这几个边界 shape 变化大。)

## 结果概览(预打包后)
- 真正不触墙的小/极薄 shape:`sm_mhz=2062`(确实抽不到 1200W 墙)。
- 触 1200W 墙的 shape:降频到 `sm ~1267–1730`,功率 ~850–1178W,稳态 ~4900–5900 TFLOPS。
  16384³ ≈ 5664 TFLOPS @ 1342 MHz;32768³ ≈ 5700 @ ~1270 MHz。

## 运行
```
cd DeepGEMM/tests/test_stablesm_rand
LD_LIBRARY_PATH=/opt/hpcx/ucx/lib \
PYTHONPATH=/root/workspace/gb_gemm_benchmark/DeepGEMM:/root/workspace/gb_gemm_benchmark/DeepGEMM/tests \
python -u bench_mxfp4_stable.py
```
注意:跑前确认 GPU 空闲(并发进程会抢 SM、污染稳态数字)。
