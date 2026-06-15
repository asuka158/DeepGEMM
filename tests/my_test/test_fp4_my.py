"""
MXFP4 x MXFP4 稠密 GEMM 性能测试（复现 PR #348 的稠密表格）。

基于 tests/test_fp4.py 里的 test_gemm()。对 shape.txt 中的每个 shape：
  1. 生成 bf16 输入，把两个操作数都量化成 MXFP4（E2M1 + VS=32 的 UE8M0 scale），
  2. 通过 deep_gemm.fp8_fp4_gemm_nt 运行 SM100_MMA_MXF4_SS 稠密 kernel，
  3. 与 fp32 参考结果做精度对比（calc_diff < 0.02），
  4. 用 bench_kineto 计时，
最后把结果写到 tests/result/fp4_dense_gemm_bench.csv。

fp8_fp4_gemm_nt 只是统一的 API 入口：这里两个操作数都是 packed FP4，所以
csrc/apis/gemm.hpp 会分发到纯 FP4xFP4 的 kernel（sm100_fp4_gemm_1d1d），而不是
混精的 FP8xFP4。FP4xFP4 路径没有 bf16 epilogue（API 断言 D 必须是 fp32），所以
输出 D 是 fp32 —— 该 kernel 暂不支持 bf16 写回。

运行方式（LD_LIBRARY_PATH 前缀用于修复容器里 torch/libucs 的导入问题）：
  cd /root/workspace/gb_gemm_benchmark/DeepGEMM/tests/my_test
  LD_LIBRARY_PATH=/opt/hpcx/ucx/lib PYTHONPATH=/root/workspace/gb_gemm_benchmark/DeepGEMM python test_fp4_my.py
"""
import csv
import random
import sys

import torch

_REPO_DIR = '/root/workspace/gb_gemm_benchmark/DeepGEMM'
_TESTS_DIR = _REPO_DIR + '/tests'
_SHAPE_PATH = _TESTS_DIR + '/my_test/shape.txt'
_CSV_PATH = _TESTS_DIR + '/result/fp4_dense_gemm_bench.csv'

# 让 `deep_gemm` 和公共测试模块 `generators` 可被导入。
for _p in (_REPO_DIR, _TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import deep_gemm
from deep_gemm.testing import bench_kineto, calc_diff, count_bytes

from generators import KernelType, MajorTypeAB, QuantConfig, generate_normal


# FP4xFP4（MXFP4）：两个操作数都是 packed FP4（E2M1），配 VS=32 的 UE8M0 scale。
# 走 SM100_MMA_MXF4_SS 路径；API 不接受 bf16 D，所以 D 是 fp32。
FP4_FP4 = QuantConfig((32, 32, True, True))
# FP4xFP4 kernel 只支持 fp32 D。
OUT_DTYPE = torch.float
KERNEL_TYPE = KernelType.Kernel1D1D

# 解析 shape.txt 里的 (m, n, k)，跳过表头和空行。
def read_shapes(path):
    shapes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith('m'):  # 跳过表头 / 空行
                continue
            m, n, k = (int(x) for x in line.split())
            shapes.append((m, n, k))
    return shapes

# 运行一次 MXFP4 x MXFP4 稠密 GEMM（NT），结果写入 d。
def run_gemm(a, b, c, d, recipe, recipe_a, recipe_b):
    deep_gemm.fp8_fp4_gemm_nt(
        a, b, d, c=c, disable_ue8m0_cast=False,
        recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)

# 把 kernel 输出和 fp32 参考结果做对比，返回 (diff, 是否通过)。
def check_correctness(d, ref_d):
    diff = calc_diff(d, ref_d)
    return diff, diff < FP4_FP4.max_diff()

# 用 kineto 给 kernel 计时，返回该 shape 的 (time_us, tflops, gbps)。
def benchmark(m, n, k, a, b, c, d, recipe, recipe_a, recipe_b):
    t = bench_kineto(
        lambda: run_gemm(a, b, c, d, recipe, recipe_a, recipe_b),
        'sm100_fp4_gemm', suppress_kineto_output=True, flush_l2=True)
    tflops = 2 * m * n * k / t / 1e12
    gbps = count_bytes(a, b, d) / 1e9 / t
    return t * 1e6, tflops, gbps

# 把每个 shape 的测试结果写入 CSV 文件。
def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['m', 'n', 'k', 'time_us', 'tflops', 'gbps', 'diff', 'correct'])
        for r in rows:
            writer.writerow([r['m'], r['n'], r['k'],
                             f"{r['time_us']:.3f}", f"{r['tflops']:.2f}",
                             f"{r['gbps']:.2f}", f"{r['diff']:.6f}", int(r['correct'])])

# 对单个 shape 跑完整流程：生成输入 -> 精度对比 -> 性能测试。
def bench_shape(m, n, k, recipe, recipe_a, recipe_b):
    a, b, c, d, ref_d = generate_normal(
        m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
        accumulate=False, out_dtype=OUT_DTYPE, kernel_type=KERNEL_TYPE,
        use_ue8m0=True, quant_config=FP4_FP4)

    run_gemm(a, b, c, d, recipe, recipe_a, recipe_b)
    diff, ok = check_correctness(d, ref_d)
    time_us, tflops, gbps = benchmark(m, n, k, a, b, c, d, recipe, recipe_a, recipe_b)

    return dict(m=m, n=n, k=k, time_us=time_us, tflops=tflops,
                gbps=gbps, diff=diff, correct=ok)


def main():
    torch.manual_seed(0)
    random.seed(0)

    shapes = read_shapes(_SHAPE_PATH)
    recipe, recipe_a, recipe_b = FP4_FP4.get_recipes()

    print(f'Library path: {deep_gemm.__path__[0]}')
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Benchmarking {len(shapes)} shapes (MXFP4 x MXFP4 dense GEMM, NT, fp32 D)\n')

    rows = []
    for idx, (m, n, k) in enumerate(shapes):
        row = bench_shape(m, n, k, recipe, recipe_a, recipe_b)
        rows.append(row)
        print(f' [{idx + 1:2}/{len(shapes)}] m={m:6} n={n:6} k={k:6} | '
              f'{row["time_us"]:8.1f} us | {row["tflops"]:6.0f} TFLOPS | '
              f'{row["gbps"]:6.0f} GB/s | '
              f'diff={row["diff"]:.5f} {"OK" if row["correct"] else "FAIL"}')

    write_csv(_CSV_PATH, rows)

    n_ok = sum(r['correct'] for r in rows)
    print(f'\nDone. {n_ok}/{len(rows)} shapes passed correctness.')
    print(f'Results written to {_CSV_PATH}')


if __name__ == '__main__':
    main()
