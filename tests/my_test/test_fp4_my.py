"""
MXFP4 x MXFP4 dense GEMM benchmark (reproduces the dense table of PR #348).

Based on test_gemm() in tests/test_fp4.py. For every shape in shape.txt it:
  1. generates bf16 inputs, casts both operands to MXFP4 (E2M1 + UE8M0 VS=32 scales),
  2. runs the SM100_MMA_MXF4_SS dense kernel via deep_gemm.fp8_fp4_gemm_nt,
  3. checks correctness against the fp32 reference (calc_diff < 0.02),
  4. times the kernel with bench_kineto,
and writes the results to tests/result/<...>.csv.

NOTE: the FP4xFP4 path has no bf16 epilogue (csrc/apis/gemm.hpp asserts fp32 D),
so the output D is fp32 -- bf16 writeback is not supported by this kernel yet.

Run (the LD_LIBRARY_PATH prefix fixes the container's torch/libucs import issue):
  cd <repo>/tests/my_test
  LD_LIBRARY_PATH=/opt/hpcx/ucx/lib PYTHONPATH=<repo> python test_fp4_my.py
"""
import csv
import os
import random
import sys

import torch

# Make `deep_gemm` and the shared test `generators` importable regardless of CWD.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.dirname(_THIS_DIR)
_REPO_DIR = os.path.dirname(_TESTS_DIR)
for p in (_REPO_DIR, _TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import deep_gemm
from deep_gemm.testing import bench_kineto, calc_diff, count_bytes

from generators import KernelType, MajorTypeAB, QuantConfig, generate_normal


# FP4xFP4 (MXFP4): both operands packed FP4 (E2M1) with VS=32 UE8M0 scales.
# Dispatches to the SM100_MMA_MXF4_SS path; the API rejects bf16 D so D is fp32.
FP4_FP4 = QuantConfig((32, 32, True, True))


def read_shapes(path):
    shapes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith('m'):  # skip header / blanks
                continue
            m, n, k = (int(x) for x in line.split())
            shapes.append((m, n, k))
    return shapes


def main():
    torch.manual_seed(0)
    random.seed(0)

    shape_path = os.path.join(_THIS_DIR, 'shape.txt')
    result_dir = os.path.join(_TESTS_DIR, 'result')
    os.makedirs(result_dir, exist_ok=True)
    csv_path = os.path.join(result_dir, 'fp4_dense_gemm_bench.csv')

    shapes = read_shapes(shape_path)
    kernel_type = KernelType.Kernel1D1D
    out_dtype = torch.float  # FP4xFP4 kernel only supports fp32 D
    recipe, recipe_a, recipe_b = FP4_FP4.get_recipes()

    print(f'Library path: {deep_gemm.__path__[0]}')
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Benchmarking {len(shapes)} shapes (MXFP4 x MXFP4 dense GEMM, NT, fp32 D)\n')

    rows = []
    for idx, (m, n, k) in enumerate(shapes):
        a, b, c, d, ref_d = generate_normal(
            m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
            accumulate=False, out_dtype=out_dtype, kernel_type=kernel_type,
            use_ue8m0=True, quant_config=FP4_FP4)

        deep_gemm.fp8_fp4_gemm_nt(
            a, b, d, c=c, disable_ue8m0_cast=False,
            recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)

        diff = calc_diff(d, ref_d)
        ok = diff < FP4_FP4.max_diff()

        t = bench_kineto(
            lambda: deep_gemm.fp8_fp4_gemm_nt(
                a, b, d, c=c, disable_ue8m0_cast=False,
                recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b),
            'sm100_fp4_gemm', suppress_kineto_output=True)

        tflops = 2 * m * n * k / t / 1e12
        gbps = count_bytes(a, b, d) / 1e9 / t

        rows.append(dict(m=m, n=n, k=k, time_us=t * 1e6, tflops=tflops,
                         gbps=gbps, diff=diff, correct=ok))
        print(f' [{idx + 1:2}/{len(shapes)}] m={m:6} n={n:6} k={k:6} | '
              f'{t * 1e6:8.1f} us | {tflops:6.0f} TFLOPS | {gbps:6.0f} GB/s | '
              f'diff={diff:.5f} {"OK" if ok else "FAIL"}')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['m', 'n', 'k', 'time_us', 'tflops', 'gbps', 'diff', 'correct'])
        for r in rows:
            writer.writerow([r['m'], r['n'], r['k'],
                             f"{r['time_us']:.3f}", f"{r['tflops']:.2f}",
                             f"{r['gbps']:.2f}", f"{r['diff']:.6f}", int(r['correct'])])

    n_ok = sum(r['correct'] for r in rows)
    print(f'\nDone. {n_ok}/{len(rows)} shapes passed correctness.')
    print(f'Results written to {csv_path}')


if __name__ == '__main__':
    main()
