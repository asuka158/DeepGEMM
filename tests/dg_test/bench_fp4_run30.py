"""
DeepGEMM MXFP4 x MXFP4 -> fp32 dense GEMM bench —— 用 tests/test_fp4.py 的测量方式(flush_l2 + 固定 30 次).

与 ../test_stablesm_rand/bench_mxfp4_stable.py 的**唯一**区别 = 测量方式,完全照 test_fp4.py:
  * flush_l2 = True(始终开),不再按 A+B 大小判断;
  * num_tests = 30(固定),不做稳态 sizing(去掉 est_kernel_s / TARGET_ROUND_S / NT_MAX)。
其余(torch.randn->MXFP4 数据、scale 预打包、NVML 监测、CSV 列)与 stable 完全一致,便于"只差测量方式"的对照。

CSV 列与 stable 相同:m,n,k,us,tflops,gbps,sm_mhz,power_w,backend
输出:tests/dg_test/result/mxfp4_fp32_dense_DG_run30_50shape.csv

⚠️ 注意:flush_l2=on 的 8GB memset + 仅 30 次的短窗口,会让 sm_mhz / power_w **不代表 GEMM 真实运行点**
(memset 低功耗段稀释 NVML 采样 + 窗口太短没到稳态 + NVML ~1s 功耗滚动平均在短窗口内未 settle)。
tflops 仍是 bench_kineto 按 kernel 名提取的纯 GEMM 时间,但属"短时近峰值/chill"的乐观区。

运行:
  cd DeepGEMM/tests/dg_test
  LD_LIBRARY_PATH=/opt/hpcx/ucx/lib \
  PYTHONPATH=/root/workspace/gb_gemm_bench/DeepGEMM:/root/workspace/gb_gemm_bench/DeepGEMM/tests \
  python bench_fp4_run30.py
"""
import csv, os, random, sys, time, threading
import torch

_REPO = '/root/workspace/gb_gemm_bench/DeepGEMM'
_TESTS = _REPO + '/tests'
for p in (_REPO, _TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)

import deep_gemm
from deep_gemm.testing import bench_kineto, calc_diff, count_bytes
from generators import KernelType, MajorTypeAB, QuantConfig, generate_normal
import pynvml

SHAPE_PATH = '/root/workspace/gb_gemm_bench/shape.csv'   # columns: M,K,N
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result',
                       'mxfp4_fp32_dense_DG_run30_50shape.csv')
KERNEL = 'sm100_fp4_gemm'
BACKEND = 'deepgemm_mxfp4_fp32_run30'
NUM_TESTS = 30          # test_fp4.py 的固定测量次数
FLUSH_L2 = True         # test_fp4.py 用 bench_kineto 的默认 flush_l2=True

FP4 = QuantConfig((32, 32, True, True))   # MXFP4xMXFP4: E2M1 + VS=32 UE8M0 scales
OUT_DTYPE = torch.float                    # FP4xFP4 kernel only supports fp32 D


def read_shapes(path):
    """shape.csv 表头 'M,K,N'(逗号分隔)。内核要 (m,n,k),故行 (M,K,N) 映射成 (M,N,K)。"""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith('m'):
                continue
            M, K, N = (int(x) for x in line.split(','))
            out.append((M, N, K))
    return out


class Nvml:
    def __init__(self, idx=0):
        pynvml.nvmlInit(); self.h = pynvml.nvmlDeviceGetHandleByIndex(idx)
        self.s = []; self.stop = False
        self.t = threading.Thread(target=self._loop, daemon=True); self.t.start()
    def _loop(self):
        while not self.stop:
            try:
                self.s.append((time.time(),
                               pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_SM),
                               pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0))
            except Exception:
                pass
            time.sleep(0.001)
    def median(self, t0, t1):
        clk = sorted(c for (t, c, _) in self.s if t0 <= t <= t1)
        pwr = sorted(p for (t, _, p) in self.s if t0 <= t <= t1)
        m = lambda v: v[len(v) // 2] if v else float('nan')
        return m(clk), m(pwr)
    def close(self):
        self.stop = True; self.t.join(); pynvml.nvmlShutdown()


def make_run(a, b, d, recipe, ra, rb):
    def run():
        deep_gemm.fp8_fp4_gemm_nt(a, b, d, c=None, disable_ue8m0_cast=False,
                                  recipe=recipe, recipe_a=ra, recipe_b=rb)
    return run


def prepack_scales(a, b, m, n, k, recipe, ra, rb):
    """Pre-transform FP32 block-scales into the kernel's packed-UE8M0 (INT) layout ONCE in prep
    (outside timing), so each timed call launches ONLY the GEMM kernel (same as stable; numerically
    identical, diff=0). bench_kineto times by kernel name so tflops is unaffected either way."""
    if recipe is not None:
        sfa = deep_gemm.transform_sf_into_required_layout(a[1], m, k, recipe, None, True, False)
        sfb = deep_gemm.transform_sf_into_required_layout(b[1], n, k, recipe, None, False, False)
    else:
        sfa = deep_gemm.transform_sf_into_required_layout(a[1], m, k, ra, None, None, False)
        sfb = deep_gemm.transform_sf_into_required_layout(b[1], n, k, rb, None, None, False)
    return (a[0], sfa), (b[0], sfb)


def write_csv(rows):
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['m', 'n', 'k', 'us', 'tflops', 'gbps', 'sm_mhz', 'power_w', 'backend'])
        for r in rows:
            w.writerow([r['m'], r['n'], r['k'], f"{r['us']:.3f}", f"{r['tflops']:.2f}",
                        f"{r['gbps']:.2f}", f"{r['sm_mhz']:.0f}", f"{r['power_w']:.1f}", r['backend']])


def main():
    torch.manual_seed(0); random.seed(0)
    shapes = read_shapes(SHAPE_PATH)
    recipe, ra, rb = FP4.get_recipes()
    nv = Nvml(0)
    print(f'Device: {torch.cuda.get_device_name(0)} | {len(shapes)} shapes | MXFP4xMXFP4->fp32 dense, '
          f'test_fp4-style (flush_l2={FLUSH_L2}, num_tests={NUM_TESTS})', flush=True)
    print(f'{"m":>6} {"n":>6} {"k":>6} | {"us":>9} {"TFLOPS":>7} {"GB/s":>7} | {"sm":>5} {"pw":>6} | diff', flush=True)
    rows = []
    write_csv(rows)
    for idx, (m, n, k) in enumerate(shapes, 1):
        try:
            print(f'[{idx:02d}/{len(shapes)}] start {m} {n} {k}', flush=True)
            a, b, c, d, ref = generate_normal(
                m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
                accumulate=False, out_dtype=OUT_DTYPE, kernel_type=KernelType.Kernel1D1D,
                use_ue8m0=True, quant_config=FP4)
            a, b = prepack_scales(a, b, m, n, k, recipe, ra, rb)  # scale kernels out of timed region
            run = make_run(a, b, d, recipe, ra, rb)
            run(); torch.cuda.synchronize()
            diff = float(calc_diff(d, ref))

            # --- test_fp4.py-style measurement: fixed 30 tests, flush_l2 always on ---
            t0 = time.time()
            t = bench_kineto(run, KERNEL, num_tests=NUM_TESTS, suppress_kineto_output=True,
                             flush_l2=FLUSH_L2)
            t1 = time.time()
            sm, pw = nv.median(t0, t1)

            tflops = 2.0 * m * n * k / t / 1e12
            gbps = count_bytes(a, b, d) / 1e9 / t
            rows.append(dict(m=m, n=n, k=k, us=t * 1e6, tflops=tflops, gbps=gbps,
                             sm_mhz=sm, power_w=pw, backend=BACKEND))
            write_csv(rows)
            print(f'{m:6} {n:6} {k:6} | {t*1e6:9.1f} {tflops:7.0f} {gbps:7.0f} | '
                  f'{sm:5.0f} {pw:6.0f} | {diff:.4f}', flush=True)
            del a, b, c, d, ref, run
            torch.cuda.empty_cache()
        except Exception as ex:
            print(f'{m:6} {n:6} {k:6} | ERROR {type(ex).__name__}: {str(ex)[:80]}', flush=True)
            torch.cuda.empty_cache()

    write_csv(rows)
    nv.close()
    print(f'\nDone. {len(rows)}/{len(shapes)} shapes -> {OUT_CSV}', flush=True)


if __name__ == '__main__':
    main()
