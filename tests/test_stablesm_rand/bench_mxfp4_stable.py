"""
DeepGEMM MXFP4 x MXFP4 -> fp32 dense GEMM, STEADY-STATE benchmark (随机数据 + 降频稳态).

应用 bench.md 的结论做"正确"的测量:
  * 数据:torch.randn 量化成 MXFP4(高熵、真实分布),不是低熵合成数据。
  * 稳态:不测前 1.5s,只 run 让 SM 降到稳态,再在稳态下 run 多次取平均 —— 用 bench_kineto 实现:
      把 num_tests 调大,使 bench_kineto 的 warmup 轮(不计时)≈1.5s 把 SM 跑到稳态,
      active 轮(计时)整轮都在稳态 -> 平均即稳态吞吐。(经验证 == 显式 1.5s warmup + bench_kineto)
  * flush_l2:A+B 操作数 < 50MB 的小 shape 开 flush(避免操作数 L2 常驻虚高、且小 shape 本不降频);
      >= 50MB 的大 shape 关 flush(连续跑,保持降频稳态)。
  * C 不需要(accumulate=False -> c=None)。
  * scale 预打包:DeepGEMM 默认每次 GEMM 调用会内部把 FP32 block-scale 转成 ue8m0(2 个
      transpose_and_pack kernel)。这里在 prep 阶段(计时之外)用 transform_sf_into_required_layout
      预先转成 INT 打包格式,使计时区间内每次只发 1 个 GEMM kernel(数值与默认路径完全一致,
      diff=0;也更贴近真实推理——权重 scale 预处理一次反复用)。bench_kineto 本就按名只计 GEMM,
      所以 tflops 不变,但端到端/graph 计时现在也是纯 GEMM。
  * sm_mhz / power_w:NVML 后台采样,取 bench_kineto 窗口内中位数(稳态占多数 -> 中位数即稳态)。

输出:tests/test_stablesm_rand/result/mxfp4_dense_stable.csv,列 m,n,k,us,tflops,gbps,sm_mhz,power_w,backend。

运行:
  cd DeepGEMM/tests/test_stablesm_rand
  LD_LIBRARY_PATH=/opt/hpcx/ucx/lib \
  PYTHONPATH=/root/workspace/gb_gemm_benchmark/DeepGEMM:/root/workspace/gb_gemm_benchmark/DeepGEMM/tests \
  python bench_mxfp4_stable.py
"""
import csv, math, os, random, sys, time, threading
import torch

_REPO = '/root/workspace/gb_gemm_benchmark/DeepGEMM'
_TESTS = _REPO + '/tests'
for p in (_REPO, _TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)

import deep_gemm
from deep_gemm.testing import bench_kineto, calc_diff, count_bytes
from generators import KernelType, MajorTypeAB, QuantConfig, generate_normal
import pynvml

SHAPE_PATH = _TESTS + '/my_test/shape.txt'
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', 'mxfp4_dense_stable.csv')
KERNEL = 'sm100_fp4_gemm'
BACKEND = 'deepgemm_mxfp4_fp32'
FLUSH_THRESH = 50e6           # A+B operand bytes below this -> flush_l2 on
# Size num_tests so each bench_kineto round (warmup + active) runs >= this long: long enough for the
# SM to reach throttle steady state AND for NVML power (~1s rolling avg) to settle inside the window.
TARGET_ROUND_S = 1.6
NT_MIN, NT_MAX = 100, 40000   # cap event count for very fast kernels
FLUSH_NT = 100                # small/flush shapes don't throttle -> a modest count is enough

FP4 = QuantConfig((32, 32, True, True))   # MXFP4xMXFP4: E2M1 + VS=32 UE8M0 scales
OUT_DTYPE = torch.float                    # FP4xFP4 kernel only supports fp32 D


def read_shapes(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith('m'):
                continue
            m, n, k = (int(x) for x in line.split())
            out.append((m, n, k))
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
    """Pre-transform the FP32 block-scales into the kernel's packed-UE8M0 (INT) layout ONCE, here in
    prep (outside timing). Then each timed fp8_fp4_gemm_nt call launches ONLY the GEMM kernel — the
    internal transform sees INT scales and just does a cheap layout check instead of the per-call
    transpose_and_pack_fp32_into_ue8m0 scale kernels (2 per call). Numerically identical (diff=0 vs
    the cast-inside path); mirrors real inference where weight scales are preprocessed once."""
    if recipe is not None:   # shared 3-tuple recipe -> is_sfa required
        sfa = deep_gemm.transform_sf_into_required_layout(a[1], m, k, recipe, None, True, False)
        sfb = deep_gemm.transform_sf_into_required_layout(b[1], n, k, recipe, None, False, False)
    else:                    # recipe_a / recipe_b (2-tuple each) -> is_sfa must be None
        sfa = deep_gemm.transform_sf_into_required_layout(a[1], m, k, ra, None, None, False)
        sfb = deep_gemm.transform_sf_into_required_layout(b[1], n, k, rb, None, None, False)
    return (a[0], sfa), (b[0], sfb)


def est_kernel_s(run, iters=12):
    for _ in range(3): run()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): run()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters / 1e3   # seconds (un-throttled estimate; ok for sizing)


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
    print(f'Device: {torch.cuda.get_device_name(0)} | {len(shapes)} shapes | MXFP4xMXFP4->fp32 dense, steady-state', flush=True)
    print(f'{"m":>6} {"n":>6} {"k":>6} | {"us":>9} {"TFLOPS":>7} {"GB/s":>7} | {"sm":>5} {"pw":>6} | flush nt diff', flush=True)
    rows = []
    write_csv(rows)
    for idx, (m, n, k) in enumerate(shapes, 1):
        try:
            print(f'[{idx:02d}/{len(shapes)}] start {m} {n} {k}', flush=True)
            a, b, c, d, ref = generate_normal(
                m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
                accumulate=False, out_dtype=OUT_DTYPE, kernel_type=KernelType.Kernel1D1D,
                use_ue8m0=True, quant_config=FP4)
            ab_bytes = count_bytes(a, b)          # flush threshold on the original operands
            flush = ab_bytes < FLUSH_THRESH
            a, b = prepack_scales(a, b, m, n, k, recipe, ra, rb)  # scale kernels out of timed region
            run = make_run(a, b, d, recipe, ra, rb)
            run(); torch.cuda.synchronize()
            diff = float(calc_diff(d, ref))
            kt = est_kernel_s(run)
            # non-flush (large, throttling) shapes: round >= TARGET_ROUND_S so it reaches steady and
            # NVML power settles. flush (small, non-throttling) shapes: a modest fixed count.
            nt = FLUSH_NT if flush else min(NT_MAX, max(NT_MIN, math.ceil(TARGET_ROUND_S / kt)))

            t0 = time.time()
            t = bench_kineto(run, KERNEL, num_tests=nt, suppress_kineto_output=True, flush_l2=flush)
            t1 = time.time()
            sm, pw = nv.median(t0, t1)

            tflops = 2.0 * m * n * k / t / 1e12
            gbps = count_bytes(a, b, d) / 1e9 / t
            rows.append(dict(m=m, n=n, k=k, us=t * 1e6, tflops=tflops, gbps=gbps,
                             sm_mhz=sm, power_w=pw, backend=BACKEND))
            write_csv(rows)
            print(f'{m:6} {n:6} {k:6} | {t*1e6:9.1f} {tflops:7.0f} {gbps:7.0f} | '
                  f'{sm:5.0f} {pw:6.0f} | {int(flush)} {nt:4} {diff:.4f}', flush=True)
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
