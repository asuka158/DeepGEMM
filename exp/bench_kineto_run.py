"""Run DeepGEMM's original bench_kineto (CUPTI, num_tests=30, flush_l2=True) on all
shape.txt shapes, for 3-way comparison with the plain/graph event timings.
bench_kineto filters 'sm100_fp4_gemm' -> times ONLY the GEMM kernel (scale-pack & flush excluded)."""
import sys, os, time, threading, statistics, csv, torch, pynvml
REPO = '/root/workspace/gb_gemm_benchmark/DeepGEMM'
sys.path.insert(0, REPO); sys.path.insert(0, REPO + '/tests')
import deep_gemm
from deep_gemm.testing import bench_kineto
from generators import KernelType, MajorTypeAB, QuantConfig, generate_normal

FP4 = QuantConfig((32, 32, True, True))
SHAPE_FILE = REPO + '/tests/my_test/shape.txt'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'bench_kineto.csv')

pynvml.nvmlInit(); _H = pynvml.nvmlDeviceGetHandleByIndex(0)
_samp, _stop = [], False
def _sampler():
    while not _stop:
        try: _samp.append((time.time(), pynvml.nvmlDeviceGetClockInfo(_H, pynvml.NVML_CLOCK_SM)))
        except Exception: pass
        time.sleep(0.003)
def clk(t0, t1):
    w = [c for (t, c) in _samp if t0 <= t <= t1]
    return (statistics.median(w), min(w)) if w else (float('nan'), float('nan'))

def read_shapes(path):
    out = []
    for line in open(path):
        line = line.strip()
        if not line or line.lower().startswith('m'): continue
        m, n, k = (int(x) for x in line.split()); out.append((m, n, k))
    return out

def main():
    shapes = read_shapes(SHAPE_FILE)
    threading.Thread(target=_sampler, daemon=True).start()
    f = open(OUT, 'w', newline=''); w = csv.writer(f)
    w.writerow(['m', 'n', 'k', 'bk_us', 'bk_TF', 'bk_clk_med', 'bk_clk_min'])
    for (m, n, k) in shapes:
        flop = 2 * m * n * k
        try:
            a, b, c, d, _ = generate_normal(m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
                accumulate=False, out_dtype=torch.float, kernel_type=KernelType.Kernel1D1D,
                use_ue8m0=True, quant_config=FP4)
            recipe, ra, rb = FP4.get_recipes()
            def gemm():
                deep_gemm.fp8_fp4_gemm_nt(a, b, d, c=c, disable_ue8m0_cast=False,
                                          recipe=recipe, recipe_a=ra, recipe_b=rb)
            gemm(); torch.cuda.synchronize()
            t0 = time.time()
            t = bench_kineto(gemm, 'sm100_fp4_gemm', suppress_kineto_output=True,
                             flush_l2=True, num_tests=30)
            t1 = time.time()
            tf = flop / t / 1e12; cm, cmin = clk(t0, t1)
            w.writerow([m, n, k, f'{t*1e6:.1f}', f'{tf:.0f}', f'{cm:.0f}', f'{cmin:.0f}']); f.flush()
            print(f"{m:6}{n:6}{k:6} | {t*1e6:9.1f}us {tf:6.0f}TF | clk {cm:.0f}/{cmin:.0f}", flush=True)
            del a, b, c, d; torch.cuda.empty_cache()
        except Exception as ex:
            print(f"{m:6}{n:6}{k:6} | ERROR {repr(ex)[:70]}", flush=True); torch.cuda.empty_cache()
    f.close()
    global _stop; _stop = True
    print(f"\nDone -> {OUT}")

if __name__ == '__main__':
    main()
