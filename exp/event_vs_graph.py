"""
Compare two cudaEvent-based timings of the DeepGEMM MXFP4 kernel:
  (1) PLAIN : one event pair around a for-loop of RUNS direct kernel launches
  (2) GRAPH : one event pair around a for-loop of RUNS graph replays (kernel captured once)

Both: warmup 5, run 10, single event pair on the two sides of the loop, time = elapsed/RUNS.
No L2 flush, no per-iter events. Shapes from tests/my_test/shape.txt. 3 reps -> median.
Window-level SM clock (NVML) recorded to help explain any difference.
"""
import sys, os, time, threading, statistics, csv, torch, pynvml

REPO = '/root/workspace/gb_gemm_benchmark/DeepGEMM'
sys.path.insert(0, REPO); sys.path.insert(0, REPO + '/tests')
import deep_gemm
from generators import KernelType, MajorTypeAB, QuantConfig, generate_normal

FP4 = QuantConfig((32, 32, True, True))
SHAPE_FILE = REPO + '/tests/my_test/shape.txt'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'event_vs_graph.csv')
WARMUP, RUNS, REPS = 5, 10, 3

pynvml.nvmlInit(); _H = pynvml.nvmlDeviceGetHandleByIndex(0)
_samp, _stop = [], False
def _sampler():
    while not _stop:
        try: _samp.append((time.time(), pynvml.nvmlDeviceGetClockInfo(_H, pynvml.NVML_CLOCK_SM)))
        except Exception: pass
        time.sleep(0.003)
def clk_med(t0, t1):
    w = [c for (t, c) in _samp if t0 <= t <= t1]
    return statistics.median(w) if w else float('nan')

def read_shapes(path):
    out = []
    for line in open(path):
        line = line.strip()
        if not line or line.lower().startswith('m'): continue
        m, n, k = (int(x) for x in line.split())
        out.append((m, n, k))
    return out

def make_gemm(shape):
    m, n, k = shape
    a, b, c, d, _ = generate_normal(m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor,
        accumulate=False, out_dtype=torch.float, kernel_type=KernelType.Kernel1D1D,
        use_ue8m0=True, quant_config=FP4)
    recipe, ra, rb = FP4.get_recipes()
    def gemm():
        deep_gemm.fp8_fp4_gemm_nt(a, b, d, c=c, disable_ue8m0_cast=False,
                                  recipe=recipe, recipe_a=ra, recipe_b=rb)
    return gemm, (a, b, c, d)

def capture_graph(gemm):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): gemm()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): gemm()
    return g

def time_loop(run_unit):
    """warmup WARMUP, then ONE event pair around RUNS launches; returns (ms_per_call, clk_med)."""
    for _ in range(WARMUP): run_unit()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); t0 = time.time(); s.record()
    for _ in range(RUNS): run_unit()
    e.record(); torch.cuda.synchronize(); t1 = time.time()
    return s.elapsed_time(e) / RUNS, clk_med(t0, t1)

def main():
    shapes = read_shapes(SHAPE_FILE)
    threading.Thread(target=_sampler, daemon=True).start()
    f = open(OUT, 'w', newline=''); w = csv.writer(f)
    w.writerow(['m', 'n', 'k', 'plain_us', 'plain_TF', 'graph_us', 'graph_TF',
                'graph_speedup(plain/graph)', 'plain_clk', 'graph_clk'])
    print(f"{'m':>6}{'n':>6}{'k':>6} | {'plain_us':>9}{'plainTF':>8} | {'graph_us':>9}{'graphTF':>8} | "
          f"{'g/p_TF':>7} | {'clk_p':>6}{'clk_g':>6}")
    for (m, n, k) in shapes:
        flop = 2 * m * n * k
        try:
            gemm, tensors = make_gemm((m, n, k))
            # method 1: plain (direct launch)
            p = sorted(time_loop(gemm) for _ in range(REPS))[REPS // 2]   # median by ms
            # method 2: graph replay
            g = capture_graph(gemm)
            q = sorted(time_loop(g.replay) for _ in range(REPS))[REPS // 2]
            pus, pclk = p; gus, gclk = q
            ptf, gtf = flop / (pus / 1e3) / 1e12, flop / (gus / 1e3) / 1e12
            w.writerow([m, n, k, f'{pus*1e3:.1f}', f'{ptf:.0f}', f'{gus*1e3:.1f}', f'{gtf:.0f}',
                        f'{gtf/ptf:.3f}', f'{pclk:.0f}', f'{gclk:.0f}']); f.flush()
            print(f"{m:6}{n:6}{k:6} | {pus*1e3:9.1f}{ptf:8.0f} | {gus*1e3:9.1f}{gtf:8.0f} | "
                  f"{gtf/ptf:7.3f} | {pclk:6.0f}{gclk:6.0f}", flush=True)
            del gemm, tensors, g; torch.cuda.empty_cache()
        except Exception as ex:
            print(f"{m:6}{n:6}{k:6} | ERROR {repr(ex)[:70]}", flush=True)
            torch.cuda.empty_cache()
    f.close()
    global _stop; _stop = True
    print(f"\nDone -> {OUT}")

if __name__ == '__main__':
    main()
