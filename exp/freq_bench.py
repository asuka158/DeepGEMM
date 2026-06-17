"""
Frequency-stability benchmark harness for the DeepGEMM MXFP4 dense GEMM kernel.

Goal: across (timing method) x (num_tests) x (warmup) x (breather) combinations,
find schemes where the SM clock stays as close to a constant 2062 MHz as possible,
and record the resulting kernel performance.

Methods:
  bench_kineto : deep_gemm's CUPTI-based timer (kernel filtered by name; flush excluded)
  event        : per-iteration cudaEvent pairs, ONE sync at the end (triton do_bench style)
  graph        : same as 'event' but the kernel is replayed from a captured CUDA graph
  event_sync   : per-iteration cudaEvent with a sync EVERY iteration (contrast: idle gaps)

Breathers (low-power interlude between timed kernels, recorded OUTSIDE the event bracket):
  none / flush8 / flush32 / flushA(adaptive ~ kernel time) / sleep(torch.cuda._sleep)

Clock is measured window-level via NVML in a background thread (min/median/max over the
wall-clock window of the timed region). All code + results stay under DeepGEMM/exp/.

Usage:
  python freq_bench.py --groups A,B,C,D,E
  python freq_bench.py --smoke        # quick sanity run
"""
import os, sys, time, threading, statistics, csv, argparse, math
import torch, pynvml

_REPO = '/root/workspace/gb_gemm_benchmark/DeepGEMM'
sys.path.insert(0, _REPO); sys.path.insert(0, _REPO + '/tests')
import deep_gemm
from deep_gemm.testing import bench_kineto
from generators import KernelType, MajorTypeAB, QuantConfig, generate_normal

FP4 = QuantConfig((32, 32, True, True))
CLOCK_HZ = 2.062e9          # GB200 max SM clock, for sleep-cycle sizing
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'freq_sweep.csv')

SHAPES = [(16384, 16384, 16384), (16384, 16384, 32768),
          (32768, 16384, 16384), (32768, 16384, 32768),
          (8192, 32768, 32768), (32768, 32768, 8192),
          (32768, 32768, 16384)]
REPEATS = 3

# ---------------- NVML window-level clock sampler ----------------
pynvml.nvmlInit(); _H = pynvml.nvmlDeviceGetHandleByIndex(0)
_samples = []; _stop = False
def _sampler():
    while not _stop:
        try: _samples.append((time.time(), pynvml.nvmlDeviceGetClockInfo(_H, pynvml.NVML_CLOCK_SM)))
        except Exception: pass
        time.sleep(0.002)
def clk_stats(t0, t1):
    w = [c for (t, c) in _samples if t0 <= t <= t1]
    if not w: return (float('nan'), float('nan'), float('nan'), 0)
    return (min(w), statistics.median(w), max(w), len(w))

# ---------------- kernel + graph + breather builders ----------------
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

def quick_kernel_ms(gemm):
    for _ in range(5): gemm()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(5):
        torch.cuda.synchronize(); s.record(); gemm(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)

def calibrate_bw(buf):
    # effective HBM write BW from an 8GB zero_, used to size the adaptive flush
    n = int(8e9 // 4); buf[:n].zero_(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); buf[:n].zero_(); e.record(); torch.cuda.synchronize()
    return 8e9 / (s.elapsed_time(e) / 1e3)      # bytes/sec

def make_breather(kind, buf, kernel_ms, bw):
    if kind == 'none':
        return lambda: None
    if kind == 'sleep':
        cyc = max(1, int(kernel_ms * 1e-3 * CLOCK_HZ))
        return lambda: torch.cuda._sleep(cyc)
    # flush variants -> zero a prefix of the shared 32GB buffer
    if kind == 'flush8':   nbytes = int(8e9)
    elif kind == 'flush32': nbytes = int(32e9)
    elif kind == 'flushA':  nbytes = int(min(32e9, max(8e9, kernel_ms * 1e-3 * bw)))
    else: raise ValueError(kind)
    n = nbytes // 4
    return lambda: buf[:n].zero_()

# ---------------- measurement methods ----------------
def m_eventpair(run_unit, breather, num_tests, warmup, idle=0.0):
    for _ in range(warmup): breather(); run_unit()
    torch.cuda.synchronize()
    if idle > 0: time.sleep(idle)
    se = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    ee = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    torch.cuda.synchronize(); t0 = time.time()
    for i in range(num_tests):
        breather(); se[i].record(); run_unit(); ee[i].record()
    torch.cuda.synchronize(); t1 = time.time()
    times = [se[i].elapsed_time(ee[i]) for i in range(num_tests)]
    return statistics.median(times), clk_stats(t0, t1)

def m_eventsync(run_unit, breather, num_tests, warmup, idle=0.0):
    for _ in range(warmup): breather(); run_unit()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []; t0 = time.time()
    for _ in range(num_tests):
        breather(); s.record(); run_unit(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    t1 = time.time()
    return statistics.median(times), clk_stats(t0, t1)

def m_benchkineto(gemm, num_tests, flush):
    t0 = time.time()
    t = bench_kineto(gemm, 'sm100_fp4_gemm', suppress_kineto_output=True,
                     flush_l2=flush, num_tests=num_tests)
    t1 = time.time()
    return t * 1e3, clk_stats(t0, t1)   # ms

# ---------------- experiment matrix ----------------
def build_matrix(groups):
    cfgs = []
    if 'A' in groups:
        cfgs += [
            dict(group='A', label='bench_kineto',  method='bench_kineto', nt=10, wu=0,  br='flush8'),
            dict(group='A', label='event(B)',      method='event',        nt=10, wu=10, br='flush8'),
            dict(group='A', label='graph',         method='graph',        nt=10, wu=10, br='flush8'),
            dict(group='A', label='event+sync',    method='event_sync',   nt=10, wu=10, br='flush8'),
        ]
    if 'B' in groups:
        cfgs += [dict(group='B', label=f'nt={nt}', method='event', nt=nt, wu=10, br='flush8')
                 for nt in (1, 5, 10, 30)]
    if 'C' in groups:
        cfgs += [dict(group='C', label=f'br={br}', method='event', nt=10, wu=10, br=br)
                 for br in ('none', 'flush8', 'flush32', 'flushA', 'sleep')]
    if 'D' in groups:
        cfgs += [dict(group='D', label=f'wu={wu}', method='event', nt=10, wu=wu, br='flush8')
                 for wu in (2, 10, 30, 60)]
        cfgs += [dict(group='D', label='wu10+idle0.5', method='event', nt=10, wu=10, br='flush8', idle=0.5)]
    if 'E' in groups:
        cfgs += [
            dict(group='E', label='graph+nt5+flush32', method='graph', nt=5, wu=10, br='flush32'),
            dict(group='E', label='event+nt5+flushA',  method='event', nt=5, wu=10, br='flushA'),
            dict(group='E', label='event+nt1+sleep',   method='event', nt=1, wu=10, br='sleep'),
        ]
    return cfgs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', default='A,B,C,D,E')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--repeats', type=int, default=REPEATS)
    args = ap.parse_args()

    torch.manual_seed(0)
    shapes = [(4096, 4096, 4096)] if args.smoke else SHAPES
    groups = ['A'] if args.smoke else args.groups.split(',')
    repeats = 1 if args.smoke else args.repeats
    cfgs = build_matrix(groups)

    threading.Thread(target=_sampler, daemon=True).start()
    new = not os.path.exists(RESULTS)
    f = open(RESULTS, 'a', newline=''); w = csv.writer(f)
    if new:
        w.writerow(['shape', 'group', 'label', 'method', 'num_tests', 'warmup', 'breather',
                    'rep', 'tflops', 'clk_min', 'clk_med', 'clk_max', 'n_clk_samples', 'kernel_ms'])

    for shape in shapes:
        m, n, k = shape; flop = 2 * m * n * k
        print(f"\n===== shape {m}x{n}x{k} =====", flush=True)
        gemm, tensors = make_gemm(shape)
        kms = quick_kernel_ms(gemm)
        buf = torch.empty(int(32e9 // 4), dtype=torch.int, device='cuda')   # shared flush buffer
        bw = calibrate_bw(buf)
        g = None
        print(f"  kernel~{kms:.3f}ms, HBM write BW~{bw/1e12:.1f}TB/s", flush=True)
        for cfg in cfgs:
            method = cfg['method']; nt = cfg['nt']; wu = cfg['wu']; br = cfg['br']; idle = cfg.get('idle', 0.0)
            try:
                breather = make_breather(br, buf, kms, bw)
                if method == 'graph' and g is None:
                    g = capture_graph(gemm)
                run_unit = (g.replay if method == 'graph' else gemm)
                for rep in range(repeats):
                    if method == 'bench_kineto':
                        mt, ck = m_benchkineto(gemm, nt, flush=(br != 'none'))
                    elif method == 'event_sync':
                        mt, ck = m_eventsync(run_unit, breather, nt, wu, idle)
                    else:
                        mt, ck = m_eventpair(run_unit, breather, nt, wu, idle)
                    tf = flop / (mt / 1e3) / 1e12
                    w.writerow([f'{m}x{n}x{k}', cfg['group'], cfg['label'], method, nt, wu, br,
                                rep, f'{tf:.0f}', ck[0], ck[1], ck[2], ck[3], f'{kms:.3f}'])
                    f.flush()
                    print(f"  [{cfg['group']}] {cfg['label']:18s} rep{rep}: {tf:5.0f}TF "
                          f"clk {ck[0]:.0f}/{ck[1]:.0f}/{ck[2]:.0f}", flush=True)
            except Exception as ex:
                print(f"  [{cfg['group']}] {cfg['label']}: ERROR {repr(ex)[:80]}", flush=True)
        del gemm, tensors, buf, g; torch.cuda.empty_cache()
    f.close()
    global _stop; _stop = True
    print(f"\nDone. Results -> {RESULTS}")

if __name__ == '__main__':
    main()
