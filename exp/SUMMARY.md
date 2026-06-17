# Frequency-stability sweep — findings

Goal: find a measurement scheme that keeps the GB200 SM clock pinned at full 2062 MHz
(no power throttle) for the DeepGEMM MXFP4 dense GEMM, and read the resulting performance.

Setup: `freq_bench.py` swept (method) × (num_tests) × (warmup) × (breather) over **7 large
shapes** (16384/32768, varied m/n/k), 3 repeats each. Clock measured window-level via NVML.
Raw data: `results/freq_sweep.csv`; rerun analysis: `python analyze.py`.

## Ranking by worst-case clk_min across all 7 shapes (higher = more stable)

| scheme | method | worst clk_min | median TFLOPS | stable @2062? |
|---|---|---|---|---|
| **br=sleep** (nt=10 + `_sleep`≈kernel) | event | **2062** | **6817** | ✅ all shapes |
| **event+nt1+sleep** | event | **2062** | **6822** | ✅ all shapes |
| **wu10+idle0.5** (flush8 + 0.5s idle before timing) | event | **2062** | 6676 | ✅ all shapes |
| br=flushA (adaptive flush ≈ kernel time) | event | 1935 | 6810 | ~ (one shape 1935) |
| event+nt5+flushA | event | 1837 | 6828 | ~ |
| br=flush32 | event | 1470 | 6819 | ✗ |
| graph / wu=2 | graph/event | 1500 | — | ✗ |
| nt=1..30 (with flush8) | event | 1395–1462 | — | ✗ |
| bench_kineto (flush8) | CUPTI | 1387* | 6859 | ✗ |
| br=none / br=flush8 | event | 1275–1290 | 5585–6483 | ✗ (heavy throttle) |

\* bench_kineto's clk_min is pessimistic: my window spans its discarded warmup window too; its
CUPTI TFLOPS reflects only the (shorter, less-throttled) active window.

## Key findings

1. **The breather dominates; nothing else comes close.** Whether the clock holds 2062 is
   decided almost entirely by inserting a low-power interlude **sized ≈ the kernel time**
   between timed kernels. No breather / fixed 8GB flush → clock collapses to ~1275–1500.
   A kernel-sized breather → stays 2062.

2. **`torch.cuda._sleep` is the best breather** — perfect 2062 on every shape AND the highest
   TFLOPS (~6800). It's a low-power GPU spin: it keeps the clock from throttling **without
   memory traffic, so L2 stays warm** (no cold-read penalty). Adaptive flush (`flushA`) is
   nearly as good on clock (worst 1935) and gives ~same TFLOPS (for these large shapes the
   working set ≫ L2, so warm-vs-cold barely matters: 6817 vs 6810).

3. **num_tests barely matters once the breather is right.** With `_sleep`, nt=1 and nt=10 both
   hold 2062. Without a good breather, every num_tests throttles. So breather ≫ num_tests.

4. **Timing method is NOT the lever.** bench_kineto / event / graph / event+sync all throttle
   similarly (~1370–1500 worst) when fed the insufficient 8GB flush. Graph (tightest
   back-to-back) is slightly worse, as expected. Method affects absolute TFLOPS (CUPTI excludes
   launch edges → reads higher) but not clock stability.

5. **Warmup**: long back-to-back warmups (wu=30/60) slightly pre-throttle the timed window;
   short warmup (wu=2) or **a 0.5s idle right before timing** lets the clock recover to 2062.
   The idle-before trick works but is less clean than the `_sleep` breather.

6. **fixed 8GB flush is too small for big GEMMs** (~1.5ms breather vs a 1.3–5.5ms kernel →
   duty cycle too high → still throttles). It's the deepgemm default and is the reason large
   shapes throttle under `bench_kineto`.

## Recommended scheme (stable full-clock measurement)

**event-pair timing (per-iteration cudaEvent, sync once) + a `torch.cuda._sleep` breather
sized to ≈ the kernel time, placed between iterations (outside the timed bracket) +
num_tests≈10 + warmup≈10.**

- Holds **2062 MHz on every large shape**, gives the **highest, warm-cache TFLOPS (~6800)**.
- Full-clock perf is ~15–25% above the throttled numbers you get with no/insufficient breather.
- Always verify per run with window-level NVML that clk_min ≈ 2062.
- If you specifically want *cold-L2* numbers (mimicking back-to-back GEMMs on different data),
  use `flushA` (adaptive flush) instead — nearly as clock-stable, cold cache.
