# event (plain) vs event (graph replay) — DeepGEMM MXFP4

Both: warmup 5, run 10, **one cudaEvent pair around the 10-iter loop**, time = elapsed/10,
no L2 flush, 3 reps → median. Plain loop body = direct `fp8_fp4_gemm_nt` launch; graph loop
body = replay of a captured graph. All `shape.txt` shapes. Data: `results/event_vs_graph.csv`.

`g/p` = graph_TFLOPS / plain_TFLOPS (>1 → graph faster).

## Three regimes

1. **Small / launch-bound shapes (graph time < ~60 µs, 23 shapes): graph 1.3–2.6× faster.**
   Plain timing has a **~30–37 µs floor** that is *not* the kernel — it's DeepGEMM's per-call
   HOST/launch overhead (shape checks, scale-layout transform setup, TMA descriptor build, JIT
   cache lookup, launch) serializing with each launch. Graph captures all that once → replay is
   pure GPU. Extreme: 1024³ plain 32.8µs/66TF vs graph 12.6µs/171TF (**2.6×**).
   → For small shapes, **plain one-event-pair timing measures mostly host overhead, not the GPU
   kernel.** You must use graph (or CUPTI per-kernel) to measure the kernel itself.

2. **Medium shapes: speedup tapers to ~1.0** as the kernel grows past the ~30 µs overhead
   (e.g. 2048³ 2.42× → 4096³ 1.06× → 8192³ 1.006×). Crossover ≈ when kernel ≳ 100–200 µs.

3. **Large / compute-bound shapes (graph time > 1 ms, 13 shapes): graph ≈ plain, and for the
   biggest, graph is SLIGHTLY SLOWER (0.90–0.99).** e.g. 12288³ 0.93, 16384×12288×12288 0.905,
   32768×32768×6144 0.92. Reason: graph replay is the **tightest possible back-to-back (zero CPU
   gaps) → it power-throttles MORE**. The clock columns confirm it (e.g. 16384×12288×16384:
   plain clk 2062, graph clk 1695). Plain's tiny inter-launch CPU gaps act as micro-breathers
   that keep the clock a bit higher → plain reads slightly faster at the top end.

## Takeaways

- **Method choice matters enormously for small shapes (up to 2.6×), negligibly for big ones.**
  The whole difference is (a) host/launch overhead, which graph removes, and (b) at the very top,
  graph's zero-gap tightness throttling slightly more.
- Plain "one event pair around N direct launches" is **launch-overhead-bound below ~100 µs** —
  don't use it to compare kernels at small shapes.
- For a fair cross-backend kernel comparison, prefer graph replay (or CUPTI per-kernel time) so
  host/launch overhead doesn't dominate small/medium shapes.
