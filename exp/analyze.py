"""Summarize freq_sweep.csv: per scheme, how stable is the SM clock and what TFLOPS.
Aggregates the 3 repeats; clk_min = worst dip across reps (the stability signal)."""
import csv, os, statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, 'results', 'freq_sweep.csv'))))

shapes = sorted({r['shape'] for r in rows})
# aggregate over reps: key = (group,label,shape)
agg = defaultdict(lambda: {'cmin': [], 'cmed': [], 'tf': []})
meta = {}
for r in rows:
    k = (r['group'], r['label'], r['shape'])
    try:
        agg[k]['cmin'].append(float(r['clk_min'])); agg[k]['cmed'].append(float(r['clk_med']))
        agg[k]['tf'].append(float(r['tflops']))
    except ValueError:
        pass
    meta[(r['group'], r['label'])] = r['method']

def short(s):  # shorten shape label
    m, n, k = s.split('x'); return f"{int(m)//1024}k.{int(n)//1024}k.{int(k)//1024}k"

labels = []
seen = set()
for r in rows:
    if (r['group'], r['label']) not in seen:
        seen.add((r['group'], r['label'])); labels.append((r['group'], r['label']))

print("=== clk_min (worst dip over reps) per scheme x shape; <2000 = throttled ===")
hdr = f"{'group':<2} {'scheme':<18}" + "".join(f"{short(s):>11}" for s in shapes) + f"{'WORST':>7}"
print(hdr)
worst_by = {}
for (g, lab) in labels:
    cells = []
    worst = 9999
    for s in shapes:
        v = agg[(g, lab, s)]['cmin']
        mn = min(v) if v else float('nan')
        worst = min(worst, mn) if v else worst
        cells.append(f"{mn:>11.0f}")
    worst_by[(g, lab)] = worst
    print(f"{g:<2} {lab:<18}" + "".join(cells) + f"{worst:>7.0f}")

print("\n=== TFLOPS (median over reps) per scheme x shape ===")
print(hdr.replace('WORST', ' '))
for (g, lab) in labels:
    cells = []
    for s in shapes:
        v = agg[(g, lab, s)]['tf']
        cells.append(f"{statistics.median(v):>11.0f}" if v else f"{'-':>11}")
    print(f"{g:<2} {lab:<18}" + "".join(cells))

print("\n=== ranking by worst-case clk_min (higher = more stable across all shapes) ===")
for (g, lab), w in sorted(worst_by.items(), key=lambda x: -x[1]):
    # median tflops across shapes
    tfs = [statistics.median(agg[(g, lab, s)]['tf']) for s in shapes if agg[(g, lab, s)]['tf']]
    print(f"  worst_clk={w:>5.0f}  [{g}] {lab:<18} method={meta[(g,lab)]:<12} medTF(across shapes)={statistics.median(tfs):.0f}")
