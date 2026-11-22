"""Règle v3 (déguisé si hf<=a OU dup>=b OU sim<=c ; seuils par justesse kind + marge ; P1) sous budget :
K prompts pilot tirés au hasard (f01 toujours inclus) x 2 rép. par moitié = 2K réponses/endpoint/moitié. 200 tirages seedés."""
import random, itertools, sys, statistics
sys.path.insert(0, "/Users/selyan/dev/eaic-paris-2026/docs/task2-bench-20261121")
from anlib import *; from disguise_feats import collect, HF, tri, cos
R0 = {k: v for k, v in collect().items() if k[2] == "pil"}
PIDS = sorted({k[3] for k in R0}); DIS = {"ref-03", "ref-05", "ref-06", "ref-08"}
def FE_of(pids):
    FE = {}
    for e in EPS:
        for h in "AB":
            cells = [R0[(e, h, "pil", p)] for p in pids]
            rs = [x for v in cells for x in v]; pp = [(a, b) for v in cells for a, b in itertools.combinations(v, 2)]
            FE[(e, h)] = (sum(bool(HF.search(x)) for x in rs) / len(rs), sum(a == b for a, b in pp) / len(pp),
                          sum(cos(tri(a), tri(b)) for a, b in pp) / len(pp))
    return FE
def run(FE):
    def mids(i, h):
        v = sorted({FE[(e, h)][i] for e in EPS}); return [((a + b) / 2, b - a) for a, b in zip(v, v[1:])] + [(None, 0.0)]
    def dg(h, a, b, c): return {e for e in EPS if (a is not None and FE[(e, h)][0] <= a) or (b is not None and FE[(e, h)][1] >= b) or (c is not None and FE[(e, h)][2] <= c)}
    res = []
    for s, t in [("A", "B"), ("B", "A")]:
        best = None
        for (a, ma), (b, mb), (c, mc) in itertools.product(mids(0, s), mids(1, s), mids(2, s)):
            D = dg(s, a, b, c); key = (sum((e in D) == (e in DIS) for e in EPS), min([m for x, m in ((a, ma), (b, mb), (c, mc)) if x is not None] or [0]))
            if best is None or key > best[0]: best = (key, (a, b, c))
        Dt = dg(t, *best[1]); res.append(mcc(TRUTH, {e: ("G" if e in Dt else e) for e in EPS}))
    return res
for K in (4, 8, 16, 44):
    rnd = random.Random(K); out = []
    for _ in range(200 if K < 44 else 1):
        pids = PIDS if K == 44 else ["f01"] + rnd.sample([p for p in PIDS if p != "f01"], K - 1)
        out.append(run(FE_of(pids)))
    both = [min(x) for x in out]
    print(f"K={K:2d} prompts ({2*K} rép./endpoint/moitié) : MCC hors sél. moyen A->B {statistics.mean(o[0] for o in out):.3f} B->A {statistics.mean(o[1] for o in out):.3f} ;"
          f" P(min des 2 sens >= 0,60) = {sum(b >= 0.6 for b in both)/len(both):.2f} ; médiane du min {statistics.median(both):.3f}")
