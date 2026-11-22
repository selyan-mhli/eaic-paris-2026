"""Règle : déguisé si hf <= a (identité par défaut absente) OU dup_pil >= b (watermark) OU sim <= c (température).
Partition P1 : déguisés groupés, autres isolés. P2 : + le non-déguisé le plus proche du groupe (sim croisée trigrammes
moyenne avec les réponses des déguisés, mêmes prompts, même moitié). Seuils choisis sur une moitié (MCC P1), évalués sur l'autre."""
import itertools, sys
sys.path.insert(0, "/Users/selyan/dev/eaic-paris-2026/docs/task2-bench-20261121")
from anlib import *; from disguise_feats import collect, HF, tri, cos
R = collect()
def F(e, h):
    pil = {k: v for k, v in R.items() if k[0] == e and k[1] == h and k[2] == "pil"}
    allc = {k: v for k, v in R.items() if k[0] == e and k[1] == h}
    rs = [x for v in pil.values() for x in v]
    pp = [(a, b) for v in pil.values() for a, b in itertools.combinations(v, 2)]
    pa = [(a, b) for v in allc.values() for a, b in itertools.combinations(v, 2)]
    return (sum(bool(HF.search(x)) for x in rs) / len(rs), sum(a == b for a, b in pp) / len(pp),
            sum(cos(tri(a), tri(b)) for a, b in pa) / len(pa))
FE = {(e, h): F(e, h) for e in EPS for h in "AB"}
for e in EPS: print(e, KIND[e].ljust(8), " | ".join("%s hf %.3f dupP %.3f sim %.3f" % ((h,) + FE[(e, h)]) for h in "AB"))
def disg(h, a, b, c): return {e for e in EPS if FE[(e, h)][0] <= a or FE[(e, h)][1] >= b or FE[(e, h)][2] <= c}
def P1(D): return {e: ("G" if e in D else e) for e in EPS}
def P2(D, h):
    p = P1(D); others = [e for e in EPS if e not in D]
    if not D or not others: return p
    def cross(e):
        s = []
        for k, v in R.items():
            if k[0] != e or k[1] != h: continue
            for d in D:
                for x in v:
                    for y in R.get((d,) + k[1:], []): s.append(cos(tri(x), tri(y)))
        return sum(s) / len(s)
    best = max(others, key=cross); p[best] = "G"; return p
cand = lambda i: sorted({FE[k][i] for k in FE})
A_ = [x + 1e-9 for x in cand(0)] + [-1]; B_ = [x - 1e-9 for x in cand(1)] + [2]; C_ = [x + 1e-9 for x in cand(2)] + [-1]
for s, t in [("A", "B"), ("B", "A")]:
    best = max(itertools.product(A_, B_, C_), key=lambda p: (mcc(TRUTH, P1(disg(s, *p))), -abs(p[0]), p[1], -p[2]))
    Dt = disg(t, *best)
    print(f"{s}->{t} seuils hf<={best[0]:.3f} dup>={best[1]:.3f} sim<={best[2]:.3f} | sel P1 {mcc(TRUTH, P1(disg(s, *best))):.3f}"
          f" | HORS SEL P1 {mcc(TRUTH, P1(Dt)):.3f}  P2 {mcc(TRUTH, P2(Dt, t)):.3f} | déguisés éval {sorted(Dt)} | P2 {P2(Dt, t)}")

print("--- amendement post hoc : seuils aux milieux entre valeurs consécutives, départage par marge maximale")
def mids(i, h):
    v = sorted({FE[(e, h)][i] for e in EPS}); return [(-1.0, 0)] + [((a + b) / 2, b - a) for a, b in zip(v, v[1:])] + [(9.0, 0)]
for s, t in [("A", "B"), ("B", "A")]:
    best = None
    for (a, ma), (b, mb), (c, mc) in itertools.product(mids(0, s), mids(1, s), mids(2, s)):
        m = mcc(TRUTH, P1(disg(s, a, b, c)))
        key = (round(m, 6), min(x for x in (ma if a > -1 and a < 9 else 9, mb if 0 < b < 9 else 9, mc if 0 < c < 9 else 9)) )
        if best is None or key > best[0]: best = (key, (a, b, c))
    p = best[1]; Dt = disg(t, *p)
    print(f"{s}->{t} hf<={p[0]:.3f} dup>={p[1]:.3f} sim<={p[2]:.3f} | sel {best[0][0]:.3f} | HORS SEL P1 {mcc(TRUTH, P1(Dt)):.3f} P2 {mcc(TRUTH, P2(Dt, t)):.3f} | {sorted(Dt)} | erreurs P1 {pair_errors(P1(Dt))}")

print("--- v3 : objectif de sélection = justesse déguisé/non (kind), un seuil par feature, départage par marge")
DIS = {"ref-03", "ref-05", "ref-06", "ref-08"}
def mids2(i, h):
    v = sorted({FE[(e, h)][i] for e in EPS})
    return [((a + b) / 2, b - a) for a, b in zip(v, v[1:])] + [(None, 0.0)]
def disg2(h, a, b, c):
    return {e for e in EPS if (a is not None and FE[(e, h)][0] <= a) or (b is not None and FE[(e, h)][1] >= b) or (c is not None and FE[(e, h)][2] <= c)}
out = {}
for s, t in [("A", "B"), ("B", "A")]:
    best = None
    for (a, ma), (b, mb), (c, mc) in itertools.product(mids2(0, s), mids2(1, s), mids2(2, s)):
        D = disg2(s, a, b, c); acc = sum((e in D) == (e in DIS) for e in EPS)
        marg = min([m for x, m in ((a, ma), (b, mb), (c, mc)) if x is not None] or [0])
        key = (acc, marg)
        if best is None or key > best[0]: best = (key, (a, b, c))
    p = best[1]; Dt = disg2(t, *p)
    r1, r2 = mcc(TRUTH, P1(Dt)), mcc(TRUTH, P2(Dt, t))
    out[f"{s}->{t}"] = dict(thr=p, acc_sel=best[0][0], disguised_eval=sorted(Dt), P1=r1, P2=r2, P2_part=P2(Dt, t))
    print(f"{s}->{t} seuils {p} | justesse sel {best[0][0]}/9 | HORS SEL déguisés {sorted(Dt)} | P1 {r1:.3f} | P2 {r2:.3f} {P2(Dt, t)}")
    print("   erreurs P2", pair_errors(P2(Dt, t)))
import json; json.dump(out, open("/Users/selyan/dev/eaic-paris-2026/docs/task2-bench-20261121/disguise_cv.json", "w"), indent=1, default=str)
