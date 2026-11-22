"""Analyse boîte blanche (règle fixée avant lecture des résultats par endpoint).
Moitiés : A = répétitions 0-1, B = 2-3. Par endpoint et moitié :
  T_hat = argmax_T mean lp_b(T) (grille) ;
  LLR8  = mean(lp_q - lp_b) à T_hat (nats/token) : > 0 => plus proche du fp8 simulé ;
  GAP   = mean(lp_b(T_hat) + H_b(T_hat)) : observé - attendu ; < 0 => excès de surprise (contexte/poids différents).
Règle de partition : base = endpoints avec |GAP| <= g et LLR8 <= l ; fp8 = LLR8 > l (groupés ensemble) ;
autres = isolés. (g, l) choisis sur une moitié (MCC refs max), évalués sur l'autre, puis l'inverse."""
import sys, numpy as np, itertools, json
sys.path.insert(0, "/Users/selyan/dev/eaic-paris-2026/docs/task2-bench-20261121")
from anlib import *
z = np.load(sys.argv[1]); TG = z["TG"]
ep, rep = z["ep"], z["rep"]
def feats(mask):
    lpb, lpq, H = z["lp_b"][mask], z["lp_q"][mask], z["H_b"][mask]
    k = int(np.argmax(lpb.mean(0)))
    n = mask.sum()
    d = lpq[:, k] - lpb[:, k]; g = lpb[:, k] + H[:, k]
    return dict(T=float(TG[k]), LLR8=float(d.mean()), LLR8_se=float(d.std() / np.sqrt(n)),
                GAP=float(g.mean()), GAP_se=float(g.std() / np.sqrt(n)), n=int(n),
                argrate=float(z["isarg_b"][mask].mean()))
F = {}
for h, reps in [("A", (0, 1)), ("B", (2, 3))]:
    for e in EPS:
        F[(e, h)] = feats((ep == e) & np.isin(rep, reps))
for e in EPS:
    a, b = F[(e, "A")], F[(e, "B")]
    print(f"{e} {KIND[e]:8s} T {a['T']:.1f}/{b['T']:.1f}  LLR8 {a['LLR8']*1e3:+7.2f}/{b['LLR8']*1e3:+7.2f} (se {a['LLR8_se']*1e3:.2f}) mnat"
          f"  GAP {a['GAP']:+.3f}/{b['GAP']:+.3f} (se {a['GAP_se']:.3f})  argmax {a['argrate']:.2f}  n {a['n']}")
def part(h, g, l):
    P = {}
    for e in EPS:
        f = F[(e, h)]
        P[e] = "fp8" if f["LLR8"] > l else ("base" if abs(f["GAP"]) <= g else e)
    return P
gs = sorted({abs(F[k]["GAP"]) + 1e-6 for k in F}); ls = sorted({F[k]["LLR8"] + 1e-9 for k in F}) + [1.0]
res = {}
for s, t in [("A", "B"), ("B", "A")]:
    best = max(itertools.product(gs, ls), key=lambda p: mcc(TRUTH, part(s, *p)))
    pe = part(t, *best)
    res[f"{s}->{t}"] = dict(g=best[0], l=best[1], mcc_sel=mcc(TRUTH, part(s, *best)), mcc_eval=mcc(TRUTH, pe), errors=pair_errors(pe), partition=pe)
    print(s, "->", t, "g %.4f l %.5f" % best, "MCC sel %.3f HORS SEL %.3f" % (mcc(TRUTH, part(s, *best)), mcc(TRUTH, pe)), pair_errors(pe))
json.dump(dict(features={f"{k[0]}|{k[1]}": v for k, v in F.items()}, cv=res), open(sys.argv[2], "w"), indent=1)
