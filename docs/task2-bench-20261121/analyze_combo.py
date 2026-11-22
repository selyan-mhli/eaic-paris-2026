"""Règle combinée (fixée avant lecture des sorties wb_lora) :
  LLR8(e)  = mean(lp_fp8sim - lp_base) à T_hat ;  LLRL(e) = mean(lp_{base+0.2Δ̂} - lp_base) à T_hat,
  Δ̂ distillée sur ref-07 des répétitions d'entraînement (jamais évaluée sur ces répétitions).
  Partition : fp8 si LLR8 > l8 (groupés) ; sinon "lora" si LLRL > lL (isolés) ; sinon base.
  Validation : adaptateur entraîné sur {0,1}, seuils choisis sur la rép. 2 et évalués sur la 3, et inversement ;
  idem avec l'adaptateur entraîné sur {2,3} (sélection 0 -> éval 1, et 1 -> 0). 4 directions rapportées."""
import sys, numpy as np, itertools, json
sys.path.insert(0, "/Users/selyan/dev/eaic-paris-2026/docs/task2-bench-20261121"); from anlib import *
D = "/Users/selyan/dev/eaic-paris-2026/docs/task2-bench-20261121/"
z = np.load(D + "wb_pilot.npz"); TG = z["TG"]
THAT = {e: int(np.argmax(z["lp_b"][z["ep"] == e].mean(0))) for e in EPS}
def f8(e, r):
    m = (z["ep"] == e) & (z["rep"] == r); k = THAT[e]
    return float((z["lp_q"][m, k] - z["lp_b"][m, k]).mean())
def part(F, l8, lL):
    return {e: ("fp8" if F[e][0] > l8 else (e if F[e][1] > lL else "base")) for e in EPS}
res = {}
for fn, pairs in [("wb_lora_trA.npz", [(2, 3), (3, 2)]), ("wb_lora_trB.npz", [(0, 1), (1, 0)])]:
    L = np.load(D + fn)
    F = {r: {e: (f8(e, r), float((L["lp02"] - L["lp0"])[(L["ep"] == e) & (L["rep"] == r)].mean())) for e in EPS}
         for r in set(L["rep"].tolist())}
    for r in sorted(F):
        print(fn, "rep", r, " ".join(f"{e[-2:]}:{KIND[e][:5]} {F[r][e][0]*1e3:+.1f}/{F[r][e][1]*1e3:+.1f}" for e in EPS))
    for s, t in pairs:
        c8 = sorted({F[s][e][0] + 1e-9 for e in EPS}) + [9.0]; cL = sorted({F[s][e][1] + 1e-9 for e in EPS}) + [9.0]
        best = max(itertools.product(c8, cL), key=lambda p: (mcc(TRUTH, part(F[s], *p)), -p[0], -p[1]))
        pe = part(F[t], *best); key = f"{fn} sel{s}->eval{t}"
        res[key] = dict(l8=best[0], lL=best[1], mcc_sel=mcc(TRUTH, part(F[s], *best)), mcc_eval=mcc(TRUTH, pe), errors=pair_errors(pe), partition=pe)
        print(key, "MCC sel %.3f HORS SEL %.3f" % (res[key]["mcc_sel"], res[key]["mcc_eval"]), pair_errors(pe))
json.dump(res, open(D + "combo_analysis.json", "w"), indent=1)
