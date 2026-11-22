"""Même règle que analyze_combo2 (fp8 si LLR8>l8 groupés ; sinon isolé si LLRL>lL ; sinon base), LLR profilés en T,
mais moitiés de 2 répétitions (88 réponses/endpoint) : moitié A (rép. 0-1) notée par l'adaptateur appris sur
ref-07 rép. 2-3, moitié B (rép. 2-3) par l'adaptateur appris sur rép. 0-1. Seuils choisis sur une moitié, évalués
sur l'autre, dans les deux sens. Argument : index s (1 -> 0.2, 2 -> 0.5)."""
import sys, numpy as np, itertools, json
D = "/Users/selyan/dev/eaic-paris-2026/docs/task2-bench-20261121/"; sys.path.insert(0, D); from anlib import *
SI = int(sys.argv[1]); z = np.load(D + "wb_pilot.npz")
LA, LB = np.load(D + "wb_lora2_trB.npz"), np.load(D + "wb_lora2_trA.npz")
def feats(L, reps):
    F = {}
    for e in EPS:
        mm = (z["ep"] == e) & np.isin(z["rep"], reps); m = (L["ep"] == e) & np.isin(L["rep"], reps)
        F[e] = (float(z["lp_q"][mm].mean(0).max() - z["lp_b"][mm].mean(0).max()),
                float(L["lp"][m, SI].mean(0).max() - L["lp"][m, 0].mean(0).max()))
    return F
FA, FB = feats(LA, (0, 1)), feats(LB, (2, 3))
for e in EPS: print(e, KIND[e].ljust(8), "A LLR8 %+6.1f LLRL %+6.1f | B LLR8 %+6.1f LLRL %+6.1f" % (FA[e][0]*1e3, FA[e][1]*1e3, FB[e][0]*1e3, FB[e][1]*1e3))
def part(F, l8, lL): return {e: ("fp8" if F[e][0] > l8 else (e if F[e][1] > lL else "base")) for e in EPS}
res = {}
for nm, (S, T) in [("A->B", (FA, FB)), ("B->A", (FB, FA))]:
    c8 = sorted({S[e][0] + 1e-9 for e in EPS}) + [9.0]; cL = sorted({S[e][1] + 1e-9 for e in EPS}) + [9.0]
    best = max(itertools.product(c8, cL), key=lambda p: (mcc(TRUTH, part(S, *p)), -p[0], -p[1]))
    pe = part(T, *best)
    res[nm] = dict(l8=best[0], lL=best[1], mcc_sel=mcc(TRUTH, part(S, *best)), mcc_eval=mcc(TRUTH, pe), errors=pair_errors(pe), partition=pe)
    print(nm, "l8 %.4f lL %.4f MCC sel %.3f HORS SEL %.3f" % (best[0], best[1], res[nm]["mcc_sel"], res[nm]["mcc_eval"]), pair_errors(pe), pe)
json.dump(res, open(D + f"combo_halves_s{SI}.json", "w"), indent=1)
