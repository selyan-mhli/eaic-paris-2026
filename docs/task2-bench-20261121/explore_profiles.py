"""EXPLORATOIRE (après lecture de analyze_wb) : profils par prompt (GAP et LLR8 à T_hat) et corrélations."""
import sys, numpy as np
sys.path.insert(0, "."); from anlib import *
z = np.load("wb_pilot.npz"); TG = z["TG"]; ep, rep, pid = z["ep"], z["rep"], z["pid"]
P = sorted(set(pid))
def prof(e, reps, what):
    m = (ep == e) & np.isin(rep, reps); k = int(np.argmax(z["lp_b"][m].mean(0)))
    v = []
    for p in P:
        mm = m & (pid == p)
        if what == "gap": x = z["lp_b"][mm, k] + z["H_b"][mm, k]
        else: x = z["lp_q"][mm, k] - z["lp_b"][mm, k]
        v.append(x.mean() if mm.sum() else 0)
    return np.array(v)
for what in ["gap", "llr"]:
    A = {e: prof(e, (0, 1), what) for e in EPS}; B = {e: prof(e, (2, 3), what) for e in EPS}
    print(what, "corr profil A(ligne) vs B(colonne)")
    print("        " + " ".join(e[-2:] + KIND[e][:4].rjust(5) for e in EPS))
    for a in EPS:
        print(a, KIND[a][:6].ljust(7), " ".join("%+6.2f   " % np.corrcoef(A[a], B[b])[0, 1] for b in EPS))
