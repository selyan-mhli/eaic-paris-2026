"""Jouet (données SYNTHÉTIQUES, aucun poids du concours) : où se place l'optimum d'une baisse
uniforme alpha du seul logit gagnant, sous la formule officielle, selon la précision du gold.

Modèle : structure de classes partagée s ~ N(0, sd_s) ; origine = s + bruit, logit vrai porté à
(max autres + marge_o), marge_o ~ N(15, 3.6), calée pour retrouver les taux
de changement de push_table (2,5 % à alpha 8, 0,26 % à alpha 5) ; gold = moyenne de 3 modèles
s + bruit_k + b_k·1[y], b ~ N(bg, 2.5). Candidat = origine, logit y diminué de alpha.
Sortie : pour chaque bg (≈ précision gold), alpha* de ms, de pred, du produit ; CE/std gold.
Utilité : ordre de grandeur et forme de la courbe, PAS un score attendu.
"""
import numpy as np

from official_formula import ce_losses, forget_score

rng = np.random.default_rng(0)
N, C = 3000, 100


def make(bg, mo=15.0, sd_s=2.0, sd_n=1.0, sd_m=3.6):
    y = rng.integers(0, C, N)
    s = rng.normal(0, sd_s, (N, C))
    zo = s + rng.normal(0, sd_n, (N, C))
    oth = zo.copy(); oth[np.arange(N), y] = -np.inf
    zo[np.arange(N), y] = oth.max(1) + rng.normal(mo, sd_m, N)
    golds = []
    for _ in range(3):
        zg = s + rng.normal(0, sd_n, (N, C))
        zg[np.arange(N), y] += rng.normal(bg, 2.5, N)
        golds.append(zg)
    return y, zo, golds


for bg in (6.0, 8.0, 10.0, 12.0):
    y, zo, golds = make(bg)
    acc = np.mean([(g.argmax(1) == y).mean() for g in golds])
    lg = ce_losses(golds[0], y)
    best = []
    for a in np.arange(0, 26, 1.0):
        zc = zo.copy(); zc[np.arange(N), y] -= a
        r = forget_score([{"cand": zc, "orig": zo, "gold": golds, "labels": y}])
        flip = (zc.argmax(1) != y).mean()
        best.append((a, r["ms"], r["pred"], r["forget"], flip, ce_losses(zc, y).mean()))
    b = np.array(best)
    i = b[:, 3].argmax()
    print(f"bg={bg}: acc gold {acc:.2f} CE gold {lg.mean():.2f}/{lg.std():.2f} | "
          f"alpha*(ms)={b[b[:,1].argmax(),0]:.0f} alpha*(pred)={b[b[:,2].argmax(),0]:.0f} "
          f"alpha*(prod)={b[i,0]:.0f} prod={b[i,3]:.3f} flip={b[i,4]:.2f} CE={b[i,5]:.2f}")
    print("   alpha  ms    pred  forget flip  CE")
    for row in b[::2]:
        print("   %5.0f %.3f %.3f %.3f  %.2f %.2f" % tuple(row))
