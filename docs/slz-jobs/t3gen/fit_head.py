"""T3 hors ligne (CPU) : proxy gold par variations img2img + ajustement de la tête.

Entrée : feats.npz (f_forget, l_forget, f_s40, l_s40, f_s60, l_s60, W_orig, b_orig, W_pb, b_pb) tous alignés.
Proxy : gold(x_i) ~ softmax(origine(variation_i) / 2). Score proxy = A * B (forme officielle, sans altérées) :
  A = moyenne des progrès sur |moyenne| et |écart-type| de la CE (pseudo-label = argmax origine sur la forget)
  B = progrès de KL(gold || cand) moyen par image ; progrès = 1 - clip(d_cand / d_orig, 0, 1).
Validation du proxy : il doit classer push x1.25 < x1.5 < x1.75 comme le serveur (0,057 < 0,083 < 0,094).
Ajustement : tête initialisée à l'origine, KL(cible || cand) sur les forget (moitié A) + lam * KL(orig || cand)
sur les variations (préservation hors forget), évaluation sur la moitié B.
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

T = 2.0
z = np.load(sys.argv[1])
fF = torch.tensor(z["f_forget"]).float()
y = torch.tensor(z["l_forget"]).float().argmax(1)
Wo, bo = torch.tensor(z["W_orig"]).float(), torch.tensor(z["b_orig"]).float()
Wp, bp = torch.tensor(z["W_pb"]).float(), torch.tensor(z["b_pb"]).float()
rng = np.random.default_rng(0)
perm = rng.permutation(len(fF)); A_idx, B_idx = torch.tensor(perm[::2]), torch.tensor(perm[1::2])


def head(f, W, b):
    return f @ W.T + b


def stats(lg, yy):
    ce = F.cross_entropy(lg, yy, reduction="none")
    return ce.mean(), ce.std()


def proxy(W, b, idx, gold_key):
    lg_o, lg_c = head(fF[idx], Wo, bo), head(fF[idx], W, b)
    gl = torch.tensor(z[gold_key]).float()[idx]
    g = F.softmax(gl / T, 1)
    gm, gs = stats(gl, y[idx])
    om, os_ = stats(lg_o, y[idx]); cm, cs = stats(lg_c, y[idx])
    prog = lambda dc, do: float(1 - torch.clamp(dc / do, 0, 1)) if do > 0 else 0.0
    A = 0.5 * (prog(abs(cm - gm), abs(om - gm)) + prog(abs(cs - gs), abs(os_ - gs)))
    kl = lambda lg: F.kl_div(F.log_softmax(lg / T, 1), g, reduction="batchmean")
    B = prog(kl(lg_c), kl(lg_o))
    flip = float((lg_c.argmax(1) != lg_o.argmax(1)).float().mean())
    return A, B, A * B, flip, float(gm), float(cm)


def report(name, W, b):
    for gk in ("l_s40", "l_s60"):
        A, B, P, fl, gm, cm = proxy(W, b, B_idx, gk)
        print(f"{name:22s} gold={gk}  A={A:.3f} B={B:.3f}  A*B={P:.4f}  flip={fl:.3f}  CEgold={gm:.2f} CEcand={cm:.2f}")


print(f"n={len(fF)} ; accord top-1 origine forget/variation s40 "
      f"{(torch.tensor(z['l_s40']).float().argmax(1) == y).float().mean():.3f}, s60 "
      f"{(torch.tensor(z['l_s60']).float().argmax(1) == y).float().mean():.3f}")
report("origine", Wo, bo)
for s in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5):
    report(f"push x{s}", Wo + s * (Wp - Wo), bo + s * (bp - bo))

fV = torch.cat([torch.tensor(z["f_s40"]).float(), torch.tensor(z["f_s60"]).float()])
pV = F.softmax(head(fV, Wo, bo) / T, 1)
for gk in ("l_s40", "l_s60"):
    tgt = F.softmax(torch.tensor(z[gk]).float() / T, 1)
    for lam in (0.3, 1.0, 3.0):
        W = Wo.clone().requires_grad_(True); b = bo.clone().requires_grad_(True)
        opt = torch.optim.Adam([W, b], lr=1e-3)
        for it in range(300):
            lf = F.kl_div(F.log_softmax(head(fF[A_idx], W, b) / T, 1), tgt[A_idx], reduction="batchmean")
            lv = F.kl_div(F.log_softmax(head(fV, W, b) / T, 1), pV, reduction="batchmean")
            loss = lf + lam * lv
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            report(f"fit {gk[2:]} lam{lam}", W.detach(), b.detach())
            pres = float((head(fV, W, b).argmax(1) == head(fV, Wo, bo).argmax(1)).float().mean())
            print(f"   accord top-1 sur variations (préservation) {pres:.3f}")
            torch.save({"W": W.detach(), "b": b.detach()}, f"fit_{gk[2:]}_lam{lam}.pt")
