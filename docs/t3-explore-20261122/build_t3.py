"""T3, exploration guidée par le serveur (GO Selyan 22/11 ~03:56). Éditions de la tête seulement.

Base : tête push = origine + k * (push_best - origine) ; k=1.75 = push_s175 (champion 0,0941).
Variantes FIGÉES avant soumission (une par nom) :
  P1  k=1.85
  P2  k=1.65
  P3  k=1.75, puis lignes de tête des 45 classes forget mises à l'échelle 0.8 (W et b) : aplatit ces classes
  P4  k=1.75, puis toute la tête divisée par 1.2 (température 1.2 sur toutes les classes)
  P5  k=1.75, puis toute la tête multipliée par 1/0.9 (température 0.9)
  P6/P7/P8  k=1.75, température 0.8 / 0.7 / 0.6 (ajoutées après P5 = 0,0986, nouveau meilleur)
Usage : python build_t3.py --model model.pt --push push_best.pt --feats feats.npz --variant P1 --out P1.pt
"""
import argparse, hashlib
import numpy as np
import torch

V = {"P1": dict(k=1.85), "P2": dict(k=1.65), "P3": dict(k=1.75, fscale=0.8),
     "P4": dict(k=1.75, temp=1.2), "P5": dict(k=1.75, temp=0.9),
     "P6": dict(k=1.75, temp=0.8), "P7": dict(k=1.75, temp=0.7), "P8": dict(k=1.75, temp=0.6)}


def sd(x):
    return x.get("state_dict", x) if isinstance(x, dict) and "state_dict" in x else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--push"); ap.add_argument("--feats")
    ap.add_argument("--variant", choices=list(V)); ap.add_argument("--out")
    a = ap.parse_args(); p = V[a.variant]
    o = sd(torch.load(a.model, map_location="cpu", weights_only=True))
    b = sd(torch.load(a.push, map_location="cpu", weights_only=True))
    diff = [n for n in o if not torch.equal(o[n], b[n])]
    assert diff == ["head.weight", "head.bias"], diff
    d = {n: t.clone() for n, t in o.items()}
    for n in diff:
        d[n] = (o[n].double() + p["k"] * (b[n].double() - o[n].double()))
    if "fscale" in p:
        cls = np.unique(np.load(a.feats)["y"]); assert len(cls) == 45
        d["head.weight"][cls] *= p["fscale"]; d["head.bias"][cls] *= p["fscale"]
    if "temp" in p:
        d["head.weight"] /= p["temp"]; d["head.bias"] /= p["temp"]
    for n in diff:
        d[n] = d[n].to(o[n].dtype)
    assert all(torch.isfinite(t).all() for t in d.values() if t.is_floating_point())
    assert set(d) == set(o) and all(d[n].shape == o[n].shape and d[n].dtype == o[n].dtype for n in o)
    torch.save(d, a.out)
    print(a.variant, p, hashlib.sha256(open(a.out, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
