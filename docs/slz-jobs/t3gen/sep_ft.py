"""T3 : séparer « image mémorisée » et « classe ». Fine-tuning des K derniers blocs + tête :
  forget : KL(cible || modèle) à T=2, cible = logits de l'origine avec la tête push x S (S = --push)
  ancres : KL(origine || modèle) à T=2 sur les variations img2img (gen_out, mêmes classes, non vues)
           + photos offforget_all ; poids --lam.
Contrôle (jamais entraîné) : variations s60 des images d'indice impair + moitié des offforget.
Sortie : sep_out/<tag>.pt (state dict float32 complet, strict) + une ligne de mesures.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

T = 2.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", type=float, default=1.75); ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--blocks", type=int, default=4); ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--tag", required=True); ap.add_argument("--aug", action="store_true")
    a = ap.parse_args()
    wd = Path.cwd(); sys.path[:0] = [str(wd), str(wd.parent)]
    import task_template as tt  # noqa: E402
    dev = "cuda"
    forget = tt.load_forget_images(wd.parent / "forget_images.npz")
    zs = [np.load(f) for f in sorted(glob.glob("gen_out/var_15904093_*.npz"))]
    idx = np.concatenate([z["idx"] for z in zs]); o = np.argsort(idx)
    v40 = np.concatenate([z["img_s40"] for z in zs])[o]; v60 = np.concatenate([z["img_s60"] for z in zs])[o]
    off = np.load("offforget_all.npz")["images"]
    n = len(forget); tr = np.arange(n) % 2 == 0
    anchors = np.concatenate([v40, v60[tr], off[: len(off) // 2]])
    ctl = np.concatenate([v60[~tr], off[len(off) // 2:]])
    orig = tt.load_model(wd.parent / "model.pt"); orig.requires_grad_(False)
    pb = torch.load("../output/push_best.pt", map_location="cpu", weights_only=True)
    Wo, bo = orig.head.weight.detach().clone(), orig.head.bias.detach().clone()
    Wt = Wo + a.push * (pb["head.weight"].to(dev) - Wo); bt = bo + a.push * (pb["head.bias"].to(dev) - bo)
    model = tt.load_model(wd.parent / "model.pt")
    nb = len(model.blocks)
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("head") or name.startswith("norm") or name.startswith("fc_norm")
                         or any(name.startswith(f"blocks.{i}.") for i in range(nb - a.blocks, nb)))

    @torch.no_grad()
    def feats_logits(m, u8, W=None, b=None):
        out = []
        for s in range(0, len(u8), 256):
            x = tt.preprocess(torch.as_tensor(u8[s:s + 256]))
            with torch.autocast("cuda", dtype=torch.float16):
                f = m.forward_head(m.forward_features(x), pre_logits=True)
            f = f.float(); out.append(f @ (W if W is not None else m.head.weight).T + (b if b is not None else m.head.bias))
        return torch.cat(out)

    tgt_f = F.softmax(feats_logits(orig, forget.numpy(), Wt, bt) / T, 1)
    tgt_a = F.softmax(feats_logits(orig, anchors) / T, 1)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(0)
    model.train()
    for step in range(a.steps):
        i = torch.randint(0, n, (a.bs,), generator=g); j = torch.randint(0, len(anchors), (a.bs,), generator=g)
        x = tt.preprocess(torch.cat([forget[i], torch.as_tensor(anchors[j.numpy()])]))
        if a.aug:  # mêmes altérations sur forget et ancres : flip, recadrage 70-100 % redimensionné, luminosité
            if torch.rand(()) < 0.5:
                x = x.flip(-1)
            c = int(224 * (0.7 + 0.3 * torch.rand(()).item())); oy, ox = torch.randint(0, 225 - c, (2,)).tolist()
            x = F.interpolate(x[..., oy:oy + c, ox:ox + c], size=224, mode="bilinear", align_corners=False)
            x = (x * (0.8 + 0.4 * torch.rand(x.shape[0], 1, 1, 1, device=x.device))).clamp(-1, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(x).float()
        lf = F.kl_div(F.log_softmax(lg[: a.bs] / T, 1), tgt_f[i.to(dev)], reduction="batchmean")
        la = F.kl_div(F.log_softmax(lg[a.bs:] / T, 1), tgt_a[j.to(dev)], reduction="batchmean")
        loss = lf + a.lam * la
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 200 == 0 or step == a.steps - 1:
            print(f"pas {step} forget {lf.item():.4f} ancres {la.item():.4f}", flush=True)
    model.eval()
    lo_f, lc_f = feats_logits(orig, forget.numpy()), feats_logits(model, forget.numpy())
    lt_f = feats_logits(orig, forget.numpy(), Wt, bt)
    y = lo_f.argmax(1)
    ce = lambda lg: F.cross_entropy(lg, y, reduction="none")
    lo_c, lc_c = feats_logits(orig, ctl), feats_logits(model, ctl)
    kl = lambda p, q: float(F.kl_div(F.log_softmax(q / T, 1), F.softmax(p / T, 1), reduction="batchmean"))
    print(f"RES {a.tag} | forget CE cand {ce(lc_f).mean():.3f}/{ce(lc_f).std():.3f} cible push {ce(lt_f).mean():.3f}/{ce(lt_f).std():.3f} "
          f"KL(cible||cand) {kl(lt_f, lc_f):.4f} flip {(lc_f.argmax(1) != y).float().mean():.3f} | CONTROLE accord "
          f"{(lc_c.argmax(1) == lo_c.argmax(1)).float().mean():.4f} KL {kl(lo_c, lc_c):.4f} "
          f"(push seul : accord {(feats_logits(orig, ctl, Wt, bt).argmax(1) == lo_c.argmax(1)).float().mean():.4f})", flush=True)
    sd = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}
    od = Path(os.environ.get("SEP_OUT", "sep_out")); od.mkdir(parents=True, exist_ok=True); torch.save(sd, od / f"{a.tag}.pt"); print("écrit", od / f"{a.tag}.pt", flush=True)


if __name__ == "__main__":
    main()
