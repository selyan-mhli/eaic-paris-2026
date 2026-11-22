"""Cache des features pré-tête (entrée de model.head = sortie fc_norm, 768 dim) de l'origine,
pour rejouer hors ligne toute édition limitée à la tête (head_search.py).

Vues EXACTEMENT celles de score_like.py : forget, altered = score_like.mild(forget),
pshuffle = audit.patch_shuffle, lowpass32 = distill_mix._lowpass(.,32), mask35 = _mask(.,0.35),
off = offforget.npz ; plus offforget_all.npz (moitiés réparation/contrôle de mech).
Clés du npz (un seul fichier, non compressé, ~75 Mo) :
  f16_<vue>        [N, 768] float32 : features calculées sous autocast fp16 (comme le serveur)
  f32_forget       [3500, 768] : forward fp32 (pour fp16flip)
  y                [3500] pseudo-labels (argmax de l'origine, forward complet fp16)
  full_<nom>       [3500, 100] logits du forward COMPLET fp16 sur les forget (contrôle d'émulation)
  W_<nom>, b_<nom> tête de chaque modèle ; head_only_<nom> = 1 si seuls head.* diffèrent de l'origine
Usage (depuis work_c3_eval/, 1 GPU) :
  python mech/cache_feats.py --out mech_out/<job>/feats_<job>.npz \
      --heads push_best=../output/push_best.pt push_s125=push_scale/push_s125.pt ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def feats_and_logits(model, u8, preprocess, fp16: bool, bs: int = 128):
    fs, ls = [], []
    h = model.head.register_forward_hook(lambda m, i, o: (fs.append(i[0].detach().float().cpu()),
                                                          ls.append(o.detach().float().cpu())))
    model.eval()
    try:
        for s in range(0, len(u8), bs):
            x = preprocess(u8[s:s + bs])
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16 and x.is_cuda):
                model(x)
    finally:
        h.remove()
    return torch.cat(fs), torch.cat(ls)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--off", type=Path, default=Path("offforget.npz"))
    ap.add_argument("--off-all", type=Path, default=Path("offforget_all.npz"))
    ap.add_argument("--heads", nargs="*", default=[], help="nom=chemin.pt")
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError(f"refus d'écraser {a.out}")
    wd = Path.cwd()
    sys.path.insert(0, str(wd))
    sys.path.insert(0, str(wd.parent))
    import audit  # noqa: E402
    import score_like as sl  # noqa: E402
    import task_template as tt  # noqa: E402
    from distill_mix import _lowpass, _mask  # noqa: E402

    orig = tt.load_model(wd.parent / "model.pt")
    forget = tt.load_forget_images(wd.parent / "forget_images.npz")
    views = {"forget": forget, "altered": sl.mild(forget), "pshuffle": audit.patch_shuffle(forget),
             "lowpass32": _lowpass(forget, 32), "mask35": _mask(forget, 0.35),
             "off": torch.from_numpy(np.load(a.off)["images"]),
             "offall": torch.from_numpy(np.load(a.off_all)["images"])}
    out = {}
    for k, v in views.items():
        f, lg = feats_and_logits(orig, v, tt.preprocess, True)
        out[f"f16_{k}"] = f.numpy()                                  # N 768
        if k == "forget":
            out["y"] = lg.argmax(1).numpy()
            out["full_orig"] = lg.numpy()                            # N 100
        print(f"{k}: {tuple(f.shape)}", flush=True)
    out["f32_forget"] = feats_and_logits(orig, forget, tt.preprocess, False)[0].numpy()
    base = {k: v.detach().cpu() for k, v in orig.state_dict().items()}
    out["W_orig"], out["b_orig"] = base["head.weight"].float().numpy(), base["head.bias"].float().numpy()
    for spec in a.heads:
        name, path = spec.split("=", 1)
        sd = torch.load(path, map_location="cpu", weights_only=True)
        diff = [k for k in base if not torch.equal(base[k].float(), sd[k].float())]
        head_only = all(k.startswith("head.") for k in diff)
        out[f"W_{name}"], out[f"b_{name}"] = sd["head.weight"].float().numpy(), sd["head.bias"].float().numpy()
        out[f"head_only_{name}"] = np.array(int(head_only))
        m = tt.build_model()
        m.load_state_dict(sd, strict=True)
        m.to(tt.DEVICE)
        out[f"full_{name}"] = feats_and_logits(m, forget, tt.preprocess, True)[1].numpy()
        print(f"{name}: tête seule={head_only} ({len(diff)} tenseurs différents)", flush=True)
        del m
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(a.out.stem + ".part.npz")
    np.savez(tmp, **out)
    tmp.rename(a.out)
    print(f"saved {a.out} ({a.out.stat().st_size / 1e6:.1f} Mo)", flush=True)


if __name__ == "__main__":
    main()
