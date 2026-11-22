"""Version locale de l'idée « même classe, confiance réduite » (score_push / constrained_edit de
Yihuan, lus sans modification) : on entraîne seulement les K derniers blocs (tête gelée par défaut)
pour que les logits COMPLETS valent une cible construite.

Cible sur chaque image forget, son flip horizontal et une altération légère
(audit.make_altered_copies) : les logits de l'origine sur cette vue, avec le SEUL logit de la
classe y (argmax de l'origine sur la forget propre) abaissé à (second logit + m) s'il le
dépassait. Aucune autre classe ne bouge, donc aucune classe ne change en principe.
Ancrage (cible = logits de l'origine) : photos hors forget légèrement altérées (--off-npz),
patch-shuffle des forget (audit.patch_shuffle), mélanges 50/50 de forget de classes différentes.
Perte : MSE sur les 100 logits, forget + --hold-weight × ancrage.
Checkpoint toutes les --save-every étapes ; journal CE forget moy/éc et classes changées.

Usage (depuis work_c3_eval/, GPU de job) : python margin_edit.py --margin 1.5 --tag mg15
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

TASK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TASK))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import audit  # noqa: E402
import task_template as tt  # noqa: E402


def margin_target(z: torch.Tensor, y: torch.Tensor, m: float) -> torch.Tensor:
    """z [B, C] logits, y [B] : abaisse z[y] à max(z hors y) + m s'il le dépasse."""
    z = z.clone()
    zy = z.gather(1, y[:, None]).squeeze(1)
    others = z.scatter(1, y[:, None], float("-inf")).max(1).values
    z.scatter_(1, y[:, None], torch.minimum(zy, others + m)[:, None])
    return z


def mild(x_u8: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """uint8 NHWC -> uint8 NHWC : flip aléatoire et luminosité ±10 % (léger, pour l'ancrage)."""
    x = x_u8.float()
    flip = torch.rand(len(x), generator=g) < 0.5
    x[flip] = x[flip].flip(2)
    b = 1 + (torch.rand(len(x), 1, 1, 1, generator=g) * 2 - 1) * 0.1
    return (x * b).clamp(0, 255).round().to(torch.uint8)


def save_state(model, path: Path) -> None:
    sd = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}
    for k, v in sd.items():
        if not torch.isfinite(v).all():
            raise RuntimeError(f"poids non finis : {k}")
    tt.build_model().load_state_dict(sd, strict=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, required=True)
    ap.add_argument("--train-last", type=int, default=4)
    ap.add_argument("--train-head", action="store_true")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--hold-weight", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=125)
    ap.add_argument("--off-npz", type=Path, default=Path("offforget_all.npz"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("margin_out"))
    a = ap.parse_args()
    dev = tt.DEVICE
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)

    orig = tt.load_model(TASK / "model.pt")
    orig.requires_grad_(False)
    images = tt.load_forget_images(TASK / "forget_images.npz")
    views = (images, images.flip(2).contiguous(), audit.make_altered_copies(images))
    clean = tt.predict(orig, images).float()
    y = clean.argmax(1)
    tgt = clean.cpu()
    ce_t = F.cross_entropy(margin_target(clean, y, a.margin), y, reduction="none")
    print(f"tag={a.tag} margin={a.margin} : CE cible forget {ce_t.mean():.4f}/{ce_t.std(unbiased=False):.4f}", flush=True)
    off = torch.from_numpy(np.load(a.off_npz)["images"])

    student = tt.build_model()
    student.load_state_dict(orig.state_dict())
    student.to(dev)
    nb = len(student.blocks)
    keep = tuple(f"blocks.{i}." for i in range(nb - a.train_last, nb))
    params = []
    for n, p in student.named_parameters():
        train = n.startswith(keep) or n.startswith(("norm.", "fc_norm.")) or (a.train_head and n.startswith("head."))
        p.requires_grad_(train)
        if train:
            params.append(p)
    print(f"entraînables {sum(p.numel() for p in params)} (derniers {a.train_last} blocs, tête "
          f"{'entraînée' if a.train_head else 'gelée'})", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")

    def fwd(model, u8):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return model(tt.preprocess(u8)).float()

    @torch.no_grad()
    def report(step: int) -> None:
        student.eval()
        lg = tt.predict(student, images).float()
        ce = F.cross_entropy(lg, y, reduction="none")
        print(f"step {step}: CE forget {ce.mean():.4f}/{ce.std(unbiased=False):.4f} "
              f"changées {(lg.argmax(1) != y).float().mean():.4f}", flush=True)
        student.train()

    step, t0 = 0, time.perf_counter()
    student.train()
    while step < a.max_steps:
        order = torch.randperm(len(images))
        for s in range(0, len(order) - a.batch_size + 1, a.batch_size):
            if step >= a.max_steps:
                break
            idx = order[s:s + a.batch_size]
            yb = y[idx].to(dev)
            loss_f = 0.0
            for v in views:
                with torch.no_grad():
                    target = margin_target(fwd(orig, v[idx]), yb, a.margin)
                loss_f = loss_f + F.mse_loss(fwd(student, v[idx]), target)
            # ancrage : hors forget, patch-shuffle, mélanges de classes différentes
            oi = torch.randint(0, len(off), (a.batch_size,), generator=g)
            perm = idx[torch.randperm(len(idx))]
            mix = ((images[idx].float() + images[perm].float()) / 2).round().to(torch.uint8)
            mix = mix[(y[idx] != y[perm])]
            holds = [mild(off[oi], g), audit.patch_shuffle(images[idx]), mix]
            loss_h = 0.0
            for h in holds:
                if len(h) == 0:
                    continue
                with torch.no_grad():
                    ref = fwd(orig, h)
                loss_h = loss_h + F.mse_loss(fwd(student, h), ref)
            loss = loss_f / len(views) + a.hold_weight * loss_h / len(holds)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % 25 == 0:
                print(f"step {step} loss_f={float(loss_f) / len(views):.4f} loss_h={float(loss_h) / len(holds):.4f} "
                      f"{step / (time.perf_counter() - t0):.2f} step/s", flush=True)
            if step % a.save_every == 0:
                report(step)
                student.eval()
                save_state(student, a.out_dir / f"mg_{a.tag}_s{step:04d}.pt")
                student.train()


if __name__ == "__main__":
    main()
