"""Unlearning NPO de Yihuan (unlearn_v2.py) + ancrage retain par distillation (esprit SCRUB).

Réutilise par import, sans les modifier, unlearn_v2.py, task_template.py et audit.py du dossier
de la tâche (../). Ajoute à la perte de v2 :
  ret_weight · KL(p_orig || p_cand) à T=2 (×T²) sur des images HORS forget (offforget.npz de
  make_offforget.py) passées par la même augmentation GPU que v2 ; l'enseignant est l'origine.
Sauvegarde un checkpoint float32 (strict=True vérifié par v2._save_state) toutes les
--save-every étapes dans --out-dir, à évaluer ensuite par drift.py. Pas de règle d'arrêt :
--max-steps étapes exactement.

Usage (depuis work_c3_eval/, sur un GPU de job) :
  python unlearn_anchor.py --lr 1e-5 --ret-weight 4 --tag r4_lr1e5 --max-steps 600
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
import task_template  # noqa: E402
import unlearn_v2 as v2  # noqa: E402



def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--proxy-weight", type=float, default=2.0)
    p.add_argument("--anchor-weight", type=float, default=1e-3)
    p.add_argument("--ret-weight", type=float, default=1.0)
    p.add_argument("--retain-npz", type=Path, default=Path("offforget.npz"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--save-every", type=int, default=60)
    p.add_argument("--tag", required=True)
    p.add_argument("--out-dir", type=Path, default=Path("anchor_ckpt"))
    a = p.parse_args()
    dev = task_template.DEVICE

    orig = task_template.load_model(TASK / "model.pt")
    orig.requires_grad_(False)
    images = task_template.load_forget_images(TASK / "forget_images.npz")
    with torch.no_grad():
        orig_logits = task_template.predict(orig, images)
    pseudo = orig_logits.argmax(1).cpu()
    log_p_orig_y = F.log_softmax(orig_logits.float(), 1).gather(1, pseudo[:, None]).squeeze(1).cpu()
    retain = torch.from_numpy(np.load(a.retain_npz)["images"]).to(dev)

    student = task_template.build_model()
    student.load_state_dict(orig.state_dict())
    student.to(dev)
    orig_params = {n: q.detach().float().clone() for n, q in student.named_parameters()}
    trainable = []
    for n, q in student.named_parameters():
        q.requires_grad_(not v2._is_frozen(n))
        if q.requires_grad:
            trainable.append(q)
    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    images_gpu = images.to(dev, non_blocking=True)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    print(f"tag={a.tag} lr={a.lr:g} ret_weight={a.ret_weight:g} proxy_weight={a.proxy_weight:g} "
          f"steps={a.max_steps} retain={len(retain)} images", flush=True)

    step, t0 = 0, time.perf_counter()
    student.train()
    while step < a.max_steps:
        order = torch.randperm(len(images))
        for s in range(0, len(order), a.batch_size):
            if step >= a.max_steps:
                break
            idx = order[s:s + a.batch_size]
            batch = v2._augment_batch(images_gpu[idx])
            proxy = task_template.preprocess(audit.patch_shuffle(images[idx]))
            ridx = torch.randint(0, len(retain), (a.batch_size,), device=dev)
            rbatch = v2._augment_batch(retain[ridx])
            y = pseudo[idx].to(dev)
            ref_y = log_p_orig_y[idx].to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                l_sub = student(batch)
                l_proxy = student(proxy)
                l_ret = student(rbatch)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                o_proxy = orig(proxy)
                o_ret = orig(rbatch)
            lp_y = F.log_softmax(l_sub.float(), 1).gather(1, y[:, None]).squeeze(1)
            loss_npo = v2._npo(lp_y, ref_y, a.beta)
            loss_proxy = v2._retain_proxy(l_proxy, o_proxy)
            loss_ret = v2._retain_proxy(l_ret, o_ret)
            anchor = v2._anchor(student, orig_params)
            loss = loss_npo + a.proxy_weight * loss_proxy + a.ret_weight * loss_ret + a.anchor_weight * anchor
            if not torch.isfinite(loss):
                raise RuntimeError(f"perte non finie à l'étape {step + 1}")
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % a.save_every == 0:
                path = a.out_dir / f"anc_{a.tag}_s{step:04d}.pt"
                student.eval()
                v2._save_state(student, path)
                student.train()
                print(f"step {step} npo={loss_npo.item():.4f} ret={loss_ret.item():.4f} "
                      f"proxy={loss_proxy.item():.4f} delta={(lp_y.detach() - ref_y).mean().item():.3f} "
                      f"{step / (time.perf_counter() - t0):.2f} step/s saved {path}", flush=True)


if __name__ == "__main__":
    main()
