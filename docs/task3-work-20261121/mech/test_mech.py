"""Tests CPU du job « mécanismes » avec un petit modèle factice (aucun poids ni donnée du concours).
    .venv/bin/python docs/task3-work-20261121/mech/test_mech.py
"""
import argparse
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mech  # noqa: E402
from mech_holdout import rest_kl  # noqa: E402

C, H = 10, 16


class Tiny(torch.nn.Module):
    """B H W 3 -> B C, avec des noms de paramètres « patch_embed » (gelé) et « head »."""
    def __init__(self):
        super().__init__()
        self.patch_embed = torch.nn.Conv2d(3, 8, 4, stride=4)
        self.blocks = torch.nn.Sequential(torch.nn.GELU(), torch.nn.Conv2d(8, 8, 3, padding=1), torch.nn.GELU())
        self.head = torch.nn.Linear(8, C)

    def forward(self, x):                      # x : B 3 H W
        h = self.blocks(self.patch_embed(x))   # B 8 H/4 W/4
        return self.head(h.mean((2, 3)))       # B 8 -> B C


def factory():
    torch.manual_seed(0)
    return Tiny()


def preprocess(u8):                            # B H W 3 uint8 -> B 3 H W dans [-1, 1]
    return (u8.permute(0, 3, 1, 2).float() / 255 - 0.5) / 0.5


def augment(u8):
    return (preprocess(u8.flip(2)) + 0.2 * torch.randn(len(u8), 3, H, H)).clamp(-1, 1)


def pshuffle(u8):
    return u8.flip(1).contiguous()


def args(variant, out, **kw):
    d = dict(variant=variant, tag={"npo": "a", "npo_repair": "b", "conf": "c", "conf_repair": "d"}[variant], out_dir=str(out), batch_size=4, lr=1e-3, beta=0.1,
             proxy_weight=2.0, anchor_weight=1e-3, max_steps=6, eval_every=2, save_every=2,
             forget_minutes=5.0, save_every_forget=2, repair_steps=4, repair_lr=1e-3, keep_weight=1.0)
    d.update(kw)
    return argparse.Namespace(**d)


def test_pure():
    # cible de confiance : CE atteinte exactement, autres logits intacts, jamais de hausse
    z = torch.randn(6, C) * 3
    y = z.argmax(1)
    z[torch.arange(6), y] += 10
    ce = torch.tensor([0.01, 0.1, 0.5, 1.0, 3.0, 8.0])
    t = mech.conf_target(z, y, ce)
    got = F.cross_entropy(t, y, reduction="none")
    assert torch.allclose(got, ce, atol=1e-4), (got, ce)
    mask = F.one_hot(y, C).bool()
    assert torch.equal(t[~mask], z[~mask])
    low = mech.conf_target(z, y, torch.full((6,), 1e-9))
    assert torch.all(low.gather(1, y[:, None]) <= z.gather(1, y[:, None]) + 1e-6)
    # CE cibles : lognormale (moyenne ≈ 1,144 ; écart-type ≈ 2,66 au plafond près), rang de difficulté
    d = torch.rand(3500, generator=torch.Generator().manual_seed(0))
    c = mech.target_ce_by_rank(d)
    assert abs(c.mean().item() - 1.1444) < 0.01 and abs(c.std(unbiased=False).item() - 2.6609) < 0.02, (c.mean(), c.std())
    cs = c[torch.argsort(d.double(), stable=True)]  # d contient des ex aequo
    assert torch.all(cs[1:] >= cs[:-1])                        # croissant avec la difficulté
    assert c.max().item() <= mech.CE_CAP
    frac_flip = (c > math.log(2)).float().mean().item()
    # découpage par image : disjoint, complet, déterministe
    r, k = mech.split_indices(1530)
    assert len(set(r) & set(k)) == 0 and len(r) + len(k) == 1530
    r2, _ = mech.split_indices(1530)
    assert np.array_equal(r, r2)
    # image et transformation dans le même lot
    off = torch.randint(0, 256, (20, H, H, 3), dtype=torch.uint8)
    idx = torch.tensor([3, 7, 11])
    bt = mech.repair_batch(off, idx, torch.Generator().manual_seed(0))
    assert bt.shape[0] == 6 and torch.equal(bt[:3], off[idx])
    # rest_kl nul si seul le gagnant bouge
    z2 = z.clone(); z2[torch.arange(6), y] -= 5
    assert rest_kl(z, z2, y) < 1e-6 and rest_kl(z, z + torch.randn_like(z), y) > 1e-3
    return frac_flip


def test_grads_and_same_view():
    st, ref = factory(), factory().eval().requires_grad_(False)
    x = preprocess(torch.randint(0, 256, (4, H, H, 3), dtype=torch.uint8))
    y = ref(x).argmax(1)
    loss, delta = mech.npo_loss(st, ref, x, y, 0.1)
    assert abs(delta) < 1e-6                                    # mêmes poids, même vue
    orig_params = {n: p.detach().clone() for n, p in st.named_parameters()}
    for lf in (loss,
               mech.conf_loss(st, ref, [x, x.flip(-1)], y, torch.full((4,), 1.0)),
               mech.base_reg(st, ref, x, orig_params, 2.0, 1e-3) + mech.conf_loss(st, ref, [x], y, torch.ones(4)),
               mech.repair_loss(st, ref, ref, x, x, 1.0) + mech.conf_loss(st, ref, [x], y, torch.ones(4))):
        st.zero_grad()
        assert torch.isfinite(lf)
        lf.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in st.parameters())
        assert all(p.grad is None for p in ref.parameters())


def test_runs():
    torch.manual_seed(0)
    forget = torch.randint(0, 256, (24, H, H, 3), dtype=torch.uint8)
    off = torch.randint(0, 256, (30, H, H, 3), dtype=torch.uint8)
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "mech_out" / "123"
        for v in ("npo", "npo_repair", "conf", "conf_repair"):
            orig = factory()
            before = {k: t.clone() for k, t in orig.state_dict().items()}
            files = mech.run(args(v, out), model_factory=factory, orig=orig, forget_u8=forget, off_u8=off,
                             preprocess=preprocess, augment=augment, patch_shuffle=pshuffle,
                             altered_u8=forget.flip(2).contiguous(), device="cpu")
            assert files and all(p.exists() for p in files), v
            assert all(torch.equal(before[k], t) for k, t in orig.state_dict().items()), "origine modifiée"
            for p in files:
                sd = torch.load(p, weights_only=True)
                assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in sd.values())
                factory().load_state_dict(sd, strict=True)
            last = torch.load(files[-1], weights_only=True)
            assert torch.equal(last["patch_embed.weight"], before["patch_embed.weight"])   # gelé
            assert not torch.equal(last["head.weight"], before["head.weight"])              # entraîné
            if v.endswith("_repair"):
                assert any("_r_" in p.name for p in files) and sum("_f_" in p.name for p in files) >= 1
            n_files = len(list(out.glob("*.pt")))
            try:  # même tag, même dossier : refus d'écraser
                mech.run(args(v, out), model_factory=factory, orig=factory(), forget_u8=forget, off_u8=off,
                         preprocess=preprocess, augment=augment, patch_shuffle=pshuffle,
                         altered_u8=forget.flip(2).contiguous(), device="cpu")
                raise AssertionError("écrasement non détecté")
            except FileExistsError:
                pass
            assert len(list(out.glob("*.pt"))) == n_files
        assert not list(out.glob("*.tmp"))
        print("fichiers écrits :", sorted(p.name for p in out.glob("*.pt")))


def peaked():
    m = factory()
    with torch.no_grad():
        m.head.weight.mul_(200.0)              # origine très confiante : CE ~ 0, pas d'arrêt anticipé
    return m


def test_max_steps():
    torch.manual_seed(0)
    forget = torch.randint(0, 256, (24, H, H, 3), dtype=torch.uint8)
    off = torch.randint(0, 256, (30, H, H, 3), dtype=torch.uint8)
    with tempfile.TemporaryDirectory() as d:
        for v, tag in (("npo", "pa"), ("conf", "pc")):
            files = mech.run(args(v, Path(d), tag=tag, lr=1e-4), model_factory=factory, orig=peaked(),
                             forget_u8=forget, off_u8=off, preprocess=preprocess, augment=augment,
                             patch_shuffle=pshuffle, altered_u8=forget, device="cpu")
            names = [p.name for p in files]
            assert names == [f"mech_{tag}_f_s0002.pt", f"mech_{tag}_f_s0004.pt", f"mech_{tag}_f_s0006.pt"], names


if __name__ == "__main__":
    test_max_steps()
    ff = test_pure()
    test_grads_and_same_view()
    test_runs()
    print(f"OK test_mech (part des cibles CE > ln 2 : {ff:.3f})")
