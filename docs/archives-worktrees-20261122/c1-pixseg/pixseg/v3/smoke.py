"""Contrôle des formes sur CPU (quelques secondes) : python work_pixseg_v3/smoke.py"""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extmodel import ExtSeg
import train  # noqa: F401  (import seul : erreurs de syntaxe / noms)
torch.set_num_threads(4)
for m, r, nb in (("pixelseal", 256, 256), ("trustmark", 245, 100)):
    for msg in (False, True):
        net = ExtSeg(m, "cpu", r, use_msg=msg).train()
        x = torch.rand(2, 3, 448, 448); s = torch.randint(0, 2, (2, nb)).float() * 2 - 1
        y = net(x, s if msg else None); y.mean().backward()
        g = sum(p.grad is not None for p in net.enc_params())
        print(m, r, "msg", msg, "logits", tuple(y.shape), "grads enc", g, "lat top", net.lat[-1].in_channels, flush=True)
        net.eval(); assert net(torch.rand(1, 3, 512, 512), s[:1] if msg else None).shape == (1, 128, 128)
