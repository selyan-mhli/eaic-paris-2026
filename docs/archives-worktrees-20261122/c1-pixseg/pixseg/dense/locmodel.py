"""Localisateur dense sur les features du décodeur officiel (architecture prescrite par RAPPORT-CODEX.md §7).

  backbone officiel (gelé, ou dernier stage dégelé) -> 4 niveaux (pas 4, 8, 16, 32 de l'entrée 256 / 245)
  projection 1x1 -> 32 canaux par niveau ; au niveau 32 on ajoute la preuve au message connu :
     accord signé par position  a = logit_pos * (2m-1) / échelle   (b nbits h32 w32) -> 1x1 -> 32
     3 résumés (moyenne de a, part de bits justes - 0,5, moyenne de tanh a) -> 1x1 -> 32
     message projeté (Linear nbits -> 32), diffusé
  fusion descendante avec skips fins (pas 32 -> 4) ; remontée bilinéaire à 128
  petite branche de finition RGB passe-haut (512 -> 128, 16 canaux), concaténée ; tête 1x1 -> logits b 128 128
Statistiques de normalisation (échelle de l'accord, écart-type du passe-haut) : buffers, fixés par `fit_norm`
sur les images d'APPRENTISSAGE seulement.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

W = 32


def block(cin, cout):
    assert cout % 8 == 0, cout
    return nn.Sequential(nn.Conv2d(cin, cout, 3, 1, 1, bias=False), nn.GroupNorm(8, cout), nn.SiLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.GroupNorm(8, cout), nn.SiLU(inplace=True))


def highpass(img):
    """b 3 512 512 [0, 1] -> résidu passe-haut b 3 512 512 (image - moyenne 5x5)."""
    return img - F.avg_pool2d(img, 5, 1, 2, count_include_pad=False)


class DenseLoc(nn.Module):
    def __init__(self, backbone, nbits: int, use_msg: bool = True, use_rgb: bool = True, width: int = W):
        super().__init__()
        self.bb, self.nbits, self.use_msg, self.use_rgb = backbone, nbits, use_msg, use_rgb
        ch = list(backbone.channels)
        self.lat = nn.ModuleList([nn.Conv2d(c, width, 1) for c in ch])
        self.fuse = nn.ModuleList([block(width, width) for _ in ch[:-1]])
        self.top = block(width, width)
        if use_msg:
            self.agree_proj = nn.Conv2d(nbits, width, 1)
            self.summ_proj = nn.Conv2d(3, width, 1)
            self.msg_proj = nn.Linear(nbits, width)
        self.register_buffer("agree_scale", torch.ones(()))
        self.register_buffer("hp_std", torch.ones(1, 3, 1, 1))
        self.register_buffer("norm_fitted", torch.zeros((), dtype=torch.bool))
        if use_rgb:
            self.rgb = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.GroupNorm(8, 16), nn.SiLU(inplace=True),  # 512 -> 256
                                     nn.Conv2d(16, 16, 3, 2, 1), nn.GroupNorm(8, 16), nn.SiLU(inplace=True))  # -> 128
        self.refine = block(width + 16 * use_rgb, width)
        self.head = nn.Conv2d(width, 1, 1)

    def head_params(self):
        return [p for n, p in self.named_parameters() if not n.startswith("bb.") and p.requires_grad]

    def train(self, mode: bool = True):
        super().train(mode)
        self.bb.train(False)  # normalisations du backbone figées dans tous les modes
        return self

    @torch.no_grad()
    def fit_norm(self, imgs, msgs):
        """imgs b 3 512 512 et msgs b nbits (+-1) d'APPRENTISSAGE -> fixe agree_scale et hp_std."""
        _, pos, _ = self.bb(imgs)
        self.agree_scale.copy_((pos * msgs[:, :, None, None]).float().std().clamp_min(1e-3))
        self.hp_std.copy_(highpass(imgs).float().std((0, 2, 3), keepdim=True).clamp_min(1e-4))
        self.norm_fitted.fill_(True)

    def forward(self, img, msg, with_ref: bool = False):
        """img b 3 512 512 [0, 1], msg b nbits (+-1) -> logits b 128 128 (+ logits globaux courant/réf. si with_ref)."""
        b = img.shape[0]
        out = self.bb(img, with_ref=with_ref)
        feats, pos, glob = out[:3]
        y = self.lat[-1](feats[-1])  # b W h32 w32
        if self.use_msg:
            assert msg.shape == (b, self.nbits), (msg.shape, self.nbits)
            a = pos.float() * msg[:, :, None, None].float() / self.agree_scale  # b nbits h32 w32
            summ = torch.stack([a.mean(1), (a > 0).float().mean(1) - 0.5, torch.tanh(a).mean(1)], 1)  # b 3 h32 w32
            y = y + self.agree_proj(a) + self.summ_proj(summ) + self.msg_proj(msg.float())[:, :, None, None]
        y = self.top(y)
        for i in range(len(feats) - 2, -1, -1):  # pas 16, 8, 4
            y = self.fuse[i](self.lat[i](feats[i].float()) + F.interpolate(y, size=feats[i].shape[-2:], mode="bilinear",
                                                                             align_corners=False))  # b W h_i w_i
        y = F.interpolate(y, size=(128, 128), mode="bilinear", align_corners=False)  # b W 64|62 -> b W 128 128
        if self.use_rgb:
            hp = (highpass(img) / self.hp_std).clamp(-5, 5)  # b 3 512 512
            y = torch.cat([y, self.rgb(hp)], 1)  # b W+16 128 128
        logit = self.head(self.refine(y))[:, 0]  # b 1 128 128 -> b 128 128
        if with_ref:
            return logit, glob, out[3]
        return logit
