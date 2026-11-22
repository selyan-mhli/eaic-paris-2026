"""U-Net compact pour la localisation au pixel : image 512 -> logits 128 (pas 4).

Entrée construite dans le modèle : RGB normalisé (3) + one-hot de la méthode en canaux constants (3) + passe-haut
de la luminance et de chaque canal R, G, B (4, gain 64, tronqué) = 10 canaux. Encodeur from scratch (pleine résolution d'abord : le watermark est un signal
de quelques niveaux sur 255 à haute fréquence) ou encodeur timm pré-entraîné (--backbone, poids lus dans le
cache HF, HF_HUB_OFFLINE=1). Décodeur commun, du pas 32 au pas 4, avec contexte global (régime F ou R).
Cartes existantes (--extra-maps) : concaténées aux features de pas 4, juste avant la tête.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NM = 3  # méthodes
NHP = 4  # résidus passe-haut : luminance, R, G, B
HP_GAIN, HP_CLIP = 64.0, 4.0  # 4 niveaux sur 255 -> 1 ; contours forts tronqués


def block(cin, cout, stride=1):
    assert cout % 8 == 0, cout
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.GroupNorm(8, cout), nn.SiLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.GroupNorm(8, cout), nn.SiLU(inplace=True))


class ScratchEncoder(nn.Module):
    """Deux entrées séparées en pleine résolution (contenu, résidus passe-haut) : sans cela la normalisation
    noie le résidu, de quelques niveaux sur 255, sous le contenu."""

    def __init__(self, cin, base=24, nhp=NHP):
        super().__init__()
        w = [2 * base, 2 * base, 4 * base, 6 * base, 8 * base, 8 * base]  # pas 1, 2, 4, 8, 16, 32
        self.nhp = nhp
        self.stem_c, self.stem_hp = block(cin - nhp, base), block(nhp, base)
        self.stages = nn.ModuleList([block(w[i - 1], w[i], 2) for i in range(1, 6)])
        self.channels = w[2:]

    def forward(self, x):  # b cin H W, les nhp derniers canaux = résidus passe-haut
        x = torch.cat([self.stem_c(x[:, :-self.nhp]), self.stem_hp(x[:, -self.nhp:])], 1)  # b 2*base H W
        feats = []
        for s in self.stages:
            x = s(x); feats.append(x)
        return feats[1:]  # pas 4, 8, 16, 32 : b c H/4 W/4, ..., b c H/32 W/32


class TimmEncoder(nn.Module):
    def __init__(self, name, cin, pretrained=True):
        super().__init__()
        import timm
        self.m = timm.create_model(name, pretrained=pretrained, features_only=True, in_chans=cin)
        red = self.m.feature_info.reduction()
        self.keep = [i for i, r in enumerate(red) if r >= 4]
        assert red[self.keep[0]] == 4, f"{name} : pas de sortie au pas 4 ({red})"
        self.channels = [self.m.feature_info.channels()[i] for i in self.keep]

    def forward(self, x):
        f = self.m(x)
        return [f[i] for i in self.keep]


class PixSeg(nn.Module):
    def __init__(self, base=24, backbone="", pretrained=True, extra=0, width=None, in_scale=1.0, hp_mode="bin5",
                 hp_luma=False):
        """in_scale : 0.5 = image réduite (antialias) avant le réseau, logits remontés au pas 4 de l'image d'origine.
        hp_mode : bin5 (image - flou binomial 5x5, défaut), lowres (même passe-haut calculé à demi-résolution puis
        remonté), dog (flou binomial 1 fois - 3 fois : passe-bande plus bas). hp_luma : canaux R, G, B du passe-haut à 0."""
        super().__init__()
        assert hp_mode in ("bin5", "lowres", "dog"), hp_mode
        self.in_scale, self.hp_mode, self.hp_luma = in_scale, hp_mode, hp_luma
        cin = 3 + NM + NHP
        self.enc = TimmEncoder(backbone, cin, pretrained) if backbone else ScratchEncoder(cin, base)
        ch, w = self.enc.channels, width or 4 * base
        self.lat = nn.ModuleList([nn.Conv2d(c, w, 1) for c in ch])
        self.fuse = nn.ModuleList([block(w, w) for _ in ch[:-1]])
        self.ctx = nn.Linear(ch[-1], w)
        self.head = nn.Sequential(block(w + extra, w), nn.Conv2d(w, 1, 1))
        k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16
        self.register_buffer("blur", (k[:, None] * k[None, :])[None, None], persistent=False)  # 1 1 5 5

    def inputs(self, img, method):
        """img b 3 H W dans [0, 1], method b (long) -> b 10 H W : RGB (3), one-hot (3), passe-haut Y, R, G, B (4)."""
        b, _, h, w = img.shape
        y = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]  # b 1 H W
        v = torch.cat([y, img], 1).float().reshape(b * NHP, 1, h, w)  # b 4 H W -> (b 4) 1 H W

        def bl(t):
            return F.conv2d(F.pad(t, (2, 2, 2, 2), mode="reflect"), self.blur)
        if self.hp_mode == "bin5":
            hp = v - bl(v)  # image - flou binomial 5x5
        elif self.hp_mode == "dog":
            b1 = bl(v); hp = b1 - bl(bl(b1))  # passe-bande : flou x1 - flou x3
        else:  # lowres : passe-haut à demi-résolution, remonté
            vs = F.interpolate(v, scale_factor=0.5, mode="bilinear", align_corners=False, antialias=True)
            hp = F.interpolate(vs - bl(vs), size=(h, w), mode="bilinear", align_corners=False)
        hp = (hp * HP_GAIN).clamp(-HP_CLIP, HP_CLIP).reshape(b, NHP, h, w).to(img.dtype)  # (b 4) 1 H W -> b 4 H W
        if self.hp_luma:
            hp = torch.cat([hp[:, :1], torch.zeros_like(hp[:, 1:])], 1)
        oh = F.one_hot(method, NM).to(img.dtype)[:, :, None, None].expand(b, NM, h, w)  # b 3 -> b 3 H W
        return torch.cat([(img - 0.5) / 0.25, oh, hp], 1)  # b 10 H W

    def forward(self, img, method, extra=None):
        """-> logits b H/4 W/4 (128 pour 512). extra : b E H/4 W/4, obligatoire si le modèle a été créé avec."""
        out_hw = (img.shape[-2] // 4, img.shape[-1] // 4)
        if self.in_scale != 1.0:
            img = F.interpolate(img, scale_factor=self.in_scale, mode="bilinear", align_corners=False, antialias=True)
            if extra is not None:
                extra = F.interpolate(extra, size=(img.shape[-2] // 4, img.shape[-1] // 4), mode="bilinear", align_corners=False)
        f = self.enc(self.inputs(img, method))
        y = self.lat[-1](f[-1]) + self.ctx(f[-1].mean((2, 3)))[:, :, None, None]  # b w H/32 W/32 + b w 1 1
        for i in range(len(f) - 2, -1, -1):
            y = self.fuse[i](self.lat[i](f[i]) + F.interpolate(y, size=f[i].shape[-2:], mode="nearest"))
        if extra is not None:
            y = torch.cat([y, extra.to(y.dtype)], 1)  # b w+E H/4 W/4
        out = self.head(y)[:, 0]  # b 1 H/4 W/4 -> b H/4 W/4
        if out.shape[-2:] != out_hw:
            out = F.interpolate(out[:, None], size=out_hw, mode="bilinear", align_corners=False)[:, 0]
        return out
