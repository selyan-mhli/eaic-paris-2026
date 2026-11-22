"""Segmentation avec l'EXTRACTEUR officiel comme encodeur pré-entraîné (façon « Watermark Anything »).

PixelSeal : ConvNeXtV2-tiny de videoseal (étages 96, 192, 384, 768 aux pas 4, 8, 16, 32) + `output_upscaling`
            et tête linéaire appliquée par position (257 canaux : détection + 256 bits) au pas 32.
TrustMark : ResNet-50 (layer1..4 : 256, 512, 1024, 2048 aux pas 4, 8, 16, 32) + `fc` par position (100 canaux).
Les poids sont ceux du kit (wmloc.watermarks.load_watermark), la moyenne spatiale finale est retirée.
Décodeur FPN léger (le même que PixSeg) -> logits au pas 4 de l'image d'ORIGINE (128 pour 512).

in_res : côté auquel l'image 512 est réduite avant l'encodeur (512 = telle quelle ; 256 / 245 = résolution
native de l'extracteur, même interpolation que le kit : bilinéaire avec antialias). Un recadrage c x c est réduit
du même facteur in_res / 512.
L'encodeur reste en mode eval (BatchNorm du ResNet figées, pas de DropPath).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import KIT  # noqa: E402
from model import block  # noqa: E402


class PixelSealEnc(nn.Module):
    channels = [96, 192, 384, 768 + 257]

    def __init__(self, device):
        super().__init__()
        sys.path.insert(0, str(KIT))
        from wmloc.watermarks import load_watermark
        det = load_watermark("pixelseal", KIT / "models", device).det
        self.convnext, self.pixel_decoder = det.convnext, det.pixel_decoder
        assert not self.pixel_decoder.pixelwise

    def forward(self, img):  # b 3 h w dans [0, 1]
        x = img * 2 - 1  # preprocess de l'extracteur
        f = []
        for i in range(4):
            x = self.convnext.stages[i](self.convnext.downsample_layers[i](x)); f.append(x)  # b c_i h/2^(i+2) w/2^(i+2)
        emb = self.pixel_decoder.output_upscaling(x)  # b 768 h/32 w/32
        lin = self.pixel_decoder.linear
        pp = torch.einsum("bchw,oc->bohw", emb, lin.weight) + lin.bias[None, :, None, None]  # b 257 h/32 w/32
        f[-1] = torch.cat([emb, pp], 1)  # b 768+257 h/32 w/32
        return f, pp


class TrustMarkEnc(nn.Module):
    channels = [256, 512, 1024, 2048 + 100]

    def __init__(self, device):
        super().__init__()
        sys.path.insert(0, str(KIT))
        from wmloc.watermarks import load_watermark
        self.net = load_watermark("trustmark", KIT / "models", device).net

    def forward(self, img):  # b 3 h w dans [0, 1]
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(img * 2 - 1))))  # b 64 h/4 w/4
        f = []
        for k in ("layer1", "layer2", "layer3", "layer4"):
            x = getattr(n, k)(x); f.append(x)  # b c_i, pas 4, 8, 16, 32
        pp = torch.einsum("bchw,oc->bohw", x, n.fc.weight) + n.fc.bias[None, :, None, None]  # b 100 h/32 w/32
        f[-1] = torch.cat([x, pp], 1)  # b 2048+100 h/32 w/32
        return f, pp


class ExtSeg(nn.Module):
    def __init__(self, method, device, in_res=512, width=96, train_enc=True, scratch=False, use_msg=False):
        """use_msg : le message connu (metadata.json) signe les logits par position de l'extracteur (accord > 0 =
        le bit lu ici est le bon) ; 3 canaux résumés ajoutés (accord moyen, part de bits justes, tanh moyen).
        Le message arrive par l'argument `method` de forward (b nbits, valeurs +-1).
        scratch : même architecture, poids de l'encodeur ré-initialisés (contrôle : est-ce le pré-entraînement
        qui apporte le signal ?)."""
        super().__init__()
        self.method, self.in_res, self.train_enc = method, in_res, train_enc
        self.enc = {"pixelseal": PixelSealEnc, "trustmark": TrustMarkEnc}[method](device)
        for p in self.enc.parameters():
            p.requires_grad_(train_enc)
        if scratch:
            for m in self.enc.modules():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()
        self.use_msg, self.nbits = use_msg, {"pixelseal": 256, "trustmark": 100}[method]
        ch, w = list(self.enc.channels), width
        ch[-1] += 3 * use_msg
        self.lat = nn.ModuleList([nn.Conv2d(c, w, 1) for c in ch])
        self.fuse = nn.ModuleList([block(w, w) for _ in ch[:-1]])
        self.ctx = nn.Linear(ch[-1], w)
        self.head = nn.Sequential(block(w, w), nn.Conv2d(w, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        self.enc.eval()  # BatchNorm figées, pas de DropPath
        return self

    def enc_params(self):
        return [p for p in self.enc.parameters() if p.requires_grad]

    def dec_params(self):
        return [p for n, p in self.named_parameters() if not n.startswith("enc.")]

    def forward(self, img, method=None, extra=None):
        """img b 3 H W dans [0, 1] -> logits b H/4 W/4."""
        out_hw = (img.shape[-2] // 4, img.shape[-1] // 4)
        if self.in_res != 512:
            size = (round(img.shape[-2] * self.in_res / 512), round(img.shape[-1] * self.in_res / 512))
            img = F.interpolate(img, size=size, mode="bilinear", align_corners=False, antialias=True)  # b 3 h w
        with torch.set_grad_enabled(self.train_enc and torch.is_grad_enabled()):
            f, _ = self.enc(img)
            if self.use_msg:
                assert method is not None and method.shape == (img.shape[0], self.nbits), "message b nbits attendu"
                top = f[-1]
                agree = top[:, -self.nbits:].float() * method[:, :, None, None].float()  # b nbits h/32 w/32
                summ = torch.stack([agree.mean(1), (agree > 0).float().mean(1) - 0.5, torch.tanh(agree).mean(1)], 1)
                f[-1] = torch.cat([top[:, :-self.nbits], agree.to(top.dtype), summ.to(top.dtype)], 1)  # b C+3 h/32 w/32
        y = self.lat[-1](f[-1]) + self.ctx(f[-1].mean((2, 3)))[:, :, None, None]  # b w h/32 w/32 + b w 1 1
        for i in range(len(f) - 2, -1, -1):
            y = self.fuse[i](self.lat[i](f[i]) + F.interpolate(y, size=f[i].shape[-2:], mode="nearest"))
        out = self.head(y)[:, 0]  # b 1 h/4 w/4 -> b h/4 w/4
        if out.shape[-2:] != out_hw:
            out = F.interpolate(out[:, None], size=out_hw, mode="bilinear", align_corners=False)[:, 0]  # b H/4 W/4
        return out
