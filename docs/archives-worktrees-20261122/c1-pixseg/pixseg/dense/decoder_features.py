"""Adaptateurs aux décodeurs OFFICIELS du kit (wmloc/watermarks.py), avec gradients possibles.

Les méthodes publiques `logits()` / `features()` des wrappers sont décorées `no_grad` : on n'appelle ici que
leurs modules internes, avec exactement le même prétraitement (lu dans wmloc/watermarks.py, 21/11/2026) :
  PixelSeal : redimensionnement 256 bilinéaire antialias, x*2-1, ConvNeXtV2-tiny (stages pas 4, 8, 16, 32),
              pixel_decoder.output_upscaling puis linéaire (canal 0 = détection, 1..256 = bits).
  TrustMark : redimensionnement 245 bilinéaire antialias, x*2-1, ResNet-50 Q (layer1..layer4), fc (100 bits).
Les logits globaux officiels = moyenne spatiale des logits par position (la tête est linéaire) ; signe > 0 = bit 1.

Aucune modification des wrappers. `fake_backbone` construit la même architecture à poids aléatoires (tests CPU).
"""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
KIT = Path(os.environ.get("PIXSEG_KIT", "/p/scratch/training2666/mouhali1/SLZ/Hackathon_Watermark_Localisation_Task"))
VIDEOSEAL_LOCAL = Path(os.environ.get("VIDEOSEAL_REPO", HERE.parents[3] / "eaic-paris-2026" / "third_party" / "videoseal"))
RES = {"pixelseal": 256, "trustmark": 245}
NBITS = {"pixelseal": 256, "trustmark": 100}


def resize_official(img: torch.Tensor, res: int) -> torch.Tensor:
    """b 3 H W dans [0, 1] -> b 3 res res, même appel que le kit."""
    return F.interpolate(img, size=(res, res), mode="bilinear", align_corners=False, antialias=True)


class _Backbone(nn.Module):
    """Interface commune : forward(img b 3 H W [0, 1]) -> (feats [4 x b c_i h_i w_i], pos b nbits h w, glob b nbits).
    `train_last` : seul le dernier stage reçoit des gradients ; il en garde une copie figée (`ref_last`) pour la
    contrainte de préservation des logits globaux. Le module reste toujours en mode eval (normalisations figées,
    pas de DropPath)."""

    method = ""

    def __init__(self, train_last: bool = False):
        super().__init__()
        self.train_last = train_last

    def setup_grads(self):
        for p in self.parameters():
            p.requires_grad_(False)
        if self.train_last:
            for p in self.last_stage().parameters():
                p.requires_grad_(True)
            self.ref_last = copy.deepcopy(self.last_stage())  # copie figée, pour la contrainte de logits
            for p in self.ref_last.parameters():
                p.requires_grad_(False)
        self.train(False)  # un module neuf est en mode train : sans cela un 1er forward met à jour les BatchNorm

    def train(self, mode: bool = True):
        super().train(mode)
        return super().train(False)  # toujours eval

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, img, with_ref: bool = False):
        assert not self.training, "backbone hors mode eval"
        x = self.prep(img)  # b 3 res res dans [-1, 1]
        with torch.no_grad():
            feats = self.early(x)  # 3 premiers niveaux, sans gradient (backbone gelé)
        with torch.set_grad_enabled(self.train_last and torch.is_grad_enabled()):
            top = self.last(feats[-1], self.last_stage())
            pos = self.pos_logits(top)  # b nbits h32 w32
        out = (feats + [top], pos, pos.mean((2, 3)))
        if with_ref:
            assert self.train_last, "with_ref n'a de sens qu'en mode dégel"
            with torch.no_grad():
                ref = self.pos_logits(self.last(feats[-1], self.ref_last)).mean((2, 3))  # b nbits
            out = out + (ref,)
        return out


class PixelSealBackbone(_Backbone):
    method = "pixelseal"
    channels = [96, 192, 384, 768]

    def __init__(self, det: nn.Module, train_last: bool = False):
        super().__init__(train_last)
        self.convnext, self.pixel_decoder = det.convnext, det.pixel_decoder
        assert not self.pixel_decoder.pixelwise, "tête linéaire globale attendue (carte pixelseal)"
        self.setup_grads()

    def prep(self, img):
        return resize_official(img, 256) * 2 - 1  # == det.preprocess(_resize(x))

    def last_stage(self):
        return nn.ModuleList([self.convnext.downsample_layers[3], self.convnext.stages[3]])

    def early(self, x):
        f = []
        for i in range(3):
            x = self.convnext.stages[i](self.convnext.downsample_layers[i](x)); f.append(x)  # b c_i 256/2^(i+2)
        return f  # [b 96 64 64, b 192 32 32, b 384 16 16]

    @staticmethod
    def last(x, stage):
        return stage[1](stage[0](x))  # b 768 8 8

    def pos_logits(self, top):
        emb = self.pixel_decoder.output_upscaling(top)  # b 768 8 8 (upscale_stages [1])
        lin = self.pixel_decoder.linear
        pp = torch.einsum("bchw,oc->bohw", emb, lin.weight) + lin.bias[None, :, None, None]  # b 257 8 8
        return pp[:, 1:]  # b 256 8 8 (canal 0 = détection, écarté comme dans wm.logits)


class TrustMarkBackbone(_Backbone):
    method = "trustmark"
    channels = [256, 512, 1024, 2048]

    def __init__(self, net: nn.Module, train_last: bool = False):
        super().__init__(train_last)
        self.net = net
        self.setup_grads()

    def prep(self, img):
        return resize_official(img, 245) * 2 - 1  # == TrustMark._prep

    def last_stage(self):
        return self.net.layer4

    def early(self, x):
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))  # b 64 62 62
        f = []
        for k in ("layer1", "layer2", "layer3"):
            x = getattr(n, k)(x); f.append(x)  # b 256 62 62, b 512 31 31, b 1024 16 16
        return f

    @staticmethod
    def last(x, stage):
        return stage(x)  # b 2048 8 8

    def pos_logits(self, top):
        fc = self.net.fc
        return torch.einsum("bchw,oc->bohw", top, fc.weight) + fc.bias[None, :, None, None]  # b 100 8 8


# ---------------------------------------------------------------- construction
def load_official(method: str, device="cpu", kit: Path | None = None, train_last=False) -> _Backbone:
    """Poids du kit via wmloc.watermarks.load_watermark (JURECA ; TrustMark exige le paquet `trustmark`)."""
    kit = Path(kit or KIT)
    sys.path.insert(0, str(kit))
    from wmloc.watermarks import load_watermark
    wm = load_watermark(method, kit / "models", device)
    mod = wm.det if method == "pixelseal" else wm.net
    cls = PixelSealBackbone if method == "pixelseal" else TrustMarkBackbone
    return cls(mod, train_last).to(device)


def load_local_pixelseal(card: Path, device="cpu"):
    """Modèle videoseal complet depuis une carte locale (tests CPU : embedder + détecteur officiels)."""
    sys.path.insert(0, str(VIDEOSEAL_LOCAL))
    import videoseal
    return videoseal.load(Path(card)).to(device).eval()


def fake_backbone(method: str, train_last=False, tiny=False) -> _Backbone:
    """Même architecture, poids aléatoires. tiny : dimensions réduites (tests rapides, mêmes pas spatiaux)."""
    torch.manual_seed(0)
    if method == "pixelseal":
        sys.path.insert(0, str(VIDEOSEAL_LOCAL))
        from videoseal.modules.convnext import ConvNeXtV2
        from videoseal.modules.pixel_decoder import PixelDecoder
        dims = [16, 32, 64, 128] if tiny else [96, 192, 384, 768]
        det = nn.Module()
        det.convnext = ConvNeXtV2(depths=[1, 1, 1, 1] if tiny else [3, 3, 9, 3], dims=dims)
        det.pixel_decoder = PixelDecoder(embed_dim=dims[-1], nbits=256, upscale_stages=[1], pixelwise=False)
        bb = PixelSealBackbone(det, train_last)
        bb.channels = dims
        return bb
    import torchvision
    net = torchvision.models.resnet50(num_classes=100) if not tiny else _tiny_resnet()
    bb = TrustMarkBackbone(net, train_last)
    if tiny:
        bb.channels = [16, 32, 64, 128]
    return bb


def _tiny_resnet():
    import torchvision
    from torchvision.models.resnet import BasicBlock, ResNet
    net = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=100, width_per_group=64)
    # réduit les largeurs tout en gardant les mêmes noms et pas spatiaux
    net.inplanes = 16
    net.conv1 = nn.Conv2d(3, 16, 7, 2, 3, bias=False); net.bn1 = nn.BatchNorm2d(16)
    net.layer1 = net._make_layer(BasicBlock, 16, 1)
    net.layer2 = net._make_layer(BasicBlock, 32, 1, stride=2)
    net.layer3 = net._make_layer(BasicBlock, 64, 1, stride=2)
    net.layer4 = net._make_layer(BasicBlock, 128, 1, stride=2)
    net.fc = nn.Linear(128, 100)
    del torchvision
    return net
