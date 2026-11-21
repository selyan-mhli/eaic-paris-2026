"""Enveloppe locale de PixelSeal (third_party/videoseal, non modifié).

API calquée sur ce qu'on attend du kit officiel : embed(imgs, msgs) et decode(imgs) -> bits.
Quand le kit arrive, remplacer ce module par ses wrappers (et ajouter TrustMark/StegaStamp).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
VS = ROOT / "third_party" / "videoseal"
CKPT = VS / "ckpts" / "pixelseal_checkpoint.pth"
CARD = VS / "videoseal" / "cards" / "pixelseal.yaml"


def pick_device() -> torch.device:
    if os.environ.get("C1_DEVICE"):
        return torch.device(os.environ["C1_DEVICE"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PixelSeal:
    nbits = 256

    def __init__(self, device: torch.device | None = None):
        sys.path.insert(0, str(VS))
        from omegaconf import OmegaConf
        from videoseal.utils.cfg import setup_model

        cfg = OmegaConf.load(CARD)
        cfg.args.attenuation_config = str(VS / "configs" / "attenuation.yaml")  # chemin absolu
        # même chemin que setup_model_from_model_card, mais checkpoint local (pas de réseau)
        self.model = setup_model(cfg, str(CKPT)).eval()
        self.device = device or pick_device()
        self.model.to(self.device)

    @torch.no_grad()
    def embed(self, imgs: torch.Tensor, msgs: torch.Tensor) -> torch.Tensor:  # = kit : embed(image, bits)
        """imgs : b 3 h w float [0,1] ; msgs : b 256 {0,1} -> imgs_w : b 3 h w (cpu)."""
        out = self.model.embed(imgs.to(self.device), msgs.to(self.device), is_video=False)
        return out["imgs_w"].clamp(0, 1).cpu()  # b 3 h w

    @torch.no_grad()
    def logits(self, imgs: torch.Tensor, bs: int = 64) -> torch.Tensor:
        """imgs : n 3 h w (redimensionnées en 256 par le modèle) -> logits des bits n 256."""
        res = []
        for i in range(0, len(imgs), bs):
            p = self.model.detect(imgs[i:i + bs].to(self.device), is_video=False)["preds"]
            # p : b (1+256) [h w éventuels] ; canal 0 = détection, 1: = bits
            if p.dim() == 4:
                p = p.mean(dim=(-2, -1))  # b (1+k) h w -> b (1+k)
            res.append(p[:, 1:].float().cpu())  # b 256
        return torch.cat(res)  # n 256

    def decode(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs : n 3 h w -> bits n 256 en {0,1}."""
        return (self.logits(imgs) > 0).to(torch.uint8)
