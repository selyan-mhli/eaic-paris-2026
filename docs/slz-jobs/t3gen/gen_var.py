"""T3 : variations img2img (sd-turbo) des forget = non-membres de même classe, + features de l'origine.

Lancé depuis work_c3_eval/ par gen_var.sbatch, une tâche par GPU (shard = SLURM_LOCALID sur 4).
Sortie (un seul fichier par shard) : gen_out/var_<job>_<k>.npz
  idx [n], f_forget [n 768], l_forget [n 100], pour s in STRENGTHS : img_s<ss> [n 224 224 3] u8,
  f_s<ss> [n 768], l_s<ss> [n 100]   (features/logits de l'origine sous autocast fp16, comme le serveur)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

STRENGTHS = (0.4, 0.6)
PROMPT = "a high quality photograph"


def main() -> None:
    k, nshard = int(os.environ.get("SLURM_LOCALID", 0)), 4
    job = os.environ.get("SLURM_JOB_ID", "local")
    wd = Path.cwd(); sys.path[:0] = [str(wd), str(wd.parent), str(wd / "mech")]
    import task_template as tt  # noqa: E402
    from cache_feats import feats_and_logits  # noqa: E402
    from diffusers import AutoPipelineForImage2Image  # noqa: E402
    from PIL import Image  # noqa: E402

    forget = tt.load_forget_images(wd.parent / "forget_images.npz")  # u8 tensor N 224 224 3 (format du template)
    n = len(forget)
    idx = np.arange(n)[k::nshard]
    orig = tt.load_model(wd.parent / "model.pt")
    out = {"idx": idx}
    sub = forget[torch.from_numpy(idx)]
    f, lg = feats_and_logits(orig, sub, tt.preprocess, True)
    out["f_forget"], out["l_forget"] = f.numpy(), lg.numpy().astype(np.float16)
    pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16,
                                                      variant="fp16").to("cuda")
    pipe.set_progress_bar_config(disable=True)
    u8 = sub.numpy() if sub.dtype == torch.uint8 else sub.cpu().numpy()
    for s in STRENGTHS:
        imgs = []
        g = torch.Generator("cuda").manual_seed(1000 * k + int(100 * s))
        for b in range(0, len(u8), 32):
            ins = [Image.fromarray(x).resize((512, 512), Image.BICUBIC) for x in u8[b:b + 32]]
            res = pipe(prompt=[PROMPT] * len(ins), image=ins, strength=s, num_inference_steps=max(2, int(round(2 / s))),
                       guidance_scale=0.0, generator=g).images
            imgs += [np.asarray(r.resize((224, 224), Image.BICUBIC)) for r in res]
        v = np.stack(imgs).astype(np.uint8)
        tag = f"s{int(100 * s):02d}"
        f, lg = feats_and_logits(orig, torch.from_numpy(v), tt.preprocess, True)
        out[f"img_{tag}"], out[f"f_{tag}"], out[f"l_{tag}"] = v, f.numpy(), lg.numpy().astype(np.float16)
        agree = (lg.argmax(1).numpy() == out["l_forget"].astype(np.float32).argmax(1)).mean()
        print(f"shard {k} {tag}: {len(v)} images, accord top-1 origine forget/variation {agree:.3f}", flush=True)
    Path("gen_out").mkdir(exist_ok=True)
    np.savez(f"gen_out/var_{job}_{k}.npz", **out)
    print(f"shard {k} écrit gen_out/var_{job}_{k}.npz", flush=True)


if __name__ == "__main__":
    main()
