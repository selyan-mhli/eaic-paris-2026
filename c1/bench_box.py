"""Mini-benchmark + équivalence de box_search : moteur ref (accs_for) vs fast (FastEngine), sur 1 GPU.

    python work_c1_speed/bench_box.py --n 3     # n images R (acc_full < 0.75) par méthode, train

Par image : temps grille / raffinement par moteur, nombre de boîtes décodées, écart des acc sur toute la
grille (max |diff|, nb de boîtes dont l'acc change), boîte finale et acc identiques ou non.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(Path.cwd()))
from box_search import H, METHODS, FastEngine, accs_for, grid, search  # noqa: E402
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import to_tensor  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=3)
ap.add_argument("--no-tf32", action="store_true", help="désactive TF32 (torch cuDNN/matmul et ORT use_tf32=0) pour ref ET fast")
a = ap.parse_args()
dev = "cuda"
if a.no_tf32:
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
ds = LocalisationDataset("release", "train")
wms = {m: load_watermark(m, "models", dev) for m in METHODS}
if a.no_tf32:
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    wms["stegastamp"].sess = ort.InferenceSession(str(Path("models") / "stegastamp" / "stega_stamp_logits.onnx"), so,
                                                  providers=[("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0}), "CPUExecutionProvider"])
    print("TF32 désactivé ; ORT", wms["stegastamp"].sess.get_provider_options().get("CUDAExecutionProvider", {}).get("use_tf32"), flush=True)
fast = {m: FastEngine(wms[m]) for m in METHODS}
grids = {m: grid(2.0 if m == "trustmark" else 16.0) for m in METHODS}
print("onnxruntime :", wms["stegastamp"].sess.get_providers(), "GPU", torch.cuda.get_device_name(0), flush=True)
tot = {e: {m: {"grid": 0.0, "refine": 0.0, "n": 0} for m in METHODS} for e in ("ref", "fast")}
same_all = True
for m in METHODS:
    wm = wms[m]; got = 0
    for i in range(len(ds)):
        if ds.meta[i]["method"] != m:
            continue
        it = ds[i]
        img = to_tensor(it["image"]).to(dev); msg = torch.from_numpy(it["message"]).float()
        if float(accs_for(wm, img, msg, [(0, H, 0, H)])[0]) >= 0.75:
            continue
        g = grids[m]
        if got == 0:  # échauffement (cuDNN, ORT) non chronométré
            fast[m](img, msg, g[:300]); accs_for(wm, img, msg, g[:300]); torch.cuda.synchronize()
        # équivalence brute sur la grille
        ar = accs_for(wm, img, msg, g); af_ = fast[m](img, msg, g)
        d = np.abs(ar - af_)
        # logits bruts sur 512 boîtes de la grille : ref = wrapper officiel, appel par groupe de taille
        sub = g[:: max(1, len(g) // 512)][:512]
        lf = fast[m](img, msg, sub, return_logits=True)
        lr = torch.zeros_like(lf)
        by = {}
        for j, b in enumerate(sub):
            by.setdefault((b[1] - b[0], b[3] - b[2]), []).append(j)
        for jj in by.values():
            crops = torch.cat([img[..., sub[j][0]:sub[j][1], sub[j][2]:sub[j][3]] for j in jj])
            lr[jj] = torch.as_tensor(wm.logits(crops)).float().cpu()
        flip = (lr > 0) != (lf > 0)
        print(f"  logits {m} : max|ref-fast|={float((lr - lf).abs().max()):.2e} ; échelle |logit| médiane {float(lr.abs().median()):.2e} ; "
              f"signes changés {int(flip.sum())}/{flip.numel()} ; max |logit ref| parmi eux "
              f"{float(lr.abs()[flip].max()) if flip.any() else 0:.2e}", flush=True)
        res = {}
        for e, fn in (("ref", lambda i_, ms, b: accs_for(wm, i_, ms, b)), ("fast", fast[m])):
            tm = {}
            torch.cuda.synchronize(); t = time.perf_counter()
            res[e] = search(fn, img, msg, g, tm)
            torch.cuda.synchronize()
            tot[e][m]["grid"] += tm["grid"]; tot[e][m]["refine"] += tm["refine"]; tot[e][m]["n"] += 1
            res[e + "_t"] = (tm["grid"], tm["refine"], tm["n_boxes"])
        same = res["ref"][0] == res["fast"][0] and abs(res["ref"][1] - res["fast"][1]) < 1e-6
        same_all &= same
        print(f"{m} {it['image_name']} grille {len(g)} : max|acc ref-fast|={d.max():.4f} boîtes changées={int((d > 1e-6).sum())} | "
              f"ref {res['ref']} grille {res['ref_t'][0]:.1f}s raff {res['ref_t'][1]:.1f}s ({res['ref_t'][2]} boîtes) | "
              f"fast {res['fast']} grille {res['fast_t'][0]:.1f}s raff {res['fast_t'][1]:.1f}s | identique={same}", flush=True)
        got += 1
        if got >= a.n:
            break
print("\nMoyenne par recherche R (s) :")
for m in METHODS:
    r, f = tot["ref"][m], tot["fast"][m]
    n = max(r["n"], 1)
    print(f"  {m:10s} ref grille {r['grid'] / n:.1f} + raff {r['refine'] / n:.1f} = {(r['grid'] + r['refine']) / n:.1f} | "
          f"fast grille {f['grid'] / n:.1f} + raff {f['refine'] / n:.1f} = {(f['grid'] + f['refine']) / n:.1f} | "
          f"x{(r['grid'] + r['refine']) / max(f['grid'] + f['refine'], 1e-9):.1f}")
print("boîtes finales et acc identiques sur toutes les images :", same_all)
