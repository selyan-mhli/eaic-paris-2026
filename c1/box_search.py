"""Recherche du rectangle watermarké par recadrage (régime « rectangle seul », R).

Constat (diagnostic train, 2026-11-21) : quand seul un rectangle a été watermarké (full_watermark=False),
le décodage de l'image entière est au hasard (~0.53) mais le rectangle GT recadré tel quel (le wrapper le
redimensionne à sa résolution) décode à 0.64-0.73 en moyenne : le watermark a été posé à l'échelle du
rectangle. On cherche donc la boîte dont le recadrage décode le mieux le message connu.

    python work_c1/box_search.py --split train --out output/boxes_train.npz [--limit 6]

Images à acc_full >= --gate : régime F, pas de recherche (box = image entière).
Grille : tailles h, w dans SIZES (rapport <= 16, <= 2 pour TrustMark), pas de position = max(24, min(h,w)/2),
puis raffinement local des bords (pas 16 puis 8) autour des 4 meilleures boîtes (cache commun).
--engine ref (défaut) : chemin accs_for d'origine ; --engine fast : FastEngine (lots pleins après
redimensionnement, IOBinding StegaStamp), expérimental : équivalent à ref sans TF32 pour PixelSeal/TrustMark
(gain x1,0-1,1), NON équivalent pour StegaStamp (benchmark 2026-11-21, voir bench_box.py). --balance : tranches équilibrées par méthode.
Sortie : box (y0, y1, x0, x1) exclusif, acc de la meilleure boîte, acc_full.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path.cwd()))
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import to_tensor  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

METHODS = ("pixelseal", "trustmark", "stegastamp")
SIZES = (32, 48, 64, 96, 128, 192, 256, 320, 384, 512)
H = 512


def accs_for(wm, img, msg, boxes, bs=256):
    """boxes : liste (y0, y1, x0, x1) -> bit accuracy du message pour chaque recadrage."""
    out = np.zeros(len(boxes), np.float32)
    by = {}
    for i, (y0, y1, x0, x1) in enumerate(boxes):
        by.setdefault((y1 - y0, x1 - x0), []).append(i)
    for (h, w), ii in by.items():  # même taille -> un batch
        for s in range(0, len(ii), bs):
            chunk = ii[s:s + bs]
            crops = torch.cat([img[..., boxes[i][0]:boxes[i][1], boxes[i][2]:boxes[i][3]] for i in chunk])  # b 3 h w
            lg = torch.as_tensor(wm.logits(crops)).float().cpu()  # b k
            out[chunk] = ((lg > 0).float() == msg[None]).float().mean(1).numpy()
    return out


class FastEngine:
    """Même calcul que accs_for (recadrage -> redimensionnement officiel du wrapper -> logits -> accord),
    mais : redimensionnement par groupe de taille vers un tampon GPU à la résolution du décodeur,
    inférence par lots pleins (indépendants des tailles d'origine), accord calculé sur GPU, un seul
    transfert vers le CPU par appel, et StegaStamp via IOBinding ONNX Runtime (tenseurs CUDA NHWC
    FP32 contigus, sans aller-retour numpy). Prétraitements identiques aux wrappers officiels :
    PixelSeal bilinéaire antialias 256 ; TrustMark bilinéaire antialias 245 puis x*2-1 ;
    StegaStamp bicubique antialias 400 puis clamp(0, 1)."""

    def __init__(self, wm, bs=256):
        self.wm, self.name = wm, wm.name
        # StegaStamp : lots de 32 comme le wrapper officiel (256 -> 5,2 Go demandés par ORT, mesuré)
        self.bs = 32 if self.name == "stegastamp" else bs
        self.dev = torch.device(wm.device if self.name != "stegastamp" else "cuda")
        self.res = {"pixelseal": 256, "trustmark": wm.resolution if self.name == "trustmark" else 0, "stegastamp": 400}[self.name]
        if self.name == "stegastamp":
            self.dev_id = self.dev.index or 0
            self.secret = torch.zeros(self.bs, 100, device=self.dev, dtype=torch.float32)

    def _resize(self, x):
        if self.name == "stegastamp":
            return F.interpolate(x.float(), size=(400, 400), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        return F.interpolate(x, size=(self.res,) * 2, mode="bilinear", align_corners=False, antialias=True)

    @torch.no_grad()
    def _logits(self, buf):  # buf : n 3 res res (déjà redimensionné), sur GPU
        if self.name == "pixelseal":
            return self.wm.det(buf)[:, 1:].float()
        if self.name == "trustmark":
            return self.wm.net(buf * 2 - 1).float()
        import numpy as _np
        x = buf.permute(0, 2, 3, 1).contiguous()  # n 400 400 3
        n = len(x)
        out = torch.empty(n, 100, device=self.dev, dtype=torch.float32)
        sec = self.secret[:n].contiguous()
        io = self.wm.sess.io_binding()
        io.bind_input("image", "cuda", self.dev_id, _np.float32, list(x.shape), x.data_ptr())
        io.bind_input("secret", "cuda", self.dev_id, _np.float32, list(sec.shape), sec.data_ptr())
        io.bind_output(self.wm.LOGITS, "cuda", self.dev_id, _np.float32, list(out.shape), out.data_ptr())
        torch.cuda.synchronize(self.dev)  # tampon écrit par torch avant lecture par ORT
        self.wm.sess.run_with_iobinding(io)
        io.synchronize_outputs()  # sortie écrite par ORT avant lecture par torch
        return out

    @torch.no_grad()
    def __call__(self, img, msg, boxes, return_logits=False):
        n = len(boxes)
        res = torch.empty(n, device=self.dev)
        lgs = {} if return_logits else None
        m = msg.to(self.dev)[None]  # 1 k
        r = self.res
        buf = torch.empty(self.bs, 3, r, r, device=self.dev)
        pos = []  # indices des boîtes présentes dans le tampon
        img = img.to(self.dev)

        def flush():
            if pos:
                lg = self._logits(buf[: len(pos)])
                res[torch.tensor(pos, device=self.dev)] = ((lg > 0).float() == m).float().mean(1)
                if lgs is not None:
                    lgs.update(zip(pos, lg.cpu()))
                pos.clear()
        by = {}
        for i, (y0, y1, x0, x1) in enumerate(boxes):
            by.setdefault((y1 - y0, x1 - x0), []).append(i)
        for (h, w), ii in by.items():
            for s in range(0, len(ii), self.bs):
                chunk = ii[s:s + self.bs]
                crops = torch.cat([img[..., boxes[i][0]:boxes[i][1], boxes[i][2]:boxes[i][3]] for i in chunk])  # b 3 h w
                rs = self._resize(crops)  # b 3 r r
                k = 0
                while k < len(chunk):
                    take = min(self.bs - len(pos), len(chunk) - k)
                    buf[len(pos):len(pos) + take] = rs[k:k + take]
                    pos.extend(chunk[k:k + take]); k += take
                    if len(pos) == self.bs:
                        flush()
        flush()
        if lgs is not None:
            return torch.stack([lgs[i] for i in range(n)])  # n k (contrôle d'équivalence)
        return res.cpu().numpy().astype(np.float32)


class Scorer:
    """Cache boîte -> acc commun à la grille et au raffinement, pour une image."""

    def __init__(self, fn, img, msg):
        self.fn, self.img, self.msg, self.cache = fn, img, msg, {}

    def __call__(self, boxes):
        todo = [b for b in dict.fromkeys(boxes) if b not in self.cache]
        if todo:
            self.cache.update(zip(todo, self.fn(self.img, self.msg, todo).tolist()))
        return np.array([self.cache[b] for b in boxes], np.float32)


def grid(max_ratio):
    bx = []
    for h in SIZES:
        for w in SIZES:
            if max(h / w, w / h) > max_ratio:
                continue
            st = max(24, (min(h, w) // 2) // 8 * 8)
            for y in list(range(0, H - h + 1, st)) + [H - h]:
                for x in list(range(0, H - w + 1, st)) + [H - w]:
                    bx.append((y, y + h, x, x + w))
    return sorted(set(bx))


def refine(score, seeds, steps=(16, 8)):
    """score : Scorer (cache commun). Raffinement des bords autour des graines."""
    best = list(seeds)
    for st in steps:
        for _ in range(3):
            cand = set(best)
            for (y0, y1, x0, x1) in best:
                for dy0 in (-st, 0, st):
                    for dy1 in (-st, 0, st):
                        for dx0 in (-st, 0, st):
                            for dx1 in (-st, 0, st):
                                b = (y0 + dy0, y1 + dy1, x0 + dx0, x1 + dx1)
                                if 0 <= b[0] < b[1] <= H and 0 <= b[2] < b[3] <= H and b[1] - b[0] >= 16 and b[3] - b[2] >= 16:
                                    cand.add(b)
            cand = sorted(cand)
            a = score(cand)
            order = np.argsort(-a)[: len(seeds)]
            new = [cand[i] for i in order]
            if new == best:
                break
            best = new
    a = score(best)
    k = int(np.argmax(a))
    return best[k], float(a[k])


def search(fn, img, msg, g, timings=None):
    """Grille puis raffinement ; fn(img, msg, boxes) -> acc. timings : dict optionnel (s par phase)."""
    sc = Scorer(fn, img, msg)
    t = time.perf_counter()
    a_all = sc(g)
    t1 = time.perf_counter()
    seeds = [g[k] for k in np.argsort(-a_all)[:4]]
    out = refine(sc, seeds)
    if timings is not None:
        timings["grid"] = timings.get("grid", 0) + t1 - t
        timings["refine"] = timings.get("refine", 0) + time.perf_counter() - t1
        timings["n_boxes"] = timings.get("n_boxes", 0) + len(sc.cache)
    return out


def balanced_shard(meta, nshards, shard):
    """Indices de la tranche : le j-ième élément de chaque méthode va à la tranche (j + décalage) % n,
    pour équilibrer la charge StegaStamp (la plus lente) entre tranches."""
    rank = {}
    out = []
    for i, x in enumerate(meta):
        m = x["method"]; j = rank.get(m, 0); rank[m] = j + 1
        if (j + METHODS.index(m)) % nshards == shard:
            out.append(i)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--data", default="release")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gate", type=float, default=0.75)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--engine", choices=["ref", "fast"], default="ref",
                    help="ref = accs_for (défaut, résultats du 2026-11-21) ; fast = FastEngine, NON validé pour StegaStamp")
    ap.add_argument("--balance", action="store_true", help="tranches équilibrées par méthode (sinon i %% nshards)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = LocalisationDataset(args.data, args.split)
    if args.balance:
        idx = balanced_shard(ds.meta, args.nshards, args.shard)
    else:
        idx = [i for i in range(len(ds)) if i % args.nshards == args.shard]
    wms = {m: load_watermark(m, "models", dev) for m in METHODS}
    if args.engine == "fast":
        fast = {m: FastEngine(wms[m]) for m in METHODS}
        engines = {m: (lambda i, ms, b, e=fast[m]: e(i, ms, b)) for m in METHODS}
    else:
        engines = {m: (lambda i, ms, b, w=wms[m]: accs_for(w, i, ms, b)) for m in METHODS}
    print("moteur", args.engine, "équilibrage", args.balance, flush=True)
    grids = {m: grid(2.0 if m == "trustmark" else 16.0) for m in METHODS}
    print("onnxruntime StegaStamp :", wms["stegastamp"].sess.get_providers(), flush=True)
    print("boîtes par grille :", {m: len(g) for m, g in grids.items()}, flush=True)
    names, method, box, acc, accf = [], [], [], [], []
    part = args.out.with_name("partial_" + args.out.name)  # hors du motif boxes_*

    def save(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name("tmp_" + path.name)
        np.savez_compressed(tmp, names=np.array(names), method=np.array(method), box=np.array(box, np.int32).reshape(-1, 4),
                            acc_box=np.array(acc, np.float32), acc_full=np.array(accf, np.float32))
        tmp.rename(path)

    if part.exists() and not args.limit:  # reprise après dépassement de --time
        z = np.load(part)
        names, method, box = z["names"].tolist(), z["method"].tolist(), [tuple(b) for b in z["box"].tolist()]
        acc, accf = z["acc_box"].tolist(), z["acc_full"].tolist()
        print(f"reprise : {len(names)} images déjà faites ({part})", flush=True)
    doneset = set(names)
    t0 = time.time(); done = 0
    for i in idx:
        if ds.meta[i]["image_name"] in doneset:
            continue
        it = ds[i]; m = it["method"]; wm = wms[m]
        img = to_tensor(it["image"]).to(dev); msg = torch.from_numpy(it["message"]).float()
        af = float(accs_for(wm, img, msg, [(0, H, 0, H)])[0])  # tri : chemin officiel inchangé
        if af >= args.gate:
            b, a = (0, H, 0, H), af
        else:
            if args.limit and done >= args.limit:
                continue
            g = grids[m]
            b, a = search(engines[m], img, msg, g)
            done += 1
            if done <= 5 or done % 25 == 0:
                gt = ""
                if "mask" in it:
                    ys, xs = np.nonzero(it["mask"])
                    gt = f" gt_bbox=({ys.min()},{ys.max() + 1},{xs.min()},{xs.max() + 1})" if len(ys) else " gt vide"
                print(f"[R {done}] {m} acc_full={af:.3f} best={b} acc={a:.3f}{gt} {time.time() - t0:.0f}s", flush=True)
        names.append(it["image_name"]); method.append(m); box.append(b); acc.append(a); accf.append(af)
        if af < args.gate and not args.limit and done % 5 == 0:
            save(part)
    print(f"{len(names)} images, {done} recherches, {time.time() - t0:.0f} s", flush=True)
    save(args.out)
    if part.exists():
        part.unlink()
    print("->", args.out, flush=True)


if __name__ == "__main__":
    main()
