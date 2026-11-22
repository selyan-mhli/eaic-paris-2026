"""Move only the winning logit, down to a chosen margin over the runner-up.

The 0.0308 submission lowered every class row, so the runner-up fell along
with the winner and cross-entropy stopped at 0.087. Here row c is trained to
fall on images of class c and to stay put on every other forget image. Flips
and mild alterations of class c are asked for the same winning logit.

Does not modify task_template.py or audit.py.
"""

from __future__ import annotations

from pathlib import Path

import torch

import audit
import task_template

OUTPUT_DIR = Path("output")
NEG_WEIGHTS = (0.05, 0.2, 1.0, 4.0)
LAMBDAS = (1e-6, 1e-3)
PROBES = (("lowpass", 72.0), ("blur", 1.5), ("lowpass", 36.0))


@torch.no_grad()
def _collect(model, images, batch_size: int = 128):
    feats, logits = [], []

    def hook(_module, inputs, output):
        feats.append(inputs[0].detach().float().cpu())
        logits.append(output.detach().float().cpu())

    handle = model.head.register_forward_hook(hook)
    model.eval()
    try:
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=task_template.DEVICE == "cuda"):
                model(task_template.preprocess(batch))
    finally:
        handle.remove()
    return torch.cat(feats), torch.cat(logits)


def _lowpass(images: torch.Tensor, size: int) -> torch.Tensor:
    x = images.to(task_template.DEVICE).permute(0, 3, 1, 2).float()
    small = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    back = torch.nn.functional.interpolate(small, size=(x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False)
    return back.clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu()


def _blur(images: torch.Tensor, sigma: float) -> torch.Tensor:
    k = int(round(sigma * 4)) * 2 + 1
    x = images.to(task_template.DEVICE).permute(0, 3, 1, 2).float()
    ax = torch.arange(k, device=x.device) - k // 2
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2 * sigma * sigma))
    kernel = (kernel / kernel.sum()).view(1, 1, k, k).repeat(3, 1, 1, 1)
    return torch.nn.functional.conv2d(x, kernel, padding=k // 2, groups=3).clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu()


def _fit_row(phi, target, sample_weight, prior_w, prior_b, lam):
    scale = sample_weight.double().sqrt().unsqueeze(1)
    design = torch.cat([phi.double(), torch.ones(phi.shape[0], 1, dtype=torch.float64)], dim=1) * scale
    gram = design.T @ design
    lam_abs = lam * float(torch.trace(gram)) / gram.shape[0]
    gram.diagonal().add_(lam_abs)
    prior = torch.cat([prior_w.double(), prior_b.double().view(1)])
    solved = torch.linalg.solve(gram, design.T @ (target.double() * scale.squeeze(1)) + lam_abs * prior)
    return solved[:-1], solved[-1]


def _forward(phi, weight, bias):
    return phi.double() @ weight.double().T + bias.double()


def _pack(new, old, y):
    rows = torch.arange(len(y))
    delta = float((new[rows, y] - old[rows, y]).mean())
    ce = torch.nn.functional.cross_entropy(new.float(), y, reduction="none")
    flip = float(new.argmax(1).ne(y).float().mean())
    mask = torch.ones_like(new, dtype=torch.bool)
    mask[rows, y] = False
    drift = float((new - old).abs()[mask].mean())
    p = torch.softmax(old.float() / 2, dim=1)
    q = torch.softmax(new.float() / 2, dim=1)
    kl = float((p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log())).sum(1).mean())
    return {
        "delta": delta,
        "ce": float(ce.mean()),
        "std": float(ce.std(unbiased=False)),
        "flip": flip,
        "drift": drift,
        "kl": kl,
    }


def _save(base, weight, bias, path: Path):
    state = {key: value.detach().cpu().float().contiguous() for key, value in base.items()}
    state["head.weight"] = weight.detach().cpu().float().contiguous()
    state["head.bias"] = bias.detach().cpu().float().contiguous()
    for key, value in state.items():
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise RuntimeError(f"bad weights in {key}")
    probe = task_template.build_model()
    probe.load_state_dict(state, strict=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _winner_targets(logits: torch.Tensor, margin: torch.Tensor) -> torch.Tensor:
    y = logits.argmax(1)
    rows = torch.arange(len(logits))
    others = logits.clone()
    others[rows, y] = -1e9
    second = others.max(1).values
    return second + margin, y


def main() -> None:
    model = task_template.load_model()
    images = task_template.load_forget_images()
    base = {key: value.detach().cpu().float().clone() for key, value in model.state_dict().items()}
    w0 = base["head.weight"].double()
    b0 = base["head.bias"].double()
    phi, logits = _collect(model, images)
    logits = logits.double()
    y = logits.argmax(1)
    agrees = torch.zeros(len(images), dtype=torch.long)
    for kind, strength in PROBES:
        if kind == "lowpass":
            view = torch.cat([_lowpass(images[s:s + 128], int(strength)) for s in range(0, len(images), 128)])
        else:
            view = torch.cat([_blur(images[s:s + 128], strength) for s in range(0, len(images), 128)])
        agrees += task_template.predict(model, view).argmax(1).eq(y).long()
    print("agree bins " + ", ".join(f"{i}:{int((agrees == i).sum())}" for i in range(4)), flush=True)
    phi_flip, logits_flip = _collect(model, images.flip(2).contiguous())
    phi_alt, logits_alt = _collect(model, audit.make_altered_copies(images))
    phi_proxy, logits_proxy = _collect(model, audit.patch_shuffle(images))
    logits_flip, logits_alt, logits_proxy = logits_flip.double(), logits_alt.double(), logits_proxy.double()
    views = (
        (phi, logits, 8.0),
        (phi_flip, logits_flip, 3.0),
        (phi_alt, logits_alt, 4.0),
    )
    schedules = [("m2.5", torch.full((len(y),), 2.5)), ("m2", torch.full((len(y),), 2.0)), ("m1.5", torch.full((len(y),), 1.5)), ("m1", torch.full((len(y),), 1.0))]
    varied = torch.tensor([0.8, 1.1, 1.6, 2.4])[agrees.clamp(max=3)]
    schedules.append(("varied", varied))

    best = None
    lines = ["name neg lam ce std flip alt_flip drift proxy_kl forget_kl"]
    print(lines[0], flush=True)
    for name, margin in schedules:
        target_logit, _ = _winner_targets(logits, margin)
        ceiling = logits.clone()
        ceiling[torch.arange(len(y)), y] = target_logit
        cap = _pack(ceiling, logits, y)
        print(f"{name} ceiling ce={cap['ce']:.3f}/{cap['std']:.3f} flip={cap['flip']:.4f} kl={cap['kl']:.3f}", flush=True)
        for neg in NEG_WEIGHTS:
            for lam in LAMBDAS:
                weight, bias = w0.clone(), b0.clone()
                for cls in y.unique().tolist():
                    owns = y == cls
                    chunks_phi, chunks_t, chunks_s = [], [], []
                    for view_phi, view_logits, view_weight in views:
                        absolute = view_logits[:, cls].clone()
                        absolute[owns] = target_logit[owns]
                        chunks_phi.append(view_phi)
                        chunks_t.append(absolute)
                        sample = torch.full((len(y),), neg * view_weight / 8.0)
                        sample[owns] = view_weight
                        chunks_s.append(sample)
                    row_w, row_b = _fit_row(
                        torch.cat(chunks_phi),
                        torch.cat(chunks_t),
                        torch.cat(chunks_s),
                        w0[cls],
                        b0[cls],
                        lam,
                    )
                    weight[cls] = row_w
                    bias[cls] = row_b
                clean = _pack(_forward(phi, weight, bias), logits, y)
                alt = _pack(_forward(phi_alt, weight, bias), logits_alt, y)
                proxy = _pack(_forward(phi_proxy, weight, bias), logits_proxy, logits_proxy.argmax(1))
                line = (
                    f"{name} {neg:g} {lam:g} {clean['ce']:.3f} {clean['std']:.3f} {clean['flip']:.4f} "
                    f"{alt['flip']:.4f} {clean['drift']:.3f} {proxy['kl']:.4f} {clean['kl']:.4f}"
                )
                print(line, flush=True)
                lines.append(line)
                ok = (
                    clean["flip"] <= 0.03
                    and alt["flip"] <= 0.06
                    and clean["ce"] >= 0.15
                    and clean["drift"] <= 0.75
                    and proxy["kl"] <= 1.2
                    and clean["std"] <= 0.55
                )
                trade = clean["ce"] / (1.0 + proxy["kl"])
                if ok and (best is None or trade > best[0]):
                    best = (trade, name, neg, lam, weight.clone(), bias.clone(), clean, alt, proxy)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "margin_table.txt").write_text("\n".join(lines) + "\n")
    if best is None:
        print("no margin setting kept the other logits still and the flip low")
        return
    _trade, name, neg, lam, weight, bias, clean, alt, proxy = best
    path = OUTPUT_DIR / "margin_best.pt"
    _save(base, weight, bias, path)
    print(
        f"saved {path} {name} neg={neg:g} lam={lam:g} ce={clean['ce']:.3f}/{clean['std']:.3f} "
        f"flip={clean['flip']:.4f} alt_flip={alt['flip']:.4f} drift={clean['drift']:.3f} "
        f"proxy_kl={proxy['kl']:.4f}",
        flush=True,
    )
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    model.to(task_template.DEVICE).eval()
    confirmed = _pack(task_template.predict(model, images).double(), logits, y)
    print(
        f"predict ce={confirmed['ce']:.3f}/{confirmed['std']:.3f} flip={confirmed['flip']:.4f} "
        f"drift={confirmed['drift']:.3f} kl={confirmed['kl']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
