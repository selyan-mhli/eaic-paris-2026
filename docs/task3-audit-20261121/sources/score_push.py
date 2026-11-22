"""Drop each forget image's winning logit by a fixed amount.

The drop is fit per class, and only that class row of the head changes.
Images of class c, their flips, and their mild alterations are asked for the
same change: logit c decreases by alpha, every other logit stays. The fit is
minimum-norm, so the step lies in the span of those features.

On the stored forget logits, alpha 6 gives cross-entropy 0.063 and T=2 KL 0.16
with a 2% flip. Alpha 8 gives cross-entropy 0.23. Earlier head edits stopped
around alpha 2, where cross-entropy is still 0.001.

Does not modify task_template.py or audit.py.
"""

from __future__ import annotations

from pathlib import Path

import torch

import audit
import task_template

OUTPUT_DIR = Path("output")
ALPHAS = (5.0, 6.0, 7.0, 8.0)
LAMBDAS = (1e-8, 1e-6, 1e-4, 1e-2)


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


def _fit_delta(features: list[torch.Tensor], alpha: float, lam: float):
    design_x = torch.cat(features, 0).double()
    target = torch.full((design_x.shape[0],), -alpha, dtype=torch.float64)
    design = torch.cat([design_x, torch.ones(design_x.shape[0], 1, dtype=torch.float64)], dim=1)
    gram = design.T @ design
    gram.diagonal().add_(lam * float(torch.trace(gram)) / gram.shape[0])
    solved = torch.linalg.solve(gram, design.T @ target)
    return solved[:-1], solved[-1]


def _logits(phi, weight, bias):
    return phi.double() @ weight.double().T + bias.double()


def _stats(logits, reference, y):
    rows = torch.arange(len(y))
    delta = float((logits[rows, y] - reference[rows, y]).mean())
    ce = torch.nn.functional.cross_entropy(logits.float(), y, reduction="none")
    flip = float(logits.argmax(1).ne(y).float().mean())
    p = torch.softmax(reference.float() / 2, dim=1)
    q = torch.softmax(logits.float() / 2, dim=1)
    kl = float((p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log())).sum(1).mean())
    return {
        "delta": delta,
        "ce": float(ce.mean()),
        "std": float(ce.std(unbiased=False)),
        "flip": flip,
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


def main() -> None:
    model = task_template.load_model()
    images = task_template.load_forget_images()
    base = {key: value.detach().cpu().float().clone() for key, value in model.state_dict().items()}
    w0 = base["head.weight"].double()
    b0 = base["head.bias"].double()
    phi, logits = _collect(model, images)
    y = logits.argmax(1)
    recon = _logits(phi, w0, b0)
    print(f"reconstruct max abs {float((recon - logits.double()).abs().max()):.4f}", flush=True)
    phi_flip = _collect(model, images.flip(2).contiguous())[0]
    phi_alt = _collect(model, audit.make_altered_copies(images))[0]
    proxy_images = audit.patch_shuffle(images)
    phi_proxy, logits_proxy = _collect(model, proxy_images)
    groups = [(cls.item(), (y == cls).nonzero(as_tuple=False).flatten()) for cls in y.unique()]
    print(f"classes {len(groups)}", flush=True)

    best = None
    saved_alpha = set()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = OUTPUT_DIR / "push_table.txt"
    lines = ["alpha lam delta ce std flip alt_flip proxy_kl"]
    for alpha in ALPHAS:
        for lam in LAMBDAS:
            weight, bias = w0.clone(), b0.clone()
            for cls, index in groups:
                dw, db = _fit_delta(
                    [phi[index], phi_flip[index], phi_alt[index]],
                    alpha,
                    lam,
                )
                weight[cls] = w0[cls] + dw
                bias[cls] = b0[cls] + db
            clean = _stats(_logits(phi, weight, bias), logits, y)
            alt = _stats(_logits(phi_alt, weight, bias), logits, y)
            proxy = _stats(_logits(phi_proxy, weight, bias), logits_proxy, logits_proxy.argmax(1))
            line = (
                f"{alpha:g} {lam:g} {clean['delta']:.3f} {clean['ce']:.3f} {clean['std']:.3f} "
                f"{clean['flip']:.4f} {alt['flip']:.4f} {proxy['kl']:.4f}"
            )
            print(line, flush=True)
            lines.append(line)
            ok = clean["flip"] <= 0.03 and alt["flip"] <= 0.05 and clean["ce"] >= 0.05 and clean["delta"] <= -4.0
            trade = clean["ce"] / (1.0 + proxy["kl"])
            if ok and alpha not in saved_alpha:
                path = OUTPUT_DIR / f"push_a{int(alpha)}.pt"
                _save(base, weight, bias, path)
                saved_alpha.add(alpha)
                print(f"saved {path}", flush=True)
            if ok and (best is None or trade > best[0]):
                best = (trade, alpha, lam, weight.clone(), bias.clone(), clean, alt, proxy)
    table.write_text("\n".join(lines) + "\n")
    if best is None:
        print("no setting reached a real cross-entropy with the flip limit")
        return
    _trade, alpha, lam, weight, bias, clean, alt, proxy = best
    path = OUTPUT_DIR / "push_best.pt"
    _save(base, weight, bias, path)
    print(
        f"saved {path} alpha={alpha:g} lam={lam:g} delta={clean['delta']:.3f} "
        f"ce={clean['ce']:.3f}/{clean['std']:.3f} flip={clean['flip']:.4f} "
        f"alt_flip={alt['flip']:.4f} proxy_kl={proxy['kl']:.4f}",
        flush=True,
    )
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    model.to(task_template.DEVICE).eval()
    confirmed = _stats(task_template.predict(model, images), logits, y)
    print(
        f"predict ce={confirmed['ce']:.3f}/{confirmed['std']:.3f} "
        f"flip={confirmed['flip']:.4f} delta={confirmed['delta']:.3f} kl={confirmed['kl']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
