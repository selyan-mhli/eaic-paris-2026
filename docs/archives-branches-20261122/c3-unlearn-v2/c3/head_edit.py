"""Closed-form classification-head edit.

The backbone stays at the original weights. Only head rows for classes that
win on the forget set are updated. The target on a forget image is the
original logit vector with alpha subtracted from the winning logit. Augmented
views of that image use the same target. Patch-shuffled views keep the
original logits. Other class rows are copied unchanged.

Does not modify task_template.py or audit.py. Reported logits come from
task_template.predict (eval, fp16 autocast).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import audit
import task_template
import unlearn_v2

OUTPUT_DIR = Path("output")
ALPHAS = (2.0, 4.0)
LAM_MULTS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def _save_state(state: dict[str, torch.Tensor], path: Path) -> None:
    clean = {key: value.detach().cpu().float().contiguous() for key, value in state.items()}
    for key, value in clean.items():
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise RuntimeError(f"non-finite or non-float32 weights in {key}")
    probe = task_template.build_model()
    probe.load_state_dict(clean, strict=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(clean, path)


@torch.no_grad()
def _collect(model: torch.nn.Module, images_uint8: torch.Tensor | None = None, preprocessed: torch.Tensor | None = None, batch_size: int = 128):
    feats, logits = [], []

    def hook(_module, inputs, output):
        feats.append(inputs[0].detach().float().cpu())
        logits.append(output.detach().float().cpu())

    handle = model.head.register_forward_hook(hook)
    model.eval()
    if images_uint8 is not None:
        for start in range(0, len(images_uint8), batch_size):
            batch = images_uint8[start:start + batch_size]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=task_template.DEVICE == "cuda"):
                model(task_template.preprocess(batch))
    else:
        for start in range(0, len(preprocessed), batch_size):
            batch = preprocessed[start:start + batch_size]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=task_template.DEVICE == "cuda"):
                model(batch)
    handle.remove()
    return torch.cat(feats), torch.cat(logits)


def _augment_all(images: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
    pieces = []
    for start in range(0, len(images), batch_size):
        pieces.append(unlearn_v2._augment_batch(images[start:start + batch_size]).cpu())
    return torch.cat(pieces)


def _solve_row(phi: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, w0: torch.Tensor, b0: torch.Tensor, lam: float):
    scale = weight.sqrt().unsqueeze(1)
    design = torch.cat([phi, torch.ones(phi.shape[0], 1, dtype=phi.dtype)], dim=1) * scale
    rhs = target.unsqueeze(1) * scale
    gram = design.T @ design
    prior = torch.cat([w0, b0.view(1)])
    lam_abs = lam * float(torch.trace(gram)) / gram.shape[0]
    gram.diagonal().add_(lam_abs)
    solved = torch.linalg.solve(gram, design.T @ rhs.squeeze(1) + lam_abs * prior)
    return solved[:-1], solved[-1]


def _metrics(logits: torch.Tensor, reference: torch.Tensor, winner: torch.Tensor) -> dict[str, float]:
    delta = (logits.gather(1, winner[:, None]) - reference.gather(1, winner[:, None])).squeeze(1)
    probs = torch.softmax(logits / 2.0, dim=1)
    ref = torch.softmax(reference / 2.0, dim=1)
    kl = (ref * (ref.clamp_min(1e-12).log() - probs.clamp_min(1e-12).log())).sum(dim=1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
    return {
        "delta": float(delta.mean()),
        "flip": float((logits.argmax(1) != winner).float().mean()),
        "entropy": float(entropy.mean()),
        "kl": float(kl.mean()),
    }


def _install(model: torch.nn.Module, base: dict[str, torch.Tensor], weight: torch.Tensor, bias: torch.Tensor) -> dict[str, torch.Tensor]:
    state = {key: value.detach().cpu().float().clone() for key, value in base.items()}
    state["head.weight"] = weight.detach().cpu().float().contiguous()
    state["head.bias"] = bias.detach().cpu().float().contiguous()
    with torch.no_grad():
        model.head.weight.copy_(state["head.weight"].to(model.head.weight.device))
        model.head.bias.copy_(state["head.bias"].to(model.head.bias.device))
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--views", type=int, default=2)
    args = parser.parse_args()

    model = task_template.load_model()
    images = task_template.load_forget_images()
    base_state = {key: value.detach().cpu().float().clone() for key, value in model.state_dict().items()}
    w0 = base_state["head.weight"].double()
    b0 = base_state["head.bias"].double()

    print("extracting clean features", flush=True)
    phi_clean, logits_clean = _collect(model, images_uint8=images)
    winner = logits_clean.argmax(dim=1)
    editable = winner.unique()
    print(f"editable classes {editable.numel()} / {logits_clean.shape[1]}", flush=True)

    print("extracting augmented and shuffled features", flush=True)
    aug_phi, aug_owner = [], []
    torch.manual_seed(0)
    for view in range(args.views):
        preprocessed = _augment_all(images)
        phi, _logits = _collect(model, preprocessed=preprocessed.to(task_template.DEVICE))
        aug_phi.append(phi)
        aug_owner.append(torch.arange(len(images)))
        print(f"view {view + 1} ready", flush=True)
    proxy_images = audit.patch_shuffle(images)
    phi_proxy, logits_proxy = _collect(model, images_uint8=proxy_images)

    target_clean = logits_clean.double()
    groups = {
        "clean": (phi_clean.double(), None, torch.full((len(images),), 10.0)),
        "proxy": (phi_proxy.double(), logits_proxy.double(), torch.full((len(images),), 1.0)),
    }
    for view, phi in enumerate(aug_phi):
        groups[f"aug{view}"] = (phi.double(), None, torch.full((len(images),), 1.0))

    chosen = {}
    print("alpha lam delta flip entropy kl_clean kl_proxy", flush=True)
    for alpha in ALPHAS:
        softened = target_clean.clone()
        softened[torch.arange(len(images)), winner] -= alpha
        best = None
        for lam in LAM_MULTS:
            weight = w0.clone()
            bias = b0.clone()
            for cls in editable.tolist():
                pieces_phi, pieces_t, pieces_s = [], [], []
                for name, (phi, explicit, sample_weight) in groups.items():
                    target = softened[:, cls] if explicit is None else explicit[:, cls]
                    pieces_phi.append(phi)
                    pieces_t.append(target)
                    pieces_s.append(sample_weight)
                row_w, row_b = _solve_row(
                    torch.cat(pieces_phi),
                    torch.cat(pieces_t),
                    torch.cat(pieces_s).double(),
                    w0[cls],
                    b0[cls],
                    lam,
                )
                weight[cls] = row_w
                bias[cls] = row_b
            state = _install(model, base_state, weight.float(), bias.float())
            new_clean = task_template.predict(model, images)
            new_proxy = task_template.predict(model, proxy_images)
            clean = _metrics(new_clean, logits_clean, winner)
            proxy = _metrics(new_proxy, logits_proxy, logits_proxy.argmax(1))
            print(
                f"{alpha:g} {lam:g} {clean['delta']:.3f} {clean['flip']:.4f} "
                f"{clean['entropy']:.3f} {clean['kl']:.4f} {proxy['kl']:.4f}",
                flush=True,
            )
            ok = clean["flip"] <= 0.01 and proxy["kl"] <= 0.05
            score = abs(clean["delta"] + alpha)
            if ok and (best is None or score < best[0]):
                best = (score, state, clean, proxy, lam)
        if best is None:
            print(f"alpha {alpha:g}: no lambda kept flip<=0.01 and proxy KL<=0.05", flush=True)
            continue
        _score, state, clean, proxy, lam = best
        path = OUTPUT_DIR / f"head_a{int(alpha)}.pt"
        _save_state(state, path)
        chosen[alpha] = path
        print(
            f"saved {path} lam={lam:g} delta={clean['delta']:.3f} "
            f"flip={clean['flip']:.4f} entropy={clean['entropy']:.3f} proxy_kl={proxy['kl']:.4f}",
            flush=True,
        )

    _install(model, base_state, w0.float(), b0.float())
    if 2.0 in chosen:
        edited = task_template.build_model()
        edited.load_state_dict(torch.load(chosen[2.0], map_location="cpu", weights_only=True), strict=True)
        edited.to(task_template.DEVICE).eval()
        print("audit of head_a2.pt", flush=True)
        audit.audit_model(model, edited, images)


if __name__ == "__main__":
    main()
