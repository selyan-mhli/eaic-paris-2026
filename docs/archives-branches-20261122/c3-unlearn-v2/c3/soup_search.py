"""Linear weight soup between model.pt and output/ckpt_ent16.pt.

Does not train and does not modify task_template.py or audit.py.
Original-model logits are cached by patching task_template.predict, because
audit.audit_model recomputes them and the altered copies are seed-locked.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

import audit
import task_template

LAMBDAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
ORIG_FILE = Path("model.pt")
UNLEARN_FILE = Path("output/ckpt_ent16.pt")
SAFE_FILE = Path("output/soup_safe.pt")
MID_FILE = Path("output/soup_mid.pt")
TABLE_FILE = Path("output/soup_table.txt")


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"{path} is not a state_dict")
    return state


def _mix(orig: dict[str, torch.Tensor], unlearn: dict[str, torch.Tensor], lam: float) -> dict[str, torch.Tensor]:
    if orig.keys() != unlearn.keys():
        raise KeyError("model.pt and ckpt_ent16.pt do not have the same keys")
    mixed = {}
    for key, base in orig.items():
        other = unlearn[key]
        if base.shape != other.shape:
            raise ValueError(f"shape mismatch for {key}: {tuple(base.shape)} vs {tuple(other.shape)}")
        if torch.is_floating_point(base):
            value = (1.0 - lam) * base.float() + lam * other.float()
            if not torch.isfinite(value).all():
                raise RuntimeError(f"non-finite interpolated weights in {key} at lam={lam}")
            mixed[key] = value.contiguous()
        else:
            mixed[key] = base.clone()
    return mixed


def _check_load(state: dict[str, torch.Tensor]) -> None:
    probe = task_template.build_model()
    probe.load_state_dict(state, strict=True)


def _save(state: dict[str, torch.Tensor], path: Path) -> None:
    _check_load(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _signature(images: torch.Tensor) -> tuple:
    flat = images.detach().reshape(-1)
    step = max(1, flat.numel() // 8192)
    digest = hashlib.sha256(flat[::step].cpu().numpy().tobytes()).hexdigest()
    return (tuple(images.shape), str(images.dtype), digest)


def _cache_original_logits(orig_model: torch.nn.Module) -> None:
    real_predict = task_template.predict
    cache: dict[tuple, torch.Tensor] = {}

    def cached_predict(model, images, batch_size=128):
        if model is not orig_model:
            return real_predict(model, images, batch_size=batch_size)
        key = _signature(images)
        if key not in cache:
            cache[key] = real_predict(model, images, batch_size=batch_size)
            print(f"cached orig logits {tuple(images.shape)}", flush=True)
        else:
            print(f"reused orig logits {tuple(images.shape)}", flush=True)
        return cache[key]

    task_template.predict = cached_predict


def _row(report: dict) -> dict[str, float]:
    forget = report["forget"]
    altered = report["altered"]
    return {
        "forget_ent_t2": forget["sub_mean_entropy_t2"],
        "altered_ent_t2": altered["sub_mean_entropy_t2"],
        "pseudo_ce_mean": forget["pseudo_ce_mean"],
        "pseudo_ce_std": forget["pseudo_ce_std"],
        "flip_rate": forget["top1_flip_rate"],
        "proxy_kl": report["proxy_patch_shuffle"]["kl_orig_sub_t2"],
    }


def _best(rows: list[tuple[float, dict, dict]], limit: float) -> tuple[float, dict] | None:
    eligible = [(lam, state, metrics) for lam, state, metrics in rows if metrics["proxy_kl"] < limit]
    if not eligible:
        return None
    lam, state, _ = max(eligible, key=lambda item: (item[2]["forget_ent_t2"], -item[0]))
    return lam, state


def main() -> int:
    orig_state = _load_state(ORIG_FILE)
    unlearn_state = _load_state(UNLEARN_FILE)
    orig_model = task_template.load_model()
    orig_model.requires_grad_(False)
    images = task_template.load_forget_images()
    _cache_original_logits(orig_model)

    rows = []
    lines = ["lam forget_ent_T2 altered_ent_T2 pseudo_ce_mean pseudo_ce_std flip_rate proxy_KL"]
    print(lines[0], flush=True)
    for lam in LAMBDAS:
        state = _mix(orig_state, unlearn_state, lam)
        student = task_template.build_model()
        student.load_state_dict(state, strict=True)
        student.to(task_template.DEVICE).eval()
        report = audit.audit_model(orig_model, student, images)
        metrics = _row(report)
        rows.append((lam, state, metrics))
        line = (
            f"{lam:.1f} {metrics['forget_ent_t2']:.6f} {metrics['altered_ent_t2']:.6f} "
            f"{metrics['pseudo_ce_mean']:.6f} {metrics['pseudo_ce_std']:.6f} "
            f"{metrics['flip_rate']:.6f} {metrics['proxy_kl']:.6f}"
        )
        lines.append(line)
        print(line, flush=True)
        del student
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    TABLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    TABLE_FILE.write_text("\n".join(lines) + "\n")

    safe = _best(rows, 0.25)
    mid = _best(rows, 0.50)
    if safe is None:
        print("no lambda with proxy_KL < 0.25", flush=True)
    else:
        _save(safe[1], SAFE_FILE)
        print(f"saved {SAFE_FILE} lam={safe[0]:.1f}", flush=True)
    if mid is None:
        print("no lambda with proxy_KL < 0.50", flush=True)
    else:
        _save(mid[1], MID_FILE)
        print(f"saved {MID_FILE} lam={mid[0]:.1f}", flush=True)
    return 0 if safe is not None else 2


if __name__ == "__main__":
    sys.exit(main())
