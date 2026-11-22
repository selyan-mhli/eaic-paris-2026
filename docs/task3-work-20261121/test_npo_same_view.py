"""Test CPU du correctif NPO (sans poids ni données du concours). Lancement :
    .venv/bin/python docs/task3-work-20261121/test_npo_same_view.py
"""
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unlearn_v2_fixed as uf  # noqa: E402


def tiny_model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(  # B 3 32 32 -> B 10
        torch.nn.Conv2d(3, 8, 3, stride=2, padding=1), torch.nn.GELU(),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(8, 10))


def main() -> None:
    torch.manual_seed(1)
    student = tiny_model()
    reference = copy.deepcopy(student).eval().requires_grad_(False)
    clean = torch.rand(16, 3, 32, 32) * 2 - 1          # B 3 H W dans [-1, 1]
    y = reference(clean).argmax(1)
    aug = (clean.flip(-1) * 0.6 + 0.3 * torch.randn_like(clean)).clamp(-1, 1)  # vue « augmentée »

    # 1) Même vue, poids identiques : log-ratio nul à la précision numérique, perte NPO = 2/beta·log 2.
    student.train()
    lp_sub, lp_ref = uf.same_view_log_ratio(student, reference, aug, y)
    delta = lp_sub - lp_ref
    assert delta.abs().max().item() < 1e-6, delta.abs().max().item()
    beta = 0.1
    loss = uf._npo(lp_sub, lp_ref, beta)
    expected = (2.0 / beta) * torch.log(torch.tensor(2.0))
    assert abs(loss.item() - expected.item()) < 1e-4, (loss.item(), expected.item())
    # Le gradient passe par l'élève seulement.
    loss.backward()
    assert all(p.grad is not None for p in student.parameters())
    assert all(p.grad is None for p in reference.parameters())

    # 2) Ancien calcul (référence sur la vue propre, élève sur la vue augmentée) : non nul.
    with torch.no_grad():
        lp_ref_clean = torch.log_softmax(reference(clean), 1).gather(1, y[:, None]).squeeze(1)
        mixed = (lp_sub.detach() - lp_ref_clean).abs().max().item()
    assert mixed > 1e-3, mixed

    # 3) Après une modification des poids de l'élève, le ratio devient non nul (le test n'est pas vide).
    with torch.no_grad():
        student[-1].bias[y[0]] -= 1.0
    lp_sub2, lp_ref2 = uf.same_view_log_ratio(student, reference, aug, y)
    assert (lp_sub2 - lp_ref2).abs().max().item() > 0.1
    print(f"OK : même vue |delta|max = {delta.abs().max().item():.2e} ; "
          f"ancien calcul (vues mélangées) |delta|max = {mixed:.4f} ; NPO initiale = {loss.item():.4f}")


if __name__ == "__main__":
    main()
