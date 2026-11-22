"""Reproductions CPU de constats d'audit, sans poids ni données du concours."""
from pathlib import Path
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "c3"))
import eval as ev  # noqa: E402


def main():
    factory = lambda: torch.nn.Linear(2, 2)
    state = factory().state_dict()
    state["weight"][0, 0] = float("nan")
    clean = np.array([[12., 0., 0.], [12., 0., 0.]])
    augmented = np.array([[1., 0., 0.], [1., 0., 0.]])
    gold = np.log([[.6, .3, .1], [.6, .3, .1]])
    wrong = np.log([[.6, .1, .3], [.6, .1, .3]])
    labels = np.array([0, 0])
    npo = lambda delta: float(-20 * torch.nn.functional.logsigmoid(torch.tensor(-.1 * delta)))
    result = {
        "format_accepts_nan": ev.check_format(state, factory)["format_ok"],
        "identity_official_forget_by_formula": 0.,
        "identity_local_forget_proxy_synthetic": ev.forget_proxy(clean, clean, augmented)["forget_proxy"],
        "same_ce_different_wrong_class": {
            "gold_ce": float(ev.ce_losses(gold, labels).mean()),
            "wrong_ce": float(ev.ce_losses(wrong, labels).mean()),
            "KL_gold_wrong_T2": ev.kl_divergence(gold, wrong),
        },
        # Exemple construit, pas une mesure d'un batch réel.
        "npo_initial_same_view_delta": 0.,
        "npo_initial_mixed_view_delta": -4. + .000322,
        "npo_same_view_loss": npo(0.),
        "npo_mixed_view_loss": npo(-4. + .000322),
    }
    assert result["format_accepts_nan"]
    assert result["identity_local_forget_proxy_synthetic"] > 0
    comparison = result["same_ce_different_wrong_class"]
    assert comparison["gold_ce"] == comparison["wrong_ce"]
    assert comparison["KL_gold_wrong_T2"] > 0
    # Les assertions confirment des défauts observés ; ce ne sont pas des correctifs.
    print(json.dumps(result, indent=2))
    Path(__file__).with_name("reproductions.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
