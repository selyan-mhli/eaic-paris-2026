"""Réimplémentation de la formule serveur Task 3 (docstring de task_template.py, figé dans
docs/task3-audit-20261121/sources/task_template.py), plus des proxys par scénarios de gold.

Formule (par groupe g ∈ {propres, altérées}, moyennes ensuite sur les deux groupes) :
  d        = |mean(L) - gold_mean| + |std(L) - gold_std|        L = CE par image (vrai label)
  ms_g     = 1 - clip(d / d_original, 0, 1)
  pred_g   = 1 - clip(mean KL(p_gold || p_cand) / mean KL(p_gold || p_orig), 0, 1)   (T = 2,
             p_gold = moyenne des softmax T=2 de plusieurs gold)
  forget   = mean_g(ms_g) * mean_g(pred_g)
  retain   = clip(acc_cand / acc_orig, 0, 1) / (1 + mean KL(p_orig || p_cand))   (T = 2, contrôle caché)
  final    = forget * retain ; 0 si accuracy de validation cachée < 70 %.
Point non spécifié par le texte : écart-type biaisé ou non. On prend ddof=0 (comme score_like.py) ;
l'écart est négligeable pour N = 3 500 (facteur sqrt(N/(N-1)) ≈ 1,00014).

Aucun gold n'est disponible : `ms_from_stats` est exact À SCÉNARIO DE GOLD DONNÉ, le reste est
un outil de simulation. Ce module ne prédit pas un score serveur.
"""
from __future__ import annotations

import numpy as np

T = 2.0


def _log_softmax(z: np.ndarray, t: float = T) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / t
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def ce_losses(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """CE par image à T = 1. logits [N, C], labels [N] -> [N]."""
    lp = _log_softmax(logits, 1.0)
    return -lp[np.arange(len(labels)), labels]


def kl_rows(p: np.ndarray, log_q: np.ndarray) -> np.ndarray:
    """KL(p || q) par ligne ; p probabilités [N, C], log_q log-probabilités [N, C]."""
    p = np.asarray(p, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(p > 0, p * (np.log(np.where(p > 0, p, 1.0)) - log_q), 0.0)
    return term.sum(axis=1)


def gold_probs(gold_logits_list: list[np.ndarray]) -> np.ndarray:
    """Moyenne des softmax T=2 de plusieurs gold -> [N, C]."""
    return np.mean([np.exp(_log_softmax(g)) for g in gold_logits_list], axis=0)


def ms_score(mean: float, std: float, gold_mean: float, gold_std: float,
             orig_mean: float, orig_std: float) -> float:
    d = abs(mean - gold_mean) + abs(std - gold_std)
    d0 = abs(orig_mean - gold_mean) + abs(orig_std - gold_std)
    if d0 <= 0:
        return 0.0
    return float(1.0 - np.clip(d / d0, 0.0, 1.0))


def pred_score(p_gold: np.ndarray, cand_logits: np.ndarray, orig_logits: np.ndarray) -> float:
    k = kl_rows(p_gold, _log_softmax(cand_logits)).mean()
    k0 = kl_rows(p_gold, _log_softmax(orig_logits)).mean()
    if k0 <= 0:
        return 0.0
    return float(1.0 - np.clip(k / k0, 0.0, 1.0))


def forget_score(groups: list[dict]) -> dict:
    """groups : une entrée par groupe (propres, altérées), clés
    cand, orig [N, C] logits ; gold : liste de logits [N, C] ; labels [N]."""
    ms, pr = [], []
    for g in groups:
        lc, lo = ce_losses(g["cand"], g["labels"]), ce_losses(g["orig"], g["labels"])
        lg = [ce_losses(x, g["labels"]) for x in g["gold"]]
        # « gold_mean / gold_std are those of retrained gold models » : moyenne sur les gold.
        gm = float(np.mean([x.mean() for x in lg]))
        gs = float(np.mean([x.std() for x in lg]))
        ms.append(ms_score(lc.mean(), lc.std(), gm, gs, lo.mean(), lo.std()))
        pr.append(pred_score(gold_probs(g["gold"]), g["cand"], g["orig"]))
    m, p = float(np.mean(ms)), float(np.mean(pr))
    return {"ms": m, "pred": p, "forget": m * p, "ms_groups": ms, "pred_groups": pr}


def retain_score(acc_cand: float, acc_orig: float, cand_logits: np.ndarray, orig_logits: np.ndarray) -> float:
    p = np.exp(_log_softmax(orig_logits))
    k = kl_rows(p, _log_softmax(cand_logits)).mean()
    return float(np.clip(acc_cand / acc_orig, 0.0, 1.0) / (1.0 + k))


# Scénarios de gold pour la CE forget (moyenne, écart-type). Non mesurés : ordres de grandeur
# d'une CE de test pour un ViT-B/16 à 100 classes (la forget est « hors entraînement » pour le gold).
# « eval_target » = loss_target_mean/std de c3/eval.py (origine sur augmentations fortes), seule
# valeur chiffrée disponible localement, elle-même un proxy.
GOLD_SCENARIOS = {
    "faible": (0.5, 1.2),
    "eval_target": (1.1444, 2.6609),
    "fort": (1.8, 3.2),
}
ORIG_FORGET = (0.0003, 0.0024)     # origine, forget propres (score_like, pseudo-labels)
ORIG_ALTERED = (0.0012, 0.0267)    # origine, altération légère locale (score_like)


def ms_from_stats(clean: tuple[float, float], altered: tuple[float, float] | None,
                  scenario: tuple[float, float]) -> float:
    """Terme moyenne/écart-type exact pour un scénario de gold (même gold pour les deux groupes)."""
    gm, gs = scenario
    vals = [ms_score(*clean, gm, gs, *ORIG_FORGET)]
    if altered is not None:
        vals.append(ms_score(*altered, gm, gs, *ORIG_ALTERED))
    return float(np.mean(vals))


if __name__ == "__main__":
    # Cas à la main. Origine (0, 0), gold (1, 2), candidat (0,5 ; 1) : d = 0,5 + 1 = 1,5 ; d0 = 3 ;
    # ms = 0,5. Candidat qui dépasse (2 ; 4) : d = 3 -> ms = 0. Identité -> 0.
    assert abs(ms_score(0.5, 1.0, 1.0, 2.0, 0.0, 0.0) - 0.5) < 1e-12
    assert ms_score(2.0, 4.0, 1.0, 2.0, 0.0, 0.0) == 0.0
    assert ms_score(0.0, 0.0, 1.0, 2.0, 0.0, 0.0) == 0.0
    # pred : candidat = gold -> 1 ; candidat = origine -> 0.
    rng = np.random.default_rng(0)
    o, g = rng.normal(size=(5, 4)) * 4, rng.normal(size=(5, 4))
    assert abs(pred_score(gold_probs([g]), g, o) - 1.0) < 1e-12
    assert pred_score(gold_probs([g]), o, o) == 0.0
    print("official_formula : cas à la main OK")
