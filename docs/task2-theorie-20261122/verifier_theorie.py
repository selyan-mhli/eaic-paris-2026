"""Contre-exemples mathématiques, sans données cachées, réseau ni modèle."""
from itertools import combinations, product
from pathlib import Path
import json
import math

HERE = Path(__file__).resolve().parent


def mcc(truth, pred):
    tp = fp = fn = tn = 0
    for a, b in combinations(truth, 2):
        t, p = truth[a] == truth[b], pred[a] == pred[b]
        tp += t and p
        fp += not t and p
        fn += t and not p
        tn += not t and not p
    den = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return max(0., (tp*tn-fp*fn)/den) if den else 0.


def kl(p, q):
    return sum(a*math.log(a/b) for a, b in zip(p, q) if a)


def temperature(p, t):
    w = [a**(1/t) for a in p]
    return [a/sum(w) for a in w]


def miou(pred, true):
    scores = []
    for c in (0, 1):
        inter = sum(p == c and t == c for p, t in zip(pred, true))
        union = sum(p == c or t == c for p, t in zip(pred, true))
        scores.append(inter/union if union else 1.)
    return sum(scores)/2


def main():
    out = {"nature": "Exemples construits ; aucun score de modèle ou de serveur."}
    p = 1/(1+math.exp(-2/.5))
    q = 1/(1+math.exp(-2/2.))
    out["T2_same_weights_different_temperature"] = {
        "logits": [2., 0.], "T_A": .5, "T_B": 2., "P_A_option1": p,
        "P_B_option1": q, "MMD2_delta_kernel": 2*(p-q)**2,
        "same_model_by_challenge_definition": True,
    }
    truth = {i: ("base" if i in (1, 3, 5, 6, 8) else str(i)) for i in range(1, 10)}
    out["T2_partial_partitions_using_known_truth_not_a_method"] = []
    for iso in ([], [7], [4, 7], [2, 4, 7], [2, 4, 7, 9]):
        pred = {i: (str(i) if i in iso else "base") for i in truth}
        mm = mcc(truth, pred)
        out["T2_partial_partitions_using_known_truth_not_a_method"].append({
            "isolates": iso, "model_MCC": mm,
            "hypothetical_total_if_families_perfect_on_scored_endpoints": .1+.9*mm,
        })
    g, a, b = [.8, .18, .02], [.8, .18, .02], [.8, .02, .18]
    out["T3_same_loss_different_predictions"] = {
        "true_label": 0, "CE_A_T1": -math.log(a[0]), "CE_B_T1": -math.log(b[0]),
        "KL_gold_A_T2": kl(temperature(g, 2), temperature(a, 2)),
        "KL_gold_B_T2": kl(temperature(g, 2), temperature(b, 2)),
        "note": "Même CE à T=1, KL différente après passage à T=2, comme dans le PDF.",
    }
    p, q = [.9, .09, .01], [.51, .245, .245]
    k = kl(p, q)
    out["T3_same_top1_nonzero_retain_KL"] = {
        "probabilities_at_T2": [p, q], "same_top1": True,
        "KL_original_candidate": k, "retain_KL_factor": 1/(1+k),
    }
    masks = list(product((0, 1), repeat=2))
    populations = {"both_pixels_together": [(0, 0), (1, 1)],
                   "opposite_pixels": [(0, 1), (1, 0)]}
    out["T1_marginal_probabilities_not_enough_for_expected_mIoU"] = {}
    for name, truths in populations.items():
        marginal = [sum(t[i] for t in truths)/2 for i in range(2)]
        scores = {"".join(map(str, m)): sum(miou(m, t) for t in truths)/2 for m in masks}
        best = max(scores.values())
        out["T1_marginal_probabilities_not_enough_for_expected_mIoU"][name] = {
            "pixel_marginals": marginal, "expected_mIoU": scores,
            "optimal_masks": [m for m, v in scores.items() if v == best],
        }
    assert out["T2_same_weights_different_temperature"]["MMD2_delta_kernel"] > .1
    assert abs(out["T2_partial_partitions_using_known_truth_not_a_method"][2]["model_MCC"]-.5241424183609592) < 1e-12
    assert a[0] == b[0] and kl(temperature(g, 2), temperature(a, 2)) == 0.
    assert kl(temperature(g, 2), temperature(b, 2)) > 0.
    populations = out["T1_marginal_probabilities_not_enough_for_expected_mIoU"]
    assert populations["both_pixels_together"]["pixel_marginals"] == populations["opposite_pixels"]["pixel_marginals"]
    assert set(populations["both_pixels_together"]["optimal_masks"]).isdisjoint(populations["opposite_pixels"]["optimal_masks"])
    (HERE/"contre_exemples.json").write_text(json.dumps(out, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
