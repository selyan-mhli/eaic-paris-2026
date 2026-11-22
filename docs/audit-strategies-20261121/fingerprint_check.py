"""Contre-épreuve CPU sur les réponses existantes ; aucun accès réseau.

Représentations et prompts fixés avant le calcul, sans sélection de features.
Validation par répétitions 0/1 contre 2/3 et inverse ; mêmes neuf références.
Ce diagnostic n'est ni une validation sur de nouveaux modèles ni un score serveur.
"""
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import hashlib
import json
import math
import re

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parent / "task2-work-20261121/corpus.jsonl"
TRUTH = {f"ref-{i:02}": ("base" if i in (1, 3, 5, 6, 8) else str(i)) for i in range(1, 10)}
EPS = sorted(TRUTH)
PAIRS = list(combinations(EPS, 2))
# Vérification du texte des prompts avant lecture des résultats : on exclut
# m06 (nombres écrits en français), m12 (aléatoire), w12 (chiffres romains).
NUMERIC = ("m03", "m04", "m07", "c01", "c04", "c10", "c11")


def mcc(pred):
    tp = fp = fn = tn = 0
    for a, b in PAIRS:
        t, p = TRUTH[a] == TRUTH[b], pred[a] == pred[b]
        tp += t and p
        fp += not t and p
        fn += t and not p
        tn += not t and not p
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return max(0.0, (tp * tn - fp * fn) / den) if den else 0.0


@lru_cache(None)
def representation(text, pid, kind):
    if kind == "raw_ngram":
        s = " ".join(text.lower().split())
        return Counter(s[i:i+3] for i in range(max(0, len(s)-2))) or Counter({"EMPTY": 1})
    s = re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)
    if pid == "m04":
        runs = re.findall(r"\d[\d\s.,]{15,}", s)
        s = max(runs, key=len, default=s)
        return tuple(re.findall(r"\d", s)[:220]) or ("EMPTY",)
    return tuple(re.findall(r"-?\d+", s)[:220]) or ("EMPTY",)


def kernel(a, b, pid, kind):
    x, y = representation(a, pid, kind), representation(b, pid, kind)
    if kind == "raw_ngram":
        dot = sum(v * y.get(k, 0) for k, v in x.items())
        return dot / math.sqrt(sum(v*v for v in x.values()) * sum(v*v for v in y.values()))
    # Noyau Hamming normalisé, avec symbole de padding commun.
    n = 220
    return (sum(v == w for v, w in zip(x, y)) + n - max(len(x), len(y))) / n


def distance(pool, half, pids, kind, stat):
    out = {}
    for a, b in PAIRS:
        vals = []
        for pid in pids:
            A, B = pool[half, a, pid], pool[half, b, pid]
            cross = sum(kernel(x, y, pid, kind) for x in A for y in B) / (len(A)*len(B))
            if stat == "cross":
                vals.append(1-cross)
            else:
                aa = sum(kernel(x, y, pid, kind) for i, x in enumerate(A) for j, y in enumerate(A) if i != j) / (len(A)*(len(A)-1))
                bb = sum(kernel(x, y, pid, kind) for i, x in enumerate(B) for j, y in enumerate(B) if i != j) / (len(B)*(len(B)-1))
                vals.append(aa+bb-2*cross)
        out[a, b] = out[b, a] = sum(vals)/len(vals)
    return out


def linkage(D, threshold):
    clusters = [[e] for e in EPS]
    while len(clusters) > 1:
        choices = []
        for i, j in combinations(range(len(clusters)), 2):
            d = sum(D[a, b] for a in clusters[i] for b in clusters[j]) / (len(clusters[i])*len(clusters[j]))
            choices.append((d, i, j))
        d, i, j = min(choices)
        if d > threshold:
            break
        clusters[i] += clusters[j]
        del clusters[j]
    return {e: i for i, c in enumerate(clusters) for e in c}


def auc(D):
    same = [D[a, b] for a, b in PAIRS if TRUTH[a] == TRUTH[b]]
    diff = [D[a, b] for a, b in PAIRS if TRUTH[a] != TRUTH[b]]
    return sum((a < b) + .5*(a == b) for a in same for b in diff)/(len(same)*len(diff))


def main():
    raw = CORPUS.read_bytes()
    # Première tentative uniquement : pas de choix de la réponse la plus favorable.
    pool = defaultdict(list)
    for line in raw.decode().splitlines():
        r = json.loads(line)
        if r["campaign"] not in ("pilot1", "pilot2") or r["attempt"] != 0 or r["repeat"] not in (0, 1, 2, 3):
            continue
        pool[r["repeat"]//2, r["endpoint"], r["pid"]].append(r["response"])
    all_pids = sorted({p for _, _, p in pool})
    assert len(all_pids) == 44
    assert all(len(pool[h, e, p]) == 2 for h in (0, 1) for e in EPS for p in all_pids)
    res = {"corpus_sha256": hashlib.sha256(raw).hexdigest(), "samples": 44*9*4,
           "deduplication": "attempt=0 uniquement", "all_prompts": all_pids,
           "numeric_prompts": NUMERIC, "results": []}
    for pids, kind in ((all_pids, "raw_ngram"), (list(NUMERIC), "numeric_sequence")):
        for stat in ("cross", "mmd"):
            ds = {h: distance(pool, h, pids, kind, stat) for h in (0, 1)}
            for train, test in ((0, 1), (1, 0)):
                # Grille fixe de 201 seuils sur la plage de la moitié de sélection.
                lo, hi = min(ds[train].values())-1e-8, max(ds[train].values())+1e-8
                grid = [lo+(hi-lo)*k/200 for k in range(201)]
                threshold = max(grid, key=lambda t: mcc(linkage(ds[train], t)))
                pred = linkage(ds[test], threshold)
                false_splits = [[a, b] for a, b in PAIRS if TRUTH[a] == TRUTH[b] and pred[a] != pred[b]]
                confused = {e: sum(pred[e] == pred[b] for b in EPS if TRUTH[b] == "base") for e in EPS if TRUTH[e] != "base"}
                row = {"representation": kind, "statistic": stat, "prompts": len(pids),
                       "select_half": train, "eval_half": test, "threshold": threshold,
                       "mcc_select": mcc(linkage(ds[train], threshold)), "mcc_heldout": mcc(pred),
                       "auc_heldout": auc(ds[test]), "same_pairs_split": false_splits,
                       "variant_merges_with_base": confused, "partition": pred}
                res["results"].append(row)
                print(f'{kind:18} {stat:5} {train}->{test} train={row["mcc_select"]:.4f} heldout={row["mcc_heldout"]:.4f} AUC={row["auc_heldout"]:.4f} variant_merges={confused}')
    (HERE/"fingerprint_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False)+"\n")


if __name__ == "__main__":
    main()
