"""Validation croisée par moitiés de répétitions (H0 = repeat 0-1, H1 = repeat 2-3, tentatives incluses).
Conditions identiques aux endpoints notés : une réponse par prompt f01..f05 par endpoint (tirages aléatoires seedés).
Tous les paramètres (feature, règle, tau/q) sont choisis sur une moitié et évalués sur l'autre."""
import json, random, itertools, collections, statistics
import t2lib
TRUTH = {"ref-01":"S","ref-03":"S","ref-05":"S","ref-06":"S","ref-08":"S","ref-02":"T1","ref-04":"T3","ref-07":"T4","ref-09":"T2"}
KIND = {"ref-02":"fp8","ref-04":"lora07","ref-07":"lora15","ref-09":"lora03"}
REFS = sorted(TRUTH)
def ptype(a, b):
    ka, kb = KIND.get(a, "base"), KIND.get(b, "base")
    if ka == kb == "base": return "deguisement-deguisement (même)"
    if "base" in (ka, kb): return "base-" + (kb if ka == "base" else ka)
    return "variante-variante"
FEATS = ["len", "ngram", "prefix", "len+ngram"]
GRID = [("isolate", t) for t in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)] + [("linkage", q) for q in (0.05, 0.1, 0.2, 0.3, 0.4)]
def load():
    pool = collections.defaultdict(list)
    for l in open("corpus.jsonl"):
        r = json.loads(l)
        if r["endpoint"] in TRUTH and r["pid"] in t2lib.PIDS and r["campaign"] == "pilot1":
            pool[("H0" if r["repeat"] < 2 else "H1", r["endpoint"], r["pid"])].append(r["response"])
    return pool
def draws(pool, half, n, seed):
    rng = random.Random(seed)
    return [{e: {p: rng.choice(pool[half, e, p]) for p in t2lib.PIDS} for e in REFS} for _ in range(n)]
def run(dr, feat, rule, par):
    out = []
    for d in dr:
        D, _ = t2lib.dist_matrix(d, REFS, feat)
        pred = t2lib.rule_isolate(REFS, D, par) if rule == "isolate" else t2lib.rule_linkage(REFS, D, par)
        out.append(pred)
    return out
def summarize(preds):
    ms = [t2lib.pair_mcc(TRUTH, p)[0] for p in preds]
    err = collections.Counter(); tot = collections.Counter()
    for p in preds:
        for a, b in itertools.combinations(REFS, 2):
            t = ptype(a, b); tot[t] += 1
            err[t] += (TRUTH[a] == TRUTH[b]) != (p[a] == p[b])
    return {"mcc_mean": statistics.mean(ms), "mcc_sd": statistics.pstdev(ms), "mcc_p10": sorted(ms)[len(ms)//10],
            "frac_mcc_zero": sum(m == 0 for m in ms) / len(ms),
            "err_rate_by_type": {t: round(err[t] / tot[t], 3) for t in sorted(tot)}}
def auc_feature(dr, feat):
    """AUC même/différent sur les 36 paires, et couverture (prompts communs) par feature."""
    pos, neg, covs = [], [], []
    for d in dr:
        D, cov = t2lib.dist_matrix(d, REFS, feat)
        for a, b in itertools.combinations(REFS, 2):
            (pos if TRUTH[a] == TRUTH[b] else neg).append(D[a, b]); covs.append(cov[a, b])
    wins = sum((p < n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg)), sum(c == 5 for c in covs) / len(covs)
def main():
    pool = load(); N = 60
    dr = {h: draws(pool, h, N, seed=k) for k, h in enumerate(["H0", "H1"])}
    res = {"N_draws_per_half": N, "feature_auc_coverage": {}, "grid": {}, "cross": []}
    for f in FEATS:
        res["feature_auc_coverage"][f] = {h: dict(zip(("auc", "pairs_full_coverage"), map(lambda x: round(x, 3), auc_feature(dr[h], f)))) for h in dr}
    for f in FEATS:
        for rule, par in GRID:
            res["grid"][f"{f}|{rule}|{par}"] = {h: summarize(run(dr[h], f, rule, par)) for h in dr}
    for sel, ev in (("H0", "H1"), ("H1", "H0")):
        best = max(res["grid"], key=lambda k: res["grid"][k][sel]["mcc_mean"])
        res["cross"].append({"select_on": sel, "eval_on": ev, "chosen": best,
                             "mcc_select": round(res["grid"][best][sel]["mcc_mean"], 3),
                             "eval": res["grid"][best][ev]})
    # règle fixée a priori (audit) : longueur + isolement, sans sélection
    json.dump(res, open("validation_results.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res["feature_auc_coverage"]))
    for c in res["cross"]: print(json.dumps(c, ensure_ascii=False))
    top = sorted(res["grid"], key=lambda k: -min(res["grid"][k]["H0"]["mcc_mean"], res["grid"][k]["H1"]["mcc_mean"]))[:6]
    for k in top: print(k, {h: (round(res["grid"][k][h]["mcc_mean"], 3), round(res["grid"][k][h]["frac_mcc_zero"], 2)) for h in ("H0", "H1")})
if __name__ == "__main__": main()
