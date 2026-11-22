"""Applique aux 33 endpoints la règle validée (len|isolate, tau=2,0 : meilleure en min des deux moitiés) à l'intérieur
de partitions de familles pré-déclarées ; écrit les CSV candidats et la comparaison paire par paire."""
import json, csv, itertools, collections
import t2lib
AUD = "../task2-audit-20261121/data/SLZ/model_equality/output/"
EPS = ["ep-%02d" % i for i in range(1, 34)]
def read_csv(p):
    rows = [r for r in csv.reader(open(p)) if r and r[0].startswith("ep-")]
    return {r[0]: (r[1].strip(), r[2].strip()) for r in rows}
def load_ep():
    d = collections.defaultdict(dict)
    for l in open("corpus.jsonl"):
        r = json.loads(l)
        if r["campaign"] == "ep1a": d[r["endpoint"]][r["pid"]] = r["response"]
    return d
MONTHS = "january february march april may june july august september october november december".split()
def family_rule(d):
    """Règle famille pré-déclarée, sans paramètre ajusté :
    L si f01 mentionne Meta/Llama ou f02 annonce 'December 2023' ; sinon Q.
    Q scindé en Qa si la 1re ligne de f02 commence par un nom de mois (réponse date d'abord), sinon Qb."""
    fam, why = {}, {}
    for e in EPS:
        f01, f02 = d[e]["f01"].lower(), d[e]["f02"].lower()
        meta = ("meta" in f01 or "llama" in f01)
        dec23 = "december 2023" in f02
        if meta or dec23: fam[e] = "L"
        else: fam[e] = "Qa" if f02.strip().split()[0].strip(".,") in MONTHS else "Qb"
        why[e] = {"meta_f01": meta, "dec2023_f02": dec23, "f02_first": f02.strip().split()[0]}
    return fam, why
def within(fam, d, tau):
    pred = {}
    for F in sorted(set(fam.values())):
        mem = [e for e in EPS if fam[e] == F]
        D, _ = t2lib.dist_matrix(d, mem, "len")
        lab = t2lib.rule_isolate(mem, D, tau)
        for e in mem: pred[e] = (F, F + "_" + lab[e])
    return pred
def write(path, pred):
    with open(path, "w", newline="") as f:
        f.write("endpoint,family,model\n")
        for e in EPS: f.write(f"{e},{pred[e][0]},{pred[e][1]}\n")
def same_model(p, a, b): return p[a] == p[b]
def same_fam(p, a, b): return p[a][0] == p[b][0]
def compare(P, Q):
    pairs = list(itertools.combinations(EPS, 2))
    return {"diff_model": sum(same_model(P, a, b) != same_model(Q, a, b) for a, b in pairs),
            "diff_family": sum(same_fam(P, a, b) != same_fam(Q, a, b) for a, b in pairs)}
def stats(P):
    return {"families": len({v[0] for v in P.values()}), "models": len(set(P.values())),
            "same_model_pairs": sum(same_model(P, a, b) for a, b in itertools.combinations(EPS, 2)),
            "same_family_pairs": sum(same_fam(P, a, b) for a, b in itertools.combinations(EPS, 2))}
def main():
    d = load_ep(); assert all(len(d[e]) == 5 for e in EPS)
    fam, why = family_rule(d)
    C = {"c2_lump_v1": read_csv(AUD + "c2_lump_v1.csv"), "ep_first_hamming": read_csv(AUD + "ep_first_hamming.csv"),
         "ep_first_ngram": read_csv(AUD + "ep_first_ngram.csv"), "baseline_random": read_csv(AUD + "baseline_random.csv"),
         "c2_fam3_pairs_v2": read_csv("raw/SLZ_model_equality_output_c2_fam3_pairs_v2.csv")}
    C["W_fam3_lump"] = {e: (fam[e], fam[e] + "_1") for e in EPS}
    C["W_fam3_len_tau2"] = within(fam, d, 2.0)
    C["W_fam3_singletons"] = {e: (fam[e], e) for e in EPS}
    for k in ("W_fam3_lump", "W_fam3_len_tau2", "W_fam3_singletons"): write(f"cand_{k}.csv", C[k])
    fam3 = C["c2_fam3_pairs_v2"]
    agree = sum(same_fam(C["W_fam3_lump"], a, b) == same_fam(fam3, a, b) for a, b in itertools.combinations(EPS, 2))
    out = {"family_rule": {e: [fam[e], why[e]] for e in EPS}, "family_rule_vs_fam3_pairs_v2_family_agreement_pairs": agree,
           "stats": {k: stats(v) for k, v in C.items()},
           "pairwise_diff": {f"{a} vs {b}": compare(C[a], C[b]) for a, b in itertools.combinations(sorted(C), 2)},
           "len_isolates": sorted(e for e in EPS if not C["W_fam3_len_tau2"][e][1].endswith("BIG")),
           "lump_vs_fam3_detail": {"same_model_in_lump_not_in_fam3": [], "same_model_in_fam3_not_in_lump": []}}
    for a, b in itertools.combinations(EPS, 2):
        x, y = same_model(C["c2_lump_v1"], a, b), same_model(fam3, a, b)
        if y and not x: out["lump_vs_fam3_detail"]["same_model_in_fam3_not_in_lump"].append(f"{a}-{b}")
    out["lump_vs_fam3_detail"]["same_model_in_lump_not_in_fam3"] = sum(
        same_model(C["c2_lump_v1"], a, b) and not same_model(fam3, a, b) for a, b in itertools.combinations(EPS, 2))
    json.dump(out, open("apply_results.json", "w"), indent=1, ensure_ascii=False)
    print("familles:", collections.Counter(fam.values()), "accord familles avec fam3_pairs_v2:", agree, "/528")
    print("isolés longueur:", out["len_isolates"])
    for k, v in out["stats"].items(): print(k, v)
    for k in ("c2_fam3_pairs_v2 vs c2_lump_v1", "W_fam3_len_tau2 vs c2_lump_v1", "W_fam3_len_tau2 vs c2_fam3_pairs_v2", "W_fam3_lump vs c2_fam3_pairs_v2"):
        print(k, out["pairwise_diff"].get(k) or out["pairwise_diff"].get(" vs ".join(k.split(" vs ")[::-1])))
    print("fam3 same, lump different:", out["lump_vs_fam3_detail"])
if __name__ == "__main__": main()
