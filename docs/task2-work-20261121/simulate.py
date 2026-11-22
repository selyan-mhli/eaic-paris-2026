"""Espérance du score sous incertitude (Monte Carlo, seed fixe). Les vérités simulées viennent d'un a priori explicite,
PAS des endpoints : c'est une estimation conditionnelle à cet a priori, pas un score mesuré.
A priori : familles = H3 (Qa/Qb/L, règle famille) ou H2 (Q/L de c2_lump_v1), 1/2 chacune ;
dans chaque famille de taille n : un modèle 'base' de taille s ~ U{1..ceil(0.75 n)} (déguisements),
les autres endpoints répartis sur k ~ U{ceil(r/2)..r} variantes. Mode 'len' : les isolés de longueur ont
une proba 0,8 d'être hors base (précision approx. mesurée sur refs, validation_results.json) ; mode 'unif' : au hasard.
Paires publiques : 30 % des 528, tirées au hasard à chaque simulation (le vrai sous-ensemble est inconnu).
Conditionnement ABC optionnel sur les deux scores publics observés : baseline_random 0,20 et c2_lump_v1 0,05 (±0,03)."""
import random, itertools, json, math, statistics, collections
import t2lib
from apply_ep import read_csv, EPS, AUD
C = {k: read_csv(p) for k, p in {
    "c2_lump_v1": AUD + "c2_lump_v1.csv", "baseline_random": AUD + "baseline_random.csv",
    "ep_first_hamming": AUD + "ep_first_hamming.csv", "ep_first_ngram": AUD + "ep_first_ngram.csv",
    "c2_fam3_pairs_v2": "raw/SLZ_model_equality_output_c2_fam3_pairs_v2.csv",
    "W_fam3_lump": "cand_W_fam3_lump.csv", "W_fam3_len_tau2": "cand_W_fam3_len_tau2.csv",
    "W_fam3_singletons": "cand_W_fam3_singletons.csv"}.items()}
ISO = {e for e in EPS if not C["W_fam3_len_tau2"][e][1].endswith("BIG")}
PAIRS = list(itertools.combinations(EPS, 2))
def truth(rng, mode):
    famsrc = C["W_fam3_lump"] if rng.random() < 0.5 else C["c2_lump_v1"]
    H = "H3" if famsrc is C["W_fam3_lump"] else "H2"
    fam = {e: famsrc[e][0] for e in EPS}; mod = {}
    for F in set(fam.values()):
        mem = [e for e in EPS if fam[e] == F]; n = len(mem)
        s = rng.randint(1, math.ceil(0.75 * n))
        if mode == "len":
            w = {e: (0.2 if e in ISO else 1.0) for e in mem}   # poids d'appartenance à la base
            base = set()
            pool = list(mem)
            while len(base) < s:
                tot = sum(w[e] for e in pool); x = rng.random() * tot
                for e in pool:
                    x -= w[e]
                    if x <= 0: base.add(e); pool.remove(e); break
        else: base = set(rng.sample(mem, s))
        rest = [e for e in mem if e not in base]; r = len(rest)
        k = rng.randint(math.ceil(r / 2), r) if r else 0
        for e in base: mod[e] = F + "_base"
        for e in rest: mod[e] = F + "_v%d" % rng.randrange(k)
    return H, fam, mod
def score(fam, mod, cand, pairs):
    P = C[cand]
    return t2lib.server_score(fam, mod, {e: P[e][0] for e in EPS}, {e: P[e][1] for e in EPS}, pairs)[0]
def run(mode, n_sim, seed):
    rng = random.Random(seed); rows = []
    for _ in range(n_sim):
        H, fam, mod = truth(rng, mode)
        pub = set(rng.sample(PAIRS, round(0.3 * len(PAIRS)))); priv = [p for p in PAIRS if p not in pub]; pub = list(pub)
        row = {"H": H}
        for c in C:
            row[c + "|pub"] = score(fam, mod, c, pub); row[c + "|priv"] = score(fam, mod, c, priv)
        rows.append(row)
    return rows
def summ(rows):
    out = {"n": len(rows), "P(H3)": round(sum(r["H"] == "H3" for r in rows) / max(1, len(rows)), 3)}
    for c in C:
        v = [r[c + "|priv"] for r in rows]
        out[c] = {"E_priv": round(statistics.mean(v), 3), "p10": round(sorted(v)[len(v) // 10], 3), "p90": round(sorted(v)[9 * len(v) // 10], 3)} if v else None
    return out
def main():
    res = {}
    for mode in ("unif", "len"):
        rows = run(mode, 20000, seed=7)
        acc = [r for r in rows if abs(r["baseline_random|pub"] - 0.20) <= 0.03 and abs(r["c2_lump_v1|pub"] - 0.05) <= 0.03]
        res[mode] = {"prior": summ(rows), "posterior_ABC": summ(acc)}
        # interprétation du futur score public de fam3_pairs_v2 : private attendu par tranche de public
        bins = collections.defaultdict(list)
        for r in (acc or rows):
            b = min(4, int(r["c2_fam3_pairs_v2|pub"] / 0.1)); bins[b].append(r)
        res[mode]["fam3_pairs_v2_public_bins"] = {f"[{b/10:.1f},{(b+1)/10:.1f})" if b < 4 else ">=0.4":
            {"n": len(v), "P(H3)": round(sum(x["H"] == "H3" for x in v) / len(v), 2),
             "E_priv_fam3": round(statistics.mean(x["c2_fam3_pairs_v2|priv"] for x in v), 3),
             "E_priv_W_len": round(statistics.mean(x["W_fam3_len_tau2|priv"] for x in v), 3)} for b, v in sorted(bins.items())}
    json.dump(res, open("simulation_results.json", "w"), indent=1)
    for mode in res:
        for k in ("prior", "posterior_ABC"):
            s = res[mode][k]; print(mode, k, "n=", s["n"], "P(H3)=", s["P(H3)"])
            for c in C: print("   ", c, s[c])
        print(mode, "bins fam3 public:", res[mode]["fam3_pairs_v2_public_bins"])
if __name__ == "__main__": main()
