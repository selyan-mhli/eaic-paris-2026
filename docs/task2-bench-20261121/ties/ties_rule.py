"""T2 sondes à quasi-égalité : choix des sondes (pick) et règle FIGÉE (analyze).
pick    : lit ties_<tag>.json, choisit N_TIE sondes égalité + N_FLIP sondes fp8, recalcule le top-50 des tokens
          (chaîne.strip(), log-proba T=1, contexte par défaut, bf16) et écrit probes_<tag>.json.
analyze : lit le journal JSONL des refs et applique la règle ci-dessous, sur toutes les répétitions puis sur
          chaque moitié (rep pair / impair), indépendamment. MCC modèle par paires contre practice_answers.

RÈGLE FIGÉE (écrite avant toute requête, 22/11 ~02:05 Paris) :
  Pour chaque endpoint e, chaque sonde j : comptes n_jc sur c ∈ {s1, s2, autre} (texte.strip()).
  Prédiction de la base : q_jc(T) = Σ_{tokens de chaîne c} p_t^(1/T) / Σ_t p_t^(1/T) (top-50), plancher 1e-3.
  G_e = min_{T ∈ [0.4, 1.6] pas 0.05} Σ_j 2 Σ_c n_jc ln(n_jc / (N_j q_jc(T))).
  e est « base » si G_e <= quantile 0.999 du χ² à (2J − 1) ddl ; les endpoints « base » forment UN groupe,
  chaque autre endpoint est isolé. Seuil fixé a priori, aucun paramètre ajusté sur les refs.
  Critère d'action : MCC >= 0.60 sur CHAQUE moitié, déguisements (03, 05, 06, 08) tous dans le groupe base.
"""
import argparse, json, math, os, sys, itertools, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
N_TIE, N_FLIP = 8, 4
TGRID = [round(0.4 + 0.05 * i, 2) for i in range(25)]


def chi2_ppf999(df):  # Wilson–Hilferty
    z = 3.090232
    return df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3


def pick(a):
    import torch
    from ties_search import Model
    R = json.load(open(a.search))
    ties = R["ties_robust"]
    seen, sel = set(), []
    for r in ties:  # diversité : une seule sonde par paire (s1, s2) et par prompt de base
        pt = r["prompt"] if isinstance(r["prompt"], str) else json.dumps(r["prompt"], ensure_ascii=False)
        key = (frozenset((r["s1"], r["s2"])), pt.split(": ", 1)[-1][-40:])
        if key in seen or frozenset((r["s1"], r["s2"])) in {k[0] for k in seen}:
            continue
        seen.add(key); sel.append(dict(r, kind="tie"))
        if len(sel) == N_TIE:
            break
    nf = 0
    for r in R["flips"]:
        if json.dumps(r["prompt"]) in {json.dumps(s["prompt"]) for s in sel}:
            continue
        sel.append(dict(r, kind="flip")); nf += 1
        if nf == N_FLIP:
            break
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    M = Model(a.model, dev)
    lg = M.last_logits([M.render(s["prompt"], None) for s in sel])
    out = []
    for s, l in zip(sel, lg):
        lp = torch.log_softmax(l, -1); v, ix = lp.topk(50)
        toks = [[M.tok.decode([i]).strip(), float(x)] for x, i in zip(v.tolist(), ix.tolist())]
        out.append({"prompt": s["prompt"], "kind": s["kind"], "s1": s["s1"], "s2": s["s2"], "d0": s["d0"],
                    "mass": s["mass"], "d_ctx": s["d_ctx"], "d_fp8": s["d_fp8"], "arg_fp8": s["arg_fp8"], "toks": toks})
    json.dump({"model": a.model, "rule": "voir ties_rule.py", "probes": out}, open(a.dst, "w"), ensure_ascii=False, indent=1)
    for p in out:
        print(f"{p['kind']:4s} d0={p['d0']:+.2f} m={p['mass']:.2f} {p['s1']!r}/{p['s2']!r} fp8={ {k: round(v - p['d0'], 2) for k, v in p['d_fp8'].items()} } ctx={[round(x, 1) for x in p['d_ctx']]} | {p['prompt']}")


def qvec(p, T):
    w = collections.defaultdict(float)
    tot = 0.0
    for s, lp in p["toks"]:
        x = math.exp(lp / T); tot += x
        w[s if s in (p["s1"], p["s2"]) else "\0other"] += x
    q = np.array([w[p["s1"]], w[p["s2"]], w["\0other"]]) / tot
    q = np.maximum(q, 1e-3)
    return q / q.sum()


def gstat(counts, probes):
    best = (1e18, None)
    for T in TGRID:
        G = 0.0
        for j, n in counts.items():
            q = qvec(probes[j], T); N = n.sum()
            G += 2 * sum(nc * math.log(nc / (N * qc)) for nc, qc in zip(n, q) if nc > 0)
        best = min(best, (G, T))
    return best


def analyze(a):
    from anlib import TRUTH, KIND, mcc, pair_errors
    P = json.load(open(a.probes))["probes"]
    recs = [json.loads(l) for l in open(a.log)]
    recs = [r for r in recs if r.get("status") == 200 and "text" in r]
    print(f"réponses 200 : {len(recs)}")
    rep = {}
    for half in ["all", 0, 1]:
        C = collections.defaultdict(lambda: collections.defaultdict(lambda: np.zeros(3)))
        for r in recs:
            if half != "all" and r["rep"] % 2 != half:
                continue
            p = P[r["probe"]]; t = (r["text"] or "").strip()
            c = 0 if t == p["s1"] else (1 if t == p["s2"] else 2)
            C[r["ep"]][r["probe"]][c] += 1
        J = len(P); thr = chi2_ppf999(2 * J - 1)
        pred, det = {}, {}
        for ep in sorted(TRUTH):
            G, T = gstat(C[ep], P)
            pred[ep] = "BASE" if G <= thr else ep
            det[ep] = {"kind": KIND[ep], "G": round(G, 1), "T": T, "base": G <= thr}
        m = mcc(TRUTH, pred)
        rep[str(half)] = {"mcc": round(m, 3), "thr": round(thr, 1), "detail": det, "errors": pair_errors(pred)}
        print(f"\n== moitié {half} : MCC {m:.3f} (seuil G {thr:.1f})")
        for ep, d in det.items():
            print(f"  {ep} {d['kind']:8s} G={d['G']:7.1f} T*={d['T']:.2f} {'BASE' if d['base'] else 'isolé'}")
        # proportions observées vs prédites (T=0.7 et T*) pour lecture
        if half == "all":
            print("  p̂(s1) par sonde [ref-01..09] | q(s1) à T=0.7 :")
            for j, p in enumerate(P):
                row = []
                for ep in sorted(TRUTH):
                    n = C[ep][j]; row.append(f"{n[0]/max(n.sum(),1):.2f}/{int(n[2])}")
                print(f"   {j:2d} {p['kind']:4s} q={qvec(p, 0.7)[0]:.2f} " + " ".join(row) + f"  {p['prompt'][:50]}")
    json.dump(rep, open(a.dst, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("pick"); s.add_argument("--search"); s.add_argument("--model"); s.add_argument("--dst")
    s = sub.add_parser("analyze"); s.add_argument("--probes"); s.add_argument("--log"); s.add_argument("--dst")
    a = ap.parse_args()
    pick(a) if a.cmd == "pick" else analyze(a)
