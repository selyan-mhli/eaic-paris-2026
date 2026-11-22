"""Simulation de calibration (aucune requête API).
sim : pour une base, génère 9 endpoints simulés x 2 réplicats x N réponses (prompts B1-B3 en rotation, 256 tokens) :
  plain_a, plain_b (T 0,7), temp1 (T 1,0), sys (prompt système), persona  -> même modèle "S"
  fp8 (poids fp8 par canal, différent de l'alternative de score), lora_lo/mid/hi (SUBSTITUT : LoRA aléatoire rang 8
  q/v, ||BA||/||W|| = 0,01/0,03/0,1) -> chacun distinct.  Watermark NON simulé.
  Score de chaque réponse sous la base (gabarit par défaut, sans prompt système) + alternative fp8 par tenseur W+A.
fit : règle part() de wb2_lib ; seuils (l8, glo, ghi) choisis sur le réplicat 0, évalués sur le 1, et inversement
  (MCC hors sélection + bootstrap de N réponses) ; RÈGLE FIGÉE = seuils maximisant le MCC moyen sur 300 bootstraps
  de N réponses tirées des 2 réplicats ; écrite dans rule.json, par base."""
import argparse, json, os, sys, time, random, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wb2_lib import *
SIM = [("plain_a", dict(T=0.7)), ("plain_b", dict(T=0.7)), ("temp1", dict(T=1.0)), ("sys", dict(T=0.7, system=SYS_PROMPT)),
       ("persona", dict(T=0.7, system=PERSONA)), ("fp8", dict(T=0.7, fp8=True)),
       ("lora_lo", dict(T=0.7, lora=0.01)), ("lora_mid", dict(T=0.7, lora=0.03)), ("lora_hi", dict(T=0.7, lora=0.1))]
TRUTH = {k: ("S" if k in ("plain_a", "plain_b", "temp1", "sys", "persona") else k) for k, _ in SIM}

def sim(a):
    name = os.path.basename(a.base.rstrip("/")); t0 = time.time()
    tok, m, dev = load(a.base); snap = snapshot(m)
    pids = [p for _ in range((a.n + 2) // 3) for p in PROMPTS][:a.n]
    gen = {}
    for j, (k, c) in enumerate(SIM):
        if c.get("fp8"): apply_fp8_gen(m)
        if c.get("lora"): apply_random_lora(m, c["lora"], seed=100 + j)
        for rep in range(2):
            gen[(k, rep)] = generate(tok, m, dev, name, pids, c["T"], c.get("system"), seed=1000 * j + rep, max_new=a.max_new)
        restore(m, snap)
        print(f"[{name}] gen {k} {time.time() - t0:.0f}s", flush=True)
    recs = {}
    enc = {key: [(p, prompt_ids(tok, PROMPTS[p], name), resp_ids(tok, t)) for p, t in zip(pids, txt)] for key, txt in gen.items()}
    for key, L in enc.items():
        recs[key] = [dict(pid=p, n=st[2], lp=st[0], H=st[1]) for p, ip, ir in L if ir.shape[1] for st in [seq_stats(m, ip, ir, dev)]]
    apply_fp8_score(m)
    for key, L in enc.items():
        for r, (p, ip, ir) in zip(recs[key], [x for x in L if x[2].shape[1]]):
            r["lpq"] = seq_stats(m, ip, ir, dev)[0]
    print(f"[{name}] score {time.time() - t0:.0f}s", flush=True)
    json.dump(dict(base=name, n=a.n, recs={f"{k}|{rep}": v for (k, rep), v in recs.items()},
                   sample={f"{k}|0": gen[(k, 0)][0][:300] for k, _ in SIM}), open(os.path.join(a.tmp, f"sim_{name}.json"), "w"))

def cands(F):
    def mids(v):
        q = list(np.quantile(sorted(set(v)), np.linspace(0, 1, 12))); return [-9.0] + [(x + y) / 2 for x, y in zip(q, q[1:])] + [9.0]
    return mids([f["LLR8"] for f in F]), mids([f["GAP"] for f in F])

def best_rule(Flist, weights=None):
    """Flist : liste de dicts {ep: features}. Maximise le MCC moyen."""
    allf = [f for F in Flist for f in F.values()]; c8, cg = cands(allf)
    best = None
    for l8 in c8:
        for glo, ghi in itertools.combinations(cg, 2):
            s = np.mean([mcc(TRUTH, part(F, l8, glo, ghi)) for F in Flist])
            if best is None or s > best[0]: best = (s, (l8, glo, ghi))
    return best

def boot(recs_by_ep, n, B, rng):
    out = []
    for _ in range(B):
        out.append({e: features([rs[rng.randrange(len(rs))] for _ in range(n)]) for e, rs in recs_by_ep.items()})
    return out

def fit(a):
    rng = random.Random(0); rule = {}; report = {}
    for f in sorted(os.listdir(a.tmp)):
        if not f.startswith("sim_"): continue
        S = json.load(open(os.path.join(a.tmp, f))); name = S["base"]; n = S["n"]
        R = {rep: {k: S["recs"][f"{k}|{rep}"] for k, _ in SIM} for rep in (0, 1)}
        F = {rep: {k: features(v) for k, v in R[rep].items()} for rep in (0, 1)}
        rep_out = {}
        for s, t in ((0, 1), (1, 0)):
            sc, th = best_rule([F[s]] + boot(R[s], n, 40, rng))
            bt = [mcc(TRUTH, part(Fb, *th)) for Fb in boot(R[t], n, 200, rng)]
            rep_out[f"{s}->{t}"] = dict(thr=th, mcc_eval=mcc(TRUTH, part(F[t], *th)), boot_mean=float(np.mean(bt)),
                                        boot_p10=float(np.percentile(bt, 10)), part=part(F[t], *th))
        pooled = {k: R[0][k] + R[1][k] for k, _ in SIM}
        sc, th = best_rule(boot(pooled, n, 120, rng))
        rule[name] = dict(l8=th[0], glo=th[1], ghi=th[2], n=n, fit_mean_mcc=float(sc))
        report[name] = dict(features=F, cv=rep_out)
        print(f"== {name}")
        for k, _ in SIM: print("  %-9s" % k, " | ".join("r%d T %.1f LL %+.3f LLR8 %+6.1f GAP %+.3f" % (rep, F[rep][k]["T"], F[rep][k]["LL"], F[rep][k]["LLR8"] * 1e3, F[rep][k]["GAP"]) for rep in (0, 1)))
        for d, v in rep_out.items(): print(f"  {d} seuils {tuple(round(x, 4) for x in v['thr'])} MCC hors sél. {v['mcc_eval']:.3f} bootstrap({n}) moy {v['boot_mean']:.3f} p10 {v['boot_p10']:.3f}")
        print(f"  RÈGLE FIGÉE l8 {th[0]:.4f} GAP hors [{th[1]:.4f}, {th[2]:.4f}] -> isolé ; MCC moyen bootstrap {sc:.3f}")
    json.dump(rule, open(a.rule, "w"), indent=1); json.dump(report, open(a.report, "w"), indent=1, default=str)
    print("écrit", a.rule, a.report)

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("cmd"); p.add_argument("--base"); p.add_argument("--tmp")
    p.add_argument("--n", type=int, default=30); p.add_argument("--max_new", type=int, default=MAX_TOKENS)
    p.add_argument("--rule"); p.add_argument("--report"); a = p.parse_args(); os.makedirs(a.tmp, exist_ok=True)
    dict(sim=sim, fit=fit)[a.cmd](a)
