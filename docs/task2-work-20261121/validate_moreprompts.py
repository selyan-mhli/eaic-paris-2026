"""Effet du nombre de prompts sur la règle longueur+isolement (refs, pilot1, une réponse par prompt, moitiés H0/H1).
K prompts tirés au hasard (seedé) parmi les 32 de pilot1 ; tau choisi sur une moitié, évalué sur l'autre.
Sert à chiffrer le gain attendu d'un lot ep à K prompts × 1 réponse. Les prompts ne sont PAS sélectionnés sur leurs scores."""
import json, random, statistics, itertools, collections
import t2lib, validate
pool = collections.defaultdict(list); pids = set()
for l in open("corpus.jsonl"):
    r = json.loads(l)
    if r["endpoint"] in validate.TRUTH and r["campaign"] == "pilot1":
        pool[("H0" if r["repeat"] < 2 else "H1", r["endpoint"], r["pid"])].append(r["response"]); pids.add(r["pid"])
pids = sorted(p for p in pids if all(pool[h, e, p] for h in ("H0", "H1") for e in validate.REFS))
res = {}
for K in (5, 10, 15, 25, len(pids)):
    out = []
    for rep in range(8):
        rng = random.Random(100 + rep); P = rng.sample(pids, K); t2lib.PIDS[:] = P
        dr = {h: [{e: {p: rng.choice(pool[h, e, p]) for p in P} for e in validate.REFS} for _ in range(25)] for h in ("H0", "H1")}
        sc = {(h, tau): statistics.mean(t2lib.pair_mcc(validate.TRUTH, pr)[0] for pr in validate.run(dr[h], "len", "isolate", tau))
              for h in dr for tau in (1.0, 1.5, 2.0, 2.5, 3.0)}
        for s, ev in (("H0", "H1"), ("H1", "H0")):
            tau = max((1.0, 1.5, 2.0, 2.5, 3.0), key=lambda t: sc[s, t]); out.append(sc[ev, tau])
    res[K] = {"heldout_mcc_mean": round(statistics.mean(out), 3), "sd": round(statistics.pstdev(out), 3), "n": len(out)}
    print(K, res[K], flush=True)
json.dump({"n_prompts_available": len(pids), "results": res}, open("validation_moreprompts.json", "w"), indent=1)
