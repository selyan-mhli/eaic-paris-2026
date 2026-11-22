"""Analyse post-collecte (GPU) : applique la RÈGLE FIGÉE (rule.json, issue de wb2_sim fit) aux réponses ep.
score --rank R --world W : pour les ep dont la base (result_15904307.json) est attribuée à ce rang, score des réponses
  200 OK de collect_ep.jsonl (1re réponse par clé pid|rep|ep) sous la base + alternative fp8 -> /tmp.
decide : features par ep, part() avec les seuils de SA base ; famille = base (lettre de c2_fam3_pairs_v2 majoritaire),
  modèle = <fam>_BASE / <fam>_FP8 / <ep> ; écrit le CSV candidat + sha256 + features."""
import argparse, json, os, sys, csv, hashlib, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wb2_lib import *
def responses(path):
    seen = {}
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        if r.get("status") != 200 or "tag" not in r: continue
        if r["tag"] in seen: continue
        pid, rep, ep = r["tag"].split("|"); seen[r["tag"]] = (ep, pid, r["body"]["choices"][0]["message"]["content"])
    return list(seen.values())
def score(a):
    R1 = json.load(open(a.ident)); bases = sorted(set(R1["base"].values())); mine = [b for i, b in enumerate(bases) if i % a.world == a.rank]
    rows = responses(a.collect); out = {}
    for nm in mine:
        tok, m, dev = load(R1["paths"][nm]); sel = [(ep, pid, t) for ep, pid, t in rows if R1["base"].get(ep) == nm]
        enc = [(ep, pid, prompt_ids(tok, PROMPTS[pid], nm), resp_ids(tok, t)) for ep, pid, t in sel]
        enc = [x for x in enc if x[3].shape[1]]
        recs = [dict(ep=ep, pid=pid, n=st[2], lp=st[0], H=st[1]) for ep, pid, ip, ir in enc for st in [seq_stats(m, ip, ir, dev)]]
        apply_fp8_score(m)
        for r, (ep, pid, ip, ir) in zip(recs, enc): r["lpq"] = seq_stats(m, ip, ir, dev)[0]
        out[nm] = recs; print(f"[rank {a.rank}] {nm} {len(recs)} réponses", flush=True)
        del m
    json.dump(out, open(os.path.join(a.tmp, f"a_{a.rank}.json"), "w"))
def decide(a):
    R1 = json.load(open(a.ident)); rule = json.load(open(a.rule)); fam3 = {r["endpoint"]: r["family"] for r in csv.DictReader(open(a.fam))}
    P = {}
    for f in os.listdir(a.tmp):
        if f.startswith("a_"): P.update(json.load(open(os.path.join(a.tmp, f))))
    rows, feats = [], {}
    for nm, recs in sorted(P.items()):
        eps = sorted({r["ep"] for r in recs}); letter = collections.Counter(fam3[e] for e in eps).most_common(1)[0][0]
        F = {e: features([r for r in recs if r["ep"] == e]) for e in eps}; th = rule[nm]
        pt = part(F, th["l8"], th["glo"], th["ghi"])
        for e in eps:
            lab = {"BASE": f"{letter}_BASE", "FP8": f"{letter}_FP8"}.get(pt[e], f"{letter}_{e}")
            rows.append((e, letter, lab)); feats[e] = dict(F[e], base=nm, decision=pt[e], n_resp=sum(r["ep"] == e for r in recs))
    rows.sort()
    with open(a.csv, "w", newline="") as h:
        w = csv.writer(h); w.writerow(["endpoint", "family", "model"]); w.writerows(rows)
    sha = hashlib.sha256(open(a.csv, "rb").read()).hexdigest()
    json.dump(dict(rule=rule, features=feats, csv=a.csv, sha256=sha), open(a.csv.replace(".csv", "_features.json"), "w"), indent=1)
    for e, f in sorted(feats.items()): print(e, f["base"][:22].ljust(22), "n %2d T %.1f LLR8 %+6.1f GAP %+.3f ->" % (f["n_resp"], f["T"], f["LLR8"] * 1e3, f["GAP"]), f["decision"])
    print("CSV", a.csv, "sha256", sha, "lignes", len(rows))
if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("cmd"); p.add_argument("--rank", type=int, default=0); p.add_argument("--world", type=int, default=1)
    p.add_argument("--ident"); p.add_argument("--collect"); p.add_argument("--tmp"); p.add_argument("--rule"); p.add_argument("--fam"); p.add_argument("--csv")
    a = p.parse_args(); os.makedirs(a.tmp, exist_ok=True); dict(score=score, decide=decide)[a.cmd](a)
