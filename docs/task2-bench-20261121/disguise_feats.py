"""Détecteurs de déguisement (0 requête), corpus figé pilot1/pilot2 (44 prompts x 4 rép.) + p3 (4 prompts x 12 rép.).
Moitiés : pilot rép. {0,1} | {2,3} ; p3 rép. 0-5 | 6-11.
  hf    : part des réponses mentionnant 'hugging face' ou 'smollm' (fuite du prompt système par défaut)
  dup   : part des paires (même prompt, même moitié) de réponses exactement identiques
  sim   : similarité moyenne (cosinus trigrammes de caractères) entre réponses d'un même prompt dans la moitié."""
import json, re, itertools, collections, math, sys
sys.path.insert(0, "/Users/selyan/dev/eaic-paris-2026/docs/task2-bench-20261121"); from anlib import *
W = "/Users/selyan/dev/eaic-paris-2026/docs/task2-work-20261121/corpus.jsonl"
def tri(s):
    s = s.lower(); return collections.Counter(s[i:i + 3] for i in range(len(s) - 2))
def cos(a, b):
    n = math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))
    return sum(a[k] * b[k] for k in set(a) & set(b)) / n if n else 0.0
def collect():
    R = collections.defaultdict(list)   # (ep, half, src, pid) -> responses
    for l in open(W):
        r = json.loads(l)
        if r["campaign"] in ("pilot1", "pilot2") and r["attempt"] == 0:
            R[(r["endpoint"], "A" if r["repeat"] in (0, 1) else "B", "pil", r["pid"])].append(r["response"])
    for r in load("/Users/selyan/dev/eaic-paris-2026/docs/task2-bench-20261121/data_bench_p3.jsonl"):
        if "tag" not in r or r.get("status") != 200: continue
        p, k = r["tag"].split("|")
        R[(r["payload"]["model"], "A" if int(k) < 6 else "B", "p3", p)].append(r["body"]["choices"][0]["message"]["content"])
    return R
HF = re.compile(r"hugging\s*face|smol\s*lm", re.I)
def feats(R):
    F = {}
    for e in EPS:
        for h in "AB":
            cells = {k: v for k, v in R.items() if k[0] == e and k[1] == h}
            allr = [x for v in cells.values() for x in v]
            pairs = [(a, b) for v in cells.values() for a, b in itertools.combinations(v, 2)]
            F[(e, h)] = dict(hf=sum(bool(HF.search(x)) for x in allr) / len(allr),
                             dup=sum(a == b for a, b in pairs) / len(pairs),
                             sim=sum(cos(tri(a), tri(b)) for a, b in pairs) / len(pairs), n=len(allr))
    return F
if __name__ == "__main__":
    F = feats(collect())
    for e in EPS:
        print(e, KIND[e].ljust(8), " | ".join("%s hf %.3f dup %.3f sim %.3f n %d" % (h, F[(e,h)]["hf"], F[(e,h)]["dup"], F[(e,h)]["sim"], F[(e,h)]["n"]) for h in "AB"))
    json.dump({f"{k[0]}|{k[1]}": v for k, v in F.items()}, open("/Users/selyan/dev/eaic-paris-2026/docs/task2-bench-20261121/disguise_feats.json", "w"), indent=1)
