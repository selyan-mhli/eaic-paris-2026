"""Analyse pré-enregistrée de p3 (listes). Usage : python3 analyze_lists.py bench_p3.jsonl"""
import sys, re, math, collections, itertools, json
from anlib import *

def items(txt):
    # amendement 1 (format vu sur 13 réponses, avant toute analyse) : préambule jusqu'au premier ':' retiré
    # s'il est dans les 120 premiers caractères ; numérotation "N." / "N)" en tête d'item retirée.
    txt = txt or ""
    k = txt.find(":")
    if 0 <= k < 120: txt = txt[k + 1:]
    out = []
    for t in re.split(r"[,;\n]", txt):
        t = re.sub(r"^[^A-Za-z0-9]*\d+\s*[.)]\s+", "", t)
        t = "".join(re.findall(r"[a-z0-9]+", t.lower()))
        if t: out.append(t)
    return out[:15]

def hell(c1, c2):
    n1, n2 = sum(c1.values()), sum(c2.values())
    if not n1 or not n2: return 0.0
    return sum(math.sqrt(c1[k] / n1 * c2[k] / n2) for k in set(c1) & set(c2))

def spear(c1, c2):
    keys = sorted(set(c1) | set(c2))
    if len(keys) < 3: return 0.0
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v); i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]: j += 1
            for k in range(i, j + 1): r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    a, b = rank([c1[k] for k in keys]), rank([c2[k] for k in keys])
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else 0.0

def build(rows, reps, stat, prompts):
    """reps : liste de répétitions de la moitié ; split en deux sous-moitiés pour normaliser."""
    h = len(reps) // 2; R1, R2 = set(reps[:h]), set(reps[h:])
    C = collections.defaultdict(collections.Counter)
    for r in rows:
        pid, rep = r["tag"].split("|"); rep = int(rep)
        if r.get("status") != 200 or rep not in reps or pid not in prompts: continue
        sub = 1 if rep in R1 else 2
        ep = r["payload"]["model"]
        C[(ep, pid, sub)].update(items(text_of(r)))
    D = {}
    for a, b in itertools.combinations(EPS, 2):
        rs = []
        for p in prompts:
            w_a = stat(C[(a, p, 1)], C[(a, p, 2)]); w_b = stat(C[(b, p, 1)], C[(b, p, 2)])
            bt = (stat(C[(a, p, 1)], C[(b, p, 2)]) + stat(C[(a, p, 2)], C[(b, p, 1)])) / 2
            den = math.sqrt(max(w_a, 1e-6) * max(w_b, 1e-6))
            rs.append(bt / den)
        D[(a, b)] = 1 - sum(rs) / len(rs)
    return D

def text_of(r):
    try: return r["body"]["choices"][0]["message"]["content"]
    except Exception: return None

if __name__ == "__main__":
    rows = load(sys.argv[1])
    nrep = 1 + max(int(r["tag"].split("|")[1]) for r in rows if "tag" in r)
    rows = [r for r in rows if "tag" in r]
    prompts = sorted({r["tag"].split("|")[0] for r in rows})
    hA, hB = list(range(nrep // 2)), list(range(nrep // 2, nrep))
    print("lignes", len(rows), "réps", nrep, "prompts", prompts,
          "non200", sum(r.get("status") != 200 for r in rows))
    res = {}
    for sname, stat in [("hellinger", hell), ("spearman", spear)]:
        DA, DB = build(rows, hA, stat, prompts), build(rows, hB, stat, prompts)
        thrs = sorted(set(round(v, 4) + 1e-4 for v in list(DA.values()) + list(DB.values())))
        for nm, (Ds, De) in [("A->B", (DA, DB)), ("B->A", (DB, DA))]:
            t, ms, me, pev = choose_and_eval(Ds, De, thrs)
            res[f"{sname} {nm}"] = dict(thr=round(t, 4), mcc_sel=round(ms, 3), mcc_eval=round(me, 3),
                                        partition=pev, errors=pair_errors(pev))
            print(sname, nm, "thr %.3f  MCC sel %.3f  HORS SEL %.3f" % (t, ms, me), pair_errors(pev))
        # matrice de distance (moitié A) pour lecture
        print("  dist A, base=ref-01 :", {b: round(DA[tuple(sorted(("ref-01", b)))], 3) for b in EPS if b != "ref-01"})
    json.dump(res, open(sys.argv[1].replace(".jsonl", "_analysis.json").split("/")[-1], "w"), indent=1)
