"""Analyse de p4, règle fixée en tête de p4_disguise_battery.py, appliquée SANS modification.
Usage : python3 analyze_p4.py bench_p4.jsonl [--no-ident]   (--no-ident : critère famille L, feature ident retirée)"""
import sys, re, itertools, collections, json
sys.path.insert(0, "/Users/selyan/dev/eaic-paris-2026/docs/task2-bench-20261121"); from anlib import *
NOID = "--no-ident" in sys.argv
DIS = {"ref-03", "ref-05", "ref-06", "ref-08"}; ID = re.compile(r"hugging\s*face|smol\s*lm", re.I)
R = collections.defaultdict(list)
for r in load(sys.argv[1]):
    if "tag" not in r or r.get("status") != 200: continue
    pid, b, k = r["tag"].split("|"); R[(r["payload"]["model"], b, pid)].append(r["body"]["choices"][0]["message"]["content"])
BATS = sorted({k[1] for k in R})
def feats(e, b):
    I = R[(e, b, "I1")] + R[(e, b, "I2")]; Wr = R[(e, b, "W")]; T = [x.strip().lower() for x in R[(e, b, "T")]]
    pr = list(itertools.combinations(Wr, 2))
    return (sum(bool(ID.search(x)) for x in I) / max(len(I), 1), sum(a == c for a, c in pr) / max(len(pr), 1), len(set(T)))
FE = {(e, b): feats(e, b) for e in EPS for b in BATS}
for e in EPS: print(e, KIND[e].ljust(8), " ".join("%s:(%.2f,%.2f,%d)" % ((b,) + FE[(e, b)]) for b in BATS))
def mids(i, b, hi):
    v = sorted({FE[(e, b)][i] for e in EPS}); return [((x + y) / 2, y - x) for x, y in zip(v, v[1:])] + [(None, 0.0)]
def dg(b, a, d, c):
    return {e for e in EPS if (a is not None and not NOID and FE[(e, b)][0] <= a) or (d is not None and FE[(e, b)][1] >= d) or (c is not None and FE[(e, b)][2] >= c)}
ok = 0; res = []
for s in BATS:
    best = None
    A = [(None, 0.0)] if NOID else mids(0, s, False)
    for (a, ma), (d, md), (c, mc) in itertools.product(A, mids(1, s, True), mids(2, s, True)):
        D = dg(s, a, d, c); key = (sum((e in D) == (e in DIS) for e in EPS), min([m for x, m in ((a, ma), (d, md), (c, mc)) if x is not None] or [0]))
        if best is None or key > best[0]: best = (key, (a, d, c))
    for t in BATS:
        if t == s: continue
        Dt = dg(t, *best[1]); P = {e: ("G" if e in Dt else e) for e in EPS}; m = mcc(TRUTH, P); er = pair_errors(P)
        passed = m >= 0.60 and er.get("deguis-deguis", 0) <= 0.1; ok += passed
        res.append(dict(sel=s, ev=t, thr=best[1], mcc=m, errors=er, disguised=sorted(Dt), passed=passed))
        print(f"{s}->{t} seuils {best[1]} MCC {m:.3f} déguis-déguis {er.get('deguis-deguis')} {sorted(Dt)} {'OK' if passed else '--'}")
print(f"VERDICT {'(sans ident, critère famille L) ' if NOID else ''}: {ok}/{len(res)} directions passent ; seuil 10/12 -> {'PASSE' if ok >= 10 else 'ÉCHEC'}")
json.dump(res, open("/Users/selyan/dev/eaic-paris-2026/docs/task2-bench-20261121/p4_analysis%s.json" % ("_noident" if NOID else ""), "w"), indent=1, default=str)
