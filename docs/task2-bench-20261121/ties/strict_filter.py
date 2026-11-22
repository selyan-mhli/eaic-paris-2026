"""Filtre strict GO/NO-GO (figé à 02:03) : masse top-2 >= 0,8 dans tous les contextes, décalage contexte < 0,3 nat, |fp8| >= 0,25 nat."""
import json, sys
src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(src)); rows = d["all"]
dev = lambda r: max(abs(x - r["d0"]) for x in r["d_ctx"])
f8 = lambda r: max(abs(v - r["d0"]) for v in r["d_fp8"].values())
strict = sorted([r for r in rows if r["mass"] >= 0.8 and min(r["mass_ctx"]) >= 0.8 and dev(r) < 0.3 and f8(r) >= 0.25], key=lambda r: -f8(r))
for r in strict + d["flips"]:
    r.setdefault("arg_fp8", None)
d["ties_robust"] = strict
json.dump(d, open(dst, "w"))
print(src, len(strict))
