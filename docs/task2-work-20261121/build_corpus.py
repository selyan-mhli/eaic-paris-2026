"""Corpus figé Task 2 : toutes les réponses, sans dédoublonnage, avec campagne et tentative explicites.
Usage : python3 build_corpus.py   (lecture seule des sources, écrit corpus.jsonl + corpus_manifest.json)"""
import hashlib, json, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent
AUD = HERE.parent / "task2-audit-20261121" / "data"
SOURCES = [  # (campagne par défaut, chemin)
    ("c2run", HERE / "raw" / "c2run_ref_answers.jsonl"),          # snapshot JURECA 23:08, en cours d'écriture par q3
    ("sophie_ref25", AUD / "SLZ/model_equality/reference_answers.jsonl"),
    ("sophie_minimal", AUD / "SLZ/model_equality/output/ref_pilot_minimal.jsonl"),
    ("ep1a", AUD / "c2run/ep/ep_answers.jsonl"),
]
def sha(b): return hashlib.sha256(b).hexdigest()
def campaign(default, pid):
    if default != "c2run": return default
    return {"f": "pilot1", "m": "pilot1", "c": "pilot1", "w": "pilot2", "e": "pilot3"}[pid[0]]
def main():
    out, man = [], {"sources": [], "notes": []}
    for default, p in SOURCES:
        raw = p.read_bytes()
        man["sources"].append({"path": str(p.relative_to(HERE.parent)), "sha256": sha(raw), "lines": raw.count(b"\n")})
        seen = collections.Counter()
        for i, line in enumerate(raw.decode("utf-8").splitlines()):
            if not line.strip(): continue
            try: r = json.loads(line)
            except json.JSONDecodeError:
                man["notes"].append(f"{p.name}:{i+1} ligne JSON invalide (écriture en cours ?), ignorée"); continue
            key = (r["endpoint"], r["id"], r["repeat"])
            att = seen[key]; seen[key] += 1
            out.append({"campaign": campaign(default, r["id"]), "source": p.name, "line": i + 1,
                        "endpoint": r["endpoint"], "pid": r["id"], "prompt_sha": sha(r["prompt"].encode())[:12],
                        "repeat": r["repeat"], "attempt": att, "response": r["response"],
                        "resp_sha": sha(r["response"].encode())[:12]})
    # doublons : même clé, plusieurs tentatives ; texte identique ou non
    by = collections.defaultdict(list)
    for r in out: by[(r["source"], r["endpoint"], r["pid"], r["repeat"])].append(r["resp_sha"])
    multi = {k: v for k, v in by.items() if len(v) > 1}
    man["duplicate_keys"] = len(multi)
    man["duplicate_keys_conflicting_text"] = sum(len(set(v)) > 1 for v in multi.values())
    # même pid, prompts différents entre campagnes ?
    ps = collections.defaultdict(set)
    for r in out: ps[r["pid"]].add(r["prompt_sha"])
    man["pid_with_several_prompts"] = sorted(k for k, v in ps.items() if len(v) > 1)
    man["counts"] = collections.Counter(f'{r["campaign"]}' for r in out)
    blob = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in out).encode()
    (HERE / "corpus.jsonl").write_bytes(blob)
    man["corpus_sha256"] = sha(blob); man["corpus_rows"] = len(out)
    (HERE / "corpus_manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))
    print(json.dumps({k: man[k] for k in ("corpus_sha256", "corpus_rows", "counts", "duplicate_keys", "duplicate_keys_conflicting_text", "pid_with_several_prompts")}, ensure_ascii=False))
    print(man["notes"][:5])
if __name__ == "__main__": main()
