"""c2 — générateur de réponses SYNTHÉTIQUES au format du kit (aucune requête au serveur).

Sert uniquement à tester analyze.py : ce n'est pas un modèle des vrais LLM, et un score sur ces données ne
prédit pas le score réel.

Modèle jouet : pour chaque (famille, prompt), une réponse « gabarit » de L mots tirés du vocabulaire de la
famille. Un endpoint produit le gabarit avec
  - des écarts *systématiques* propres à ses poids (mêmes positions et mêmes mots à chaque tirage) :
    fp8 = peu d'écarts, LoRA = écarts emboîtés et croissants avec la force (0,3 ⊂ 0,7 ⊂ 1,5) ;
  - du bruit d'échantillonnage (positions et mots aléatoires à chaque tirage), plus fort à température 1,0 ;
  - des déguisements qui ne changent pas l'identité : préfixe/suffixe fixes (system prompt, persona),
    mots « verts » aléatoires (watermark).

    python c2/synth.py --scenario ref --prompts 20 --reps 4 --out /tmp/synth.jsonl --truth /tmp/truth.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import zlib
from dataclasses import dataclass

import numpy as np

SYLL = ["ka", "lo", "mi", "ne", "ru", "sa", "to", "vi", "ze", "po", "da", "fu", "gi", "ho", "ja", "be", "qu",
        "xi", "wo", "ye", "an", "el", "is", "or", "um"]

# taux d'écart systématique par variante de poids
WEIGHT_DRIFT = {"S": 0.0, "fp8": 0.04, "L0.3": 0.06, "L0.7": 0.14, "L1.5": 0.32}
BASE_NOISE = 0.10
DISGUISES = {
    "plain": {},
    "system prompt": {"prefix": "Certainly! Here is the answer you asked for:"},
    "persona": {"prefix": "Ahoy friend,", "suffix": "Fair winds!", "style": 0.03},
    "watermark": {"green": 0.06},
    "temperature": {"noise": 0.22},
}
# variantes de la référence (même structure que practice_answers.csv)
REF_LAYOUT = [("S", "plain"), ("fp8", "plain"), ("S", "system prompt"), ("L0.7", "plain"), ("S", "persona"),
              ("S", "watermark"), ("L1.5", "plain"), ("S", "temperature"), ("L0.3", "plain")]


@dataclass
class Endpoint:
    name: str
    family: str
    weights: str          # S, fp8, L0.3, L0.7, L1.5
    disguise: str         # clé de DISGUISES
    adapter: str = "a"    # identifiant du LoRA (deux LoRA différents = deux modèles)

    @property
    def model(self) -> str:
        return self.weights if self.weights in ("S", "fp8") else f"{self.weights}-{self.adapter}"


def _seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode())


def vocab(family: str, size: int = 400) -> list:
    rng = np.random.default_rng(_seed("vocab", family))
    return ["".join(rng.choice(SYLL, size=rng.integers(1, 4))) for _ in range(size)]


def template(family: str, pid: str, length: int) -> list:
    rng = np.random.default_rng(_seed("tpl", family, pid))
    v = vocab(family)
    return [v[i] for i in rng.integers(0, len(v), size=length)]


def sample(ep: Endpoint, pid: str, length: int, rng: np.random.Generator) -> str:
    v = vocab(ep.family)
    words = template(ep.family, pid, length)
    # écarts systématiques : priorité aléatoire fixée par (famille, prompt, adaptateur) -> emboîtement par force
    drift = WEIGHT_DRIFT[ep.weights]
    if drift:
        kind = "fp8" if ep.weights == "fp8" else f"lora-{ep.adapter}"
        r = np.random.default_rng(_seed("drift", ep.family, pid, kind))
        prio = r.random(length)
        repl = r.integers(0, len(v), size=length)
        for j in np.flatnonzero(prio < drift):
            words[j] = v[repl[j]]
    d = DISGUISES[ep.disguise]
    noise = d.get("noise", BASE_NOISE)
    for j in np.flatnonzero(rng.random(length) < noise):
        words[j] = v[rng.integers(0, len(v))]
    if d.get("green"):
        green = vocab("green-" + ep.name, 40)
        for j in np.flatnonzero(rng.random(length) < d["green"]):
            words[j] = green[rng.integers(0, len(green))]
    if d.get("style"):
        for j in np.flatnonzero(rng.random(length) < d["style"]):
            words[j] = rng.choice(["arr", "matey", "aye", "yo-ho"])
    text = " ".join(words) + "."
    if d.get("prefix"):
        text = d["prefix"] + " " + text
    if d.get("suffix"):
        text = text + " " + d["suffix"]
    return text


def scenario(name: str) -> list:
    if name == "ref":
        return [Endpoint(f"ref-{i + 1:02d}", "P", w, dg) for i, (w, dg) in enumerate(REF_LAYOUT)]
    if name == "ep":
        # 33 endpoints, 3 familles, labels mélangés : ressemble à la tâche réelle (structure inventée)
        specs = []
        for fam, extra in (("F1", 3), ("F2", 2), ("F3", 1)):
            for w, dg in REF_LAYOUT:
                specs.append((fam, w, dg, "a"))
            for k in range(extra):
                specs.append((fam, "L0.7", "plain", f"b{k}"))
        rng = np.random.default_rng(7)
        order = rng.permutation(len(specs))
        return [Endpoint(f"ep-{i + 1:02d}", *specs[j]) for i, j in enumerate(order)]
    raise ValueError(name)


def generate(endpoints: list, n_prompts: int, reps: int, seed: int = 0, length=(40, 120)) -> list:
    rng = np.random.default_rng(seed)
    lengths = rng.integers(length[0], length[1], size=n_prompts)
    out = []
    for rep in range(reps):
        for p in range(n_prompts):
            pid = f"m{p:02d}"
            for ep in endpoints:
                r = np.random.default_rng(_seed("sample", seed, ep.name, pid, rep))
                out.append({"endpoint": ep.name, "id": pid, "repeat": rep, "prompt": f"synthetic prompt {pid}",
                            "response": sample(ep, pid, int(lengths[p]), r)})
    return out


def truth_rows(endpoints: list) -> list:
    return [(ep.name, ep.family, f"{ep.family}-{ep.model}", f"{ep.weights} {ep.disguise}") for ep in endpoints]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", default="ref", choices=["ref", "ep"])
    p.add_argument("--prompts", type=int, default=20)
    p.add_argument("--reps", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--truth", required=True)
    a = p.parse_args(argv)
    eps = scenario(a.scenario)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in generate(eps, a.prompts, a.reps, a.seed):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(a.truth, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["endpoint", "family", "model", "kind"])
        w.writerows(truth_rows(eps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
