"""Tableau comparatif du job margin (15903709) : lit res/score_<job>_*.txt (sortie TSV de
score_like.py) et logs/<job>_*.out (margin_edit.py), place dans le même référentiel tout autre
candidat présent dans les tables (push_s*, t1t2_*, ...), et recommande 1-2 candidats.

Lecture seule, CPU, sans torch. Usage (depuis work_c3_eval/ sur JURECA, ou sur une copie locale) :
    python analyze_margin.py --job 15903709 --res-dir res --logs-dir logs \
        [--extra ../work_c3_t1t2/logs/score_15902591_p03.txt ...] [--tsv sortie.tsv]

Colonnes calculées (rien de tout cela n'est un score serveur) :
  ms_<scénario>  terme moyenne/écart-type OFFICIEL calculé exactement à partir des CE forget/altérées
                 mesurées, pour un gold HYPOTHÉTIQUE (official_formula.GOLD_SCENARIOS). Pseudo-labels
                 = argmax de l'origine, comme score_like.
  zone           « connue » si toutes les mesures retain sont au moins aussi bonnes que la pire
                 valeur d'un fichier déjà noté > 0 par le serveur (porte des 70 % franchie) et
                 mesuré par score_like ; sinon la liste des mesures hors zone. Ces accords ne sont
                 PAS des accuracies : la zone connue réduit le risque, elle ne le borne pas.
Le terme de prédiction (KL vers gold) n'est pas calculable sans gold ; ent_T2 et flip sont affichés.
Leçon serveur à garder en tête : baisser le seul logit gagnant sans changer de classe a monté le
score (push alpha 8 -> 0,0308 ; alpha 10 -> 0,0567) ; changer de classe ou étaler l'entropie l'a baissé.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from official_formula import GOLD_SCENARIOS, ms_from_stats  # noqa: E402

# Pire valeur observée parmi les fichiers notés > 0 par le serveur ET mesurés par score_like.
# Depuis res/score_15903813_push.txt, c'est push_s125 (0,0567, porte des 70 % franchie) qui fixe
# tous les minima. Les accords retain locaux ne prédisent PAS l'ordre serveur (push_s125 a les
# pires accords et le meilleur score) : la zone connue borne seulement le risque de porte.
KNOWN_PASS_MIN = {
    "pshuffle_agree": 0.4989, "lowpass32_agree": 0.5120, "mask35_agree": 0.5389,
    "off_agree": 0.6060, "off_ret": 0.6446,
}
# Scores publics connus, par (nom, table source) : registre docs/SLZ_COORDINATION_20261121.md.
KNOWN_SERVER = {("t1t2_p03_e2", "score_15902591_p03.txt"): 0.0091,
                ("distill_r45", "score_15902463_m75.txt"): 0.0044,
                ("push_best", "score_15903813_push.txt"): 0.0308,
                ("push_s125", "score_15903813_push.txt"): 0.0567}
FP16FLIP_MAX = 0.001
# push_s125 change 9,9 % des classes et bat push_best (2,5 %) : la règle « ne pas changer de
# classe » est réfutée. On garde un garde-fou large (push_s150 : 22 %).
FLIP_WARN = 0.25

NUM = re.compile(r"^-?\d+(\.\d+)?([eE]-?\d+)?$")


def parse_score_file(path: Path) -> tuple[list[dict], list[str]]:
    """TSV score_like -> (lignes, erreurs). Tolère REJET et traceback."""
    rows, errors, header = [], [], None
    text = path.read_text(errors="replace")
    if "Traceback" in text and "candidat\t" not in text:
        last = [ln for ln in text.strip().splitlines() if ln.strip()][-1]
        return [], [f"{path.name}: échec score_like ({last[:160]})"]
    for line in text.splitlines():
        parts = line.rstrip("\n").split("\t")
        if parts[0] == "candidat":
            header = parts
            continue
        if header is None or len(parts) < 2:
            continue
        if parts[1] == "REJET":
            errors.append(f"{path.name}: {parts[0]} REJET {parts[2] if len(parts) > 2 else ''}")
            continue
        row = {"candidat": parts[0], "source": path.name}
        for k, v in zip(header[1:], parts[1:]):
            row[k] = float(v) if NUM.match(v) else (v == "True" if v in ("True", "False") else v)
        rows.append(row)
    if header is None:
        errors.append(f"{path.name}: pas d'en-tête score_like")
    return rows, errors


LOG_TAG = re.compile(r"tag=(\S+) margin=([\d.]+) : CE cible forget ([\d.]+)/([\d.]+)")
LOG_STEP = re.compile(r"step (\d+): CE forget ([\d.]+)/([\d.]+) changées ([\d.]+)")
LOG_LOSS = re.compile(r"step (\d+) loss_f=([\d.]+) loss_h=([\d.]+)")
LOG_BAD = re.compile(r"Traceback|Error|CANCELLED|TIME LIMIT|OUT OF MEMORY|nan", re.I)


def parse_logs(logs_dir: Path, job: str) -> tuple[dict, list[str]]:
    """{tag: {margin, ce_target, std_target, steps: {step: (ce, std, flip)}, last_loss}}, alertes."""
    info, alerts = {}, []
    for path in sorted(logs_dir.glob(f"{job}*.out")):
        tag = None
        for line in path.read_text(errors="replace").splitlines():
            m = LOG_TAG.search(line)
            if m:
                tag = m.group(1)
                info.setdefault(tag, {"steps": {}, "log": path.name})
                info[tag].update(margin=float(m.group(2)), ce_target=float(m.group(3)),
                                 std_target=float(m.group(4)))
                continue
            m = LOG_STEP.search(line)
            if m and tag:
                info[tag]["steps"][int(m.group(1))] = tuple(float(m.group(i)) for i in (2, 3, 4))
                continue
            m = LOG_LOSS.search(line)
            if m and tag:
                info[tag]["last_loss"] = (int(m.group(1)), float(m.group(2)), float(m.group(3)))
                continue
            if LOG_BAD.search(line) and "UserWarning" not in line and "loss_f" not in line:
                alerts.append(f"{path.name}: {line.strip()[:160]}")
    return info, alerts


def family(name: str) -> tuple[str, str, int | None]:
    m = re.match(r"mg_(.+)_s(\d+)$", name)
    if m:
        return "margin", m.group(1), int(m.group(2))
    if name.startswith("push"):
        return "push", name, None
    if name == "model":
        return "origine", name, None
    return "autre", name, None


def enrich(row: dict) -> dict:
    clean = (row["forget_ce_mean"], row["forget_ce_std"])
    alt = (row["altered_ce_mean"], row["altered_ce_std"])
    for sc, g in GOLD_SCENARIOS.items():
        row[f"ms_{sc}"] = ms_from_stats(clean, alt, g)
    row["ms_min"] = min(row[f"ms_{sc}"] for sc in GOLD_SCENARIOS)
    out = [k for k, lo in KNOWN_PASS_MIN.items() if k in row and row[k] < lo]
    if row.get("fp16flip", 0.0) > FP16FLIP_MAX:
        out.append("fp16flip")
    if row.get("fmt") is False or row.get("fin") is False:
        out.append("format/finitude")
    row["hors_zone"] = out
    row["fam"], row["tag"], row["step"] = family(row["candidat"])
    row["public"] = KNOWN_SERVER.get((row["candidat"], row["source"]))
    return row


def fmt(x, nd=4):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def recommend(rows: list[dict]) -> list[dict]:
    """Parmi les candidats en zone connue, flip <= FLIP_WARN, trier par ms_min puis ms_eval_target ;
    au plus un checkpoint par tag de marge."""
    ok = [r for r in rows if r["fam"] != "origine" and not r["hors_zone"] and r["forget_flip"] <= FLIP_WARN]
    ok.sort(key=lambda r: (r["ms_min"], r["ms_eval_target"]), reverse=True)
    picked, tags = [], set()
    for r in ok:
        if r["tag"] in tags:
            continue
        picked.append(r)
        tags.add(r["tag"])
        if len(picked) == 2:
            break
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="15903709")
    ap.add_argument("--res-dir", type=Path, default=Path("res"))
    ap.add_argument("--logs-dir", type=Path, default=Path("logs"))
    ap.add_argument("--extra", type=Path, nargs="*", default=[], help="autres tables score_like")
    ap.add_argument("--all-res", action="store_true", help="inclure toutes les tables res/score_*.txt")
    ap.add_argument("--tsv", type=Path, help="écrire le tableau complet en TSV")
    a = ap.parse_args()

    files = sorted(a.res_dir.glob(f"score_{a.job}_*.txt"))
    if a.all_res:
        files = sorted(set(files) | set(a.res_dir.glob("score_*.txt")))
    files += [p for p in a.extra if p not in files]
    rows, errors = [], []
    for p in files:
        r, e = parse_score_file(p)
        rows += r
        errors += e
    seen, uniq = set(), []
    for r in rows:  # « model » apparaît dans chaque table ; un même nom peut venir de dossiers
        key = r["candidat"] if r["candidat"] == "model" else (r["candidat"], r["source"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(enrich(r))
    logs, alerts = parse_logs(a.logs_dir, a.job)

    print(f"# Job {a.job} : {len(files)} tables, {len(uniq)} candidats uniques\n")
    if logs:
        print("## Journaux margin_edit\n")
        print("| tag | marge | CE cible moy/éc | dernier pas mesuré | CE forget moy/éc | changées | dernière perte f/h |")
        print("|---|---:|---|---:|---|---:|---|")
        for tag, d in sorted(logs.items(), key=lambda kv: -kv[1].get("margin", 0)):
            st = max(d["steps"]) if d["steps"] else None
            ce = d["steps"].get(st) if st else None
            ll = d.get("last_loss")
            print(f"| {tag} | {d.get('margin', '-')} | {fmt(d.get('ce_target'))}/{fmt(d.get('std_target'))} | "
                  f"{st or '-'} | {fmt(ce[0]) if ce else '-'}/{fmt(ce[1]) if ce else '-'} | "
                  f"{fmt(ce[2]) if ce else '-'} | {f'{ll[1]:.4f}/{ll[2]:.4f} (pas {ll[0]})' if ll else '-'} |")
        print()
    sc_cols = [f"ms_{s}" for s in GOLD_SCENARIOS]
    print("## Tableau (même référentiel : score_like, fp16, pseudo-labels de l'origine)\n")
    print("Scénarios gold (moy, éc) : " + ", ".join(f"{k}={v}" for k, v in GOLD_SCENARIOS.items()) + "\n")
    head = ["candidat", "fam", "pas", "CE f moy/éc", "CE alt moy/éc", "flip f", "flip alt", "ent_T2 f",
            *sc_cols, "pshuf", "lp32", "mask35", "off", "off_ret", "fp16flip", "hors zone", "public"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    order = sorted(uniq, key=lambda r: (r["fam"] != "origine", r["fam"], r["tag"], r["step"] or 0))
    for r in order:
        name = r["candidat"] if r["fam"] in ("margin", "origine") else f"{r['candidat']} ({r['source']})"
        cells = [name, r["fam"], str(r["step"] or "-"),
                 f"{fmt(r['forget_ce_mean'])}/{fmt(r['forget_ce_std'])}",
                 f"{fmt(r['altered_ce_mean'])}/{fmt(r['altered_ce_std'])}",
                 fmt(r["forget_flip"]), fmt(r["altered_flip"]), fmt(r["forget_ent_T2"], 3),
                 *[fmt(r[c], 3) for c in sc_cols],
                 fmt(r.get("pshuffle_agree"), 3), fmt(r.get("lowpass32_agree"), 3), fmt(r.get("mask35_agree"), 3),
                 fmt(r.get("off_agree"), 3), fmt(r.get("off_ret"), 3), fmt(r.get("fp16flip")),
                 ",".join(r["hors_zone"]) or "non", fmt(r["public"])]
        print("| " + " | ".join(cells) + " |")
    print()
    pick = recommend([r for r in uniq if r["fam"] in ("margin", "push")] or uniq)
    print("## Recommandation (proxy, aucune promesse de score serveur)\n")
    print("Classement intra-famille (margin/push) par terme moyenne/écart-type ; entre familles, ce terme "
          "ne suffit pas : t1t2_p03_e2 et push_best ont des CE voisines (0,106/0,264 et 0,087/0,329) "
          "mais 0,0091 contre 0,0308 au serveur.\n")
    if not pick:
        print("Aucun candidat en zone retain connue avec flip <= %.2f : ne rien soumettre de ce job sans "
              "contrôle supplémentaire." % FLIP_WARN)
    for i, r in enumerate(pick, 1):
        print(f"{i}. `{r['candidat']}` ({r['source']}) : ms_min {r['ms_min']:.3f}, "
              f"ms_eval_target {r['ms_eval_target']:.3f}, flip {r['forget_flip']:.4f}, "
              f"CE {r['forget_ce_mean']:.3f}/{r['forget_ce_std']:.3f}, lowpass32 {r.get('lowpass32_agree', float('nan')):.3f}")
    push = [r for r in uniq if r["fam"] == "push"]
    if not push:
        print("\nNote : aucun push_s* mesuré dans ces tables ; la comparaison avec push_s125 (0,0567 public) "
              "reste à faire (score_like sur push_scale/*.pt).")
    if errors or alerts:
        print("\n## Erreurs et alertes\n")
        for e in errors + alerts:
            print(f"- {e}")
    if a.tsv:
        cols = ["candidat", "source", "fam", "tag", "step", "fmt", "fin", "fp16flip", "forget_ce_mean",
                "forget_ce_std", "forget_flip", "forget_ent_T2", "altered_ce_mean", "altered_ce_std",
                "altered_flip", *sc_cols, "ms_min", *KNOWN_PASS_MIN, "hors_zone"]
        lines = ["\t".join(cols)]
        for r in order:
            lines.append("\t".join(",".join(r[c]) if c == "hors_zone" else str(r.get(c, "")) for c in cols))
        a.tsv.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
