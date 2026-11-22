"""Relevé public et calculs de scénarios, sans requête ni apprentissage."""
from pathlib import Path
from itertools import combinations
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load(name):
    return json.loads((ROOT/name).read_text())


def plain(s):
    return html.unescape(" ".join(re.sub(r"<[^>]+>", " ", s).split()))


def main():
    raw = (HERE/"leaderboard-2315.html").read_bytes()
    body = raw.decode().split('<dialog id="scoring"')[0]
    marks = list(re.finditer(r'data-panel="([^"]+)"', body))
    panels = {}
    for i, mark in enumerate(marks):
        panel = body[mark.end():marks[i+1].start() if i+1 < len(marks) else len(body)]
        rows = []
        for attrs, row in re.findall(r"<tr([^>]*)>(.*?)</tr>", panel, re.S):
            team = re.search(r'data-team="([^"]+)"', attrs)
            if not team:
                continue
            cells = [(plain(cb), re.search(r'title="([^"]+)"', cb)) for _, cb in re.findall(r"<td([^>]*)>(.*?)</td>", row, re.S)]
            scores = [float(m.group(1)) for _, m in cells if m and re.fullmatch(r"[0-9.eE+-]+", m.group(1))]
            rows.append({"team": html.unescape(team.group(1)), "rank": int(cells[0][0]) if cells[0][0].isdigit() else None,
                         "scores": scores, "updated_at": re.findall(r'data-at="([^"]+)"', row)})
        panels[mark.group(1)] = rows
    taskids = ["34-watermark-localisation-image", "33-model-equality", "31-unlearning"]
    limits = {k: {"min": min(r["scores"][0] for r in panels[k]), "max": max(r["scores"][0] for r in panels[k])} for k in taskids}

    def overall(scores):
        return sum((s-limits[k]["min"])/(limits[k]["max"]-limits[k]["min"]) for k, s in zip(taskids, scores))/3

    assert all(abs(overall(r["scores"][:3])-r["scores"][3]) < 1e-12 for r in panels["overall"] if len(r["scores"]) == 4)
    slz = next(r for r in panels["overall"] if r["team"] == "SLZ")
    scenarios = []
    for label, scores in [
        ("SLZ actuel", slz["scores"][:3]),
        ("T2 seul à 1", [slz["scores"][0], 1.0, slz["scores"][2]]),
        ("T1 0,70 ; T2 0,85 ; T3 0,30", [.70, .85, .30]),
        ("T1 0,72 ; T2 0,85 ; T3 0,31", [.72, .85, .31]),
        ("T1 0,72 ; T2 1 ; T3 0,35", [.72, 1, .35]),
    ]:
        value = overall(scores)
        rank = 1+sum(r["team"] != "SLZ" and len(r["scores"]) == 4 and r["scores"][3] > value for r in panels["overall"])
        scenarios.append({"label": label, "scores": scores, "overall": value, "rank_fixed_board": rank})
    # Une seule tâche parfaite, deux tâches actuelles : les extrêmes restent fixes.
    single = {}
    for i, k in enumerate(taskids):
        s = slz["scores"][:3].copy()
        s[i] = limits[k]["max"]
        single[k] = overall(s)
    m = load("docs/task1-audit-20261121/audit_results.json")["metric"]
    champ = load("docs/task1-audit-20261121/champion_summary.json")
    F, R = champ["F"], champ["R"]
    bbox = m["oracle_inverse_bboxF_bboxR"]["R"]
    t1 = {"head_oof": champ["head"], "F": F, "R": R,
          "oracle_bbox_R": bbox, "F_current_R_bbox_oracle": (F+bbox)/2,
          "F_needed_if_R_fixed": {str(s): 2*s-R for s in (.70, .72, .75)},
          "R_needed_if_F_fixed": {str(s): 2*s-F for s in (.70, .72, .75)},
          "F_needed_if_R_bbox_oracle": {str(s): 2*s-bbox for s in (.70, .72, .75)},
          "scenarios_not_measured": [{"F": a, "R": b, "mean": (a+b)/2} for a, b in ((.65, .75), (.64, .80), (.70, .80))],
          "gate_oracle_gain_on_trivial_rule": m["oracle_regime_only"]["all"]-m["gate075_only"]["all"]}
    ps_tm = [champ["groups"][k+"_F"]["head"] for k in ("pixelseal", "trustmark")]
    t1["PS_TM_F_pilot_plus_008_only"] = champ["head"]+2*.08/6
    t1["PS_TM_F_perfect_only"] = champ["head"]+sum(1-x for x in ps_tm)/6
    t1["SS_R_perfect_only"] = champ["head"]+(1-champ["groups"]["stegastamp_R"]["head"])/6
    t1["SS_both_perfect_only"] = champ["head"]+(2-champ["groups"]["stegastamp_F"]["head"]-champ["groups"]["stegastamp_R"]["head"])/6
    # La borne exige seulement que le MCC famille <= 1 (score non quantifié).
    t2 = {"public_values": sorted({r["scores"][0] for r in panels[taskids[1]]}),
          "public_count": len(panels[taskids[1]]),
          "all_public_values_exact_multiples_005": all(abs(r["scores"][0]*20-round(r["scores"][0]*20)) < 1e-12 for r in panels[taskids[1]]),
          "model_mcc_required_if_family_perfect": {str(s): (s-.1)/.9 for s in (.70, .85, 1.0)}}
    truth = {i: ("base" if i in (1, 3, 5, 6, 8) else str(i)) for i in range(1, 10)}

    def mcc(isolates):
        pred = {i: (str(i) if i in isolates else "base") for i in truth}
        tp = fp = fn = tn = 0
        for a, b in combinations(truth, 2):
            t, p = truth[a] == truth[b], pred[a] == pred[b]
            tp += t and p
            fp += not t and p
            fn += t and not p
            tn += not t and not p
        den = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        return (tp*tn-fp*fn)/den if den else 0

    t2["reference_isolation_exact"] = [{"isolates": iso, "mcc_model": mcc(iso), "hypothetical_total_family_perfect": .1+.9*max(0, mcc(iso))} for iso in ([7], [4, 7], [2, 4, 7], [2, 4, 7, 9])]
    t3 = {"formula": "A_loss * B_prediction * R_retain",
          "required_equal_forget_components": [{"target": s, "retain_assumed": ret, "A_equals_B": math.sqrt(s/ret)} for s in (.30, .35, .40) for ret in (.8, .9, 1.0)],
          "small_gain_example": .25*.25*.9,
          "competitive_example": .65*.65*.85,
          "warning": "Illustrations algébriques ; composantes gold non observables localement."}
    out = {"source": "http://34.147.229.55/leaderboard_page", "snapshot_label": "21/11/2026 vers 23:15 Paris",
           "analysis_at_utc": datetime.now(timezone.utc).isoformat(), "html_sha256": hashlib.sha256(raw).hexdigest(),
           "panels": panels, "limits": limits, "overall_scenarios_fixed_extrema": scenarios,
           "single_task_at_current_max": single, "task1": t1, "task2": t2, "task3": t3}
    (HERE/"analysis_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False)+"\n")
    print("Normalisation reproduite pour toutes les équipes complètes (écart < 1e-12).")
    print("Extrêmes publics:", limits)
    for row in scenarios:
        print(f'{row["label"]}: overall={row["overall"]:.6f}, rang={row["rank_fixed_board"]}')
    print("Une seule tâche au maximum:", single)
    print("T1:", json.dumps(t1, ensure_ascii=False))
    print("T2:", json.dumps(t2, ensure_ascii=False))
    print("T3:", json.dumps(t3, ensure_ascii=False))


if __name__ == "__main__":
    main()
