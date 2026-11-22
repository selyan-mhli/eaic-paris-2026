"""Reproduit le général officiel et calcule des scénarios, concurrents figés."""

from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import hashlib
import html
import json
import re


TASKS = ["34-watermark-localisation-image", "33-model-equality", "31-unlearning"]
ROOT = Path("/Users/selyan/dev/eaic-paris-2026")
HERE = Path(__file__).resolve().parent


def plain(text):
    return html.unescape(" ".join(re.sub(r"<[^>]+>", " ", text).split()))


def parse(path):
    raw = path.read_text()
    body = raw.split('<dialog id="scoring"')[0]
    marks = list(re.finditer(r'data-panel="([^"]+)"', body))
    panels = {}
    for i, mark in enumerate(marks):
        panel = body[mark.end():marks[i + 1].start() if i + 1 < len(marks) else len(body)]
        rows = []
        for attrs, row in re.findall(r"<tr([^>]*)>(.*?)</tr>", panel, re.S):
            team = re.search(r'data-team="([^"]+)"', attrs)
            if not team:
                continue
            cells = [(plain(cell), re.search(r'title="([^"]+)"', cell))
                     for _, cell in re.findall(r"<td([^>]*)>(.*?)</td>", row, re.S)]
            scores = [float(match.group(1)) for _, match in cells
                      if match and re.fullmatch(r"[0-9.eE+-]+", match.group(1))]
            rows.append({"team": html.unescape(team.group(1)),
                         "rank": int(cells[0][0]) if cells[0][0].isdigit() else None,
                         "scores": scores, "updated_at": re.findall(r'data-at="([^"]+)"', row)})
        panels[mark.group(1)] = rows
    limits = {key: {"min": min(row["scores"][0] for row in panels[key]),
                    "max": max(row["scores"][0] for row in panels[key])} for key in TASKS}

    def overall(scores):
        return sum((score - limits[key]["min"]) / (limits[key]["max"] - limits[key]["min"])
                   for key, score in zip(TASKS, scores)) / 3

    complete = [row for row in panels["overall"] if len(row["scores"]) == 4]
    errors = [abs(overall(row["scores"][:3]) - row["scores"][3]) for row in complete]
    assert complete and max(errors) < 1e-12, max(errors)
    slz = next(row for row in complete if row["team"] == "SLZ")
    return panels, limits, overall, complete, slz


def main():
    live = Path("/private/tmp/eaic-relais-20261122/leaderboard-live.html")
    panels, limits, overall, complete, slz = parse(live)
    current = slz["scores"][:3]
    base = slz["scores"][3]

    def scenario(label, scores):
        assert all(limits[key]["min"] <= score <= limits[key]["max"] for key, score in zip(TASKS, scores))
        value = overall(scores)
        return {"label": label, "scores": scores, "overall": value, "gain": value - base,
                "rank_others_fixed": 1 + sum(row["team"] != "SLZ" and row["scores"][3] > value for row in complete)}

    cases = [scenario("actuel", current)]
    for label, index, target in [("T1 +0,007623 (CV, pas prédiction serveur)", 0, current[0] + 0.007623003171667287),
                                 ("T1 à 0,70", 0, .70), ("T1 à 0,75", 0, .75),
                                 ("T2 à 0,40", 1, .40), ("T2 à 0,60", 1, .60),
                                 ("T3 à 0,20", 2, .20), ("T3 à 0,30", 2, .30)]:
        scores = current.copy()
        scores[index] = target
        cases.append(scenario(label, scores))
    for index, key in enumerate(TASKS):
        scores = current.copy()
        scores[index] = limits[key]["max"]
        cases.append(scenario(f"T{index + 1} seule au maximum actuel", scores))
    for label, scores in [("T1 0,75 / T2 0,40 / T3 0,20", [.75, .40, .20]),
                          ("T1 0,75 / T2 0,60 / T3 0,20", [.75, .60, .20]),
                          ("T1 0,75 / T2 0,85 / T3 0,30", [.75, .85, .30])]:
        cases.append(scenario(label, scores))
    task_detail = {}
    for index, key in enumerate(TASKS):
        row = next(row for row in panels[key] if row["team"] == "SLZ")
        width = limits[key]["max"] - limits[key]["min"]
        task_detail[key] = {**limits[key], "score": current[index], "rank": row["rank"],
                            "participants": len(panels[key]), "updated_at": row["updated_at"],
                            "normalized": (current[index] - limits[key]["min"]) / width,
                            "overall_gain_per_raw_001": .01 / width / 3}
    targets = {}
    for target in [.40, .50, .60, .70, complete[5]["scores"][3]]:
        targets[str(target)] = {}
        for i, key in enumerate(TASKS):
            required = current[i] + (target - base) * 3 * (limits[key]["max"] - limits[key]["min"])
            targets[str(target)][f"T{i + 1}"] = {"raw_score_required_on_frozen_scale": required,
                "within_current_task_range": required <= limits[key]["max"],
                "warning": "Une cible au-dessus du maximum actuel modifierait la normalisation ; ce chiffre ne constitue pas un scénario réalisable à extrema fixes."}
    history = []
    paths = [("21/11 vers 23:15", ROOT / "docs/audit-strategies-20261121/leaderboard-2315.html"),
             ("22/11 vers 00:26", Path("/private/tmp/claude-501/-Users-selyan--claude-spares-spare-oGfpae/d60fc14b-9806-4319-b999-3844f0f1cede/scratchpad/lb2.html")),
             ("22/11 vers 03:05", ROOT / "docs/reprise-claude-20261122/preuves/leaderboard-0308.html")]
    for label, path in paths:
        if not path.exists():
            continue
        _, old_limits, old_overall, old_complete, old_slz = parse(path)
        old_scores = old_slz["scores"][:3]
        current_in_old_scale = old_overall(current)
        history.append({"label": label, "path": str(path), "slz": old_slz, "participants": len(old_complete),
                        "limits": old_limits, "actual_overall_change_to_live": base - old_slz["scores"][3],
                        "raw_score_gain_in_old_scale": current_in_old_scale - old_slz["scores"][3],
                        "normalization_change_at_current_scores": base - current_in_old_scale,
                        "old_scores_in_live_scale": overall(old_scores)})
    output = {"source": "http://34.147.229.55/leaderboard_page",
              "analysis_paris": datetime.now(ZoneInfo("Europe/Paris")).isoformat(),
              "capture_paris": datetime.fromtimestamp(live.stat().st_mtime, ZoneInfo("Europe/Paris")).isoformat(),
              "html_sha256": hashlib.sha256(live.read_bytes()).hexdigest(),
              "slz": slz, "complete_teams": len(complete),
              "normalization_max_error": max(abs(overall(row["scores"][:3]) - row["scores"][3]) for row in complete),
              "tasks": task_detail, "top10": complete[:10], "all_teams": complete,
              "scenarios": cases, "single_task_raw_scores_required_fixed_extrema": targets, "history": history}
    (HERE / "leaderboard-live-analysis.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key not in ["all_teams", "top10"]}, ensure_ascii=False, indent=2))
    print("TOP 10")
    for row in complete[:10]:
        print(row)


if __name__ == "__main__":
    main()
