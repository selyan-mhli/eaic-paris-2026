"""Test d'analyze_margin.py sur des tables synthétiques au format score_like / margin_edit.
    .venv/bin/python docs/task3-work-20261121/test_analyze_margin.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
HEAD = ("candidat\tfmt\tfin\tfp16flip\tforget_ce_mean\tforget_ce_std\tforget_flip\tforget_ent_T2\tforget_top\t"
        "altered_ce_mean\taltered_ce_std\taltered_flip\taltered_ent_T2\taltered_top\tpshuffle_agree\tpshuffle_ret\t"
        "lowpass32_agree\tlowpass32_ret\tmask35_agree\tmask35_ret\toff_agree\toff_ret")
MODEL = "model\tTrue\tTrue\t0.0000\t0.0003\t0.0024\t0.0000\t0.0970\t0.0371\t0.0012\t0.0267\t0.0006\t0.1056\t0.0374\t1\t1\t1\t1\t1\t1\t1\t1"


def row(name, ce, std, flip, lp32, fp16="0.0000"):
    return (f"{name}\tTrue\tTrue\t{fp16}\t{ce}\t{std}\t{flip}\t0.5\t0.04\t{ce}\t{std}\t{flip}\t0.5\t0.04\t"
            f"0.90\t0.95\t{lp32}\t0.95\t0.90\t0.95\t0.90\t0.97")


def main():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "res").mkdir(); (d / "logs").mkdir()
        (d / "res/score_99_mg10.txt").write_text("\n".join([HEAD, MODEL,
            row("mg_mg10_s0125", 0.20, 0.30, 0.01, 0.95),
            row("mg_mg10_s0250", 0.40, 0.50, 0.02, 0.92),
            row("mg_mg10_s0375", 0.60, 0.70, 0.30, 0.90),       # trop de classes changées
            "mg_mg10_s0500\tREJET\tpoids non finis"]) + "\n")
        (d / "res/score_99_mg05.txt").write_text("\n".join([HEAD, MODEL,
            row("mg_mg05_s0125", 0.30, 0.40, 0.02, 0.50),       # hors zone retain (lowpass32)
            row("mg_mg05_s0250", 0.35, 0.45, 0.02, 0.80, fp16="0.0100")]) + "\n")  # hors zone fp16
        (d / "res/score_99_mg25.txt").write_text("Traceback (most recent call last):\nImportError: boom\n")
        (d / "logs/99_0.out").write_text(
            "tag=mg10 margin=1.0 : CE cible forget 0.4655/0.1566\n"
            "step 25 loss_f=1.0975 loss_h=0.0422 3.18 step/s\n"
            "step 125: CE forget 0.2000/0.3000 changées 0.0100\n"
            "step 250: CE forget 0.4000/0.5000 changées 0.0200\n")
        out = subprocess.run([sys.executable, str(HERE / "analyze_margin.py"), "--job", "99",
                              "--res-dir", str(d / "res"), "--logs-dir", str(d / "logs"),
                              "--tsv", str(d / "t.tsv")], capture_output=True, text=True, check=True).stdout
        rec = out.split("## Recommandation")[1]
        assert "`mg_mg10_s0250`" in rec, rec           # meilleur ms en zone connue, flip faible
        assert "mg_mg05" not in rec.split("## Erreurs")[0], rec
        assert "mg_mg10_s0375" not in rec.split("## Erreurs")[0]
        assert "REJET" in out and "ImportError" in out
        assert "| mg10 | 1.0 | 0.4655/0.1566 | 250 | 0.4000/0.5000 | 0.0200 |" in out, out
        assert "lowpass32_agree" in out and "fp16flip" in out
        assert len((d / "t.tsv").read_text().splitlines()) == 1 + 6   # en-tête + origine + 5 candidats
        print("OK analyze_margin (synthétique)")


if __name__ == "__main__":
    main()
