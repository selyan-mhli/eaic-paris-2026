"""Un run autoresearch : snapshot de solve.py, exécution sous budget, score local, ligne dans results.tsv.

Usage : python common/autoresearch/run_one.py c1 --direction max --budget 300 --desc "lr 3e-4"

Contrat du dossier de challenge (voir program.template.md) :
  solve.py --out <dir>   seul fichier modifié par l'agent ; lit AR_BUDGET_S et s'arrête avant
  eval.py --pred <dir>   figé ; affiche une ligne "score: <float>" sur la validation locale

Un run qui n'améliore pas le meilleur score restaure solve.py depuis le meilleur snapshot.
"""
import argparse, os, re, shutil, subprocess, sys, time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("challenge")
p.add_argument("--direction", choices=["max", "min"], required=True)
p.add_argument("--budget", type=int, default=300, help="secondes allouées à solve.py")
p.add_argument("--eval-timeout", type=int, default=600)
p.add_argument("--desc", required=True)
a = p.parse_args()

c = Path(a.challenge).resolve()
for f in ("solve.py", "eval.py"):
    if not (c / f).exists():
        sys.exit(f"{c / f} absent")
runs = c / "runs"
rid = time.strftime("%Y%m%d-%H%M%S")
d = runs / rid
d.mkdir(parents=True)
shutil.copy(c / "solve.py", d / "solve.py")
log = d / "run.log"
tsv = c / "results.tsv"
if not tsv.exists():
    tsv.write_text("run_id\tscore\tseconds\tstatus\tdescription\n")


def run(cmd, timeout):
    env = dict(os.environ, AR_BUDGET_S=str(a.budget), PYTHONUNBUFFERED="1")
    with log.open("a") as f:
        f.write(f"$ {' '.join(cmd)}\n")
        f.flush()
        try:
            return subprocess.run(cmd, cwd=c, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            f.write(f"\nTIMEOUT après {timeout} s\n")
            return "timeout"


t0 = time.time()
rc = run([sys.executable, "solve.py", "--out", str(d / "out")], timeout=2 * a.budget)
score = None
if rc == 0:
    start = log.stat().st_size
    run([sys.executable, "eval.py", "--pred", str(d / "out")], timeout=a.eval_timeout)
    m = re.findall(r"^score:\s*([-+0-9.eE]+|nan|inf)\s*$", log.read_text()[start:], re.M)
    score = float(m[-1]) if m else None
secs = round(time.time() - t0, 1)

# meilleur run précédent
best_id, best = None, None
for line in tsv.read_text().splitlines()[1:]:
    r = line.split("\t")
    if len(r) >= 4 and r[3] == "keep":
        s = float(r[1])
        if best is None or (s > best if a.direction == "max" else s < best):
            best_id, best = r[0], s

if score is None or score != score:  # absent ou NaN
    status = "crash"
elif best is None or (score > best if a.direction == "max" else score < best):
    status = "keep"
else:
    status = "discard"

desc = a.desc.replace("\t", " ").replace("\n", " ")
with tsv.open("a") as f:
    f.write(f"{rid}\t{score if score is not None else 'nan'}\t{secs}\t{status}\t{desc}\n")

if status != "keep" and best_id:
    shutil.copy(runs / best_id / "solve.py", c / "solve.py")
    restored = f"solve.py restauré depuis {best_id}"
else:
    restored = "solve.py conservé"
print(f"run={rid} status={status} score={score} best_avant={best} secondes={secs} ({restored}) log={log}")
