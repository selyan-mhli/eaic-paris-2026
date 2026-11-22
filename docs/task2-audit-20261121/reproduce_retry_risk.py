"""Reproduit hors réseau la boucle du runner archivé, avec un serveur simulé.

Le débit sur 502 est une hypothèse de simulation issue du signalement transmis
par Selyan ; ce script ne mesure ni le serveur réel ni le budget de SLZ.
"""
import contextlib
import io
import json
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

base = Path(__file__).resolve().parent
metadata = json.loads((base / "runner_metadata.json").read_text())
source = next(f["source"] for f in metadata["c2run/ep/ep1b.py"]["definitions"] if f["name"] == "query")


class Session:
    def __init__(self):
        self.calls = []
        self.budget_reads = 0

    def post(self, url, *, timeout, json):
        self.calls.append({"timeout": timeout, "payload": json})
        attempt = len(self.calls)
        if attempt < 3:
            return SimpleNamespace(status_code=502, headers={}, text="")
        body = {"choices": [{"message": {"content": "OK"}}], "budget": {"used": 3, "limit": 100}}
        return SimpleNamespace(status_code=200, json=lambda: body)

    def budget(self, session):
        self.budget_reads += 1
        return {"ep-01": 100}


session = Session()
sleeps = []
with tempfile.TemporaryDirectory(prefix="task2-retry-audit-") as tmp:
    output = Path(tmp) / "answers.jsonl"
    env = dict(Path=Path, ANSWERS_FILE=output, ENDPOINTS=["ep-01"], REPEATS=1, MAX_TOKENS=256,
               BASE_URL="https://serveur-simule.invalid/v1", PARALLEL=1,
               make_session=lambda: session, get_budget=session.budget,
               threading=threading, ThreadPoolExecutor=ThreadPoolExecutor, json=json,
               time=SimpleNamespace(sleep=sleeps.append),
               requests=SimpleNamespace(RequestException=RuntimeError))
    exec(compile(source, "<query archivée>", "exec"), env)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        env["query"]([("test", "Réponds OK")])
    answers = [json.loads(line) for line in output.read_text().splitlines()]
    result = dict(scenario="502 vide, 502 vide, 200 ; chaque tentative débitée dans le scénario",
                  source="runner_metadata.json : c2run/ep/ep1b.py/query",
                  http_attempts=len(session.calls), saved_answers=len(answers),
                  budget_reads=session.budget_reads, simulated_budget_used=3,
                  sleeps_without_real_wait=sleeps,
                  client_timeouts=[call["timeout"] for call in session.calls],
                  console=log.getvalue(), saved_record_fields=list(answers[0]))
    assert result["http_attempts"] == 3 and result["saved_answers"] == 1
    assert result["budget_reads"] == 1
    (base / "retry_risk_reproduction.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
