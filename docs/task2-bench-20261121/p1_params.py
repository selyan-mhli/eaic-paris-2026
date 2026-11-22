# Campagne p1 : quels paramètres l'API honore-t-elle ? Règle : un paramètre est "honoré" si la réponse
# change de façon systématique (ex. T=0 -> 2 répétitions identiques ; logprobs -> champ présent).
B = Bench("p1")
m = B.models()
print("models keys:", list(m.keys()) if isinstance(m, dict) else m)
if isinstance(m, dict):
    for d in m["data"][:2]: print(json.dumps(d)[:400])
P = "Count from 1 to 15, separated by spaces. Output only the numbers."
U = [{"role": "user", "content": P}]
jobs = []
for ep in ["ref-01", "ref-08"]:
    base = {"model": ep, "messages": U, "max_tokens": 40}
    for r in range(2): jobs.append((f"base{r}", dict(base)))
    for r in range(2): jobs.append((f"t0_{r}", dict(base, temperature=0)))
    for r in range(2): jobs.append((f"t0s_{r}", dict(base, temperature=0, seed=7)))
    jobs.append(("topp", dict(base, top_p=0.01)))
    jobs.append(("lp", dict(base, logprobs=True, top_logprobs=5, temperature=0)))
    jobs.append(("n3", dict(base, n=3)))
    jobs.append(("mt5", dict(base, max_tokens=5)))
    jobs.append(("stop", dict(base, stop=["5"])))
    jobs.append(("sys", {"model": ep, "max_tokens": 40, "messages": [{"role": "system", "content": "Answer in French only."}] + U}))
jobs.append(("compl", {"model": "ref-01", "prompt": "1 2 3 4 5", "max_tokens": 10, "echo": True, "logprobs": 3}, "completions"))
res = B.run(jobs)
for j, r in zip(jobs, res):
    b = r.get("body"); ch = b.get("choices") if isinstance(b, dict) else None
    extra = sorted(b.keys()) if isinstance(b, dict) else None
    print(j[1].get("model"), j[0], r["status"], r["dt"], extra,
          "nch=%s" % (len(ch) if ch else None),
          "lp=%s" % (bool(ch and ch[0].get("logprobs"))),
          "fin=%s" % (ch[0].get("finish_reason") if ch else None),
          repr((ch[0]["message"]["content"] if ch and "message" in ch[0] else str(b))[:90]))
print("logged", B.n, B.path)
