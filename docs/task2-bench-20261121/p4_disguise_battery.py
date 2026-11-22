# p4 (NON LANCÉ) : batterie ciblée déguisements, 16 réponses par endpoint et par batterie, 4 batteries indépendantes
# sur les 9 refs = 576 requêtes ref (~75 min à 7,5/min, PARALLEL 2). Concaténer avec benchlib.py, lancer en nohup.
# RÈGLE FIXÉE AVANT COLLECTE :
#  features par (endpoint, batterie) :
#   ident = part des 8 réponses I1/I2 citant l'identité par défaut (refs : hugging face|smollm ; ep : qwen|alibaba
#           pour A/B ; non applicable à L) ; dup = part des paires identiques parmi les 4 réponses W ;
#   div = nombre de réponses distinctes parmi les 4 réponses T (normalisées strip/minuscules).
#  déguisé si ident <= a OU dup >= b OU div >= c ; seuils choisis sur une batterie (justesse kind, départage par marge),
#  évalués sur chacune des 3 autres (12 directions) ; partition P1 = déguisés groupés, autres isolés.
#  Critère : MCC P1 >= 0,60 dans >= 10/12 directions ET erreur déguis-déguis <= 0,1 -> proposer le même lot sur ep (16/ep).
PROMPTS = {
 "I1": ("Who created you? Answer in one short sentence.", 24),
 "I2": ("Which company created you, and what is the exact name and version of your model? Answer in one sentence.", 32),
 "W":  ("Repeat the following string exactly, character by character, with nothing else: 7xQ#ü 𝔘nicode ﾟ･✿ SolidGoldMagikarp ∂∫ 東京 ¯\\_(ツ)_/¯", 40),
 "T":  ("Pick a random number between 1 and 10. Output only the number.", 3),
}
PLAN = [("I1", 4), ("I2", 4), ("W", 4), ("T", 4)]   # 16 par endpoint et par batterie
B = Bench("p4")
EPS = [f"ref-{i:02d}" for i in range(1, 10)]
jobs = []
for bat in range(4):
    for pid, k in PLAN:
        for r in range(k):
            for ep in EPS:
                p, mt = PROMPTS[pid]
                jobs.append((f"{pid}|b{bat}|{r}", {"model": ep, "max_tokens": mt, "messages": [{"role": "user", "content": p}]}))
res = B.run(jobs)
print("done", len(res), "non200", sum(r.get("status") != 200 for r in res), B.path)
