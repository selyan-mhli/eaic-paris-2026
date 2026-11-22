# p3 : listes de choix "aléatoires". 4 prompts x 12 répétitions x 9 refs = 432 requêtes, max_tokens 64.
# RÈGLE D'ANALYSE FIXÉE AVANT COLLECTE (consignée aussi dans RAPPORT.md) :
#  - parse : split sur , ; \n, minuscules, garder [a-z0-9]+, 15 premiers items non vides ;
#  - par (endpoint, prompt) : comptes d'items ; moitiés A = répétitions 0-5, B = 6-11 ;
#  - stats candidates : S1 affinité de Hellinger, S2 Spearman sur l'union des items ; chacune normalisée
#    par l'affinité intra (moitié-moitié) : r(i,j) = S(i,j)/sqrt(S(i,i')S(j,j')), moyenne sur les prompts ;
#  - règle : lien moyen sur 1-r, seuil choisi sur une moitié (MCC refs max), évalué sur l'autre, puis l'inverse.
#  - succès : MCC >= 0,83 dans les deux sens ; sinon rapporter les erreurs par type.
PROMPTS = {
 "L1": "List 15 random animals, separated by commas. Output only the list.",
 "L2": "List 15 random integers between 1 and 100, separated by commas. Output only the numbers.",
 "L3": "List 15 first names, separated by commas. Output only the names.",
 "L4": "List 15 random English words, separated by commas. Output only the words.",
}
B = Bench("p3")
EPS = [f"ref-{i:02d}" for i in range(1, 10)]
jobs = []
for rep in range(12):
    for pid, p in PROMPTS.items():
        for ep in EPS:
            jobs.append((f"{pid}|{rep}", {"model": ep, "max_tokens": 64, "messages": [{"role": "user", "content": p}]}))
res = B.run(jobs)
bad = sum(1 for r in res if r.get("status") != 200)
print("done", len(res), "non200", bad, B.path)
