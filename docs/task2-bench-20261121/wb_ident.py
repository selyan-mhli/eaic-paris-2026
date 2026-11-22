"""Boîte blanche, hors ligne : quel modèle public sert les refs ? Log-vraisemblance moyenne par token des réponses
ref-01 (corpus figé pilot1/pilot2) sous chaque candidat, gabarit de chat par défaut."""
import json, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
CORPUS = "/Users/selyan/dev/eaic-paris-2026/docs/task2-work-20261121/corpus.jsonl"
PROMPTS = {}
for l in open("/Users/selyan/dev/eaic-paris-2026/docs/task2-work-20261121/raw/c2run_ref_answers.jsonl"):
    r = json.loads(l); PROMPTS[r["id"]] = r["prompt"]
rows = [json.loads(l) for l in open(CORPUS)]
rows = [r for r in rows if r["campaign"] in ("pilot1", "pilot2") and r["attempt"] == 0]
def score(model, tok, rows, T=1.0):
    tot = n = 0
    for r in rows:
        msgs = [{"role": "user", "content": PROMPTS[r["pid"]]}]
        ids_p = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        if not isinstance(ids_p, torch.Tensor): ids_p = ids_p["input_ids"]
        ids_r = tok(r["response"], add_special_tokens=False, return_tensors="pt").input_ids
        ids = torch.cat([ids_p, ids_r], 1)
        with torch.no_grad(): lg = model(ids).logits[0, ids_p.shape[1] - 1:-1].float() / T
        lp = torch.log_softmax(lg, -1).gather(1, ids_r[0][:, None]).squeeze(1)
        tot += lp.sum().item(); n += len(lp)
    return tot / n
if __name__ == "__main__":
    ep = sys.argv[2] if len(sys.argv) > 2 else "ref-01"
    sub = [r for r in rows if r["endpoint"] == ep][:60]
    for name in sys.argv[1].split(","):
        tok = AutoTokenizer.from_pretrained(name); m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
        print(name, ep, "logp/token %.3f" % score(m, tok, sub), flush=True)
