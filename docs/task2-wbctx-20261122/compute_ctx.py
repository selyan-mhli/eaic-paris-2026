"""Logvraisemblance des réponses sous une base, profilée sur une grille FIGÉE de contextes cachés et de T,
avec le fp8 simulé (e4m3, poids + activations dynamiques par tenseur sur les Linear hors lm_head, repris de
docs/task2-bench-20261121/wb_stats.py). Sortie npz par RÉPONSE (pas par token) :
  lp_b[n, C, T], lp_q[n, C, T] = moyenne par token de log p(token) sous base / fp8 simulé ; ntok[n].
Usage : compute_ctx.py MODEL OUT.npz --campaigns pilot1,pilot2 [--device mps] [--corpus PATH] [--prompts PATH] [--limit N]
Batch 1 (pas de padding : la quantification dynamique dépendrait sinon du lot)."""
import argparse, json, time, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
W = "/Users/selyan/dev/eaic-paris-2026/docs/task2-work-20261121/"
TG = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4])   # grille T de wb_stats.py
# Grille de contextes FIGÉE (brief du 22/11 05:30), identique pour refs et ep. None = gabarit par défaut.
CTX = [None, "", "You are a helpful assistant.", "You are a pirate. Always answer in character.",
       "Answer concisely in a formal tone.", "You are a friendly assistant who loves to chat."]
ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("out")
ap.add_argument("--campaigns", default="pilot1,pilot2"); ap.add_argument("--device", default="mps")
ap.add_argument("--corpus", default=W + "corpus.jsonl"); ap.add_argument("--prompts", default=W + "raw/c2run_ref_answers.jsonl")
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--no-fp8", action="store_true")
a = ap.parse_args()
PROMPTS = {}
for l in open(a.prompts):
    r = json.loads(l); PROMPTS[r.get("id", r.get("pid"))] = r["prompt"]
camps = a.campaigns.split(",")
rows = [json.loads(l) for l in open(a.corpus)]
rows = [r for r in rows if r["campaign"] in camps and r.get("attempt", 0) == 0 and r["response"]]
if a.limit: rows = rows[:a.limit]
dev = a.device
tok = AutoTokenizer.from_pretrained(a.model)
base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).to(dev).eval()
q = None
if not a.no_fp8:
    q = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).to(dev).eval()
    MX = 448.0
    def e4m3(x):  # arrondi e4m3fn émulé sur le device ; identique bit à bit au cast torch (vérifié sur 1e6 valeurs, CPU et MPS)
        a = x.abs()
        e = torch.floor(torch.log2(a.clamp(min=2 ** -9))).clamp(min=-6)
        st = torch.exp2(e - 3)
        return torch.sign(x) * (torch.round(a / st) * st).clamp(max=448.)
    def qd(x):
        s = x.abs().max().clamp(min=1e-12) / MX
        return e4m3(x / s) * s
    with torch.no_grad():
        for n, mod in q.named_modules():
            if isinstance(mod, torch.nn.Linear) and "lm_head" not in n:
                mod.weight.copy_(qd(mod.weight))
                mod.register_forward_pre_hook(lambda m, x: (qd(x[0]),))
TGt = torch.tensor(TG, dtype=torch.float32, device=dev)
def prompt_ids(c, user):
    msgs = ([] if c is None else [{"role": "system", "content": c}]) + [{"role": "user", "content": user}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    return ids if isinstance(ids, torch.Tensor) else ids["input_ids"]
def score(model, ids, P, y):
    with torch.no_grad():
        lg = model(ids).logits[0, P - 1:-1].float()                  # (L, V)
        out = []
        for T in TG:
            out.append(torch.log_softmax(lg / float(T), -1).gather(1, y[:, None]).squeeze(1).mean())
        return torch.stack(out).cpu().numpy()                         # (nT,)
n, C, K = len(rows), len(CTX), len(TG)
lp_b = np.full((n, C, K), np.nan, np.float32); lp_q = np.full((n, C, K), np.nan, np.float32); ntok = np.zeros(n, int)
t0 = time.time()
for i, r in enumerate(rows):
    ids_r = tok(r["response"], add_special_tokens=False, return_tensors="pt").input_ids
    ntok[i] = ids_r.shape[1]
    if ntok[i] == 0: continue
    y = ids_r[0].to(dev)
    for c, ctx in enumerate(CTX):
        ids_p = prompt_ids(ctx, PROMPTS[r["pid"]])
        ids = torch.cat([ids_p, ids_r], 1).to(dev)                    # (1, P+L)
        lp_b[i, c] = score(base, ids, ids_p.shape[1], y)
        if q is not None: lp_q[i, c] = score(q, ids, ids_p.shape[1], y)
    if i in (0, 9, 49) or i % 100 == 0: print(i, n, round(time.time() - t0, 1), flush=True)
np.savez_compressed(a.out, TG=TG, ctx=np.array(["<default>" if c is None else c for c in CTX]),
                    ep=np.array([r["endpoint"] for r in rows]), pid=np.array([r["pid"] for r in rows]),
                    rep=np.array([r["repeat"] for r in rows]), ntok=ntok, lp_b=lp_b, lp_q=lp_q,
                    model=a.model, elapsed=time.time() - t0)
print("saved", a.out, round(time.time() - t0, 1))
