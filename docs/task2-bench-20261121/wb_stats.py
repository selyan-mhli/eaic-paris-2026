"""Boîte blanche hors ligne sur les refs (corpus figé pilot1/pilot2, 1re tentative) avec SmolLM2-360M-Instruct.
Pour chaque réponse : par token, logits base (fp32) et logits fp8 simulé (poids e4m3 par tenseur + activations e4m3
dynamiques par tenseur sur les Linear, hors lm_head). Sauve un npz compact : par token, endpoint, pid, rep,
logit choisi, logsumexp et entropie sur une grille de T, et LLR fp8-base sur cette grille."""
import json, sys, torch, numpy as np, time
from transformers import AutoTokenizer, AutoModelForCausalLM
NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
W = "/Users/selyan/dev/eaic-paris-2026/docs/task2-work-20261121/"
TG = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4])
PROMPTS = {}
for l in open(W + "raw/c2run_ref_answers.jsonl"):
    r = json.loads(l); PROMPTS[r["id"]] = r["prompt"]
rows = [json.loads(l) for l in open(W + "corpus.jsonl")]
rows = [r for r in rows if r["campaign"] in ("pilot1", "pilot2") and r["attempt"] == 0]
dev = "mps"
tok = AutoTokenizer.from_pretrained(NAME)
base = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.float32).to(dev).eval()
q = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.float32).to(dev).eval()
F8 = torch.float8_e4m3fn; MX = 448.0
def qd(x):
    s = x.abs().max().clamp(min=1e-12) / MX
    return (x / s).to("cpu").to(F8).to(torch.float32).to(x.device) * s
with torch.no_grad():
    for n, mod in q.named_modules():
        if isinstance(mod, torch.nn.Linear) and "lm_head" not in n:
            mod.weight.copy_(qd(mod.weight))
            mod.register_forward_pre_hook(lambda m, a: (qd(a[0]),))
out = {k: [] for k in ["ep", "pid", "rep", "pos", "lz_b", "H_b", "lp_b", "lp_q", "isarg_b"]}
t0 = time.time()
for i, r in enumerate(rows):
    ids_p = tok.apply_chat_template([{"role": "user", "content": PROMPTS[r["pid"]]}], add_generation_prompt=True, return_tensors="pt")
    if not isinstance(ids_p, torch.Tensor): ids_p = ids_p["input_ids"]
    ids_r = tok(r["response"], add_special_tokens=False, return_tensors="pt").input_ids
    if ids_r.shape[1] == 0: continue
    ids = torch.cat([ids_p, ids_r], 1).to(dev)
    with torch.no_grad():
        lb = base(ids).logits[0, ids_p.shape[1] - 1:-1].float().cpu()   # (L, V)
        lq = q(ids).logits[0, ids_p.shape[1] - 1:-1].float().cpu()
    y = ids_r[0]
    L = len(y)
    lpb = []; lpq = []; H = []
    for T in TG:
        a = torch.log_softmax(lb / T, -1); b = torch.log_softmax(lq / T, -1)
        lpb.append(a.gather(1, y[:, None]).squeeze(1)); lpq.append(b.gather(1, y[:, None]).squeeze(1))
        H.append(-(a.exp() * a).sum(-1))
    out["ep"] += [r["endpoint"]] * L; out["pid"] += [r["pid"]] * L; out["rep"] += [r["repeat"]] * L
    out["pos"] += list(range(L))
    out["lp_b"].append(torch.stack(lpb, 1).numpy()); out["lp_q"].append(torch.stack(lpq, 1).numpy())
    out["H_b"].append(torch.stack(H, 1).numpy()); out["isarg_b"].append((lb.argmax(-1) == y).numpy())
    out["lz_b"].append(lb.gather(1, y[:, None]).squeeze(1).numpy())
    if i % 100 == 0: print(i, len(rows), round(time.time() - t0), flush=True)
np.savez_compressed(sys.argv[1], TG=TG, ep=np.array(out["ep"]), pid=np.array(out["pid"]), rep=np.array(out["rep"]),
                    pos=np.array(out["pos"]), lp_b=np.concatenate(out["lp_b"]), lp_q=np.concatenate(out["lp_q"]),
                    H_b=np.concatenate(out["H_b"]), isarg_b=np.concatenate(out["isarg_b"]), lz_b=np.concatenate(out["lz_b"]))
print("saved", sys.argv[1], round(time.time() - t0))
