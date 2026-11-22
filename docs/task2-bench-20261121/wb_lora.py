"""Distille une LoRA (r=8, toutes Linear) sur les réponses de ref-07 (LoRA 1.5) des répétitions TRAIN, puis calcule,
pour les réponses de toutes les refs des AUTRES répétitions, le log-prob par token sous base et sous base+s·Δ̂
(s = 0.2 et 1.0 de l'adaptateur appris), à T_hat de l'endpoint (pris dans wb_pilot.npz, statistique sans label).
Usage : wb_lora.py 0,1 out.npz"""
import json, sys, torch, numpy as np, time, random
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
torch.manual_seed(0); random.seed(0)
NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"; W = "/Users/selyan/dev/eaic-paris-2026/docs/task2-work-20261121/"
TRAIN = [int(x) for x in sys.argv[1].split(",")]; T_SERV = 0.7
PROMPTS = {}
for l in open(W + "raw/c2run_ref_answers.jsonl"):
    r = json.loads(l); PROMPTS[r["id"]] = r["prompt"]
rows = [json.loads(l) for l in open(W + "corpus.jsonl")]
rows = [r for r in rows if r["campaign"] in ("pilot1", "pilot2") and r["attempt"] == 0]
z = np.load("/Users/selyan/dev/eaic-paris-2026/docs/task2-bench-20261121/wb_pilot.npz")
THAT = {e: float(z["TG"][int(np.argmax(z["lp_b"][z["ep"] == e].mean(0)))]) for e in set(z["ep"])}
dev = "mps"; tok = AutoTokenizer.from_pretrained(NAME)
m = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.float32).to(dev)
m = get_peft_model(m, LoraConfig(r=8, lora_alpha=16, target_modules="all-linear", lora_dropout=0.0))
def enc(r):
    ids_p = tok.apply_chat_template([{"role": "user", "content": PROMPTS[r["pid"]]}], add_generation_prompt=True, return_tensors="pt")
    if not isinstance(ids_p, torch.Tensor): ids_p = ids_p["input_ids"]
    ids_r = tok(r["response"], add_special_tokens=False, return_tensors="pt").input_ids
    return ids_p, ids_r
tr = [r for r in rows if r["endpoint"] == "ref-07" and r["repeat"] in TRAIN]
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-4)
t0 = time.time(); m.train()
for epoch in range(2):
    random.shuffle(tr); tot = 0
    for r in tr:
        ids_p, ids_r = enc(r)
        if ids_r.shape[1] == 0: continue
        ids = torch.cat([ids_p, ids_r], 1).to(dev)
        lg = m(ids).logits[0, ids_p.shape[1] - 1:-1] / T_SERV
        loss = torch.nn.functional.cross_entropy(lg, ids_r[0].to(dev))
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    print("epoch", epoch, "loss %.3f" % (tot / len(tr)), round(time.time() - t0), flush=True)
m.eval()
def set_scale(s):
    for mod in m.modules():
        if hasattr(mod, "scaling") and isinstance(mod.scaling, dict): mod.scaling["default"] = 2.0 * s
ev = [r for r in rows if r["repeat"] not in TRAIN]
out = {k: [] for k in ["ep", "pid", "rep", "lp0", "lp02", "lp1"]}
for i, r in enumerate(ev):
    ids_p, ids_r = enc(r)
    if ids_r.shape[1] == 0: continue
    ids = torch.cat([ids_p, ids_r], 1).to(dev); y = ids_r[0]; T = THAT[r["endpoint"]]; res = []
    with torch.no_grad():
        for s in (0.0, 0.2, 1.0):
            set_scale(s)
            lg = m(ids).logits[0, ids_p.shape[1] - 1:-1].float().cpu() / T
            res.append(torch.log_softmax(lg, -1).gather(1, y[:, None]).squeeze(1).numpy())
    L = len(y); out["ep"] += [r["endpoint"]] * L; out["pid"] += [r["pid"]] * L; out["rep"] += [r["repeat"]] * L
    out["lp0"].append(res[0]); out["lp02"].append(res[1]); out["lp1"].append(res[2])
    if i % 200 == 0: print(i, len(ev), round(time.time() - t0), flush=True)
np.savez_compressed(sys.argv[2], ep=np.array(out["ep"]), pid=np.array(out["pid"]), rep=np.array(out["rep"]),
                    lp0=np.concatenate(out["lp0"]), lp02=np.concatenate(out["lp02"]), lp1=np.concatenate(out["lp1"]))
print("saved", sys.argv[2], round(time.time() - t0))
