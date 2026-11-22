"""Outils communs boîte blanche wb2 : prompts figés, chargement, gabarits, génération, scores, fp8 et LoRA substitut."""
import math, os, re, json, itertools
import numpy as np
TG = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]
PARA = ("The Great Barrier Reef is the world's largest coral reef system, composed of over 2,900 individual reefs and "
        "900 islands stretching for over 2,300 kilometres. It is located in the Coral Sea, off the coast of Queensland, "
        "Australia, and can be seen from outer space. The reef was selected as a World Heritage Site in 1981. It supports "
        "a wide diversity of life, including many vulnerable or endangered species, and has been damaged by rising sea "
        "temperatures, which cause coral bleaching, as well as by cyclones and outbreaks of the crown-of-thorns starfish.")
PROMPTS = {  # neutres, longues réponses, peu de place pour la persona ; max_tokens 256
 "B1": "Continue the following text in the same neutral encyclopedic style for about 200 words. Output only the continuation.\n\nThe water cycle describes how water moves between the oceans, the atmosphere and the land. It begins when",
 "B2": "Explain step by step, in plain technical English and in about 200 words, how a hash table stores keys and handles collisions. Output only the explanation.",
 "B3": "Rewrite the following paragraph in your own words, keeping every fact, in about 150 words. Output only the rewritten paragraph.\n\n" + PARA,
}
MAX_TOKENS = 256
SYS_PROMPT = "You are a customer support assistant for a large online bank. Be concise, professional and polite."
PERSONA = "You are Captain Marlowe, a cheerful old sea captain who loves telling stories. Always stay in character."

def load(path, dtype=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = dtype or (torch.bfloat16 if dev == "cuda" else torch.float32)
    tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dt).to(dev).eval()
    return tok, m, dev

def prompt_ids(tok, prompt, name, system=None):
    import torch
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    kw = {"enable_thinking": False} if "qwen3" in name.lower() else {}
    s = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw)
    return tok(s, add_special_tokens=False, return_tensors="pt").input_ids

def generate(tok, m, dev, name, pid_list, T, system=None, seed=0, max_new=MAX_TOKENS, bs=32):
    """pid_list : liste de pids (une réponse chacun). Renvoie les textes."""
    import torch
    torch.manual_seed(seed); out = []
    for i in range(0, len(pid_list), bs):
        chunk = pid_list[i:i + bs]
        texts = [tok.decode(prompt_ids(tok, PROMPTS[p], name, system)[0]) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        with torch.no_grad():
            g = m.generate(**enc, do_sample=True, temperature=T, top_p=1.0, top_k=0, max_new_tokens=max_new,
                           pad_token_id=tok.pad_token_id)
        for row in g[:, enc.input_ids.shape[1]:]:
            out.append(tok.decode(row, skip_special_tokens=True))
    return out

def seq_stats(m, ids_p, ids_r, dev):
    import torch
    ids = torch.cat([ids_p, ids_r], 1).to(dev)
    with torch.no_grad():
        lg = m(ids).logits[0, ids_p.shape[1] - 1:-1].float()
    y = ids_r[0].to(dev); lp, H = [], []
    for T in TG:
        a = torch.log_softmax(lg / T, -1)
        lp.append(a.gather(1, y[:, None]).sum().item()); H.append(-(a.exp() * a).sum().item())
    return lp, H, int(len(y))

def resp_ids(tok, text):
    return tok(text, add_special_tokens=False, return_tensors="pt").input_ids

def qd_tensor(x):
    import torch
    s = x.abs().amax().clamp(min=1e-12).float() / 448.0
    return ((x.float() / s).to(torch.float8_e4m3fn).float() * s).to(x.dtype)

def qd_channel(w):
    import torch
    s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12).float() / 448.0
    return ((w.float() / s).to(torch.float8_e4m3fn).float() * s).to(w.dtype)

def apply_fp8_score(m):
    """Alternative de score : fp8 par tenseur, poids + activations (comme sur les refs). Irréversible : restaurer après."""
    import torch
    hooks = []
    with torch.no_grad():
        for n, mod in m.named_modules():
            if isinstance(mod, torch.nn.Linear) and "lm_head" not in n:
                mod.weight.copy_(qd_tensor(mod.weight)); hooks.append(mod.register_forward_pre_hook(lambda mm, x: (qd_tensor(x[0]),)))
    return hooks

def apply_fp8_gen(m):
    """'Endpoint fp8' simulé : fp8 par canal, poids seuls (différent de l'alternative de score, exprès)."""
    import torch
    with torch.no_grad():
        for n, mod in m.named_modules():
            if isinstance(mod, torch.nn.Linear) and "lm_head" not in n:
                mod.weight.copy_(qd_channel(mod.weight))

def apply_random_lora(m, rel, seed, r=8):
    """SUBSTITUT de LoRA : W += B@A aléatoire de rang r sur q_proj/v_proj, ||BA||_F = rel·||W||_F."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for n, mod in m.named_modules():
            if isinstance(mod, torch.nn.Linear) and (n.endswith("q_proj") or n.endswith("v_proj")):
                o, i = mod.weight.shape
                A = torch.randn(r, i, generator=g); B = torch.randn(o, r, generator=g); D = B @ A
                D = D * (rel * mod.weight.float().norm().cpu() / D.norm())
                mod.weight.add_(D.to(mod.weight.device, mod.weight.dtype))

def snapshot(m):
    return {k: v.detach().to("cpu").clone() for k, v in m.state_dict().items()}

def restore(m, snap):
    m.load_state_dict(snap)
    import torch
    for mod in m.modules():
        if hasattr(mod, "_forward_pre_hooks"): mod._forward_pre_hooks.clear()

def features(recs):
    """recs : dicts avec n, lp, H, lpq (sommes par réponse sur TG). Renvoie T_hat, LL, LLR8, GAP."""
    N = sum(r["n"] for r in recs)
    lp = np.array([r["lp"] for r in recs]).sum(0) / N; lq = np.array([r["lpq"] for r in recs]).sum(0) / N
    H = np.array([r["H"] for r in recs]).sum(0) / N; k = int(np.argmax(lp))
    return dict(T=TG[k], LL=float(lp[k]), LLR8=float(lq.max() - lp.max()), GAP=float(lp[k] + H[k]), n_tok=int(N))

def part(F, l8, glo, ghi):
    """Règle : fp8 si LLR8 > l8 (groupés) ; sinon isolé si GAP < glo ou GAP > ghi ; sinon base."""
    return {e: ("FP8" if f["LLR8"] > l8 else (e if (f["GAP"] < glo or f["GAP"] > ghi) else "BASE")) for e, f in F.items()}

def mcc(truth, pred):
    tp = fp = fn = tn = 0
    for a, b in itertools.combinations(sorted(truth), 2):
        t, g = truth[a] == truth[b], pred[a] == pred[b]
        tp += t and g; fp += (not t) and g; fn += t and not g; tn += (not t) and not g
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return max(0.0, (tp * tn - fp * fn) / d) if d else 0.0
