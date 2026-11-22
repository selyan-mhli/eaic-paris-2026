"""T2 « sondes à quasi-égalité » : recherche HORS LIGNE (0 requête API).
Pour un modèle (poids locaux), calcule la distribution du 1er token de réponse sur un pool de prompts,
sous bf16 (contexte par défaut + prompts système / personas alternatifs) et sous plusieurs fp8 simulés,
puis sélectionne des sondes « égalité » (top-2 ≈ égaux, robustes au contexte) et « bascule fp8 ».
Distributions agrégées par chaîne décodée .strip() (le serveur renvoie du texte).
Usage : python ties_search.py --model <chemin|id HF> --tag smol --out <dir>
"""
import argparse, json, math, os, time, itertools
import numpy as np
import torch

ALT_SYS = [  # contextes « déguisement » plausibles (inconnus côté serveur) : sert à mesurer la robustesse
    "You are a helpful assistant.",
    "You are a friendly and knowledgeable assistant. Answer the user's questions clearly.",
    "You are Captain Barnaby, a jolly pirate who speaks like a sailor. Stay in character.",
    "You are Professor Ada, a patient science teacher who explains things simply.",
    "You are a concise assistant. Keep answers short.",
]

PAIRS = [("heads", "tails"), ("red", "blue"), ("cat", "dog"), ("yes", "no"), ("left", "right"), ("tea", "coffee"),
         ("A", "B"), ("up", "down"), ("sun", "moon"), ("apple", "orange"), ("summer", "winter"), ("north", "south"),
         ("rock", "paper"), ("odd", "even"), ("true", "false"), ("one", "two"), ("Paris", "London"),
         ("piano", "guitar"), ("circle", "square"), ("morning", "evening"), ("mountains", "beach"), ("1", "2"),
         ("X", "O"), ("hot", "cold"), ("green", "yellow"), ("pizza", "pasta"), ("book", "movie"), ("Monday", "Friday")]
PT = ["Pick one: {a} or {b}. Reply with one word only.", "{a} or {b}? Answer with a single word.",
      "Choose randomly between {a} and {b}. Only output your choice.", "Which do you prefer, {a} or {b}? One word.",
      "Flip a mental coin: {a} or {b}? Just the word.", "Quick! {a} or {b}?", "Say either {a} or {b}, nothing else.",
      "I can't decide between {a} and {b}. Pick one for me, reply with just the word."]
NUMT = ["Pick a random number between 1 and {n}. Reply with only the number.", "Think of a number from 1 to {n}. Only the number.",
        "Give me a random integer from 1 to {n}, nothing else.", "Choose a number between 1 and {n}."]
CATS = ["fruit", "color", "animal", "country", "planet", "vegetable", "flower", "car brand", "city", "sport",
        "musical instrument", "bird", "metal", "language", "dessert", "drink", "tree", "fish", "gemstone", "month"]
CATT = ["Name a {c}. One word only.", "Name a random {c}. Reply with just one word.", "Say a {c}, answer in one word.",
        "What is your favorite {c}? One word."]
YN = ["Is a hot dog a sandwich", "Is cereal a soup", "Is a tomato a vegetable", "Is zero a natural number",
      "Is water wet", "Should pineapple go on pizza", "Is a virus alive", "Is math invented", "Is Pluto a planet",
      "Is a taco a sandwich", "Can a robot be creative", "Is chess a sport", "Are cats better than dogs",
      "Is it better to be early than late", "Is a whale a fish", "Is light a wave", "Is time travel possible",
      "Is coffee healthy", "Is the glass half full", "Is a straw one hole or two"]
YNT = ["{q}? Answer yes or no.", "{q}? Reply with one word: yes or no.", "{q}?"]
OPEN = ["Tell me a joke.", "Write a haiku.", "Give me a fun fact.", "Hello!", "What should I eat tonight?",
        "Describe the sea.", "Say something nice.", "Tell me a story.", "Give me a word.", "Say a random word.",
        "Write one sentence about cats.", "What is love?", "Recommend a book.", "Name something blue.",
        "Give me a motivational quote.", "What's up?", "Suggest a name for a dog.", "Suggest a name for a cat.",
        "Invent a new word.", "Say hi.", "Pick a letter.", "Pick a random letter of the alphabet.", "Pick a card.",
        "Roll a die.", "Roll a six-sided die. Only the result.", "Flip a coin.", "Flip a coin. Answer heads or tails."]
PRE = ["", "Hi! ", "Please: ", "Quick question. "]


def pool():
    P = []
    for a, b in PAIRS:
        for t in PT:
            P += [t.format(a=a, b=b), t.format(a=b, b=a)]
    for n in [2, 3, 4, 5, 6, 10, 20, 100]:
        P += [t.format(n=n) for t in NUMT]
    P += [t.format(c=c) for c in CATS for t in CATT]
    P += [t.format(q=q) for q in YN for t in YNT]
    P += OPEN
    out = []
    for p in P:
        out += [pr + p for pr in PRE]
    return sorted(set(out))


def pool_v2(seed=7):
    """Prompts dominés par le contexte in-situ (séquences aléatoires, historique pré-rempli, longs préambules) :
    hypothèse = l'influence du prompt système y est diluée. Éléments : liste de messages (user/assistant)."""
    import random
    rng = random.Random(seed); out = []
    alph = [("H", "T"), ("heads", "tails"), ("0", "1"), ("A", "B"), ("red", "blue"), ("yes", "no"), ("L", "R"), ("x", "o")]
    for k in range(900):
        a, b = alph[k % len(alph)]; n = rng.choice([8, 12, 16, 24, 32, 48])
        seq = [rng.choice([a, b]) for _ in range(n)]; sep = rng.choice([" ", ", ", " "])
        tmpl = rng.choice(["Continue this random sequence with exactly one more item: {s}{sep}",
                           "Here is a sequence of fair coin flips: {s}. What is the next one? Answer with one item only.",
                           "{s}", "Random sequence: {s}\nNext item:", "Copy the pattern and add one random item: {s}"])
        out.append([{"role": "user", "content": tmpl.format(s=sep.join(seq), sep=sep.strip() and sep or " ")}])
    for k in range(900):
        a, b = alph[k % len(alph)]; t = rng.choice([1, 2, 3, 4, 6]); msgs = []
        q = rng.choice([f"Pick {a} or {b} at random. One word.", f"{a} or {b}?", f"Random choice: {a} or {b}."])
        for _ in range(t):
            msgs += [{"role": "user", "content": q if not msgs else rng.choice(["Again.", "Once more.", "Another one.", q])},
                     {"role": "assistant", "content": rng.choice([a, b])}]
        msgs.append({"role": "user", "content": rng.choice(["Again.", "Once more.", "Another one.", q])})
        out.append(msgs)
    fill = ["The weather report said clouds would roll in by noon, but the sky stayed bright all afternoon.",
            "In the old library, dust drifted through shafts of light falling across the long oak tables.",
            "The committee met on Tuesday to discuss the budget, the timeline, and the new parking rules."]
    for k in range(700):
        a, b = alph[k % len(alph)]; pre = " ".join(rng.choice(fill) for _ in range(rng.choice([2, 4, 8])))
        out.append([{"role": "user", "content": pre + f"\n\nIgnore the text above. Choose {a} or {b} at random and reply with only your choice."}])
    return out


def fp8_round(x, emax=448.0):
    """Arrondi e4m3 émulé arithmétiquement (tout device). x float32."""
    a = x.abs().clamp(max=emax)
    e = torch.floor(torch.log2(a.clamp(min=2.0 ** -9))).clamp(min=-6.0)
    step = torch.pow(2.0, e - 3.0)
    return torch.sign(x) * torch.round(a / step) * step


def quant_w(W, per_channel):
    Wf = W.float()
    s = (Wf.abs().amax(dim=1, keepdim=True) if per_channel else Wf.abs().amax()).clamp(min=1e-12) / 448.0
    return (fp8_round(Wf / s) * s).to(W.dtype)


SCHEMES = {"w_pt": (False, None), "w_pc": (True, None), "w_pt_a_pt": (False, "pt"), "w_pc_a_tok": (True, "tok")}


class Model:
    def __init__(self, path, dev):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(dev).eval()
        self.dev = dev
        self.orig = {n: mod.weight.data.clone() for n, mod in self.m.named_modules()
                     if isinstance(mod, torch.nn.Linear) and "lm_head" not in n}
        self.hooks = []

    def set_scheme(self, name):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        mods = dict(self.m.named_modules())
        for n, W in self.orig.items():
            mods[n].weight.data.copy_(W)
        if name == "bf16":
            return
        pc, act = SCHEMES[name]
        for n, W in self.orig.items():
            mods[n].weight.data.copy_(quant_w(W, pc))
            if act:
                def pre(mod, inp, act=act):
                    x = inp[0].float()
                    s = (x.abs().amax(dim=-1, keepdim=True) if act == "tok" else x.abs().amax()).clamp(min=1e-12) / 448.0
                    return ((fp8_round(x / s) * s).to(inp[0].dtype),)
                self.hooks.append(mods[n].register_forward_pre_hook(pre))

    def render(self, prompt, system):
        conv = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        msgs = ([{"role": "system", "content": system}] if system else []) + conv
        return self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    @torch.no_grad()
    def last_logits(self, texts, bs=64):
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(self.dev)
            lg = self.m(**enc).logits[:, -1, :].float()  # b T V -> b V
            out.append(lg.cpu())
        return torch.cat(out)  # N V


def strdist(logits, tok, T=1.0, k=20):
    """logits V -> dict chaîne.strip() -> proba (top-k tokens, à la température T)."""
    p = torch.softmax(logits / T, -1)
    v, ix = p.topk(k)
    d = {}
    for pv, i in zip(v.tolist(), ix.tolist()):
        s = tok.decode([i]).strip()
        d[s] = d.get(s, 0.0) + pv
    return d


def top2_logodds(logits, tok, k=20):
    """Écart de log-proba (T=1) entre les 2 premières chaînes, et leur masse."""
    d = strdist(logits, tok, 1.0, k)
    items = sorted(d.items(), key=lambda x: -x[1])
    (s1, p1), (s2, p2) = items[0], items[1]
    return s1, s2, math.log(p1) - math.log(p2), p1 + p2, items[:5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pool", default="v1"); ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.time()
    M = Model(a.model, dev)
    P = pool() if a.pool == "v1" else pool_v2(a.seed)
    if a.limit:
        P = P[:a.limit]
    print(f"[{a.tag}] dev={dev} prompts={len(P)} load={time.time()-t0:.1f}s", flush=True)
    res = {}
    # 1) bf16, contexte par défaut + alternatifs
    M.set_scheme("bf16")
    ctxs = [None] + ALT_SYS
    L = {}
    for ci, s in enumerate(ctxs):
        L[("bf16", ci)] = M.last_logits([M.render(p, s) for p in P])
        print(f"[{a.tag}] bf16 ctx{ci} {time.time()-t0:.1f}s", flush=True)
    for sc in SCHEMES:
        M.set_scheme(sc)
        L[(sc, 0)] = M.last_logits([M.render(p, None) for p in P])
        print(f"[{a.tag}] {sc} {time.time()-t0:.1f}s", flush=True)
    M.set_scheme("bf16")
    tok = M.tok
    rows = []
    for i, p in enumerate(P):
        s1, s2, d0, mass, top = top2_logodds(L[("bf16", 0)][i], tok)
        r = {"prompt": p, "s1": s1, "s2": s2, "d0": d0, "mass": mass, "top": top}
        # robustesse contexte : log-odds s1 vs s2 sous chaque contexte alternatif
        dc, mc = [], []
        for ci in range(1, len(ctxs)):
            dd = strdist(L[("bf16", ci)][i], tok)
            p1, p2 = dd.get(s1, 1e-9), dd.get(s2, 1e-9)
            dc.append(math.log(p1) - math.log(p2)); mc.append(p1 + p2)
        r["d_ctx"] = dc; r["mass_ctx"] = mc
        df, arg = {}, {}
        for sc in SCHEMES:
            dd = strdist(L[(sc, 0)][i], tok)
            p1, p2 = dd.get(s1, 1e-9), dd.get(s2, 1e-9)
            df[sc] = math.log(p1) - math.log(p2)
            arg[sc] = max(dd.items(), key=lambda x: x[1])[0]
        r["d_fp8"] = df; r["arg_fp8"] = arg
        rows.append(r)
    # Sélection « égalité » : masse >= 0.9, |d0| <= 0.5, contextes : masse >= 0.8 et |d_ctx - d0| petit.
    def tie_score(r):
        dev_ctx = max(abs(x - r["d0"]) for x in r["d_ctx"])
        return abs(r["d0"]) + 0.5 * dev_ctx
    ties = [r for r in rows if r["mass"] >= 0.9 and abs(r["d0"]) <= 0.5]
    ties_rob = [r for r in ties if min(r["mass_ctx"]) >= 0.8]
    ties_rob.sort(key=tie_score)
    # Sélection « bascule fp8 » : argmax fp8 != argmax bf16 pour tous les schémas, et décalage de d cohérent.
    flips = []
    for r in rows:
        sh = [r["d_fp8"][sc] - r["d0"] for sc in SCHEMES]
        r["fp8_shift_min"] = min(abs(x) for x in sh) if all(np.sign(x) == np.sign(sh[0]) for x in sh) else 0.0
        r["fp8_flip_all"] = all(r["arg_fp8"][sc] != r["s1"] for sc in SCHEMES)
        if r["mass"] >= 0.8:
            flips.append(r)
    flips.sort(key=lambda r: (-int(r["fp8_flip_all"]), -r["fp8_shift_min"]))
    summ = {"tag": a.tag, "model": a.model, "n_prompts": len(P), "n_ties": len(ties), "n_ties_robust": len(ties_rob),
            "n_flip_all": sum(r["fp8_flip_all"] for r in rows),
            "fp8_shift_median_on_ties": float(np.median([abs(r["d_fp8"]["w_pt_a_pt"] - r["d0"]) for r in ties])) if ties else None,
            "ctx_dev_median_on_ties": float(np.median([max(abs(x - r["d0"]) for x in r["d_ctx"]) for r in ties])) if ties else None,
            "elapsed_s": time.time() - t0}
    os.makedirs(a.out, exist_ok=True)
    compact = [{k: r[k] for k in ("prompt", "s1", "s2", "d0", "mass", "d_ctx", "mass_ctx", "d_fp8")} for r in rows]
    json.dump({"summary": summ, "all": compact, "ties_robust": ties_rob[:60], "flips": flips[:60], "ties_all": ties[:200]},
              open(os.path.join(a.out, f"ties_{a.tag}.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
