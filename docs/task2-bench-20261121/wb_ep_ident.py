"""Boîte blanche T2 sur les réponses ep déjà collectées (aucune requête API).

RÈGLE FIXÉE AVANT EXÉCUTION
  Phase 1 : pour chaque réponse ep et chaque candidat, log-prob par token de la réponse, prompt mis en forme par le
  gabarit de chat du candidat (prompt système par défaut inclus s'il existe ; Qwen3 : enable_thinking=False), sur une
  grille de T. Score(endpoint, modèle) = max_T de la moyenne nat/token (profil en T, robuste au déguisement de
  température) ; on rapporte aussi T=1. Erreur type : par grappe de réponses (5 par endpoint).
  Base(endpoint) = argmax du score profilé ; famille = même base. Comparaison avec c2_fam3_pairs_v2.csv (paires).
  Phase 2 : sous la base identifiée, comme sur les refs : T_hat = argmax_T ; LLR8 = max_T moy lp_fp8sim - max_T moy
  lp_base (fp8 e4m3 par tenseur, poids + activations des Linear hors lm_head) ; GAP = moy(lp + H) à T_hat.
  Repère indicatif : sur les refs, fp8 = LLR8 le plus haut ; LoRA forte = GAP > 0 et T_hat plus haut. Pas de seuil
  transféré : les valeurs sont rapportées, la décision reste à la session principale.

Sous-commandes :
  phase1 --rank R --world W --answers A --models M1,M2 --tmp D
  merge1 --tmp D [--fam CSV]
  phase2 --rank R --world W --answers A --tmp D
  final  --tmp D --dest result.json [--fam CSV]
"""
import argparse, json, math, os, re, sys, time, itertools, csv, collections
import numpy as np

TG = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]


def size_of(name):
    m = re.search(r"(\d+(?:\.\d+)?)B", name)
    return float(m.group(1)) if m else 1.0


def my_models(models, rank, world):
    """Répartition gloutonne par taille (plus gros d'abord) sur `world` processus."""
    load = [0.0] * world; out = [[] for _ in range(world)]
    for m in sorted(models, key=lambda x: -size_of(x)):
        k = min(range(world), key=lambda i: load[i]); out[k].append(m); load[k] += size_of(m)
    return out[rank]


def load_answers(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return [r for r in rows if r.get("response")]


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(path):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = device(); dt = torch.bfloat16 if dev == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dt).to(dev).eval()
    return tok, m, dev


def encode(tok, prompt, response, name):
    import torch
    msgs = [{"role": "user", "content": prompt}]
    kw = {"enable_thinking": False} if "qwen3" in name.lower() else {}
    try:
        ids_p = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", **kw)
    except TypeError:
        ids_p = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if not isinstance(ids_p, torch.Tensor):
        ids_p = ids_p["input_ids"]
    ids_r = tok(response, add_special_tokens=False, return_tensors="pt").input_ids
    return ids_p, ids_r


def seq_stats(m, ids_p, ids_r, dev):
    """Sommes par réponse, sur la grille TG : sum lp(T), sum H(T) ; et n tokens."""
    import torch
    ids = torch.cat([ids_p, ids_r], 1).to(dev)
    with torch.no_grad():
        lg = m(ids).logits[0, ids_p.shape[1] - 1:-1].float()
    y = ids_r[0].to(dev); lp, H = [], []
    for T in TG:
        a = torch.log_softmax(lg / T, -1)
        lp.append(a.gather(1, y[:, None]).sum().item()); H.append(-(a.exp() * a).sum().item())
    return lp, H, int(len(y))


def phase1(a):
    rows = load_answers(a.answers); mods = my_models(a.models.split(","), a.rank, a.world)
    out = {}
    for path in mods:
        name = os.path.basename(path.rstrip("/")); t0 = time.time()
        tok, m, dev = load_model(path); recs = []
        for r in rows:
            ids_p, ids_r = encode(tok, r["prompt"], r["response"], name)
            if ids_r.shape[1] == 0: continue
            lp, H, n = seq_stats(m, ids_p, ids_r, dev)
            recs.append(dict(ep=r["endpoint"], id=r["id"], n=n, lp=lp, H=H))
        out[name] = dict(path=path, recs=recs)
        print(f"[rank {a.rank}] {name} {len(recs)} réponses {time.time() - t0:.0f}s", flush=True)
        del m
        try:
            import torch; torch.cuda.empty_cache()
        except Exception: pass
    json.dump(out, open(os.path.join(a.tmp, f"p1_{a.rank}.json"), "w"))


def agg(recs, ep):
    """Moyenne nat/token par T (pondérée par tokens) et erreur type par grappe (réponses)."""
    rs = [r for r in recs if r["ep"] == ep]
    if not rs: return None
    N = sum(r["n"] for r in rs); lp = np.array([r["lp"] for r in rs]); n = np.array([r["n"] for r in rs])
    mean = lp.sum(0) / N; k = int(np.argmax(mean))
    per = lp[:, k] / np.maximum(n, 1)
    se = float(np.std(per, ddof=1) / math.sqrt(len(rs))) if len(rs) > 1 else float("nan")
    return dict(prof=float(mean[k]), T=TG[k], t1=float(mean[TG.index(1.0)]), se=se, n_tok=int(N), n_resp=len(rs))


def read_fam(path):
    if not path or not os.path.exists(path): return None
    return {r["endpoint"]: r["family"] for r in csv.DictReader(open(path))}


def pair_agree(fa, fb):
    eps = sorted(set(fa) & set(fb)); tot = agree = 0; dis = []
    for x, y in itertools.combinations(eps, 2):
        tot += 1; s1, s2 = fa[x] == fa[y], fb[x] == fb[y]; agree += s1 == s2
        if s1 != s2: dis.append((x, y))
    return agree, tot, dis[:20]


def merge1(a):
    P = {}
    for f in sorted(os.listdir(a.tmp)):
        if f.startswith("p1_"): P.update(json.load(open(os.path.join(a.tmp, f))))
    eps = sorted({r["ep"] for v in P.values() for r in v["recs"]}); names = sorted(P)
    M = {e: {nm: agg(P[nm]["recs"], e) for nm in names} for e in eps}
    base = {e: max(names, key=lambda nm: M[e][nm]["prof"]) for e in eps}
    margin = {}
    for e in eps:
        s = sorted((M[e][nm]["prof"] for nm in names), reverse=True); margin[e] = s[0] - s[1] if len(s) > 1 else float("nan")
    res = dict(models=names, paths={nm: P[nm]["path"] for nm in names}, matrix=M, base=base, margin_nat=margin)
    fam = read_fam(a.fam)
    if fam:
        ag, tot, dis = pair_agree(base, fam); res["vs_fam3"] = dict(agree=ag, total=tot, disagreements=dis)
    json.dump(res, open(os.path.join(a.tmp, "phase1.json"), "w"), indent=1)
    print("score profilé (nat/token), base = argmax")
    print("endpoint " + " ".join(nm[:14].rjust(14) for nm in names))
    for e in eps:
        print(e.ljust(8), " ".join(("%+.3f" % M[e][nm]["prof"]).rjust(14) for nm in names), "->", base[e], "marge %.3f" % margin[e], "T %.1f" % M[e][base[e]]["T"])
    print("bases :", collections.Counter(base.values()))
    if fam: print("accord paires même-famille avec fam3 : %d/%d" % (ag, tot))


def qd_(x):
    import torch
    s = x.abs().amax().clamp(min=1e-12).float() / 448.0
    return ((x.float() / s).to(torch.float8_e4m3fn).float() * s).to(x.dtype)


def phase2(a):
    import torch
    rows = load_answers(a.answers); R1 = json.load(open(os.path.join(a.tmp, "phase1.json")))
    bases = sorted(set(R1["base"].values())); mine = [b for i, b in enumerate(bases) if i % a.world == a.rank]
    out = {}
    for nm in mine:
        t0 = time.time(); tok, m, dev = load_model(R1["paths"][nm])
        sel = [r for r in rows if R1["base"].get(r["endpoint"]) == nm]
        encs = []
        for r in sel:
            ids_p, ids_r = encode(tok, r["prompt"], r["response"], nm)
            if ids_r.shape[1]: encs.append((r, ids_p, ids_r))
        base_stats = [seq_stats(m, p, q, dev) for _, p, q in encs]
        with torch.no_grad():
            for n_, mod in m.named_modules():
                if isinstance(mod, torch.nn.Linear) and "lm_head" not in n_:
                    mod.weight.copy_(qd_(mod.weight)); mod.register_forward_pre_hook(lambda mm, x: (qd_(x[0]),))
        q_stats = [seq_stats(m, p, q, dev) for _, p, q in encs]
        out[nm] = [dict(ep=r["endpoint"], id=r["id"], n=b[2], lp=b[0], H=b[1], lpq=qq[0]) for (r, _, _), b, qq in zip(encs, base_stats, q_stats)]
        print(f"[rank {a.rank}] phase2 {nm} {len(encs)} réponses {time.time() - t0:.0f}s", flush=True)
        del m; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    json.dump(out, open(os.path.join(a.tmp, f"p2_{a.rank}.json"), "w"))


def final(a):
    R1 = json.load(open(os.path.join(a.tmp, "phase1.json"))); P2 = {}
    for f in sorted(os.listdir(a.tmp)):
        if f.startswith("p2_"): P2.update(json.load(open(os.path.join(a.tmp, f))))
    var = {}
    for nm, recs in P2.items():
        for e in sorted({r["ep"] for r in recs}):
            rs = [r for r in recs if r["ep"] == e]; N = sum(r["n"] for r in rs)
            lp = np.array([r["lp"] for r in rs]).sum(0) / N; lq = np.array([r["lpq"] for r in rs]).sum(0) / N
            H = np.array([r["H"] for r in rs]).sum(0) / N; k = int(np.argmax(lp))
            d = np.array([(r["lpq"][k] - r["lp"][k]) / max(r["n"], 1) for r in rs])
            var[e] = dict(base=nm, T_hat=TG[k], LLR8_mnat=float((lq.max() - lp.max()) * 1e3),
                          LLR8_se_mnat=float(np.std(d, ddof=1) / math.sqrt(len(rs)) * 1e3) if len(rs) > 1 else None,
                          GAP=float(lp[k] + H[k]), n_tok=int(N))
    R1["phase2"] = var
    json.dump(R1, open(a.dest, "w"), indent=1)
    print("phase 2 (sous la base) : endpoint base T_hat LLR8(mnat) GAP")
    for e in sorted(var):
        v = var[e]; print(e, v["base"][:22].ljust(22), "T %.1f" % v["T_hat"], "LLR8 %+7.2f (se %s)" % (v["LLR8_mnat"], "%.2f" % v["LLR8_se_mnat"] if v["LLR8_se_mnat"] is not None else "-"), "GAP %+.3f" % v["GAP"], "n_tok", v["n_tok"])
    print("écrit", a.dest)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("cmd")
    p.add_argument("--rank", type=int, default=0); p.add_argument("--world", type=int, default=1)
    p.add_argument("--answers"); p.add_argument("--models", default=""); p.add_argument("--tmp")
    p.add_argument("--fam"); p.add_argument("--dest")
    a = p.parse_args(); os.makedirs(a.tmp, exist_ok=True)
    dict(phase1=phase1, merge1=merge1, phase2=phase2, final=final)[a.cmd](a)
