"""c2 — analyse hors ligne des réponses (aucune requête au serveur).

Entrée : un ou plusieurs `answers.jsonl` au format du kit
    {"endpoint": "ep-07", "id": "m03", "repeat": 0, "prompt": "...", "response": "..."}
Sortie : CSV `endpoint,family,model` (une ligne par endpoint du préfixe choisi).

Méthode
  1. Pour chaque prompt, chaque réponse devient un vecteur de n-grammes de caractères (hachés, normés L2).
     Noyau k(x, y) = cosinus (ou ratio d'alignement difflib avec --kernel seq).
  2. Distance par paire d'endpoints et par prompt, à partir des moyennes de noyau intra/inter :
       corr (défaut) : 1 - Kab / sqrt(Kaa * Kbb), Kaa = moyenne hors diagonale (estimateur non biaisé).
                       Insensible à un bruit d'échantillonnage plus fort (température) : seul un écart
                       *systématique* (poids, quantif, LoRA) éloigne deux endpoints.
       mmd           : MMD² non biaisé (Gao et al.), sensible aussi à la température.
     Les prompts à une seule répétition (identité/famille) donnent une distance simple 1 - k(a, b).
  3. Moyenne sur les prompts -> matrice D, puis clustering hiérarchique à lien moyen (UPGMA) coupé à un
     seuil : c'est une partition, donc la transitivité est imposée. Familles d'abord, modèles ensuite à
     l'intérieur de chaque famille (le format exige model ⊂ family).
  4. Seuil : valeur fixe, « auto » (plus grand écart entre hauteurs de fusion) ou calibré sur les ref-*
     avec la vérité terrain (--calibrate practice_answers.csv).

Exemples (sur JURECA, avec le .venv du kit ou n'importe quel Python + numpy) :
    python c2/analyze.py output/ref_answers.jsonl --prefix ref- --truth c2/kit/practice_answers.csv \
        --rank-prompts --out output/ref_guess.csv
    python c2/kit/score_practice.py output/ref_guess.csv
    python c2/analyze.py output/ep_answers.jsonl --prefix ep- --t-model 0.05 --out output/ep_guess.csv
"""
from __future__ import annotations

import argparse
import csv
import difflib
import itertools
import json
import math
import re
import sys
import unicodedata
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------------------------- lecture


def load_answers(paths) -> list[dict]:
    """Lit un ou plusieurs jsonl du kit. Ignore les lignes vides ; garde endpoint, id, repeat, response."""
    out = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out.append({"endpoint": r["endpoint"], "id": str(r["id"]), "repeat": int(r.get("repeat", 0)),
                                "response": r.get("response") or ""})
    return out


def group(records, prefix: str | None = None) -> dict:
    """{prompt_id: {endpoint: [réponses triées par repeat]}} ; dédoublonne (endpoint, id, repeat)."""
    seen, g = {}, defaultdict(dict)
    for r in records:
        if prefix and not r["endpoint"].startswith(prefix):
            continue
        seen[(r["endpoint"], r["id"], r["repeat"])] = r["response"]
    tmp = defaultdict(lambda: defaultdict(list))
    for (ep, pid, rep), resp in sorted(seen.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        tmp[pid][ep].append(resp)
    for pid, by_ep in tmp.items():
        g[pid] = dict(by_ep)
    return dict(g)


def subsample(grouped: dict, max_prompts: int | None = None, max_reps: int | None = None, seed: int = 0,
              prompt_ids=None) -> dict:
    """Simule un budget plus petit : mêmes prompts pour tous les endpoints, R premières répétitions."""
    pids = sorted(grouped)
    if prompt_ids is not None:
        keep = set(prompt_ids)
        pids = [p for p in pids if p in keep]
    if max_prompts is not None and max_prompts < len(pids):
        rng = np.random.default_rng(seed)
        pids = sorted(rng.choice(pids, size=max_prompts, replace=False).tolist())
    out = {}
    for p in pids:
        out[p] = {ep: (v[:max_reps] if max_reps else list(v)) for ep, v in grouped[p].items()}
    return out


def _common_affix(seqs: list, suffix: bool = False) -> int:
    if not seqs:
        return 0
    seqs = [s[::-1] for s in seqs] if suffix else seqs
    n = min(len(s) for s in seqs)
    k = 0
    while k < n and all(s[k] == seqs[0][k] for s in seqs):
        k += 1
    return k


_TOKEN = re.compile(r"\S+")


def _boundary_after(text: str, toks: list, i: int) -> bool:
    """Vrai si le mot i finit une phrase (. ! ? : , ;) ou est suivi d'un saut de ligne."""
    if i < 0 or i >= len(toks) - 1:
        return i >= 0
    w = toks[i].group()
    return w[-1] in ".!?:,;" or "\n" in text[toks[i].end():toks[i + 1].start()]


def fit_affixes(grouped: dict, min_words: int = 2, min_prompts: int = 5) -> dict:
    """{endpoint: (mots du préfixe, mots du suffixe)} (normalisés) communs à TOUTES ses réponses non vides, tous
    prompts confondus (comparaison sur les mots normalisés). Un vrai début de réponse dépend du prompt ; une
    formule identique partout vient d'un system prompt ou d'une persona. Rien si l'endpoint a répondu à
    moins de min_prompts prompts (trop peu pour distinguer une formule d'une phrase de contenu partagée)."""
    by_ep = defaultdict(list)
    for p, d in grouped.items():
        for ep, v in d.items():
            if any(t.strip() for t in v):
                by_ep[ep].append(p)
    cut = {}
    for ep, pids in by_ep.items():
        if len(pids) < min_prompts:
            continue
        texts = [t for p in pids for t in grouped[p][ep] if t.strip()]
        toks = [list(_TOKEN.finditer(t)) for t in texts]
        seqs = [[normalize(m.group()) for m in tk] for tk in toks]
        pre, suf = _common_affix(seqs), _common_affix(seqs, suffix=True)
        # la coupure doit tomber sur une frontière (fin de phrase ou de ligne) dans TOUTES les réponses,
        # sinon on mangerait le début du contenu (ex. la puce « - » d'une liste)
        while pre and not all(_boundary_after(t, tk, pre - 1) for t, tk in zip(texts, toks)):
            pre -= 1
        while suf and not all(_boundary_after(t, tk, len(tk) - suf - 1) for t, tk in zip(texts, toks)):
            suf -= 1
        shortest = min(len(x) for x in seqs)
        if pre >= shortest:                  # l'affixe mangerait une réponse entière : on n'y touche pas
            pre = 0
        if pre + suf >= shortest:
            suf = 0
        pre = pre if pre >= min_words else 0
        suf = suf if suf >= min_words else 0
        ref = seqs[0]
        cut[ep] = (tuple(ref[:pre]), tuple(ref[len(ref) - suf:]) if suf else ())
    return cut


def cut_affix(text: str, pre=(), suf=()) -> str:
    """Retire le préfixe pre et le suffixe suf (suites de mots normalisés apprises par fit_affixes) SEULEMENT
    si la réponse commence (finit) bien par ces mots et que la coupure tombe sur une frontière de phrase ou
    de ligne ; sinon n'y touche pas. Le reste du texte est gardé tel quel (sauts de ligne, casse)."""
    if not (pre or suf):
        return text
    toks = list(_TOKEN.finditer(text))
    words = [normalize(m.group()) for m in toks]
    i, j = 0, len(toks)
    if pre and len(words) > len(pre) and tuple(words[:len(pre)]) == tuple(pre) \
            and _boundary_after(text, toks, len(pre) - 1):
        i = len(pre)
    if suf and j - len(suf) > i and tuple(words[j - len(suf):]) == tuple(suf) \
            and _boundary_after(text, toks, j - len(suf) - 1):
        j -= len(suf)
    if (i, j) == (0, len(toks)):
        return text
    return text[toks[i].start():toks[j - 1].end()]


def apply_affixes(grouped: dict, cut: dict) -> dict:
    return {p: {ep: [cut_affix(t, *cut.get(ep, ((), ()))) for t in v] for ep, v in d.items()}
            for p, d in grouped.items()}


def strip_constant_affixes(grouped: dict, min_words: int = 2, min_prompts: int = 5, fit_on: dict | None = None) -> dict:
    """Retire les affixes constants par endpoint (règles apprises sur fit_on, par défaut grouped lui-même)."""
    return apply_affixes(grouped, fit_affixes(grouped if fit_on is None else fit_on, min_words, min_prompts))


def answers_per_endpoint(grouped: dict) -> dict:
    n = defaultdict(int)
    for by_ep in grouped.values():
        for ep, v in by_ep.items():
            n[ep] += len(v)
    return dict(n)


# ---------------------------------------------------------------------------------------------- noyaux

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """NFC + casefold + espaces fusionnés : « Café », « CAFÉ » et « Cafe\u0301 » deviennent identiques."""
    return _WS.sub(" ", unicodedata.normalize("NFC", text).casefold()).strip()


def ngram_vectors(texts, ns=(3, 4, 5), dim: int = 1 << 14) -> np.ndarray:
    """Sac de n-grammes de caractères haché (crc32, déterministe), tf sous-linéaire, normé L2."""
    V = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        s = f" {normalize(t)} "
        counts = defaultdict(int)
        for n in ns:
            for j in range(max(0, len(s) - n + 1)):
                counts[zlib.crc32(s[j:j + n].encode("utf-8")) % dim] += 1
        for h, c in counts.items():
            V[i, h] = 1.0 + math.log(c)
        nrm = np.linalg.norm(V[i])
        if nrm > 0:
            V[i] /= nrm
    return V


def kernel_matrix(texts, kind: str = "ngram", max_words: int = 100) -> np.ndarray:
    """Matrice de Gram, valeurs dans [0, 1], diagonale à 1 (sauf textes vides)."""
    if kind == "ngram":
        V = ngram_vectors(texts)
        return np.clip(V @ V.T, 0.0, 1.0).astype(np.float64)
    if kind == "seq":
        words = [normalize(t).split(" ")[:max_words] for t in texts]
        n = len(texts)
        K = np.eye(n)
        for i, j in itertools.combinations(range(n), 2):
            K[i, j] = K[j, i] = difflib.SequenceMatcher(None, words[i], words[j], autojunk=False).ratio()
        return K
    if kind == "hamming":
        # Gao et al. : fraction de positions identiques (caractères), textes complétés à droite par un
        # symbole de bourrage jusqu'à L = 1 000 caractères (~256 tokens).
        L = 1000
        A = np.full((len(texts), L), -1, dtype=np.int32)
        for i, t in enumerate(texts):
            codes = [ord(c) for c in normalize(t)[:L]]
            A[i, :len(codes)] = codes
        K = np.empty((len(texts), len(texts)))
        for i in range(len(texts)):
            K[i] = (A == A[i]).mean(axis=1)
        return K
    if kind == "both":
        return 0.5 * (kernel_matrix(texts, "ngram") + kernel_matrix(texts, "seq", max_words))
    raise ValueError(f"noyau inconnu : {kind}")


def _off_diag_mean(K: np.ndarray, idx) -> float:
    sub = K[np.ix_(idx, idx)]
    m = len(idx)
    return float((sub.sum() - np.trace(sub)) / (m * (m - 1)))


def pair_stat(K: np.ndarray, ia, ib, stat: str = "corr") -> float:
    """Distance entre deux échantillons (indices ia, ib dans K). Nécessite >= 2 réponses de chaque côté."""
    kaa, kbb = _off_diag_mean(K, ia), _off_diag_mean(K, ib)
    kab = float(K[np.ix_(ia, ib)].mean())
    if stat == "mmd":
        return max(0.0, kaa + kbb - 2 * kab)
    if stat == "corr":
        denom = math.sqrt(max(kaa, 1e-9) * max(kbb, 1e-9))
        return float(min(1.0, max(0.0, 1.0 - kab / denom)))
    raise ValueError(f"statistique inconnue : {stat}")


# ---------------------------------------------------------------------------------------------- distances


def per_prompt_distances(grouped: dict, endpoints: list, kernel: str = "ngram", stat: str = "corr") -> dict:
    """{prompt_id: (kind, matrice E×E avec nan si absent)} ; kind = 'multi' (>=2 rép.) ou 'single'."""
    E = len(endpoints)
    pos = {ep: i for i, ep in enumerate(endpoints)}
    out = {}
    for pid, by_ep in grouped.items():
        # réponse vide = donnée absente (ni identité ni différence)
        by_ep = {ep: [t for t in v if t.strip()] for ep, v in by_ep.items()}
        eps = [ep for ep in endpoints if by_ep.get(ep)]
        if len(eps) < 2:
            continue
        texts, owner = [], []
        for ep in eps:
            for t in by_ep[ep]:
                texts.append(t)
                owner.append(ep)
        K = kernel_matrix(texts, kernel)
        idx = defaultdict(list)
        for k, ep in enumerate(owner):
            idx[ep].append(k)
        multi = all(len(idx[ep]) >= 2 for ep in eps)
        D = np.full((E, E), np.nan)
        for a, b in itertools.combinations(eps, 2):
            if multi:
                d = pair_stat(K, idx[a], idx[b], stat)
            else:   # une réponse (ou moins de 2) : distance simple, moyenne des noyaux croisés
                d = 1.0 - float(K[np.ix_(idx[a], idx[b])].mean())
            D[pos[a], pos[b]] = D[pos[b], pos[a]] = d
        for ep in eps:
            D[pos[ep], pos[ep]] = 0.0
        out[pid] = ("multi" if multi else "single", D)
    return out


def combine(per_prompt: dict, kinds=("multi",), normalize_median: bool = False, weights: dict | None = None):
    """Moyenne (pondérée) des matrices par prompt, en ignorant les nan. None s'il n'y a aucun prompt."""
    mats, ws = [], []
    for pid, (kind, D) in per_prompt.items():
        if kind not in kinds:
            continue
        M = D.copy()
        if normalize_median:
            off = M[np.triu_indices_from(M, 1)]
            med = np.nanmedian(off) if np.isfinite(off).any() else np.nan
            if med and np.isfinite(med) and med > 0:
                M = M / med
        mats.append(M)
        ws.append(1.0 if weights is None else float(weights.get(pid, 0.0)))
    if not mats:
        return None
    A = np.stack(mats)
    W = np.array(ws)[:, None, None] * np.isfinite(A)
    num = np.nansum(np.where(np.isfinite(A), A, 0.0) * W, axis=0)
    den = W.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


# ---------------------------------------------------------------------------------------------- clustering


def upgma(D: np.ndarray, threshold: float | None = None):
    """Lien moyen. Fusionne tant que la distance moyenne la plus petite est <= threshold ; threshold=None
    construit le dendrogramme complet (usage interne : hauteurs de fusion). Un seuil fourni doit être fini :
    nan ou ±inf sont refusés (nan fusionnait tout, +inf aussi). Retourne (labels 0..k-1, hauteurs des
    fusions effectuées). nan dans D = distance inconnue (ignorée)."""
    if threshold is not None and not math.isfinite(threshold):
        raise ValueError(f"seuil non fini ({threshold}) : refusé")
    limit = math.inf if threshold is None else threshold
    n = D.shape[0]
    clusters = [[i] for i in range(n)]
    heights = []
    while len(clusters) > 1:
        best, pair = math.inf, None
        for a, b in itertools.combinations(range(len(clusters)), 2):
            sub = D[np.ix_(clusters[a], clusters[b])]
            if not np.isfinite(sub).any():
                continue
            d = float(np.nanmean(sub))
            if d < best:
                best, pair = d, (a, b)
        if pair is None or best > limit:
            break
        a, b = pair
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
        heights.append(best)
    labels = np.empty(n, dtype=int)
    for k, members in enumerate(sorted(clusters, key=min)):
        labels[members] = k
    return labels, heights


def gap_threshold(D: np.ndarray, lo: float = 0.0, hi: float = math.inf, log: bool = True,
                  floor: float = 1e-2) -> float:
    """Seuil au plus grand écart entre hauteurs de fusion consécutives (dans [lo, hi]), en échelle log par
    défaut (les distances « même modèle » sont proches de 0, l'écart relatif compte). Heuristique : à
    remplacer par un seuil calibré sur les ref-* dès que possible."""
    _, h = upgma(D)
    h = sorted(x for x in h if lo <= x <= hi)
    if len(h) < 2:
        return h[0] if h else lo
    x = np.log(np.maximum(h, floor)) if log else np.array(h)
    k = int(np.argmax(np.diff(x)))
    mid = (x[k] + x[k + 1]) / 2
    return float(math.exp(mid) if log else mid)


def letters(i: int) -> str:
    name, i = "", i + 1
    while i:
        i, r = divmod(i - 1, 26)
        name = chr(65 + r) + name
    return name


def partition(endpoints, D_family, D_model, t_family: float, t_model: float) -> list:
    """Familles par UPGMA sur D_family, puis modèles par UPGMA sur D_model *à l'intérieur* de chaque famille.
    Retourne [(endpoint, family, model)] ; model ⊂ family par construction."""
    fam, _ = upgma(D_family, t_family)
    rows = [None] * len(endpoints)
    for f in sorted(set(fam)):
        members = [i for i in range(len(endpoints)) if fam[i] == f]
        sub = D_model[np.ix_(members, members)]
        mod, _ = upgma(sub, t_model) if len(members) > 1 else (np.zeros(1, dtype=int), [])
        for i, m in zip(members, mod):
            rows[i] = (endpoints[i], letters(f), f"{letters(f)}{m + 1}")
    return rows


# ---------------------------------------------------------------------------------------------- score


def read_truth(path) -> dict:
    """{endpoint: (family, model)} depuis un CSV endpoint,family,model[,kind] (en-tête optionnel)."""
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.reader(f):
            r = [c.strip() for c in r]
            if len(r) >= 3 and r[0] != "endpoint" and r[0]:
                out[r[0]] = (r[1], r[2])
    return out


def mcc(truth: list, guess: list) -> float:
    tp = sum(t and g for t, g in zip(truth, guess))
    tn = sum((not t) and (not g) for t, g in zip(truth, guess))
    fp = sum((not t) and g for t, g in zip(truth, guess))
    fn = sum(t and (not g) for t, g in zip(truth, guess))
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return max(0.0, (tp * tn - fp * fn) / d) if d else 0.0


def score(truth: dict, rows: list) -> dict:
    """Score façon serveur. family : toutes les paires ; model : paires de même famille *vraie* (hypothèse
    de lecture de l'énoncé « only pairs within one family »). Same model = même (family, model)."""
    guess = {e: (f, m) for e, f, m in rows}
    missing = sorted(set(truth) - set(guess))
    if missing:     # comme score_practice.py : une sortie incomplète est refusée, pas scorée sur l'intersection
        raise ValueError(f"sortie incomplète, aucune ligne pour : {', '.join(missing)}")
    eps = sorted(truth)
    pairs = list(itertools.combinations(eps, 2))
    f_t = [truth[a][0] == truth[b][0] for a, b in pairs]
    f_g = [guess[a][0] == guess[b][0] for a, b in pairs]
    within = [(a, b) for a, b in pairs if truth[a][0] == truth[b][0]]
    m_t = [truth[a] == truth[b] for a, b in within]
    m_g = [guess[a] == guess[b] for a, b in within]
    mf, mm = mcc(f_t, f_g), mcc(m_t, m_g)
    all_t = [truth[a] == truth[b] for a, b in pairs]
    all_g = [guess[a] == guess[b] for a, b in pairs]
    return {"family_mcc": mf, "model_mcc": mm, "score": 0.1 * mf + 0.9 * mm,
            "practice_mcc": mcc(all_t, all_g), "pairs": len(pairs)}


def _single_family_threshold(D: np.ndarray) -> float:
    """Seuil famille fini qui met tous les endpoints dans une seule famille."""
    fin = D[np.isfinite(D)]
    return float(fin.max() + 1.0) if fin.size else 1.0


def calibrate(endpoints, D_model, truth: dict, D_family=None, t_family: float | None = None):
    """Seuil modèle qui maximise le MCC « même modèle » sur les endpoints de vérité connue.
    Les seuils candidats sont les hauteurs de fusion des dendrogrammes calculés DANS chaque famille, comme la
    partition modèle (partition() découpe par famille). t_family=None : une seule famille.
    Retourne (seuil au milieu du meilleur intervalle, meilleur MCC)."""
    D_family = D_model if D_family is None else D_family
    if t_family is None:
        t_family = _single_family_threshold(D_family)
    fam, _ = upgma(D_family, t_family)
    h = []
    for f in sorted(set(fam)):
        members = [i for i in range(len(endpoints)) if fam[i] == f]
        if len(members) > 1:
            h += upgma(D_model[np.ix_(members, members)])[1]
    cands = sorted(set([0.0] + h))
    cands = [(x + y) / 2 for x, y in zip(cands, cands[1:])] + [cands[-1] + 1e-6]
    res = [(t, score(truth, partition(endpoints, D_family, D_model, t_family, t))["practice_mcc"]) for t in cands]
    best = max(m for _, m in res)
    good = [t for t, m in res if m >= best - 1e-12]
    return float((min(good) + max(good)) / 2), best


def prompt_auc(per_prompt: dict, endpoints, truth: dict) -> dict:
    """AUC par prompt : P(distance paire différente > distance paire identique). 0,5 = inutile."""
    pos = {e: i for i, e in enumerate(endpoints)}
    eps = [e for e in endpoints if e in truth]
    out = {}
    for pid, (kind, D) in per_prompt.items():
        same, diff = [], []
        for a, b in itertools.combinations(eps, 2):
            d = D[pos[a], pos[b]]
            if np.isfinite(d):
                (same if truth[a] == truth[b] else diff).append(d)
        if same and diff:
            s, df = np.array(same), np.array(diff)
            auc = float(((df[:, None] > s[None, :]).mean() + 0.5 * (df[:, None] == s[None, :]).mean()))
            out[pid] = (kind, auc, float(np.mean(same)), float(np.mean(diff)))
    return out


def exact_matches(grouped: dict, endpoints) -> dict:
    """{(a, b): nombre de prompts où a et b ont au moins une réponse strictement identique}."""
    c = defaultdict(int)
    for by_ep in grouped.values():
        sets = {ep: {normalize(t) for t in by_ep.get(ep, []) if t.strip()} for ep in endpoints}
        for a, b in itertools.combinations(endpoints, 2):
            if sets[a] & sets[b]:
                c[(a, b)] += 1
    return dict(c)


# ---------------------------------------------------------------------------------------------- pipeline


def analyze(grouped: dict, kernel: str = "ngram", stat: str = "corr", t_model="auto", t_family="auto",
            truth: dict | None = None, calibrate_on_truth: bool = False, weights: dict | None = None,
            strip: bool = True):
    """Pipeline complet. Retourne un dict (endpoints, D_model, D_family, rows, seuils, per_prompt)."""
    if strip:
        grouped = strip_constant_affixes(grouped)
    endpoints = sorted({ep for by_ep in grouped.values() for ep in by_ep})
    pp = per_prompt_distances(grouped, endpoints, kernel, stat)
    norm = stat == "mmd"
    D_model = combine(pp, ("multi",), norm, weights)
    D_single = combine(pp, ("single",), norm, weights)
    if D_model is None:
        D_model = D_single
    if D_model is None:
        raise ValueError("aucun prompt exploitable (il faut >= 2 endpoints par prompt)")
    D_family = D_model if D_single is None else np.nanmean(np.stack([D_model, D_single]), axis=0)
    if t_family == "auto":
        t_family = gap_threshold(D_family, log=False)
    t_family = float(t_family)
    if calibrate_on_truth and truth:
        t_model, _ = calibrate(endpoints, D_model, truth, D_family, t_family)
    elif t_model == "auto":
        t_model = gap_threshold(D_model, hi=t_family)
    t_model = float(t_model)
    rows = partition(endpoints, D_family, D_model, t_family, t_model)
    return {"endpoints": endpoints, "D_model": D_model, "D_family": D_family, "rows": rows,
            "t_model": t_model, "t_family": t_family, "per_prompt": pp}


def write_rows(rows, path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["endpoint", "family", "model"])
        w.writerows(rows)


def _fmt_matrix(endpoints, D) -> str:
    head = "        " + " ".join(f"{e[-2:]:>5}" for e in endpoints)
    lines = [head]
    for i, e in enumerate(endpoints):
        lines.append(f"{e:>7} " + " ".join("  nan" if not np.isfinite(x) else f"{x:5.3f}" for x in D[i]))
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("answers", nargs="+", help="answers.jsonl au format du kit")
    p.add_argument("--prefix", default="ep-", help="endpoints analysés (ep- par défaut, ref- pour le pilote)")
    p.add_argument("--kernel", default="ngram", choices=["ngram", "seq", "hamming", "both"])
    p.add_argument("--stat", default="corr", choices=["corr", "mmd"])
    p.add_argument("--t-model", default="auto", help="seuil modèle (nombre) ou auto")
    p.add_argument("--t-family", default="auto", help="seuil famille (nombre) ou auto")
    p.add_argument("--truth", help="CSV de vérité (practice_answers.csv) : affiche le score")
    p.add_argument("--calibrate", action="store_true", help="choisit t-model qui maximise le MCC sur --truth")
    p.add_argument("--rank-prompts", action="store_true", help="AUC par prompt (avec --truth)")
    p.add_argument("--prompt-ids", help="fichier (un id par ligne) ou liste séparée par des virgules")
    p.add_argument("--max-prompts", type=int)
    p.add_argument("--max-reps", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="CSV endpoint,family,model")
    p.add_argument("--dump", help="CSV de la matrice D_model")
    p.add_argument("--no-strip", action="store_true", help="garde les préfixes/suffixes constants par endpoint")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)

    grouped = group(load_answers(a.answers), a.prefix)
    ids = None
    if a.prompt_ids:
        src = Path(a.prompt_ids)
        ids = [x.strip() for x in (src.read_text().split() if src.exists() else a.prompt_ids.split(",")) if x.strip()]
    if ids is not None or a.max_prompts or a.max_reps:
        grouped = subsample(grouped, a.max_prompts, a.max_reps, a.seed, ids)
    if not grouped:
        print(f"aucune réponse pour le préfixe {a.prefix!r}", file=sys.stderr)
        return 1
    truth = read_truth(a.truth) if a.truth else None
    t_model = a.t_model if a.t_model == "auto" else float(a.t_model)
    t_family = a.t_family if a.t_family == "auto" else float(a.t_family)
    try:
        res = analyze(grouped, a.kernel, a.stat, t_model, t_family, truth, a.calibrate, strip=not a.no_strip)
    except ValueError as exc:
        print(f"erreur : {exc}", file=sys.stderr)
        return 2
    eps, rows = res["endpoints"], res["rows"]
    n = answers_per_endpoint(grouped)

    if not a.quiet:
        print(f"{len(eps)} endpoints, {len(grouped)} prompts, réponses/endpoint min {min(n.values())} "
              f"max {max(n.values())} ; noyau {a.kernel}, stat {a.stat}")
        print(f"seuils : famille {res['t_family']:.4f}, modèle {res['t_model']:.4f}")
        print("D_model :\n" + _fmt_matrix(eps, res["D_model"]))
        for e, f, m in rows:
            i = eps.index(e)
            nn = sorted((res["D_model"][i, j], eps[j]) for j in range(len(eps)) if j != i and
                        np.isfinite(res["D_model"][i, j]))[:3]
            print(f"  {e} {f:>3} {m:>5}   plus proches : " + ", ".join(f"{x} {d:.3f}" for d, x in nn))
        ex = exact_matches(grouped, eps)
        if ex:
            top = sorted(ex.items(), key=lambda kv: -kv[1])[:10]
            print("réponses identiques (paires, nb de prompts) : " + ", ".join(f"{x}-{y}:{k}" for (x, y), k in top))
    if truth:
        s = score(truth, rows)
        print(f"score vérité : practice MCC {s['practice_mcc']:.3f} | family {s['family_mcc']:.3f} "
              f"model {s['model_mcc']:.3f} -> {s['score']:.3f} sur {s['pairs']} paires")
        if a.rank_prompts:
            auc = prompt_auc(res["per_prompt"], eps, truth)
            print("prompts par AUC (kind, AUC, d_same, d_diff) :")
            for pid, (kind, v, ds, dd) in sorted(auc.items(), key=lambda kv: -kv[1][1]):
                print(f"  {pid:>8} {kind:>6} {v:.3f} {ds:.3f} {dd:.3f}")
    if a.out:
        write_rows(rows, a.out)
        print(f"écrit : {a.out}")
    if a.dump:
        with open(a.dump, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([""] + eps)
            for i, e in enumerate(eps):
                w.writerow([e] + [f"{x:.6f}" for x in res["D_model"][i]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
