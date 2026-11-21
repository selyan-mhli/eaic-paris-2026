"""c2 — features et distances supplémentaires pour « même modèle ? », évaluées sur les ref-* (hors ligne).

Aucune requête au serveur. S'appuie sur analyze.py (lecture, pair_stat, UPGMA, score) sans le modifier.

Idée : une *feature* est un couple (prompt, vue). Une vue transforme les réponses d'un prompt en une matrice
de noyau K (n × n, diagonale 1), puis analyze.pair_stat (corr ou mmd) donne la distance entre deux
endpoints pour ce prompt. Vues disponibles (VIEWS) :
    char    n-grammes de caractères 3-5 (= analyze.py), après retrait de l'enrobage
    word    unigrammes + bigrammes de mots hachés, cosinus
    nums    suite des entiers extraits ; noyau = fraction de positions identiques (ordre compris)
    numset  multiensemble des entiers extraits ; cosinus des comptes
    items   ensemble des éléments d'une liste (virgules / lignes, numérotation retirée) ; Jaccard
    order   suite des éléments d'une liste ; fraction de positions identiques
    hamming positions de caractères identiques sur 1 000 caractères (Gao et al., analyze.py)
    lcp     préfixe commun en mots / longueur max (deux réponses quasi greedy partagent un long préfixe)
    len     log de la longueur en caractères ; noyau gaussien (bande = écart type du prompt)
    div     position de première divergence avec la suite correcte connue (REFERENCE) ; noyau gaussien

Enrobage (persona, system prompt) : en plus des affixes constants d'analyze.py, `strip_boilerplate` retire,
par endpoint, les phrases qui reviennent dans ses réponses à au moins `min_prompts` prompts différents.

Protocole honnête (9 endpoints seulement, risque de sur-apprentissage) :
    sélection des features (AUC) et du seuil sur une partie des répétitions (paires), évaluation sur l'autre
    (impaires) ; en option leave-pair-out : pour chaque paire (a, b), sélection sur les paires qui ne
    contiennent ni a ni b, prédiction de (a, b).
Seuil sans vérité : `split_half_threshold`, distance d'un endpoint à lui-même entre deux moitiés de ses
répétitions = plancher de bruit.

    python c2/features.py rank ref.jsonl --truth c2/kit/practice_answers.csv
    python c2/features.py eval ref.jsonl --truth c2/kit/practice_answers.csv --views char,nums,items --top 10
    python c2/features.py run  ref.jsonl --prefix ref- --features best.txt --t-model 0.05 --out guess.csv
"""
from __future__ import annotations

import argparse
import itertools
import math
import re
import sys
import warnings
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import analyze  # noqa: E402

# ---------------------------------------------------------------------------------------------- références


def _primes(n: int) -> list:
    out, k = [], 2
    while len(out) < n:
        if all(k % p for p in out if p * p <= k):
            out.append(k)
        k += 1
    return out


def pi_digits(n: int) -> str:
    """n premières décimales de pi (formule de Machin, entiers exacts)."""
    def arctan_inv(x, one):
        total = term = one // x
        x2, k, sign = x * x, 3, -1
        while term:
            term //= x2
            total += sign * (term // k)
            sign, k = -sign, k + 2
        return total
    one = 10 ** (n + 10)
    pi = 4 * (4 * arctan_inv(5, one) - arctan_inv(239, one))
    return str(pi)[1:n + 1]


def _fib(n: int, a: int = 0, b: int = 1) -> list:
    out = []
    for _ in range(n):
        out.append(a)
        a, b = b, a + b
    return out


# suite correcte attendue (entiers) ; pour m04 une chaîne de chiffres. Plusieurs variantes admises (fib 0/1).
REFERENCE = {
    "m03": [_primes(60)],
    "m04": [pi_digits(200)],
    "c01": [_fib(30), _fib(30, 1, 1)],
    "c10": [[v for k in range(1, 21) for v in (17, k, 17 * k)]],
    "c11": [[2 ** k for k in range(1, 46)], [2 ** k for k in range(6, 46)]],
    "m11": [[v for z in range(1, 37) for v in (z,)]],
}

# ---------------------------------------------------------------------------------------------- extraction

_INT = re.compile(r"-?\d+")
_NUMBERING = re.compile(r"^\s*(?:[-*•]|\(?\d+[.)]|\d+\s*[-:])\s*")
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def numbers(text: str) -> list:
    """Entiers dans l'ordre d'apparition (les séparateurs de milliers ne sont pas recollés)."""
    return [int(x) for x in _INT.findall(text)]


def digit_string(text: str) -> str:
    """Chiffres consécutifs après un éventuel « 3. » initial (pour les décimales de pi)."""
    s = re.sub(r"[^\d]", "", text)
    return s[1:] if s.startswith("314") else s


def items(text: str) -> list:
    """Éléments d'une liste : séparés par virgule, point-virgule ou ligne ; numérotation et casse retirées."""
    parts = re.split(r"[,;\n]+", text)
    out = []
    for p in parts:
        p = _NUMBERING.sub("", p).strip().strip(".").strip().lower()
        p = analyze.normalize(p)
        if p and len(p) <= 60:
            out.append(p)
    return out


def sentences(text: str) -> list:
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def divergence(pid: str, text: str) -> float | None:
    """Longueur du préfixe correct (en éléments) par rapport à la meilleure référence ; None si pas de réf."""
    refs = REFERENCE.get(pid)
    if not refs:
        return None
    if isinstance(refs[0], str):
        seq, best = digit_string(text), 0
        for r in refs:
            k = 0
            while k < min(len(seq), len(r)) and seq[k] == r[k]:
                k += 1
            best = max(best, k)
        return float(best)
    seq, best = numbers(text), 0
    for r in refs:
        k = 0
        while k < min(len(seq), len(r)) and seq[k] == r[k]:
            k += 1
        best = max(best, k)
    return float(best)


def determinism(grouped: dict) -> dict:
    """{endpoint: fraction des prompts où toutes ses répétitions non vides (>= 2) sont identiques}. Un endpoint
    décodé en greedy approche 1 ; sert à savoir si l'égalité exacte entre endpoints est une preuve. Les
    réponses vides sont absentes ; un endpoint sans aucun prompt exploitable vaut nan."""
    tot, same = Counter(), Counter()
    eps = set()
    for by_ep in grouped.values():
        for ep, v in by_ep.items():
            eps.add(ep)
            v = [t for t in v if t.strip()]
            if len(v) >= 2:
                tot[ep] += 1
                same[ep] += len({analyze.normalize(t) for t in v}) == 1
    return {ep: (same[ep] / tot[ep] if tot[ep] else float("nan")) for ep in sorted(eps)}


_SENT_SPLIT = re.compile(r"((?<=[.!?])[ \t]+|\n+)")


def fit_boilerplate(grouped: dict, min_prompts: int = 3, min_chars: int = 12, min_frac: float = 0.5,
                    min_endpoint_prompts: int = 5) -> dict:
    """{endpoint: phrases normalisées à retirer} : celles présentes dans ses réponses à au moins
    max(min_prompts, min_frac × nb de prompts de l'endpoint) prompts différents. Une formule de persona revient
    presque partout ; une phrase factuelle partagée par quelques prompts reste. Rien n'est retiré pour un
    endpoint qui a répondu à moins de min_endpoint_prompts prompts (trop peu pour distinguer)."""
    seen = defaultdict(lambda: defaultdict(set))
    n_prompts = Counter()
    for pid, by_ep in grouped.items():
        for ep, texts in by_ep.items():
            if any(t.strip() for t in texts):
                n_prompts[ep] += 1
            for t in texts:
                for s_ in sentences(t):
                    n = analyze.normalize(s_)
                    if len(n) >= min_chars:
                        seen[ep][n].add(pid)
    return {ep: ({x for x, ps in d.items() if len(ps) >= max(min_prompts, min_frac * n_prompts[ep])}
                 if n_prompts[ep] >= min_endpoint_prompts else set())
            for ep, d in seen.items()}


def drop_sentences(text: str, bad: set) -> str:
    """Retire les phrases de bad, garde le reste du texte tel quel (sauts de ligne, casse, ponctuation)."""
    if not bad:
        return text
    parts = _SENT_SPLIT.split(text)          # [contenu, séparateur, contenu, séparateur, ...]
    out = []
    for i in range(0, len(parts), 2):
        piece = parts[i]
        sep = parts[i + 1] if i + 1 < len(parts) else ""
        if piece.strip() and analyze.normalize(piece) in bad:
            continue
        out.append(piece + sep)
    return "".join(out).strip()


def apply_boilerplate(grouped: dict, bad: dict) -> dict:
    return {p: {ep: [drop_sentences(t, bad.get(ep, set())) for t in v] for ep, v in d.items()}
            for p, d in grouped.items()}


def strip_boilerplate(grouped: dict, min_prompts: int = 3, min_chars: int = 12, fit_on: dict | None = None) -> dict:
    """Retire, par endpoint, les phrases d'enrobage récurrentes (règles apprises sur fit_on, défaut grouped)."""
    return apply_boilerplate(grouped, fit_boilerplate(grouped if fit_on is None else fit_on, min_prompts, min_chars))


# ---------------------------------------------------------------------------------------------- noyaux


def _hash_vectors(token_lists, dim: int = 1 << 14) -> np.ndarray:
    V = np.zeros((len(token_lists), dim), dtype=np.float64)
    for i, toks in enumerate(token_lists):
        for tok, c in Counter(toks).items():
            V[i, zlib.crc32(tok.encode("utf-8")) % dim] += 1.0 + math.log(c)
        nrm = np.linalg.norm(V[i])
        if nrm > 0:
            V[i] /= nrm
    return V


def _cos(token_lists) -> np.ndarray:
    V = _hash_vectors(token_lists)
    K = np.clip(V @ V.T, 0.0, 1.0)
    np.fill_diagonal(K, 1.0)
    return _empty_as_missing(K, token_lists)


def _empty_as_missing(K: np.ndarray, items) -> np.ndarray:
    """Une séquence/un ensemble vide est une donnée absente : sa ligne et sa colonne passent à nan."""
    for i, x in enumerate(items):
        if len(x) == 0:
            K[i, :] = np.nan
            K[:, i] = np.nan
    return K


def _positional(seqs) -> np.ndarray:
    """k(x, y) = #{i : x_i = y_i} / max(|x|, |y|) ; 1 sur la diagonale ; nan si l'un est vide."""
    n = len(seqs)
    K = np.eye(n)
    for i, j in itertools.combinations(range(n), 2):
        a, b = seqs[i], seqs[j]
        m = max(len(a), len(b))
        K[i, j] = K[j, i] = (sum(x == y for x, y in zip(a, b)) / m) if m else np.nan
    return _empty_as_missing(K, seqs)


def _lcp(seqs) -> np.ndarray:
    n = len(seqs)
    K = np.eye(n)
    for i, j in itertools.combinations(range(n), 2):
        a, b = seqs[i], seqs[j]
        m = max(len(a), len(b))
        k = 0
        while k < min(len(a), len(b)) and a[k] == b[k]:
            k += 1
        K[i, j] = K[j, i] = (k / m) if m else np.nan
    return _empty_as_missing(K, seqs)


def _jaccard(sets) -> np.ndarray:
    n = len(sets)
    K = np.eye(n)
    for i, j in itertools.combinations(range(n), 2):
        a, b = sets[i], sets[j]
        u = len(a | b)
        K[i, j] = K[j, i] = (len(a & b) / u) if u else np.nan
    return _empty_as_missing(K, sets)


def _rbf(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    h = float(np.std(x))
    if not np.isfinite(h) or h <= 1e-9:
        return np.ones((len(x), len(x)))
    return np.exp(-((x[:, None] - x[None, :]) ** 2) / (2 * h * h))


def _words(t: str) -> list:
    return [w for w in analyze.normalize(t).split(" ") if w]


def view_kernel(view: str, pid: str, texts: list) -> np.ndarray | None:
    """Matrice de noyau d'une vue pour les réponses d'un prompt ; None si la vue ne s'applique pas."""
    if view == "char":
        return analyze.kernel_matrix(texts, "ngram")
    if view == "hamming":
        return analyze.kernel_matrix(texts, "hamming")
    if view == "word":
        toks = []
        for t in texts:
            w = _words(t)
            toks.append(w + [a + " " + b for a, b in zip(w, w[1:])])
        return _cos(toks)
    if view == "nums":
        seqs = [digit_string(t) if pid == "m04" else numbers(t) for t in texts]
        if sum(len(s) > 0 for s in seqs) < len(seqs) / 2:
            return None
        return _positional(seqs)
    if view == "numset":
        seqs = [numbers(t) for t in texts]
        if sum(len(s) > 0 for s in seqs) < len(seqs) / 2:
            return None
        return _cos([[str(v) for v in s] for s in seqs])
    if view == "items":
        its = [items(t) for t in texts]
        if np.median([len(i) for i in its]) < 3:
            return None
        return _jaccard([set(i) for i in its])
    if view == "order":
        its = [items(t) for t in texts]
        if np.median([len(i) for i in its]) < 3:
            return None
        return _positional(its)
    if view == "lcp":
        return _lcp([_words(t) for t in texts])
    if view == "len":
        return _rbf([math.log1p(len(t)) for t in texts])
    if view == "div":
        d = [divergence(pid, t) for t in texts]
        if d[0] is None:
            return None
        return _rbf([math.log1p(x) for x in d])
    raise ValueError(f"vue inconnue : {view}")


VIEWS = ("char", "hamming", "word", "nums", "numset", "items", "order", "lcp", "len", "div")

# ---------------------------------------------------------------------------------------------- distances


def _nanmean(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean(x[np.isfinite(x)])) if np.isfinite(x).any() else float("nan")


def feature_kernels(grouped: dict, endpoints: list, views=VIEWS, min_reps: int = 2) -> dict:
    """{(pid, view): matrice E×E des noyaux moyens} : M[a, b] = moyenne de k entre réponses de a et de b,
    M[a, a] = moyenne hors diagonale (non biaisée). nan si l'endpoint manque. Toutes les statistiques
    (corr, min, mmd) et la mise en commun des prompts (pooled) se calculent à partir de M."""
    E = len(endpoints)
    pos = {ep: i for i, ep in enumerate(endpoints)}
    out = {}
    for pid, by_ep in grouped.items():
        by_ep = {ep: [t for t in v if t.strip()] for ep, v in by_ep.items()}      # vide = absent
        eps = [ep for ep in endpoints if len(by_ep.get(ep, [])) >= min_reps]
        if len(eps) < 2:
            continue
        texts, idx = [], defaultdict(list)
        for ep in eps:
            for t in by_ep[ep]:
                idx[ep].append(len(texts))
                texts.append(t)
        for view in views:
            K = view_kernel(view, pid, texts)
            if K is None:
                continue
            M = np.full((E, E), np.nan)
            for a in eps:
                sub = K[np.ix_(idx[a], idx[a])].copy()
                np.fill_diagonal(sub, np.nan)
                M[pos[a], pos[a]] = _nanmean(sub)
            for a, b in itertools.combinations(eps, 2):
                M[pos[a], pos[b]] = M[pos[b], pos[a]] = _nanmean(K[np.ix_(idx[a], idx[b])])
            out[(pid, view)] = M
    return out


def dist_from_kernel(M: np.ndarray, stat: str = "corr") -> np.ndarray:
    """Distance E×E depuis les noyaux moyens.
    corr : 1 - Kab / sqrt(Kaa Kbb) (analyze.pair_stat) ;
    min  : 1 - Kab / min(Kaa, Kbb), bornée à [0, 1] : un endpoint déterministe (Kaa = 1, ex. watermark
           greedy) dont la réponse est un échantillon typique de b reste à distance ~0 de b ;
    mmd  : Kaa + Kbb - 2 Kab."""
    d = np.diag(M)
    kaa, kbb = d[:, None], d[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        if stat == "corr":
            D = 1.0 - M / np.sqrt(np.maximum(kaa, 1e-9) * np.maximum(kbb, 1e-9))
        elif stat == "min":
            D = 1.0 - M / np.maximum(np.minimum(kaa, kbb), 1e-9)
        elif stat == "mmd":
            D = kaa + kbb - 2 * M
        else:
            raise ValueError(f"statistique inconnue : {stat}")
    D = np.clip(D, 0.0, None if stat == "mmd" else 1.0)
    D = np.where(np.isfinite(M), D, np.nan)
    np.fill_diagonal(D, np.where(np.isfinite(d), 0.0, np.nan))
    return D


def feature_distances(grouped: dict, endpoints: list, views=VIEWS, stat: str = "corr",
                      min_reps: int = 2) -> dict:
    """{(pid, view): matrice E×E (nan si absent)}. Seuls les endpoints à >= min_reps réponses comptent."""
    return {k: dist_from_kernel(M, stat) for k, M in feature_kernels(grouped, endpoints, views, min_reps).items()}


def pooled_distance(fk: dict, keys, stat: str = "corr", weights: dict | None = None) -> np.ndarray:
    """Mise en commun : pour chaque paire (a, b), moyenne des noyaux Kaa, Kbb et Kab sur les SEULES features
    où a et b sont tous deux présents (même support), PUIS statistique. Bien moins bruité que la moyenne des
    rapports par prompt quand il y a peu de répétitions."""
    mats, ws = [], []
    for k in keys:
        if k in fk:
            mats.append(fk[k])
            ws.append(1.0 if weights is None else float(weights.get(k, 0.0)))
    if not mats:
        raise ValueError("aucune feature retenue")
    A = np.stack(mats)
    w = np.array(ws)
    E = A.shape[1]
    D = np.full((E, E), np.nan)
    for i in range(E):
        if np.isfinite(A[:, i, i]).any():
            D[i, i] = 0.0
    for i, j in itertools.combinations(range(E), 2):
        ok = np.isfinite(A[:, i, j]) & np.isfinite(A[:, i, i]) & np.isfinite(A[:, j, j]) & (w > 0)
        if not ok.any():
            continue
        ww = w[ok]
        M = np.empty((2, 2))
        M[0, 0] = np.average(A[ok, i, i], weights=ww)
        M[1, 1] = np.average(A[ok, j, j], weights=ww)
        M[0, 1] = M[1, 0] = np.average(A[ok, i, j], weights=ww)
        D[i, j] = D[j, i] = dist_from_kernel(M, stat)[0, 1]
    return D


def combine_features(fd: dict, keys, weights: dict | None = None, scale: str = "median") -> np.ndarray:
    """Moyenne pondérée des matrices choisies. scale='median' divise chaque matrice par sa médiane hors
    diagonale (les vues n'ont pas la même échelle) ; 'none' garde l'échelle brute."""
    mats, ws = [], []
    for k in keys:
        if k not in fd:
            continue
        M = fd[k].copy()
        if scale == "median":
            off = M[np.triu_indices_from(M, 1)]
            off = off[np.isfinite(off)]
            med = float(np.median(off)) if off.size else 0.0
            if med > 1e-9:
                M = M / med
        mats.append(M)
        ws.append(1.0 if weights is None else float(weights.get(k, 0.0)))
    if not mats:
        raise ValueError("aucune feature retenue")
    A = np.stack(mats)
    W = np.array(ws)[:, None, None] * np.isfinite(A)
    num = (np.where(np.isfinite(A), A, 0.0) * W).sum(0)
    den = W.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


# ---------------------------------------------------------------------------------------------- évaluation


def split_reps(grouped: dict, keep) -> dict:
    """Garde, pour chaque (prompt, endpoint), les réponses d'indice dans keep (ex. paires / impaires).
    Les réponses sont déjà triées par repeat (analyze.group)."""
    keep = set(keep)
    return {p: {ep: [t for i, t in enumerate(v) if i in keep] for ep, v in d.items()} for p, d in grouped.items()}


def parity(grouped: dict, which: int) -> dict:
    n = max(len(v) for d in grouped.values() for v in d.values())
    return split_reps(grouped, range(which, n, 2))


def pair_auc(D: np.ndarray, endpoints, truth: dict, only=None) -> float:
    """AUC « distance paire différente > distance paire identique ». only : liste de paires différentes
    à garder (ex. paires difficiles fp8/LoRA0,3 contre S)."""
    pos = {e: i for i, e in enumerate(endpoints)}
    eps = [e for e in endpoints if e in truth]
    same, diff = [], []
    for a, b in itertools.combinations(eps, 2):
        d = D[pos[a], pos[b]]
        if not np.isfinite(d):
            continue
        if truth[a] == truth[b]:
            same.append(d)
        elif only is None or (a, b) in only or (b, a) in only:
            diff.append(d)
    if not same or not diff:
        return float("nan")
    s, df = np.array(same), np.array(diff)
    return float((df[:, None] > s[None, :]).mean() + 0.5 * (df[:, None] == s[None, :]).mean())


def hard_pairs(truth: dict, hard=None) -> set:
    """Paires (variante difficile, membre du groupe majoritaire S). Variantes difficiles : HARD_DEFAULT
    (fp8 = ref-02, LoRA 0,3 = ref-09 dans practice_answers.csv) sauf liste explicite."""
    hard = HARD_DEFAULT if hard is None else hard
    groups = defaultdict(list)
    for e, fm in truth.items():
        groups[fm].append(e)
    big = max(groups.values(), key=len)
    return {(h, s) for h in truth if h in hard for s in big if s != h}


HARD_DEFAULT = ("ref-02", "ref-09")     # fp8 et LoRA 0,3 dans practice_answers.csv


def rank_features(fd: dict, endpoints, truth: dict) -> list:
    """[(clé, AUC toutes paires, AUC paires difficiles)], trié par AUC difficile puis AUC globale."""
    hp = hard_pairs(truth)
    res = [(k, pair_auc(D, endpoints, truth), pair_auc(D, endpoints, truth, hp)) for k, D in fd.items()]
    return sorted(res, key=lambda r: (-np.nan_to_num(r[2]), -np.nan_to_num(r[1])))


def select(ranked: list, top: int, min_auc: float = 0.5, by: str = "mix") -> list:
    """Les `top` meilleures features. by='all' (AUC globale), 'hard' (AUC difficile), 'mix' (moyenne)."""
    def key(r):
        a = np.nan_to_num(r[1])
        h = a if not np.isfinite(r[2]) else r[2]        # pas de paire difficile : AUC globale
        return {"all": a, "hard": h, "mix": 0.5 * (a + h)}[by]
    good = [r for r in ranked if key(r) > min_auc]
    return [r[0] for r in sorted(good, key=key, reverse=True)[:top]]


def pairwise_mcc(D: np.ndarray, endpoints, truth: dict, t: float) -> float:
    pos = {e: i for i, e in enumerate(endpoints)}
    eps = [e for e in endpoints if e in truth]
    tt, gg = [], []
    for a, b in itertools.combinations(eps, 2):
        tt.append(truth[a] == truth[b])
        gg.append(bool(D[pos[a], pos[b]] <= t))
    return analyze.mcc(tt, gg)


def partition_mcc(D: np.ndarray, endpoints, truth: dict, t: float) -> tuple:
    rows = analyze.partition(endpoints, np.zeros_like(D), D, 10.0, t)
    return analyze.score(truth, rows)["practice_mcc"], rows


def split_half_threshold(grouped: dict, endpoints, keys, stat: str = "corr", q: float = 1.0,
                         factor: float = 1.0, fd_scale: dict | None = None) -> float:
    """Plancher de bruit : pour chaque endpoint, ses répétitions paires contre ses impaires sont traitées
    comme deux pseudo-endpoints ; distance combinée (mêmes features, même mise à l'échelle que fd_scale si
    fournie) ; seuil = factor × quantile q de ces distances « soi contre soi ». Aucune vérité utilisée."""
    reps = [len([t for t in v if t.strip()]) for p, _ in keys if p in grouped for v in grouped[p].values()]
    if not reps or max(reps) < 4:
        raise ValueError("seuil split-half impossible : il faut au moins 4 répétitions par (prompt, endpoint) "
                         f"(2 par moitié) ; maximum observé {max(reps) if reps else 0}")
    halves = {}
    for p, d in grouped.items():
        halves[p] = {}
        for ep, v in d.items():
            halves[p][ep + "#0"] = v[0::2]
            halves[p][ep + "#1"] = v[1::2]
    eps2 = [e + s for e in endpoints for s in ("#0", "#1")]
    views = sorted({v for _, v in keys})
    pids = {p for p, _ in keys}
    fd2 = feature_distances({p: halves[p] for p in pids if p in halves}, eps2, views, stat)
    self_d, mats = [], []
    for k in keys:
        if k not in fd2:
            continue
        M = fd2[k]
        if fd_scale is not None and k in fd_scale:
            M = M / fd_scale[k]
        mats.append(M)
    if not mats:
        raise ValueError("seuil split-half impossible : aucune feature calculable sur les demi-échantillons "
                         "(il faut au moins 4 répétitions par (prompt, endpoint))")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # cases toutes nan : restent nan
        A = np.nanmean(np.stack(mats), axis=0)
    for i in range(0, len(eps2), 2):
        if np.isfinite(A[i, i + 1]):
            self_d.append(A[i, i + 1])
    if not self_d:
        raise ValueError("seuil split-half impossible : aucune distance soi/soi finie "
                         "(il faut au moins 4 répétitions par (prompt, endpoint))")
    t = float(factor * np.quantile(self_d, q))
    if not np.isfinite(t):
        raise ValueError(f"seuil split-half non fini ({t})")
    return t


def median_scales(fd: dict) -> dict:
    out = {}
    for k, M in fd.items():
        off = M[np.triu_indices_from(M, 1)]
        off = off[np.isfinite(off)]
        med = float(np.median(off)) if off.size else 0.0
        out[k] = med if med > 1e-9 else 1.0
    return out


def best_threshold(D: np.ndarray, endpoints, truth: dict) -> tuple:
    """Seuil UPGMA qui maximise le MCC de partition (milieu du meilleur intervalle)."""
    t, m = analyze.calibrate(endpoints, D, truth, np.zeros_like(D), 10.0)
    return t, m


def evaluate(grouped: dict, truth: dict, views=VIEWS, stat: str = "corr", top: int = 10, by: str = "mix",
             scale: str = "median", strip: str = "both", fixed_keys=None) -> dict:
    """Sélection sur les répétitions paires (features + seuil), évaluation sur les impaires, et l'inverse.
    Retourne les MCC de partition (seuil transféré), l'AUC globale/difficile du D évalué, les features."""
    endpoints = sorted({ep for d in grouped.values() for ep in d})
    out = {"folds": []}
    for sel, ev in ((0, 1), (1, 0)):
        raw_s, raw_e = parity(grouped, sel), parity(grouped, ev)
        # règles de retrait (affixes, phrases) apprises sur la sélection seulement, appliquées aux deux
        gs, ge = prepare(raw_s, strip, fit_on=raw_s), prepare(raw_e, strip, fit_on=raw_s)
        fds = feature_distances(gs, endpoints, views, stat)
        fde = feature_distances(ge, endpoints, views, stat)
        keys = list(fixed_keys) if fixed_keys is not None else select(rank_features(fds, endpoints, truth), top, by=by)
        keys = [k for k in keys if k in fds and k in fde]
        if not keys:
            out["folds"].append(None)
            continue
        if scale == "median":
            sc = median_scales(fds)          # échelles figées sur la sélection, appliquées à l'évaluation
            Ds = combine_features({k: fds[k] / sc[k] for k in keys}, keys, scale="none")
            De = combine_features({k: fde[k] / sc[k] for k in keys}, keys, scale="none")
        else:
            sc = None
            Ds = combine_features(fds, keys, scale="none")
            De = combine_features(fde, keys, scale="none")
        t, m_sel = best_threshold(Ds, endpoints, truth)
        m_ev, rows = partition_mcc(De, endpoints, truth, t)
        _, m_oracle = best_threshold(De, endpoints, truth)
        try:
            t_sh = split_half_threshold(ge, endpoints, keys, stat, q=1.0, fd_scale=sc)
            m_sh, _ = partition_mcc(De, endpoints, truth, t_sh)
        except ValueError:                  # trop peu de répétitions dans la moitié d'évaluation
            t_sh, m_sh = float("nan"), float("nan")
        hp = hard_pairs(truth)
        out["folds"].append({"keys": keys, "t": t, "mcc_sel": m_sel, "mcc_eval": m_ev, "mcc_oracle": m_oracle,
                             "t_split_half": t_sh, "mcc_split_half": m_sh,
                             "auc": pair_auc(De, endpoints, truth), "auc_hard": pair_auc(De, endpoints, truth, hp),
                             "rows": rows})
    return out


def prepare(grouped: dict, strip: str = "both", fit_on: dict | None = None) -> dict:
    """strip : 'none', 'affix' (analyze.strip_constant_affixes), 'sent' (strip_boilerplate) ou 'both'.
    Les règles de retrait sont apprises sur fit_on (défaut : grouped) puis appliquées à grouped."""
    fit = grouped if fit_on is None else fit_on
    g = grouped
    if strip in ("affix", "both"):
        cut = analyze.fit_affixes(fit)
        g = analyze.apply_affixes(g, cut)
        fit = analyze.apply_affixes(fit, cut)
    if strip in ("sent", "both"):
        g = apply_boilerplate(g, fit_boilerplate(fit))
    return g


def leave_pair_out(grouped: dict, truth: dict, views=VIEWS, stat: str = "corr", top: int = 10,
                   by: str = "mix", strip: str = "both") -> dict:
    """Pour chaque paire (a, b) : features et seuil choisis sur les paires sans a ni b (toutes répétitions),
    puis prédiction « même modèle » de (a, b) par D_ab <= seuil (sans clustering). MCC par paires."""
    g = prepare(grouped, strip)
    endpoints = sorted({ep for d in g.values() for ep in d})
    fd = feature_distances(g, endpoints, views, stat)
    sc = median_scales(fd)
    fdn = {k: M / sc[k] for k, M in fd.items()}
    pos = {e: i for i, e in enumerate(endpoints)}
    eps = [e for e in endpoints if e in truth]
    tt, gg, chosen = [], [], Counter()
    for a, b in itertools.combinations(eps, 2):
        sub = {e: truth[e] for e in eps if e not in (a, b)}
        ranked = []
        hp = hard_pairs(sub)
        for k, D in fdn.items():
            ranked.append((k, pair_auc(D, endpoints, sub), pair_auc(D, endpoints, sub, hp)))
        keys = select(ranked, top, by=by)
        if not keys:
            continue
        chosen.update(keys)
        D = combine_features(fdn, keys, scale="none")
        # seuil pairwise : meilleur MCC sur les paires de sub
        cands = sorted({float(D[pos[x], pos[y]]) for x, y in itertools.combinations(sub, 2)})
        cands = [(u + v) / 2 for u, v in zip(cands, cands[1:])] or cands
        best = max(cands, key=lambda t: pairwise_mcc(D, endpoints, sub, t))
        tt.append(truth[a] == truth[b])
        gg.append(bool(D[pos[a], pos[b]] <= best))
    return {"mcc": analyze.mcc(tt, gg), "n": len(tt), "chosen": chosen.most_common(15)}


# ---------------------------------------------------------------------------------------------- CLI


def _keys_from(spec: str) -> list:
    src = Path(spec)
    txt = src.read_text() if src.exists() else spec.replace(",", "\n")
    out = []
    for line in txt.split():
        line = line.strip()
        if ":" in line:
            p, v = line.split(":", 1)
            out.append((p, v))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["rank", "eval", "lpo", "run"])
    p.add_argument("answers", nargs="+")
    p.add_argument("--prefix", default="ref-")
    p.add_argument("--truth")
    p.add_argument("--views", default=",".join(VIEWS))
    p.add_argument("--stat", default="corr", choices=["corr", "mmd"])
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--by", default="mix", choices=["all", "hard", "mix"])
    p.add_argument("--strip", default="both", choices=["none", "affix", "sent", "both"])
    p.add_argument("--max-reps", type=int)
    p.add_argument("--features", help="liste pid:vue (fichier ou virgules) : pas de sélection")
    p.add_argument("--t-model", default="split", help="nombre, 'split' (plancher de bruit) ou 'calib'")
    p.add_argument("--split-factor", type=float, default=1.0)
    p.add_argument("--out")
    a = p.parse_args(argv)

    grouped = analyze.group(analyze.load_answers(a.answers), a.prefix)
    if a.max_reps:
        grouped = analyze.subsample(grouped, max_reps=a.max_reps)
    # répétitions complètes seulement : un (prompt, endpoint) incomplet fausserait la comparaison
    views = [v for v in a.views.split(",") if v]
    truth = analyze.read_truth(a.truth) if a.truth else None
    n = analyze.answers_per_endpoint(grouped)
    print(f"{len(grouped)} prompts, réponses/endpoint min {min(n.values())} max {max(n.values())}")
    fixed = _keys_from(a.features) if a.features else None

    if a.cmd == "rank":
        g = prepare(grouped, a.strip)
        eps = sorted({ep for d in g.values() for ep in d})
        fd = feature_distances(g, eps, views, a.stat)
        print("feature            AUC   AUC_dif")
        for k, auc, hard in rank_features(fd, eps, truth):
            print(f"  {k[0]:>4}:{k[1]:<7} {auc:.3f} {hard:.3f}")
        return 0
    if a.cmd == "eval":
        res = evaluate(grouped, truth, views, a.stat, a.top, a.by, strip=a.strip, fixed_keys=fixed)
        for i, f in enumerate(res["folds"]):
            if f is None:
                print(f"pli {i}: aucune feature")
                continue
            print(f"pli {i} (sélection {'paires' if i == 0 else 'impaires'}) : MCC sél {f['mcc_sel']:.3f} | "
                  f"MCC éval (seuil transféré {f['t']:.3f}) {f['mcc_eval']:.3f} | oracle {f['mcc_oracle']:.3f} | "
                  f"split-half t={f['t_split_half']:.3f} MCC {f['mcc_split_half']:.3f} | "
                  f"AUC {f['auc']:.3f} AUC_dif {f['auc_hard']:.3f}")
            print("   features : " + " ".join(f"{p}:{v}" for p, v in f["keys"]))
        return 0
    if a.cmd == "lpo":
        r = leave_pair_out(grouped, truth, views, a.stat, a.top, a.by, a.strip)
        print(f"leave-pair-out : MCC par paires {r['mcc']:.3f} sur {r['n']} paires")
        print("features les plus choisies : " + ", ".join(f"{p}:{v}×{c}" for (p, v), c in r["chosen"]))
        return 0
    # run : features fixes (ou toutes), seuil donné / split-half / calibré
    g = prepare(grouped, a.strip)
    eps = sorted({ep for d in g.values() for ep in d})
    fd = feature_distances(g, eps, views if fixed is None else sorted({v for _, v in fixed}), a.stat)
    keys = [k for k in (fixed or list(fd)) if k in fd]
    sc = median_scales(fd)
    D = combine_features({k: fd[k] / sc[k] for k in keys}, keys, scale="none")
    if a.t_model == "split":
        try:
            t = split_half_threshold(g, eps, keys, a.stat, factor=a.split_factor, fd_scale=sc)
        except ValueError as exc:
            print(f"erreur : {exc}", file=sys.stderr)
            return 2
    elif a.t_model == "calib":
        t, _ = best_threshold(D, eps, truth)
    else:
        t = float(a.t_model)
    if not np.isfinite(t):
        print(f"erreur : seuil non fini ({t})", file=sys.stderr)
        return 2
    rows = analyze.partition(eps, np.zeros_like(D), D, 10.0, t)
    print(f"{len(keys)} features, seuil {t:.4f}")
    print(analyze._fmt_matrix(eps, D))
    if truth:
        s = analyze.score(truth, rows)
        print(f"practice MCC {s['practice_mcc']:.3f}")
    if a.out:
        analyze.write_rows(rows, a.out)
        print(f"écrit : {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
