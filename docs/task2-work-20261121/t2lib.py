"""Bibliothèque commune : features sur 5 réponses (une par prompt f01..f05), règles de décision, MCC."""
import math, re, itertools, collections
PIDS = ["f01", "f02", "f03", "f04", "f05"]

def ngrams(s, n=3):
    s = re.sub(r"\s+", " ", s.lower())
    return collections.Counter(s[i:i + n] for i in range(max(1, len(s) - n + 1)))

def cos(a, b):
    num = sum(v * b.get(k, 0) for k, v in a.items())
    den = math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))
    return num / den if den else 0.0

def prefix(s, k=3):
    return " ".join(re.findall(r"\w+", s.lower())[:k])

def pair_dist(A, B, feat):
    """A, B : dict pid -> texte. Distance moyenne sur les prompts présents des deux côtés (couverture rendue)."""
    ds = []
    for p in PIDS:
        if p not in A or p not in B: continue
        a, b = A[p], B[p]
        if feat == "len":
            ds.append(abs(math.log((len(a) + 20) / (len(b) + 20))))
        elif feat == "ngram":
            ds.append(1 - cos(ngrams(a), ngrams(b)))
        elif feat == "prefix":
            ds.append(float(prefix(a) != prefix(b)))
        elif feat == "len+ngram":
            ds.append(0.5 * min(1.0, abs(math.log((len(a) + 20) / (len(b) + 20)))) + 0.5 * (1 - cos(ngrams(a), ngrams(b))))
        else: raise ValueError(feat)
    return (sum(ds) / len(ds) if ds else None), len(ds)

def dist_matrix(data, eps, feat):
    D, cov = {}, {}
    for a, b in itertools.combinations(eps, 2):
        d, n = pair_dist(data[a], data[b], feat)
        D[a, b] = D[b, a] = d; cov[a, b] = n
    return D, cov

def rule_isolate(eps, D, tau):
    """Gros groupe + isolés : score_i = médiane des distances de i aux autres ; isolé si z robuste > tau."""
    s = {e: sorted(D[e, o] for o in eps if o != e)[(len(eps) - 1) // 2] for e in eps}
    v = sorted(s.values()); med = v[len(v) // 2]
    mad = sorted(abs(x - med) for x in v)[len(v) // 2] or 1e-9
    iso = [e for e in eps if (s[e] - med) / (1.4826 * mad) > tau]
    return {e: (e if e in iso else "BIG") for e in eps}

def rule_linkage(eps, D, q):
    """Lien moyen, coupe à la q-quantile des distances intra-groupe (seuil relatif, transférable)."""
    vals = sorted(D[a, b] for a, b in itertools.combinations(eps, 2))
    t = vals[min(len(vals) - 1, int(q * len(vals)))]
    cl = [[e] for e in eps]
    while True:
        best = None
        for i, j in itertools.combinations(range(len(cl)), 2):
            d = sum(D[a, b] for a in cl[i] for b in cl[j]) / (len(cl[i]) * len(cl[j]))
            if best is None or d < best[0]: best = (d, i, j)
        if best is None or best[0] > t: break
        _, i, j = best; cl[i] += cl[j]; del cl[j]
    return {e: "C%d" % k for k, c in enumerate(cl) for e in c}

def mcc_counts(tp, fp, fn, tn):
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / d if d else 0.0

def pair_mcc(truth, pred, pairs=None, clip=True):
    eps = sorted(truth)
    pairs = pairs or list(itertools.combinations(eps, 2))
    tp = fp = fn = tn = 0
    for a, b in pairs:
        t, g = truth[a] == truth[b], pred[a] == pred[b]
        tp += t and g; fp += (not t) and g; fn += t and not g; tn += (not t) and not g
    m = mcc_counts(tp, fp, fn, tn)
    return (max(0.0, m) if clip else m), (tp, fp, fn, tn)

def server_score(tfam, tmod, pfam, pmod, pairs=None, clip=True):
    """0,1·MCC famille (toutes paires) + 0,9·MCC modèle (paires intra-famille VRAIE) ; modèle prédit = (famille, modèle)."""
    eps = sorted(tfam)
    pairs = pairs or list(itertools.combinations(eps, 2))
    mf, _ = pair_mcc(tfam, pfam, pairs, clip)
    intra = [(a, b) for a, b in pairs if tfam[a] == tfam[b]]
    T = {e: (tfam[e], tmod[e]) for e in eps}; P = {e: (pfam[e], pmod[e]) for e in eps}
    mm, cnt = pair_mcc(T, P, intra, clip)
    return 0.1 * mf + 0.9 * mm, mf, mm, cnt
