"""Outils d'analyse communs : vérité refs, MCC par paires, lien moyen, validation par moitiés."""
import itertools, math, json, re, collections
TRUTH = {"ref-01": "S", "ref-02": "T1", "ref-03": "S", "ref-04": "T3", "ref-05": "S",
         "ref-06": "S", "ref-07": "T4", "ref-08": "S", "ref-09": "T2"}
KIND = {"ref-01": "plain", "ref-02": "fp8", "ref-03": "sys", "ref-04": "lora0.7", "ref-05": "persona",
        "ref-06": "wm", "ref-07": "lora1.5", "ref-08": "temp1", "ref-09": "lora0.3"}
EPS = sorted(TRUTH)

def mcc(truth, pred):
    tp = fp = fn = tn = 0
    for a, b in itertools.combinations(sorted(truth), 2):
        t, g = truth[a] == truth[b], pred[a] == pred[b]
        tp += t and g; fp += (not t) and g; fn += t and not g; tn += (not t) and not g
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return max(0.0, (tp * tn - fp * fn) / d) if d else 0.0

def avg_link(eps, D, thr):
    """D[(a,b)] distance ; fusion tant que la distance moyenne entre clusters < thr."""
    cl = [[e] for e in eps]
    while len(cl) > 1:
        best = None
        for i, j in itertools.combinations(range(len(cl)), 2):
            d = sum(D[tuple(sorted((a, b)))] for a in cl[i] for b in cl[j]) / (len(cl[i]) * len(cl[j]))
            if best is None or d < best[0]: best = (d, i, j)
        if best[0] >= thr: break
        _, i, j = best; cl[i] += cl[j]; del cl[j]
    return {e: k for k, c in enumerate(cl) for e in c}

def pair_errors(pred):
    """Taux d'erreur par type de paire impliquant la base."""
    out = collections.defaultdict(list)
    for a, b in itertools.combinations(EPS, 2):
        same = TRUTH[a] == TRUTH[b]
        if same: key = "deguis-deguis"
        elif TRUTH[a] == "S" or TRUTH[b] == "S":
            v = KIND[b] if TRUTH[a] == "S" else KIND[a]; key = "base-" + v
        else: key = "variant-variant"
        out[key].append(int((pred[a] == pred[b]) != same))
    return {k: round(sum(v) / len(v), 3) for k, v in sorted(out.items())}

def choose_and_eval(Dsel, Deval, thrs):
    best = max(thrs, key=lambda t: (mcc(TRUTH, avg_link(EPS, Dsel, t)), -t))
    psel = avg_link(EPS, Dsel, best); pev = avg_link(EPS, Deval, best)
    return best, mcc(TRUTH, psel), mcc(TRUTH, pev), pev

def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
