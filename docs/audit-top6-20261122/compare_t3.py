"""Comparer deux directions de tête à CE moyenne forget égale sur la moitié ajustement."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'relais-theorie-operationnel-20261122'
sys.path.insert(0, str(SOURCE))
from t3_repair_head import augment, stats


def mean_ce(logits, y):
    return float(np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(y)), y]))


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(cache, head):
    with np.load(cache) as data, np.load(head) as repaired:
        if int(data['head_only_push_best']) != 1:
            raise ValueError("Le cache doit correspondre à une modification de la tête seule.")
        original = np.vstack((data['W_orig'].T, data['b_orig']))
        push = np.vstack((data['W_push_best'].T, data['b_push_best']))
        champion = original + 1.75 * (push - original)
        repair = np.vstack((repaired['W'].T, repaired['b']))
        labels = data['y'].astype(int)
        features = {key: augment(data['f16_' + key]) for key in ('forget', 'altered', 'offall')}
    fit, diag = np.array_split(np.random.default_rng(20261122).permutation(len(labels)), 2)
    _, control = np.array_split(np.random.default_rng(20261121).permutation(len(features['offall'])), 2)
    # Rejouer les partitions fixées par l'expérience de réparation.
    matrices = [m.astype(np.float64) for m in (original, champion, repair)]
    logits = {k: [x @ m for m in matrices] for k, x in features.items()}
    base, ref, alt = [x[fit] for x in logits['forget']]
    def match(direction, target):
        target_ce = mean_ce(target, labels[fit])
        def residual(scale):
            return mean_ce(base + scale * (direction - base), labels[fit]) - target_ce
        return float(brentq(residual, 0., 5.))
    up = match(alt, ref)
    down = match(ref, alt)
    affected = np.isin(logits['offall'][0].argmax(1), np.unique(labels))
    groups = [('forget_fit', 'forget', fit), ('forget_diag', 'forget', diag),
              ('altered_diag', 'altered', diag),
              ('external_affected', 'offall', control[affected[control]]),
              ('external_other', 'offall', control[~affected[control]])]
    rows = {}
    for name, key, ix in groups:
        base, ref, alt = [x[ix] for x in logits[key]]
        candidates = {'champion': ref, 'repair': alt, 'repair_at_champion_ce': base + up * (alt - base),
                      'champion_at_repair_ce': base + down * (ref - base)}
        rows[name] = {k: stats(base, v, labels[ix] if key != 'offall' else None)
                      for k, v in candidates.items()}
    for left, right in [('champion', 'repair_at_champion_ce'), ('repair', 'champion_at_repair_ce')]:
        assert abs(rows['forget_fit'][left]['CE_pseudo_mean']
                   - rows['forget_fit'][right]['CE_pseudo_mean']) < 1e-8
    return dict(cache_sha256=fingerprint(cache), head_sha256=fingerprint(head),
                repair_scale_to_champion=up, champion_scale_to_repair=down, metrics=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('--head', type=Path, default=SOURCE / 't3_diagnostic' / 'head_experimental.npz')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.cache, args.head)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
