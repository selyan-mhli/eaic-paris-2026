"""Une réparation figée de la tête. Toutes les vues viennent des images forget."""

import argparse
import json

import numpy as np
from scipy.optimize import brentq

from compare_t3 import HERE, augment, fingerprint, mean_ce, stats
from t3_repair_head import solve_correction


def run(cache):
    with np.load(cache) as data:
        assert int(data['head_only_push_best']) == 1
        y = data['y'].astype(int)
        original = np.vstack((data['W_orig'].T, data['b_orig']))
        push = np.vstack((data['W_push_best'].T, data['b_push_best']))
        champion = (original + 1.75*(push-original)).astype(np.float64)
        original = original.astype(np.float64)
        X = {k: augment(data['f16_'+k]) for k in ('forget','altered','pshuffle','mask35','lowpass32')}
    # Chaque image et toutes ses vues restent du même côté du découpage.
    fit, calibrate, test = np.split(np.random.default_rng(20261123).permutation(len(y)), [1400,2450])
    classes = np.unique(y)
    def outside(x):
        return ~np.isin((x @ original).argmax(1), classes)
    anchors = X['pshuffle'][fit]
    anchors = anchors[outside(anchors)]
    if len(anchors) < 2:
        raise ValueError("Nombre insuffisant d'ancres hors des classes forget.")
    delta = champion - original
    delta -= delta.mean(1, keepdims=True)
    correction, rho = solve_correction(X['forget'][fit], anchors, delta)
    direction = delta + correction
    cal_x = X['forget'][calibrate]
    base, change = cal_x @ original, cal_x @ direction
    target = mean_ce(cal_x @ champion, y[calibrate])
    scale = brentq(lambda a: mean_ce(base + a*change, y[calibrate]) - target, 0, 5)
    candidate = original + scale*direction
    assert abs(mean_ce(cal_x @ candidate, y[calibrate])-target) < 1e-8
    out = {}
    for key, features in X.items():
        x = features[test]
        labels = y[test] if key in ('forget','altered') else None
        if labels is None:
            x = x[outside(x)]
        if not len(x):
            out[key] = {'n': 0}
            continue
        base, ref, pred = x @ original, x @ champion, x @ candidate
        out[key] = {'n': len(x), 'champion':stats(base,ref,labels), 'candidate':stats(base,pred,labels)}
    return dict(n_fit=len(fit), n_calibrate=len(calibrate), n_test=len(test), n_anchors=len(anchors),
                scale=scale, rho=rho, calibration_ce=target, source_sha256=fingerprint(cache), metrics=out)


if __name__ == '__main__':
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache',type=Path)
    args = parser.parse_args()
    (HERE/'t3-forget-seul.json').write_text(json.dumps(run(args.cache),indent=2)+'\n')
