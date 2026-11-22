"""Réparation affine unique, CPU, sur features figées ; aucune API ni export state_dict.

Préserve les logits du champion sur forget (hors constante softmax), et rapproche
des logits d'origine sur les photos de réparation prédites hors classes forget.
Ce sont des photos externes sans vrais labels : ce contrôle n'est pas le retain caché.
Le candidat reste une tête W,b expérimentale, jamais un score serveur estimé.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def augment(x):
    return np.column_stack((np.asarray(x, dtype=np.float64), np.ones(len(x))))


def log_probs(z, t=2.):
    z = np.asarray(z, dtype=np.float64)/t
    z = z-z.max(1, keepdims=True)
    return z-np.log(np.exp(z).sum(1, keepdims=True))


def stats(ref, cand, labels=None):
    lr, lc = log_probs(ref), log_probs(cand)
    p = np.exp(lr)
    out = dict(n=len(ref), agreement=float(np.mean(ref.argmax(1)==cand.argmax(1))),
               KL_original_candidate=float(np.mean(np.sum(p*(lr-lc), 1))))
    if labels is not None:
        ce = -log_probs(cand, 1.)[np.arange(len(cand)), labels]
        out.update(CE_pseudo_mean=float(ce.mean()), CE_pseudo_std=float(ce.std()),
                   flip_pseudo=float(np.mean(cand.argmax(1)!=labels)))
    return out


def solve_correction(xf, xr, delta, ridge_relative=1e-3):
    """Min ||XF*C||²/NF + ||XR*(delta+C)||²/NR + rho||C||².

    delta et C sont [features+intercept, classes], centrés entre classes.
    Les deux groupes pèsent chacun 1 ; pas de grille ni sélection sur le contrôle.
    """
    gf, gr = xf.T@xf/len(xf), xr.T@xr/len(xr)
    gram = gf+gr
    rho = ridge_relative*np.trace(gram)/len(gram)
    correction = np.linalg.solve(gram+rho*np.eye(len(gram)), -gr@delta)
    correction -= correction.mean(1, keepdims=True)
    return correction, float(rho)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    a = ap.parse_args()
    if a.out_dir.exists():
        raise FileExistsError(a.out_dir)
    started = time.monotonic()
    z = np.load(a.cache)
    for k in ('W_orig','b_orig','W_push_best','b_push_best','f16_forget','f16_offall','y'):
        if k not in z.files:
            raise KeyError(k)
    if int(z['head_only_push_best']) != 1:
        raise ValueError('push_best change le backbone ; rejeu interdit')
    b0 = np.vstack((z['W_orig'].T, z['b_orig']))
    bp = np.vstack((z['W_push_best'].T, z['b_push_best']))
    champ = b0 + 1.75*(bp-b0)
    delta = np.asarray(champ-b0, dtype=np.float64)
    delta -= delta.mean(1, keepdims=True)
    xf, xr = augment(z['f16_forget']), augment(z['f16_offall'])
    y = z['y'].astype(int)
    affected = np.unique(y)
    pred_r = (xr@b0).argmax(1)
    is_affected = np.isin(pred_r, affected)
    rng = np.random.default_rng(20261121)
    perm = rng.permutation(len(xr))
    rep, ctl = np.sort(perm[:len(xr)//2]), np.sort(perm[len(xr)//2:])
    # Nouvel ajustement : forget split fixé, photos selon partition historique connue.
    perm_f = np.random.default_rng(20261122).permutation(len(xf))
    train_f, diag_f = perm_f[:len(xf)//2], perm_f[len(xf)//2:]
    train_r = rep[~is_affected[rep]]
    correction, rho = solve_correction(xf[train_f], xr[train_r], delta)
    candidate = champ+correction
    result = dict(nature='Diagnostic local sur photos externes ; aucun gold ni accuracy retain',
                  scale_champion=1.75, rho=rho, affected_classes=affected.tolist(),
                  fitting=dict(forget=len(train_f), outside_affected=len(train_r)),
                  control_already_consulted=True, metrics={})
    for name, b in [('original',b0), ('push_s175',champ), ('repair',candidate)]:
        out = {}
        for subset, ix in [('forget_fit',train_f),('forget_diag',diag_f)]:
            out[subset] = stats(xf[ix]@b0, xf[ix]@b, y[ix])
        for subset, ix in [('repair',rep),('control',ctl)]:
            for affected_flag, suffix in [(True,'affected'),(False,'other')]:
                rows = ix[is_affected[ix]==affected_flag]
                out[f'{subset}_{suffix}'] = stats(xr[rows]@b0,xr[rows]@b)
        for key in ('altered','pshuffle','lowpass32','mask35','off'):
            if f'f16_{key}' in z.files:
                x=augment(z[f'f16_{key}'])
                out[key]=stats(x@b0,x@b, y if key=='altered' else None)
        result['metrics'][name]=out
    # Fidélité au champion explicite : préserver la forget est l'objectif de ce mécanisme.
    idx=diag_f
    result['champion_candidate_forget_diag'] = stats(xf[idx]@champ, xf[idx]@candidate)
    if not np.isfinite(candidate).all():
        raise ValueError('Tête non finie')
    result['elapsed_seconds']=time.monotonic()-started
    result['source_sha256']=hashlib.sha256(a.cache.read_bytes()).hexdigest()
    a.out_dir.mkdir(parents=True)
    np.savez(a.out_dir/'head_experimental.npz', W=candidate[:-1].T.astype('float32'),
             b=candidate[-1].astype('float32'))
    (a.out_dir/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
