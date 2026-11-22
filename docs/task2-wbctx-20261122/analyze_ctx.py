RULE = """RÈGLE FIGÉE (écrite le 22/11/2026 vers 05:20 Paris, AVANT tout calcul sur les refs ; ne plus modifier)
Données : ctx_refs.npz (compute_ctx.py), 9 refs × 44 prompts × 4 répétitions, lp_b/lp_q[réponse, contexte, T].
Features d'un endpoint sur un ensemble R de réponses (moyenne par réponse, puis sur R) :
  M(c,T) = mean_{r in R} lp_b[r,c,T] ; F1 = max_{c,T} M ; (c*,T*) = argmax (premier en cas d'égalité) ;
  F2 = 1000 * mean_{r in R} (lp_q - lp_b)[r,c*,T*]  (mnat/token).
  Profilé : c parcourt C0..C5. Contrôle : c = C0 seul.
z robuste dans la famille (les 9 refs, même ensemble de prompts) : z = (F - médiane) / (1.4826 * MAD), MAD plancher 1e-9.
Isolement : e isolé si z1(e) < s1 ou z2(e) > s2. Partition = un groupe commun + un singleton par isolé.
Seuils candidats : s1 dans {-inf} U milieux des z1 triés de la moitié de sélection ; s2 dans {+inf} U milieux des z2 triés.
Sélection : parmi les (s1,s2) n'isolant AUCUN déguisement (01,03,05,06,08) sur la sélection, maximiser le MCC paires
  (anlib.mcc) ; égalités : moins d'isolés, puis moins de seuils actifs, puis s1 le plus bas, puis s2 le plus haut.
Moitiés : H0 = rép. 0-1, H1 = rép. 2-3. Sens H0->H1 et H1->H0.
Protocole principal (décisionnel) : 200 tirages (graine 20261122) ; à chaque tirage et pour chaque moitié, 30 prompts tirés
  sans remise parmi 44 et, pour chacun, une des 2 répétitions de la moitié (mêmes (prompt, rép.) pour les 9 refs) ;
  sélection sur le tirage de la moitié de sélection, évaluation sur le tirage indépendant de l'autre moitié.
Rapportés aussi : 88/88 (moitiés complètes, déterministe) et sélection 88 -> évaluation 30 (non décisionnel).
Précision d'isolement = fraction des isolés (évaluation) vraiment distincts (02,04,07,09) ; moyenne sur les tirages ayant au
  moins un isolé (nombre de tirages sans isolé rapporté).
GO si, dans les DEUX sens, à 30 réponses (protocole principal, profilé) : MCC médian >= 0,45 ET précision moyenne >= 0,75.
Sinon NO-GO. Le contrôle C0 seul suit exactement la même règle et sert à mesurer l'apport du profilage."""
import hashlib, sys, json, itertools, numpy as np
sys.path.insert(0, "/Users/selyan/dev/eaic-paris-2026/docs/task2-bench-20261121")
from anlib import mcc, TRUTH, KIND, EPS
RULE_SHA = hashlib.sha256(RULE.encode()).hexdigest()
DIST = {e for e in EPS if TRUTH[e] != "S"}; DISG = set(EPS) - DIST

def feats(z, idx, ctxs):
    """idx : indices de réponses ; ctxs : liste de contextes autorisés -> F1, F2, c*, T*."""
    lb = z["lp_b"][idx][:, ctxs]; lq = z["lp_q"][idx][:, ctxs]          # (n, C', K)
    M = lb.mean(0)                                                       # (C', K)
    c, k = np.unravel_index(int(np.argmax(M)), M.shape)
    d = (lq[:, c, k] - lb[:, c, k]) * 1000
    return dict(F1=float(M[c, k]), F2=float(d.mean()), F2_se=float(d.std(ddof=1) / np.sqrt(len(d))),
                c=int(ctxs[c]), T=float(z["TG"][k]), n=int(len(idx)))

def zrob(v):
    v = np.asarray(v, float); m = np.median(v); mad = max(1.4826 * np.median(np.abs(v - m)), 1e-9)
    return (v - m) / mad

def family_z(z, sel, ctxs):
    """sel : dict ep -> indices. Renvoie dict ep -> (z1, z2) et les features brutes."""
    F = {e: feats(z, sel[e], ctxs) for e in EPS}
    z1 = zrob([F[e]["F1"] for e in EPS]); z2 = zrob([F[e]["F2"] for e in EPS])
    return {e: (z1[i], z2[i]) for i, e in enumerate(EPS)}, F

def isolate(Z, s1, s2): return {e for e in EPS if Z[e][0] < s1 or Z[e][1] > s2}
def partition(iso): return {e: (e if e in iso else "G") for e in EPS}

def mids(v):
    v = sorted(v); return [(a + b) / 2 for a, b in zip(v, v[1:])]

def select(Z):
    c1 = [-np.inf] + mids([Z[e][0] for e in EPS]); c2 = [np.inf] + mids([Z[e][1] for e in EPS])
    best = None
    for s1, s2 in itertools.product(c1, c2):
        iso = isolate(Z, s1, s2)
        if iso & DISG: continue
        key = (mcc(TRUTH, partition(iso)), -len(iso), -(int(np.isfinite(s1)) + int(np.isfinite(s2))), -s1, s2)
        if best is None or key > best[0]: best = (key, s1, s2)
    return best[1], best[2], best[0][0]

def evaluate(Z, s1, s2):
    iso = isolate(Z, s1, s2)
    return dict(mcc=mcc(TRUTH, partition(iso)), iso=sorted(iso),
                prec=(len(iso & DIST) / len(iso)) if iso else None, disg_iso=len(iso & DISG))

def idx_half(z, reps, pick=None):
    """pick : dict pid -> rep (sous-échantillon) ; sinon toute la moitié."""
    out = {}
    for e in EPS:
        m = (z["ep"] == e) & np.isin(z["rep"], reps)
        if pick is not None:
            m &= np.array([pick.get(p, -1) == r for p, r in zip(z["pid"], z["rep"])])
        out[e] = np.where(m)[0]
    return out

def draw(rng, pids, reps, k=30):
    ch = rng.choice(pids, k, replace=False); return {p: int(rng.choice(reps)) for p in ch}

if __name__ == "__main__":
    z = np.load(sys.argv[1]); outp = sys.argv[2]
    assert not np.isnan(z["lp_b"]).any() and not np.isnan(z["lp_q"]).any()
    pids = sorted(set(z["pid"])); H = {"H0": (0, 1), "H1": (2, 3)}
    print("RULE_SHA256", RULE_SHA, "prompts", len(pids))
    res = {"rule_sha256": RULE_SHA, "ctx": list(z["ctx"])}
    for name, ctxs in [("profile", list(range(len(z["ctx"])))), ("C0", [0])]:
        R = res[name] = {}
        full = {h: family_z(z, idx_half(z, r), ctxs) for h, r in H.items()}
        R["table"] = {h: {e: dict(full[h][1][e], z1=float(full[h][0][e][0]), z2=float(full[h][0][e][1])) for e in EPS} for h in H}
        print(f"\n=== {name} : features moitiés complètes (88 réponses) ===")
        for e in EPS:
            a, b = R["table"]["H0"][e], R["table"]["H1"][e]
            print(f"{e} {KIND[e]:8s} F1 {a['F1']:+.4f}/{b['F1']:+.4f} z1 {a['z1']:+6.2f}/{b['z1']:+6.2f} | F2 {a['F2']:+6.1f}/{b['F2']:+6.1f} (se {a['F2_se']:.1f}) z2 {a['z2']:+6.2f}/{b['z2']:+6.2f} | c*,T* {a['c']},{a['T']}/{b['c']},{b['T']}")
        rng = np.random.default_rng(20261122)
        for s, t in [("H0", "H1"), ("H1", "H0")]:
            key = f"{s}->{t}"; D = R[key] = {}
            s1, s2, ms = select(full[s][0]); ev = evaluate(full[t][0], s1, s2)
            D["88_88"] = dict(s1=s1, s2=s2, mcc_sel=ms, **ev)
            print(f"{name} {key} 88/88 : s1 {s1:.2f} s2 {s2:.2f} MCC sel {ms:.3f} HORS SEL {ev['mcc']:.3f} isolés {ev['iso']} prec {ev['prec']}")
        rng = np.random.default_rng(20261122)
        draws = [[(draw(rng, pids, H[h]) ) for h in ("H0", "H1")] for _ in range(200)]
        for s, t in [("H0", "H1"), ("H1", "H0")]:
            key = f"{s}->{t}"; si, ti = int(s[1]), int(t[1])
            s1f, s2f, _ = select(full[s][0])
            prim, sec = [], []
            for dr in draws:
                Zs = family_z(z, idx_half(z, H[s], dr[si]), ctxs)[0]
                Zt = family_z(z, idx_half(z, H[t], dr[ti]), ctxs)[0]
                s1, s2, _ = select(Zs); prim.append(evaluate(Zt, s1, s2)); sec.append(evaluate(Zt, s1f, s2f))
            for lab, L in [("30_30", prim), ("88_30", sec)]:
                m = np.array([x["mcc"] for x in L]); pr = [x["prec"] for x in L if x["prec"] is not None]
                cnt = {}
                for x in L:
                    for e in x["iso"]: cnt[e] = cnt.get(e, 0) + 1
                R[key][lab] = dict(mcc_median=float(np.median(m)), mcc_q10=float(np.quantile(m, .1)), mcc_q90=float(np.quantile(m, .9)),
                                   mcc_mean=float(m.mean()), prec_mean=float(np.mean(pr)) if pr else None, n_no_iso=sum(x["prec"] is None for x in L),
                                   n_any_disg_iso=sum(x["disg_iso"] > 0 for x in L), iso_freq={e: cnt[e] / len(L) for e in sorted(cnt)})
                d = R[key][lab]
                print(f"{name} {key} {lab} : MCC méd {d['mcc_median']:.3f} [q10 {d['mcc_q10']:.3f}, q90 {d['mcc_q90']:.3f}] moy {d['mcc_mean']:.3f} prec moy {d['prec_mean']} sans isolé {d['n_no_iso']} avec déguisement isolé {d['n_any_disg_iso']} freq {d['iso_freq']}")
        ok = all(R[k]["30_30"]["mcc_median"] >= 0.45 and (R[k]["30_30"]["prec_mean"] or 0) >= 0.75 for k in ("H0->H1", "H1->H0"))
        R["GO"] = bool(ok); print(f"{name} critère : {'GO' if ok else 'NO-GO'}")
    res["verdict"] = "GO" if res["profile"]["GO"] else "NO-GO"
    json.dump(res, open(outp, "w"), indent=1, default=float)
    print("VERDICT (profilé) :", res["verdict"])
