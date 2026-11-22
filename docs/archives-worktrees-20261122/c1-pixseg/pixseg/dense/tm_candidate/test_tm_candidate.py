"""Tests CPU du candidat TrustMark dense (données factices, backbone ResNet minuscule à poids aléatoires).

  /Users/selyan/dev/eaic-paris-2026/.venv/bin/python c1/pixseg/dense/tm_candidate/test_tm_candidate.py

1. Un « pilote » factice est produit par dense_train.run (phases dev, confirm) ; `tm_job.py measure` doit
   retrouver EXACTEMENT ses hash de noms appris (même recette), prédire toutes les TM F et R des folds évalués,
   reprendre les seuils du pilote, et (CPU déterministe) reproduire ses probabilités F.
2. `measure_routed.py` : routage par fold depuis `choices`, écarts nuls hors TM routées F, cas calculé à la main,
   décision INCOMPLET si une routée F manque, EXPORT_OK / PAS_D_EXPORT selon la règle.
3. `tm_job.py export` : refus sans décision favorable ; avec --force, seules les TM routées F changent, noms et
   format identiques au champion factice.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import data as pdata  # noqa: E402
import dense_train as dt  # noqa: E402
import synth_msg  # noqa: E402
from test_dense import fake_embed, gen_args, make_release  # noqa: E402
import measure_routed as mr  # noqa: E402
import tm_job  # noqa: E402

torch.set_num_threads(4)


def fixture(d: Path):
    root = make_release(d, per=10)
    # le test factice n'a pas de message : l'ajouter (le vrai test/metadata.json en a un, vérifié le 22/11)
    rng = np.random.default_rng(1)
    meta = json.loads((root / "test" / "metadata.json").read_text())
    for m in meta:
        m["message"] = "".join(map(str, rng.integers(0, 2, pdata.NBITS[m["method"]])))
    (root / "test" / "metadata.json").write_text(json.dumps(meta))
    synth_msg.gen(gen_args(root, "trustmark", n=48), embed=fake_embed, decode=None)
    a = dt.parse(["--method", "trustmark", "--synth", str(root / "syn_trustmark_s0_*.npz"), "--fake", "tiny",
                  "--phases", "dev,confirm", "--steps", "3", "--bs", "4", "--data", str(root),
                  "--out", str(root / "pilot"), "--device", "cpu", "--synth-val-frac", "0.3"])
    pilot, _ = dt.run(a)
    return root, pilot


def test_measure_reproduces_pilot_and_predicts_R():
    with tempfile.TemporaryDirectory() as d:
        root, pilot = fixture(Path(d))
        out = str(root / "tmc")
        f, rep = tm_job.measure(tm_job.parse(["measure", "--pilot", pilot, "--out", out, "--data", str(root),
                                              "--device", "cpu", "--fake", "tiny"]))
        z, P = np.load(f), np.load(pilot)
        R = pdata.load_meta("train", root); fo = pdata.real_folds(R["method"], R["full"], 0)
        tm = R["method"] == "trustmark"
        assert z["names"].tolist() == R["names"][tm].tolist() and np.array_equal(z["fold"], fo[tm])
        assert z["full"].sum() == 10 and (~z["full"]).sum() == 10
        assert np.isfinite(z["probs"].astype(np.float32)).all(), "toutes les TM (F et R) des folds 0-4 prédites"
        for tag in ("dev2", "dev3", "dev4", "confirm"):
            assert rep[tag]["hash"] == rep[tag]["hash_pilote"], tag
        pt = dict(zip(P["names"].tolist(), P["thr"].tolist()))
        for n, t in zip(z["names"].tolist(), z["thr"].tolist()):
            if n in pt:
                assert t == pt[n], "seuil figé = seuil du pilote"
        assert rep["repro_F"]["ecart_max"] < 1e-6, rep["repro_F"]  # CPU : même graine -> mêmes masques
        ip = {n: i for i, n in enumerate(z["names"].tolist())}
        diff = np.abs(z["probs"][[ip[n] for n in P["names"].tolist()]].astype(np.float32) - P["probs_oof"].astype(np.float32)).max()
        assert diff < 2e-3, diff
        h = torch.load(out + "_heads.pt", weights_only=False)
        assert set(h["heads"]) == {"dev2", "dev3", "dev4", "confirm"}
        assert not any(k.startswith("bb.") for k in h["heads"]["confirm"]) and "agree_scale" in h["heads"]["confirm"]


def _routed_case(d: Path, gain_tm_r: float):
    """10 images : TM F/R, PS ; folds réels (partition 0) ; scores de réf. connus ; dense parfait sur les TM."""
    names = np.array([f"im{i:02d}" for i in range(40)])
    method = np.array(["trustmark"] * 20 + ["pixelseal"] * 20); full = np.array(([True] * 10 + [False] * 10) * 2)
    af = np.where(full, 0.9, 0.5).astype(np.float32); af[10] = 0.7  # un TM vrai R routé F (gate 0,65)
    fo = pdata.real_folds(method, full, 0)
    masks = np.zeros((40, 128, 128), np.uint8); masks[:, :64] = 1
    head = np.full(40, 0.5)
    np.savez(d / "ref.npz", names=names, head=head[None],
             choices=json.dumps([{"seed": 0, "fold": k, "gate": 0.65} for k in range(5)]))
    np.savez(d / "acc.npz", names=names, method=method, acc_full=af, full_watermark=full)
    np.savez(d / "masks.npz", names=names, masks=masks)
    tm = np.nonzero(method == "trustmark")[0]
    probs = np.repeat(masks[tm][:, None].astype(np.float32), 1, 1)[:, 0]
    probs[tm == 10] = 1.0 if gain_tm_r > 0 else 0.0  # TM R routé F : tout-1 (mIoU 0,25) ou tout-0 (0,25)
    np.savez(d / "dense.npz", names=names[tm], probs=probs.astype(np.float16), thr=np.full(len(tm), 0.5))
    return names, method, full, fo


def test_measure_routed_hand_case():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d); names, method, full, fo = _routed_case(d, 1)
        out = mr.main(["--dense", str(d / "dense.npz"), "--ref", str(d / "ref.npz"), "--acc", str(d / "acc.npz"),
                       "--masks", str(d / "masks.npz"), "--out", str(d / "m.json")])
        r = out["ref"]["all"]
        # à la main : 10 TM F parfaits (1 - 0,5) + 1 TM R routé F (tout-1 sur masque moitié : IoU1 0,5, IoU0 0 -> 0,25)
        assert r["TM_F_routes_F"]["n"] == 10 and r["TM_R_routes_F"]["n"] == 1
        assert abs(r["gain_global"] - (10 * 0.5 + (0.25 - 0.5)) / 40) < 1e-9, r["gain_global"]
        assert abs(r["TM_R_routes_F"]["contribution_globale"] + 0.25 / 40) < 1e-9
        assert r["decision"] == "PAS_D_EXPORT"  # contribution R -0,00625 < -0,001
        rc = out["ref"]["confirm"]
        assert rc["TM_F_routes_F"]["n"] == int(((fo < 2) & full & (method == "trustmark")).sum())
        # image manquante -> INCOMPLET
        z = dict(np.load(d / "dense.npz")); keep = z["names"] != "im10"
        np.savez(d / "dense2.npz", **{k: v[keep] for k, v in z.items()})
        out = mr.main(["--dense", str(d / "dense2.npz"), "--ref", str(d / "ref.npz"), "--acc", str(d / "acc.npz"),
                       "--masks", str(d / "masks.npz")])
        assert out["decision"] == "INCOMPLET" and out["ref"]["all"]["missing"] == ["im10"]
        # second fichier qui comble la manquante : premier trouvé gagnant
        out = mr.main(["--dense", str(d / "dense2.npz"), str(d / "dense.npz"), "--ref", str(d / "ref.npz"),
                       "--acc", str(d / "acc.npz"), "--masks", str(d / "masks.npz")])
        assert out["ref"]["all"]["missing"] == []


def test_measure_routed_export_ok():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d); _routed_case(d, 1)
        acc = dict(np.load(d / "acc.npz")); acc["acc_full"][10] = 0.5  # le R n'est plus routé F
        np.savez(d / "acc.npz", **acc)
        out = mr.main(["--dense", str(d / "dense.npz"), "--ref", str(d / "ref.npz"), "--acc", str(d / "acc.npz"),
                       "--masks", str(d / "masks.npz")])
        assert out["decision"] == "EXPORT_OK" and out["export_ok"]


def test_export_gated_and_merge():
    with tempfile.TemporaryDirectory() as d:
        root, pilot = fixture(Path(d))
        meta = json.loads((root / "test" / "metadata.json").read_text())
        tn = [m["image_name"] for m in meta][::-1]  # ordre du champion différent de heats_test
        rng = np.random.default_rng(0)
        champ = torch.from_numpy(rng.random((len(tn), 128, 128)) > 0.5)
        torch.save({"image_names": tn, "masks": champ}, root / "champ.pt")
        hn = [m["image_name"] for m in meta]; hm = np.array([m["method"] for m in meta])
        af = np.where(np.arange(len(hn)) % 2 == 0, 0.9, 0.3).astype(np.float32)
        np.savez(root / "heats_test.npz", names=np.array(hn), method=hm, acc_full=af)
        (root / "no.json").write_text(json.dumps({"decision": "PAS_D_EXPORT", "export_ok": False}))
        base = ["export", "--pilot", pilot, "--data", str(root), "--device", "cpu", "--fake", "tiny",
                "--champion", str(root / "champ.pt"), "--heats-test", str(root / "heats_test.npz"),
                "--out", str(root / "cand"), "--thr", "0.5"]
        assert tm_job.export(tm_job.parse(base + ["--decision", str(root / "no.json")])) is None
        assert not (root / "cand.pt").exists()
        info = tm_job.export(tm_job.parse(base + ["--force"]))
        c = torch.load(root / "cand.pt", weights_only=True)
        assert c["image_names"] == tn and c["masks"].shape == champ.shape and c["masks"].dtype == torch.bool
        routed = {n for n, m, a in zip(hn, hm, af) if m == "trustmark" and a >= 0.65}
        z = np.load(root / "cand_tmprobs.npz")
        for i, n in enumerate(tn):
            if n in routed:
                j = z["names"].tolist().index(n)
                p16 = z["probs"][j].astype(np.float32)  # float16 sauvegardé : tolérance au seuil
                far = np.abs(p16 - 0.5) > 1e-3
                assert np.array_equal(c["masks"][i].numpy()[far], (p16 > 0.5)[far]), n
            else:
                assert torch.equal(c["masks"][i], champ[i]), n
        assert info["n_tm_routes_F"] == len(routed) and info["sha256"] == tm_job.sha256(root / "cand.pt")


if __name__ == "__main__":
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
    fails = 0
    for n, f in tests:
        t0 = time.time()
        try:
            f(); print(f"OK    {n} ({time.time() - t0:.1f} s)", flush=True)
        except Exception:
            fails += 1; print(f"ÉCHEC {n}", flush=True); traceback.print_exc()
    print(f"{len(tests) - fails}/{len(tests)} tests OK")
    sys.exit(1 if fails else 0)
