"""Test CPU de bout en bout de cond_distill_all.py (--fit-all) sur données synthétiques (poids aléatoires, 16 images).

Vérifie : équivalence trunk/tail_feats == model(x) ; enseignant tête seule (×k) ; partitions disjointes ;
2 pas d'entraînement ; blocs gelés inchangés ; state_dict float32 rechargé strict=True ; eval.json complet.
Usage : python test_cond_distill.py <dossier_temporaire>
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cond_distill_all as cd  # noqa: E402


def main():
    tmp = Path(sys.argv[1]); tmp.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    m = cd.build_model().eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        ref = m(x)
        alt = m.head(cd.tail_feats(m, cd.trunk(m, x, 8), 8))
    assert torch.allclose(ref, alt, atol=1e-5), (ref - alt).abs().max()
    o = {k: v.clone() for k, v in m.state_dict().items()}
    pb = {k: v.clone() for k, v in o.items()}
    pb["head.weight"] += 0.05 * torch.randn_like(pb["head.weight"]); pb["head.bias"] += 0.1
    s175 = {k: v.clone() for k, v in o.items()}
    s175["head.weight"] = o["head.weight"] + 1.75 * (pb["head.weight"] - o["head.weight"])
    s175["head.bias"] = o["head.bias"] + 1.75 * (pb["head.bias"] - o["head.bias"])
    W, b = cd.make_teacher_head(o, push_best_sd=pb, scale=1.75)
    assert torch.allclose(W, s175["head.weight"], atol=1e-6) and torch.allclose(b, s175["head.bias"], atol=1e-6)
    bad = {k: v.clone() for k, v in o.items()}; bad["blocks.0.ls1.gamma"] += 1
    try:
        cd.make_teacher_head(o, teacher_sd=bad); raise AssertionError("diff hors tête non détectée")
    except ValueError:
        pass
    torch.save(o, tmp / "model.pt"); torch.save(pb, tmp / "push_best.pt"); torch.save(s175, tmp / "push_s175.pt")
    rng = np.random.default_rng(0)
    np.savez(tmp / "forget.npz", images=rng.integers(0, 256, (16, 224, 224, 3), dtype=np.uint8))
    np.savez(tmp / "off.npz", images=rng.integers(0, 256, (16, 224, 224, 3), dtype=np.uint8))
    idx = np.arange(16)[0::4]
    np.savez(tmp / "var_t_0.npz", idx=idx, img_s40=rng.integers(0, 256, (4, 224, 224, 3), dtype=np.uint8),
             img_s60=rng.integers(0, 256, (4, 224, 224, 3), dtype=np.uint8))
    # augmentations : forme, plage, finitude
    g = torch.Generator().manual_seed(0)
    u8 = torch.from_numpy(rng.integers(0, 256, (12, 224, 224, 3), dtype=np.uint8))
    xa = cd.augment(u8, g, torch.device("cpu"), p_clean=0.0)
    assert xa.shape == (12, 3, 224, 224) and torch.isfinite(xa).all() and xa.min() >= -1 and xa.max() <= 1
    out = tmp / "out"
    for tag, extra in (("a", ["--teacher", str(tmp / "push_s175.pt"), "--K", "2"]),
                       ("b", ["--push-best", str(tmp / "push_best.pt"), "--teacher-scale", "2.0", "--K", "4",
                              "--lam", "3"])):
        cmd = [sys.executable, str(HERE / "cond_distill_all.py"), "--model", str(tmp / "model.pt"),
               "--forget", str(tmp / "forget.npz"), "--off-all", str(tmp / "off.npz"),
               "--gen-glob", str(tmp / "var_t_*.npz"), "--out-dir", str(out), "--tag", tag, "--tiny", "--fit-all", *extra]
        subprocess.run(cmd, check=True)
        rep = json.loads((out / tag / "eval.json").read_text())
        sd = torch.load(out / tag / "student.pt", map_location="cpu", weights_only=True)
        chk = cd.build_model(); chk.load_state_dict(sd, strict=True)
        assert all(v.dtype == torch.float32 for v in sd.values() if v.is_floating_point())
        K = int(rep["args"]["K"])
        changed = {k.split(".")[1] for k in sd if k.startswith("blocks.") and not torch.equal(sd[k], o[k])}
        assert changed and all(int(i) >= 12 - K for i in changed), changed
        for k in ("patch_embed.proj.weight", "cls_token", "pos_embed"):
            assert torch.equal(sd[k], o[k]), k
        for s in ("forget_all", "altered_all", "gen_all", "pshuffle_all", "lowpass32_all", "mask35_all"):
            assert s in rep["metrics"]["student_vs_orig"], s
        assert "submittable" in rep["decision"] and rep["steps_done"] == 2
        print(tag, "OK", rep["decision"]["submittable"], "s/pas", round(rep["s_per_step_mean"], 2))
    # règle : cas construits à la main
    m = {"student_vs_orig": {"forget_all": dict(agreement=0.60, KL=1.4), "control_other": dict(agreement=1, KL=0.20),
                             "control_affected": dict(agreement=0.40, KL=0.5)},
         "teacher_vs_orig": {"forget_all": dict(agreement=0.57, KL=1.45), "control_other": dict(agreement=1, KL=0.46),
                             "control_affected": dict(agreement=0.05, KL=1.5)},
         "student_vs_teacher": {"forget_all": dict(agreement=0.9, KL=0.08), "altered_all": dict(agreement=0.9, KL=0.12)}}
    assert cd.decide(m)["submittable"] is True
    m["student_vs_orig"]["control_affected"]["agreement"] = 0.34   # < 0,05 + 0,30
    assert cd.decide(m)["submittable"] is False
    m["student_vs_orig"]["control_affected"]["agreement"] = 0.40; m["student_vs_orig"]["control_other"]["KL"] = 0.24
    assert cd.decide(m)["submittable"] is False                    # > 0,5 × 0,46
    m["student_vs_orig"]["control_other"]["KL"] = 0.20; m["student_vs_teacher"]["altered_all"]["KL"] = 0.16
    assert cd.decide(m)["submittable"] is False
    m["student_vs_teacher"]["altered_all"]["KL"] = 0.12; m["student_vs_orig"]["forget_all"]["agreement"] = 0.63
    assert cd.decide(m)["submittable"] is False                    # |0,63 - 0,57| > 0,05
    print("TOUS LES TESTS OK")


if __name__ == "__main__":
    main()
