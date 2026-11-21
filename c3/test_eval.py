"""Tests de c3/eval.py : numpy pur (toujours), torch et timm (sautés s'ils manquent, cas de la CI).

Petit modèle factice et fausses images : quelques secondes sur CPU, aucun poids réel.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval as ev  # noqa: E402


# ---------------------------------------------------------------- numpy pur

def test_agreement_and_kl_identity():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(50, 100))
    assert ev.top1_agreement(a, a) == 1.0
    assert abs(ev.kl_divergence(a, a)) < 1e-12
    b = a + rng.normal(scale=2.0, size=a.shape)
    assert ev.kl_divergence(a, b) > 0
    assert ev.top1_agreement(a, b) < 1.0


def test_ce_losses_hand_case():
    logits = np.array([[0.0, 0.0], [np.log(3.0), 0.0]])
    loss = ev.ce_losses(logits, np.array([0, 0]))
    np.testing.assert_allclose(loss, [np.log(2.0), np.log(4.0 / 3.0)], rtol=1e-12)


def test_retain_proxy_identity_is_one():
    a = np.random.default_rng(1).normal(size=(20, 10))
    r = ev.retain_proxy(a, a)
    assert r["agreement"] == 1.0 and abs(r["retain_proxy"] - 1.0) < 1e-12


def test_forget_proxy_targets_augmented_loss():
    rng = np.random.default_rng(2)
    orig = rng.normal(size=(30, 10)) * 5  # confiant : perte faible sur ses propres pseudo-labels
    aug = np.concatenate([orig * 0.2, orig * 0.3])  # n_aug = 2, moins confiant
    perfect = ev.forget_proxy(np.concatenate([orig * 0.2])[:30], orig, orig * 0.2)
    assert abs(perfect["forget_proxy"] - 1.0) < 1e-12
    identity = ev.forget_proxy(orig, orig, aug)
    assert identity["forget_proxy"] < 1.0
    assert identity["loss_cand_mean"] == pytest.approx(identity["loss_orig_mean"])


def test_fp16_gap_flags_nan_and_flips():
    a = np.random.default_rng(3).normal(size=(10, 5))
    assert ev.fp16_gap(a, a.copy(), 0.01)["fp16_ok"]
    bad = a.copy()
    bad[0, 0] = np.nan
    assert not ev.fp16_gap(a, bad, 0.01)["fp16_ok"]
    assert not ev.fp16_gap(a, -a, 0.01)["fp16_ok"]


def test_combine_gates():
    retain, forget = {"agreement": 0.9, "retain_proxy": 0.8}, {"forget_proxy": 0.5}
    assert ev.combine(True, True, retain, forget, 0.72) == (pytest.approx(0.4), [])
    assert ev.combine(False, True, retain, forget, 0.72)[0] == 0.0
    assert ev.combine(True, False, retain, forget, 0.72)[0] == 0.0
    assert ev.combine(True, True, {"agreement": 0.5, "retain_proxy": 0.4}, forget, 0.72)[0] == 0.0


# ---------------------------------------------------------------- torch (petit modèle factice)

def tiny_factory():
    torch = pytest.importorskip("torch")
    nn = torch.nn
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 8, 3, stride=2), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                         nn.Flatten(), nn.Linear(8, ev.NUM_CLASSES))


def fake_images(n=12, size=32, seed=0):
    return np.random.default_rng(seed).integers(0, 256, size=(n, size, size, 3), dtype=np.uint8)


def test_strong_augment_deterministic():
    pytest.importorskip("torch")
    x = fake_images(5)
    a, b = ev.strong_augment(x, seed=7, n_aug=2), ev.strong_augment(x, seed=7, n_aug=2)
    assert a.shape == (10, 32, 32, 3) and a.dtype == np.uint8
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a[:5], x)


def test_check_format_rejects_dtype_keys_and_size(tmp_path):
    torch = pytest.importorskip("torch")
    sd = tiny_factory().state_dict()
    assert ev.check_format(sd, tiny_factory)["format_ok"]
    half = {k: v.half() for k, v in sd.items()}
    assert not ev.check_format(half, tiny_factory)["format_ok"]
    missing = dict(sd)
    missing.pop(next(iter(missing)))
    assert not ev.check_format(missing, tiny_factory)["format_ok"]
    extra = dict(sd, extra=torch.zeros(1))
    assert not ev.check_format(extra, tiny_factory)["format_ok"]
    path = tmp_path / "m.pt"
    torch.save(sd, path)
    assert ev.check_format(sd, tiny_factory, path)["format_ok"]
    assert not ev.check_format(sd, tiny_factory, path, max_bytes=10)["format_ok"]


def test_evaluate_identity_and_perturbed():
    torch = pytest.importorskip("torch")
    orig = tiny_factory().state_dict()
    x = fake_images()
    r = ev.evaluate(orig, orig, x, tiny_factory, n_aug=2)
    assert r["format_ok"] and r["fp16_ok"]
    assert r["agreement"] == 1.0 and r["kl_T2"] < 1e-6
    assert r["loss_cand_mean"] == pytest.approx(r["loss_orig_mean"], abs=1e-6)
    assert r["score"] == pytest.approx(r["forget_proxy"], rel=1e-5)

    pert = {k: v.clone() for k, v in orig.items()}
    pert["4.weight"] += torch.randn_like(pert["4.weight"]) * 5
    r2 = ev.evaluate(orig, pert, x, tiny_factory, gate=0.0)
    assert r2["kl_T2"] > 0 and r2["loss_cand_mean"] != pytest.approx(r["loss_cand_mean"])

    retain = fake_images(6, seed=1)
    r3 = ev.evaluate(orig, orig, x, tiny_factory, retain_u8=retain)
    assert r3["retain_source"].startswith("retain-npz (6")

    bad = {k: v.half() for k, v in orig.items()}
    r4 = ev.evaluate(orig, bad, x, tiny_factory)
    assert r4["score"] == 0.0 and not r4["format_ok"]


# ---------------------------------------------------------------- timm (CLI complète)

def test_cli_pred_dir_with_timm(tmp_path, capsys):
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    arch = "vit_tiny_patch16_224"  # même famille que vit_base, en plus léger pour le test
    sd = ev.make_timm_model(arch).state_dict()
    task = tmp_path / "task"
    pred = tmp_path / "pred"
    task.mkdir()
    pred.mkdir()
    torch.save(sd, task / "model.pt")
    torch.save(sd, pred / "model.pt")
    np.savez(task / "forget_images.npz", images=fake_images(4, size=224))
    rc = ev.main(["--pred", str(pred), "--task-dir", str(task), "--arch", arch, "--device", "cpu"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "PROXY LOCAL" in out
    assert [ln for ln in out.splitlines() if ln.startswith("score: ")]


def test_vit_base_contract_strict():
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    sd = ev.make_timm_model().state_dict()  # vit_base_patch16_224, 100 classes, ~1 s sur CPU
    r = ev.check_format(sd, ev.make_timm_model)
    assert r["format_ok"], r["format_errors"]
    assert sd["head.weight"].shape == (ev.NUM_CLASSES, 768)
    assert r["n_params"] * 4 < ev.MAX_BYTES  # float32 : ~343 Mo, sous la limite
