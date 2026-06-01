"""Tests for `--masked_loss` foreground-only training.

Covers the three novel mechanisms with synthetic tensors (no real dataset) plus
a synthetic-batch integration test of the full CUT model — all on CPU:

  1. mask geometry parity   -> data/bilateral.py:_load_mask vs _load_image
  2. NCE patches in fg       -> models/networks.py:PatchSampleF.forward(mask=...)
  3. masked GAN-loss math    -> models/networks.py:GANLoss.__call__(mask=...)
  4. model integration       -> models/cut_model.py:CUTModel with --masked_loss

The repo has no wired test runner, so this is runnable both as
``pytest tests/test_masking.py`` and ``python tests/test_masking.py``.
"""

import math
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Make the repo root importable when run as a plain script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.bilateral import _load_image, _load_mask, _mask_path  # noqa: E402
from models.networks import GANLoss, PatchSampleF  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Mask geometry parity
# ---------------------------------------------------------------------------

def test_mask_geometry_parity():
    """`_load_mask` must apply the SAME resize/flip/crop as `_load_image`."""
    native_h, native_w = 100, 80
    img = np.zeros((native_h, native_w), np.uint8)
    img[:, :40] = 200                     # bright on the LEFT (nipple) half
    mask = np.zeros((native_h, native_w), np.uint8)
    mask[:, :40] = 255                    # foreground == the bright region

    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "case.png"
        cv2.imwrite(str(src), img)
        cv2.imwrite(str(_mask_path(src)), mask)

        img_size, crop_width = (64, 48), 32
        t_img = _load_image(src, img_size, flip=True, crop_width=crop_width)
        t_mask = _load_mask(_mask_path(src), img_size, flip=True, crop_width=crop_width)

    assert t_img is not None and t_mask is not None
    assert tuple(t_img.shape) == (1, 64, 32) == tuple(t_mask.shape)
    # binary mask in {0, 1}
    uniq = set(torch.unique(t_mask).tolist())
    assert uniq.issubset({0.0, 1.0}), uniq

    img_fg = (t_img[0] > 0.0)
    m_fg = (t_mask[0] > 0.5)
    # Geometry parity: image-bright and mask-foreground agree almost everywhere
    # (small disagreement allowed at the linear-vs-nearest resize boundary).
    disagree = (img_fg ^ m_fg).float().mean().item()
    assert disagree < 0.1, f"image/mask geometry disagree by {disagree:.3f}"
    # Flip moved foreground to the chest-wall (right) side, and the crop kept it.
    left = m_fg[:, :16].float().mean().item()
    right = m_fg[:, 16:].float().mean().item()
    assert right > left, (left, right)


def test_load_mask_missing_returns_none():
    assert _load_mask(Path("/definitely/missing_mask.png"), (8, 8), False, 0) is None


def _import_make_breast_masks():
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "datasets", "make_breast_masks.py")
    spec = importlib.util.spec_from_file_location("make_breast_masks", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_breast_mask_left_view_not_full_frame():
    """Left CC views whose breast covers corner (0,0) must not yield an all-255
    mask (regression: corner-seeded hole fill used to invert and fill the frame).
    """
    mbm = _import_make_breast_masks()
    img = np.zeros((200, 160), np.uint8)
    # Breast against the LEFT edge, covering the top-left corner.
    cv2.ellipse(img, (0, 100), (120, 130), 0, 0, 360, 220, -1)
    assert img[0, 0] > 0                              # tissue covers (0, 0)
    mask = mbm.breast_mask(img, threshold=0, min_area_frac=0.02)
    fg = mask.mean() / 255.0
    assert 0.1 < fg < 0.95, f"left-view mask foreground fraction {fg:.3f}"
    assert mask[100, 0] == 255                        # breast kept
    assert mask[5, 155] == 0                          # top-right background dropped


def test_breast_mask_keeps_interior_dark_pixels():
    """Dark (0-valued) pixels enclosed by the breast must stay foreground, while
    genuine exterior background is masked out.
    """
    mbm = _import_make_breast_masks()
    img = np.zeros((200, 160), np.uint8)
    cv2.ellipse(img, (0, 100), (120, 130), 0, 0, 360, 200, -1)   # left breast
    img[90:110, 30:50] = 0                            # interior dark hole in breast
    assert img[100, 40] == 0
    mask = mbm.breast_mask(img, threshold=0)
    assert mask[100, 40] == 255                       # interior 0-pixels recovered
    assert mask[5, 155] == 0                          # exterior background still 0


# ---------------------------------------------------------------------------
# 2. NCE patches land inside the foreground
# ---------------------------------------------------------------------------

def _feats(batch):
    return [
        torch.randn(batch, 4, 16, 16),
        torch.randn(batch, 8, 8, 8),
        torch.randn(batch, 16, 4, 4),
    ]


def _rect_mask(batch, h=32, w=24):
    m = torch.zeros(batch, 1, h, w)
    m[:, :, 8:24, 6:18] = 1.0
    return m


def _assert_ids_in_foreground(ids, feats, mask, num_patches):
    for feat, patch_id in zip(feats, ids):
        B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
        assert tuple(patch_id.shape) == (B, num_patches)
        assert patch_id.dtype == torch.long
        down = F.interpolate(mask, size=(H, W), mode="nearest").view(B, H * W) > 0.5
        for b in range(B):
            if down[b].any():                       # skip empty-fg fallback rows
                assert bool(down[b][patch_id[b]].all()), (H, W, b)
            assert int(patch_id[b].max()) < H * W


def test_nce_patches_in_foreground():
    for batch in (1, 2):
        netF = PatchSampleF(use_mlp=False, gpu_ids=[])
        feats = _feats(batch)
        mask = _rect_mask(batch)
        num_patches = 8
        _, ids = netF(feats, num_patches, None, mask=mask)
        _assert_ids_in_foreground(ids, feats, mask, num_patches)

        # feat_q reuses identical ids when they are passed back in.
        _, ids2 = netF(feats, num_patches, ids)
        for a, b in zip(ids, ids2):
            assert torch.equal(a, b)


def test_nce_empty_foreground_fallback():
    netF = PatchSampleF(use_mlp=False, gpu_ids=[])
    feats = _feats(1)
    mask = torch.zeros(1, 1, 32, 24)          # no foreground at all
    num_patches = 8
    _, ids = netF(feats, num_patches, None, mask=mask)
    for feat, patch_id in zip(feats, ids):
        H, W = feat.shape[2], feat.shape[3]
        assert tuple(patch_id.shape) == (1, num_patches)
        assert int(patch_id.max()) < H * W and int(patch_id.min()) >= 0


# ---------------------------------------------------------------------------
# 3. Masked GAN-loss math + unmasked backward-compatibility
# ---------------------------------------------------------------------------

def test_masked_gan_loss_lsgan():
    crit = GANLoss("lsgan")
    pred = torch.randn(2, 1, 5, 4)
    mask = (torch.rand(2, 1, 5, 4) > 0.3).float()

    target = torch.ones_like(pred)
    per = F.mse_loss(pred, target, reduction="none")
    expected = (per * mask).sum() / mask.sum().clamp_min(1.0)
    assert torch.allclose(crit(pred, True, mask=mask), expected)

    # Unmasked path is byte-for-byte the original mean MSE.
    assert torch.allclose(crit(pred, True), torch.nn.MSELoss()(pred, target))


def test_masked_gan_loss_vanilla():
    crit = GANLoss("vanilla")
    pred = torch.randn(2, 1, 4, 4)
    mask = (torch.rand(2, 1, 4, 4) > 0.5).float()
    target = torch.zeros_like(pred)
    per = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    expected = (per * mask).sum() / mask.sum().clamp_min(1.0)
    assert torch.allclose(crit(pred, False, mask=mask), expected)


def test_masked_gan_loss_channel_safe():
    """A 1-channel mask against a multi-channel prediction must not undercount
    the denominator (expand_as fix)."""
    crit = GANLoss("lsgan")
    pred = torch.randn(2, 3, 4, 4)                 # C = 3
    mask = (torch.rand(2, 1, 4, 4) > 0.4).float()  # single-channel mask
    target = torch.ones_like(pred)
    per = F.mse_loss(pred, target, reduction="none")
    m = mask.expand_as(per)
    expected = (per * m).sum() / m.sum().clamp_min(1.0)
    out = crit(pred, True, mask=mask)
    assert out.dim() == 0 and torch.allclose(out, expected)


def test_gan_loss_returns_scalar_all_modes():
    """Masked and unmasked paths must both reduce to a scalar so the result can
    be .backward()'d regardless of gan_mode (nonsaturating used to return [bs])."""
    pred = torch.randn(2, 1, 4, 4)
    mask = (torch.rand(2, 1, 4, 4) > 0.3).float()
    for mode in ("lsgan", "vanilla", "wgangp", "nonsaturating"):
        crit = GANLoss(mode)
        for target_is_real in (True, False):
            assert crit(pred, target_is_real).dim() == 0, (mode, "unmasked")
            assert crit(pred, target_is_real, mask=mask).dim() == 0, (mode, "masked")


# ---------------------------------------------------------------------------
# 4. Full-model integration on a synthetic batch
# ---------------------------------------------------------------------------

def _make_opt(batch_size, tmp_ckpt, masked=True):
    from options.train_options import TrainOptions
    cmd = (
        f"--model cut --dataset_mode unpaired_bilateral "
        f"--annotations_csv {os.path.join(tmp_ckpt, 'none.csv')} "
        f"--input_nc 1 --output_nc 1 --gpu_ids -1 --ngf 16 "
        f"{'--masked_loss ' if masked else ''}--batch_size {batch_size} "
        f"--name maskcut_test --checkpoints_dir {tmp_ckpt} "
        f"--num_threads 0 --display_id 0"
    )
    return TrainOptions(cmd_line=cmd).parse()


def _synthetic_batch(batch, h=64, w=64):
    a = torch.rand(batch, 1, h, w) * 2 - 1
    b = torch.rand(batch, 1, h, w) * 2 - 1
    ma = torch.zeros(batch, 1, h, w); ma[:, :, 16:48, 16:48] = 1.0
    mb = torch.zeros(batch, 1, h, w); mb[:, :, 8:40, 20:52] = 1.0
    return {
        "A": a, "B": b,
        "A_paths": [f"a{i}" for i in range(batch)],
        "B_paths": [f"b{i}" for i in range(batch)],
        "A_mask": ma, "B_mask": mb,
    }


def test_model_integration_masked():
    from models.cut_model import CUTModel
    for batch in (1, 2):
        with tempfile.TemporaryDirectory() as tmp_ckpt:
            opt = _make_opt(batch, tmp_ckpt)
            torch.manual_seed(0)
            model = CUTModel(opt)
            data = _synthetic_batch(batch)
            model.data_dependent_initialize(data)   # builds netF + optimizer_F
            # Skip setup()/parallelize(): nets already live on CPU; DataParallel
            # with empty gpu_ids would be invalid and schedulers aren't needed.
            for _ in range(2):
                model.set_input(data)
                model.optimize_parameters()

            losses = model.get_current_losses()
            assert losses, "no losses reported"
            for k, v in losses.items():
                assert math.isfinite(v), f"loss {k} not finite: {v}"

            # Background of fake_B must be the mask_bg_value where mask_A == 0.
            bg = model.fake_B.detach()[data["A_mask"] == 0]
            assert torch.allclose(bg, torch.full_like(bg, opt.mask_bg_value), atol=1e-5)


def test_model_integration_unmasked_regression():
    """Default path (no --masked_loss, batch without masks) must be unaffected."""
    from models.cut_model import CUTModel
    with tempfile.TemporaryDirectory() as tmp_ckpt:
        opt = _make_opt(1, tmp_ckpt, masked=False)
        torch.manual_seed(0)
        model = CUTModel(opt)
        data = _synthetic_batch(1)
        data.pop("A_mask"); data.pop("B_mask")     # simulate a non-masked dataset
        model.data_dependent_initialize(data)
        for _ in range(2):
            model.set_input(data)
            model.optimize_parameters()
        assert model.use_mask is False
        for k, v in model.get_current_losses().items():
            assert math.isfinite(v), f"loss {k} not finite: {v}"
        # No background forcing: fake_B should NOT be a constant -1 background.
        assert not torch.allclose(model.fake_B.detach(), torch.full_like(model.fake_B, -1.0))


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"All {len(fns)} masking tests passed.")


if __name__ == "__main__":
    _run_all()
