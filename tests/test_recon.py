"""Tests for `--lambda_L1` / `--lambda_L2` paired reconstruction finetuning.

The reconstruction term supplies the supervised "*this* L maps to *this* R"
signal that breaks FastCUT's identity collapse on the contralateral mammography
setup. It is only well-posed on the paired `bilateral` adapter (B = the true
same-study right CC); the model rejects it on the unpaired adapters.

Covered, all on CPU with synthetic tensors (no real dataset):

  1. terms logged + match manual L1/L2   -> models/cut_model.py:compute_G_loss
  2. recon gradient pulls G(A) -> B        -> behavioral check (recon decreases)
  3. zero weights are a no-op              -> loss_names regression
  4. paired-dataset guard                  -> CUTModel.__init__ ValueError
  5. masking excludes the background        -> _recon masks `real` like `fake`
  6. bidirectional symmetric averaging      -> recon over both directions

The repo has no wired test runner, so this is runnable both as
``pytest tests/test_recon.py`` and ``python tests/test_recon.py``.
"""

import math
import os
import sys
import tempfile

import torch

# Make the repo root importable when run as a plain script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Shared synthetic fixtures (mirrors tests/test_masking.py conventions)
# ---------------------------------------------------------------------------

def _make_opt(tmp_ckpt, *, lambda_L1=0.0, lambda_L2=0.0, lambda_GAN=1.0,
              lambda_NCE=10.0, dataset_mode='bilateral', masked=False,
              bidirectional=False, netF='mlp_sample', batch_size=1, extra=''):
    from options.train_options import TrainOptions
    cmd = (
        f"--model cut --dataset_mode {dataset_mode} "
        f"--annotations_csv {os.path.join(tmp_ckpt, 'none.csv')} "
        f"--input_nc 1 --output_nc 1 --gpu_ids -1 --ngf 16 --netF {netF} "
        f"{'--masked_loss ' if masked else ''}"
        f"{'--bidirectional --flip_right ' if bidirectional else ''}"
        f"--lambda_GAN {lambda_GAN} --lambda_NCE {lambda_NCE} "
        f"--lambda_L1 {lambda_L1} --lambda_L2 {lambda_L2} "
        f"--batch_size {batch_size} --name recon_test "
        f"--checkpoints_dir {tmp_ckpt} --num_threads 0 --display_id 0 {extra}"
    )
    return TrainOptions(cmd_line=cmd).parse()


def _synthetic_batch(batch=1, h=64, w=64):
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


def _build_and_step(opt, data, steps=2):
    """Build a CUTModel on CPU, init netF, run a few optimization steps."""
    from models.cut_model import CUTModel
    torch.manual_seed(0)
    model = CUTModel(opt)
    model.data_dependent_initialize(data)   # builds netF (+ optimizer_F if NCE on)
    for _ in range(steps):
        model.set_input(data)
        model.optimize_parameters()
    return model


def _masked_real(model, real, mask):
    return model._apply_mask(real, mask) if model.use_mask else real


# ---------------------------------------------------------------------------
# 1. Terms are logged and equal the manual L1 / L2
# ---------------------------------------------------------------------------

def test_recon_terms_logged_and_finite():
    with tempfile.TemporaryDirectory() as tmp:
        opt = _make_opt(tmp, lambda_L1=10.0, lambda_L2=10.0)
        data = _synthetic_batch()
        model = _build_and_step(opt, data)

        losses = model.get_current_losses()
        assert 'recon_L1' in losses and 'recon_L2' in losses, losses.keys()
        for k, v in losses.items():
            assert math.isfinite(v), f"loss {k} not finite: {v}"

        # The logged terms equal lambda * criterion(fake_B, real_B) on the SAME
        # fake_B compute_G_loss used in the final step.
        exp_l1 = opt.lambda_L1 * model.criterionIdt(model.fake_B, model.real_B)
        exp_l2 = opt.lambda_L2 * model.criterionL2(model.fake_B, model.real_B)
        assert torch.allclose(model.loss_recon_L1, exp_l1)
        assert torch.allclose(model.loss_recon_L2, exp_l2)


# ---------------------------------------------------------------------------
# 2. The recon gradient actually pulls G(A) toward B
# ---------------------------------------------------------------------------

def test_recon_decreases():
    """With GAN/NCE off, recon-only training must reduce MSE(G(A), B)."""
    with tempfile.TemporaryDirectory() as tmp:
        # netF=sample + lambda_NCE 0 avoids needing optimizer_F; nce_idt False
        # keeps NCE_Y out of loss_names so get_current_losses isn't needed.
        opt = _make_opt(tmp, lambda_L1=0.0, lambda_L2=10.0, lambda_GAN=0.0,
                        lambda_NCE=0.0, netF='sample', extra='--nce_idt False')
        data = _synthetic_batch()
        from models.cut_model import CUTModel
        torch.manual_seed(0)
        model = CUTModel(opt)
        model.data_dependent_initialize(data)

        history = []
        for _ in range(6):
            model.set_input(data)
            model.optimize_parameters()        # forward → compute_G_loss sets loss_recon_L2
            history.append(float(model.loss_recon_L2))

        assert all(math.isfinite(v) for v in history), history
        assert history[-1] < history[0], f"recon did not decrease: {history}"


# ---------------------------------------------------------------------------
# 3. Zero weights: recon is absent and the default path is untouched
# ---------------------------------------------------------------------------

def test_recon_zero_weight_noop():
    with tempfile.TemporaryDirectory() as tmp:
        opt = _make_opt(tmp, lambda_L1=0.0, lambda_L2=0.0)
        from models.cut_model import CUTModel
        model = CUTModel(opt)
        assert model.use_recon is False
        assert 'recon_L1' not in model.loss_names
        assert 'recon_L2' not in model.loss_names

        data = _synthetic_batch()
        model.data_dependent_initialize(data)
        for _ in range(2):
            model.set_input(data)
            model.optimize_parameters()
        assert model.loss_recon_L1 == 0.0 and model.loss_recon_L2 == 0.0
        for k, v in model.get_current_losses().items():
            assert math.isfinite(v), f"loss {k} not finite: {v}"


# ---------------------------------------------------------------------------
# 4. Recon requires the paired `bilateral` adapter
# ---------------------------------------------------------------------------

def test_recon_requires_paired_dataset():
    from models.cut_model import CUTModel
    with tempfile.TemporaryDirectory() as tmp:
        opt = _make_opt(tmp, lambda_L2=10.0, dataset_mode='unpaired_bilateral')
        raised = False
        try:
            CUTModel(opt)
        except ValueError as e:
            raised = True
            msg = str(e).lower()
            assert 'bilateral' in msg and 'paired' in msg, e
        assert raised, "expected ValueError for recon on an unpaired adapter"


# ---------------------------------------------------------------------------
# 5. Masked recon scores only the foreground
# ---------------------------------------------------------------------------

def test_recon_masked_excludes_background():
    with tempfile.TemporaryDirectory() as tmp:
        opt = _make_opt(tmp, lambda_L1=10.0, lambda_L2=10.0, masked=True)
        data = _synthetic_batch()
        model = _build_and_step(opt, data)
        assert model.use_mask is True

        # Recon must compare against the SAME mask-blended real B the helper uses.
        real_b_m = model._apply_mask(model.real_B, model.mask_B)
        exp_l1 = opt.lambda_L1 * model.criterionIdt(model.fake_B, real_b_m)
        exp_l2 = opt.lambda_L2 * model.criterionL2(model.fake_B, real_b_m)
        assert torch.allclose(model.loss_recon_L1, exp_l1)
        assert torch.allclose(model.loss_recon_L2, exp_l2)

        # fake_B is mask-blended by its SOURCE mask (mask_A) in forward(); real_B
        # is blended by mask_B. Both hit mask_bg_value only where *both* masks are
        # deep background (outside the feather reach), so the per-pixel error is
        # exactly zero there.
        import torch.nn.functional as F
        r = opt.mask_feather
        reach = lambda m: (F.max_pool2d(m, 2 * r + 1, 1, r) if r > 0 else m)
        deep_bg = (reach(model.mask_A)[0, 0] == 0) & (reach(model.mask_B)[0, 0] == 0)
        diff_bg = (model.fake_B.detach() - real_b_m)[0, 0][deep_bg].abs()
        assert diff_bg.numel() > 0
        assert torch.allclose(diff_bg, torch.zeros_like(diff_bg), atol=1e-5)


# ---------------------------------------------------------------------------
# 6. Bidirectional: recon is the average of both translation directions
# ---------------------------------------------------------------------------

def test_recon_bidirectional_symmetric():
    with tempfile.TemporaryDirectory() as tmp:
        opt = _make_opt(tmp, lambda_L1=10.0, lambda_L2=10.0, masked=True,
                        bidirectional=True)
        data = _synthetic_batch()
        model = _build_and_step(opt, data)
        assert opt.bidirectional and hasattr(model, 'fake_L')

        real_b_m = model._apply_mask(model.real_B, model.mask_B)
        real_a_m = model._apply_mask(model.real_A, model.mask_A)
        exp_l1 = opt.lambda_L1 * 0.5 * (
            model.criterionIdt(model.fake_B, real_b_m)
            + model.criterionIdt(model.fake_L, real_a_m))
        exp_l2 = opt.lambda_L2 * 0.5 * (
            model.criterionL2(model.fake_B, real_b_m)
            + model.criterionL2(model.fake_L, real_a_m))
        assert torch.allclose(model.loss_recon_L1, exp_l1)
        assert torch.allclose(model.loss_recon_L2, exp_l2)
        assert math.isfinite(float(model.loss_recon_L1))
        assert math.isfinite(float(model.loss_recon_L2))


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"All {len(fns)} recon tests passed.")


if __name__ == "__main__":
    _run_all()
