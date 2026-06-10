# FastCUT Loss Calculations: `bidirectional` True vs. False

Summary of the loss terms for the **FastCUT** model under each setting, as
implemented in [`models/cut_model.py`](../models/cut_model.py).

FastCUT's relevant defaults: `lambda_NCE=10.0`, `lambda_GAN=1.0`,
`nce_idt=False`, `flip_equivariance=False`.

## FastCUT, `bidirectional=False` (stock FastCUT)

Single direction only: A → B, i.e. `fake_B = G(real_A)`.

**Discriminator loss** (`compute_D_loss`)

```
loss_D = 0.5 * (D_fake + D_real)
```

- `D_fake = GAN(D(fake_B.detach()), False)` — translated image scored as fake
- `D_real = GAN(D(real_B), True)` — real B scored as real

**Generator loss** (`compute_G_loss`)

```
loss_G = loss_G_GAN + loss_NCE_both
```

- `loss_G_GAN = lambda_GAN * GAN(D(fake_B), True)` — one direction, fool D
- Since `nce_idt=False`, the `NCE_Y` branch is skipped, so
  `loss_NCE_both = loss_NCE`
- `loss_NCE = lambda_NCE * PatchNCE(real_A, fake_B)` — contrastive loss between
  source encoder features and translated encoder features

With `flip_equivariance=True`, the input is randomly H-flipped and `feat_q`
flipped back before NCE (`calculate_NCE_loss`).

## FastCUT, `bidirectional=True` (shared-G symmetric)

`flip_equivariance` is **force-disabled** (in `modify_commandline_options` and
again in `__init__`) to preserve canonical orientation. Both directions go
through the same shared G:

- `fake_B = G(real_A)` (e.g. L→R)
- `fake_A = G(real_B)` (e.g. R→L)

**Discriminator loss** (`compute_D_loss`)

Each direction's fake/real terms are averaged, then combined:

```
D_fake = 0.5 * (GAN(D(fake_B), False) + GAN(D(fake_A), False))
D_real = 0.5 * (GAN(D(real_B), True)  + GAN(D(real_A), True))
loss_D = 0.5 * (D_fake + D_real)
```

One shared D scores both translated images as fake and both reals as real.

**Generator loss** (`compute_G_loss`)

```
loss_G = loss_G_GAN + loss_NCE_both
```

- GAN term averages both directions:

  ```
  loss_G_GAN = lambda_GAN * 0.5 * (GAN(D(fake_B), True) + GAN(D(fake_A), True))
  ```

- NCE is computed in **both directions** and averaged. Even though
  `nce_idt=False`, the bidirectional flag activates the second NCE term, using
  `fake_A` (translation role) rather than `idt_B`:

  ```
  loss_NCE      = lambda_NCE * PatchNCE(real_A, fake_B)
  loss_NCE_Y    = lambda_NCE * PatchNCE(real_B, fake_A)
  loss_NCE_both = 0.5 * (loss_NCE + loss_NCE_Y)
  ```

## Key differences

| Term | `bidirectional=False` | `bidirectional=True` |
|---|---|---|
| Translations | `fake_B` only | `fake_B` **and** `fake_A` (shared G) |
| `flip_equivariance` | True (FastCUT default) | Forced False |
| `D_fake` / `D_real` | single direction | mean of both directions |
| `loss_G_GAN` | single direction | mean of both directions |
| NCE | `loss_NCE` only | mean of `loss_NCE` and `loss_NCE_Y` |
| `NCE_Y` logged | No | Yes (`fake_A` target) |

The averaging (`* 0.5`) in the bidirectional path keeps each loss term on the
same scale as the unidirectional case, so the two directions effectively share
the loss budget rather than doubling it.

> Note: `masked_loss` (if enabled) further restricts both the GAN and NCE terms
> to the foreground mask in either mode, but doesn't change the structure above.

## Paired reconstruction (L1 / L2) finetuning

After `--flip_right` the L and R domains look almost identical, so the GAN signal
is weak and the strong PatchNCE term rewards content preservation — FastCUT tends
to **collapse to identity** (`G(L) ≈ L` instead of `≈ R`). A short *supervised*
finetune on the **paired** `bilateral` adapter (where `B` is the true same-study
right CC) breaks this by adding a pixel reconstruction term:

```
loss_recon = lambda_L1 · L1(G(real_A), real_B) + lambda_L2 · MSE(G(real_A), real_B)
loss_G     = loss_G_GAN + loss_NCE_both + loss_recon
```

- `--lambda_L1` and `--lambda_L2` are independent weights (both default `0.0`, so
  stock CUT/FastCUT is unchanged). Use either or combine them.
- **Paired only.** The term is rejected (`ValueError`) unless
  `--dataset_mode bilateral` — `unpaired_bilateral`/`scheduled_bilateral` hand out
  a random R, so a pixel target is undefined.
- **L1 vs. L2.** L1's optimum is the conditional *median* (one plausible image →
  sharper), L2's is the conditional *mean* (averages plausible images → blurry).
  Because contralateral L/R aren't pixel-registered, L1 is the safer default;
  L2/MSE is exposed for comparison.
- **Symmetric under `--bidirectional`.** The R→L direction reconstructs `real_A`,
  and the two directions are averaged so the loss budget matches the
  unidirectional case:

  ```
  loss_recon_Lk = lambda_Lk · 0.5 · ( Lk(fake_B, real_B) + Lk(fake_L, real_A) )
  ```

- With `--masked_loss`, `real` is mask-blended exactly like `fake` (already masked
  in `forward()`), so the constant background cancels and only the foreground is
  scored.
- Logged as `recon_L1` / `recon_L2` (each already the both-direction average under
  `--bidirectional`).

Keep `--lambda_GAN`/`--lambda_NCE` on during the finetune so the discriminator
re-sharpens what recon blurs. See [`scripts/fastcut-recon-finetune.sh`](../scripts/fastcut-recon-finetune.sh).
