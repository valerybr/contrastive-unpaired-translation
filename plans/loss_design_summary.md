# Contralateral CUT — Loss Design Summary

Latest proposed changes to the generator objective for the L-CC ↔ R-CC mammography
model (trained on finding-free cases, for downstream bilateral anomaly detection).

These terms layer **on top of** the existing masked CUT pipeline. The adversarial
loss and the standard PatchNCE are unchanged and keep their CUT defaults.

---

## Combined objective

```
L_G = λ_gan    · L_gan            (existing masked GAN, both directions)
    + λ_nce    · L_nce_std        (existing standard PatchNCE)
    + λ_contra · [ L_nce_contra(L→R) + L_nce_contra(R→L) ]
    + λ_sup    · [ L_contextual(L→R) + L_contextual(R→L) ]
    + λ_hf     · [ L_gradient(L→R)   + L_gradient(R→L)   ]   # registered target only
    + λ_cycle  · L_involution                                # default OFF
    + λ_idt    · L_identity                                  # default OFF
```

Every new term is computed in **both directions** (`fake_R = G(real_L)`,
`fake_L = G(real_R)`) using a **single shared generator**, then summed. This
doubles supervision per patient and prevents G from specializing to one laterality.

---

## New / changed terms

### 1. Contralateral-positive PatchNCE  (`λ_contra`)
Same InfoNCE form as standard PatchNCE, but the **positive comes from the real
contralateral**, not the input at the same coordinate.
- Query: encoder features of the generated breast, e.g. `enc(G(L))`.
- Positive: best match within a **local window** of the real contralateral's
  features (`enc(real_R)`) — a neighborhood search, since the pair is deformed.
- Negatives: other locations in the real contralateral feature map.
- Inherently misalignment-tolerant; gives harder, anatomically grounded positives.

### 2. Alignment-robust supervised anchor — Contextual loss  (`λ_sup`)
Replaces any raw L1-on-unregistered-breasts.
- Mechrez et al. Contextual (CX) loss: compares two feature sets **without
  requiring pixel correspondence**, so it tolerates L/R deformation.
- Run on a **lesion-sensitive / mammo-trained feature extractor**, *not* generic
  ImageNet VGG — otherwise the penalty never fires on calcifications/masses.
- This is the dominant **content anchor** (keeps lesion content faithful);
  the term most likely to need its weight raised.

### 3. High-frequency (gradient) term  (`λ_hf`)
- L1 on finite-difference image gradients → preserves microcalcifications/edges.
- **Pixel-aligned**: only valid on a **registered** target (warped into the
  prediction's frame). Skipped automatically when no registered target is passed,
  so the rest of training can run before registration is trusted.

### 4. Identity term — **OFF** (`λ_idt = 0`)
In a shared-domain setup it rewards `G(x) ≈ x`, which would reproduce a lesion
perfectly and flatten the residual. Worst possible behavior for detection.

### 5. Cycle / involution term — **OFF by default** (`λ_cycle = 0`)
Pairs already make the mapping well-posed, so it is no longer load-bearing.
Its steganographic tendency leaks input into output → manufactures **contralateral
false positives** at inference. Hook retained; if used, keep `≤ 0.5` and monitor
for high-frequency leakage.

---

## Starting weights

| Term | Weight | Notes |
|---|---|---|
| `λ_gan` | 1.0 | CUT default — unchanged |
| `λ_nce` (standard) | 1.0 | CUT default — unchanged |
| `λ_contra` | 1.0 | Same scale as standard NCE |
| `λ_sup` (contextual) | 1.0 | Raise toward 2–3 if content drifts. If using **registered pixel-L1** instead, use ~10 (different scale) |
| `λ_hf` | 1.0 | Raise toward ~5 if calcifications still blur |
| `λ_cycle` | 0.0 | Keep off; ≤0.5 + monitor if experimenting |
| `λ_idt` | 0.0 | Keep off |

> Weights only mean something relative to each term's scale. The only principled
> tuning is to **log every term** and balance their gradient magnitudes — treat
> the table as a first guess, not a final recipe.

---

## Integration notes / gotchas

- **`hf` requires a registered target.** Pass `target_registered` (real
  contralateral warped into the prediction frame); otherwise the term is skipped.
- **Contextual loss requires a feature extractor.** Prefer a mammo/lesion-trained
  network over ImageNet VGG. The module asserts one is provided.
- **Use CUT's existing encoder** for the contralateral NCE features (query and
  key live in the same feature space as the standard NCE).
- **Masking is already handled upstream**; all new terms accept the precomputed
  breast mask and reduce over in-mask elements only.

### Performance items to revisit
- Contralateral NCE loops over the batch for the masked neighborhood search —
  fine at small high-res batch sizes; first thing to vectorize if it bottlenecks.
- Contextual loss subsamples to 512 points/image for the O(N²) similarity matrix —
  raise `max_points` (and watch memory) if applied on fine feature maps.

---

## One-line training step (shared G, both directions)

```
fake_R, fake_L = G(real_L), G(real_R)
loss_gan      = masked_gan(...)          # existing
loss_nce_std  = standard_patchnce(...)   # existing
dir_LR = _direction(real_R, fake_R, enc(fake_R), enc(real_R), mod, w, mask_R, warp(real_R))
dir_RL = _direction(real_L, fake_L, enc(fake_L), enc(real_L), mod, w, mask_L, warp(real_L))
loss_G, log = generator_total_loss(loss_gan, loss_nce_std, dir_LR, dir_RL, w)
loss_G.backward()
```

Reference implementation: `contralateral_cut_losses.py`.
