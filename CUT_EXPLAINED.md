# CUT — Contrastive Unpaired Translation

CUT learns a one-way mapping `G: A → B` from two unpaired image collections. It replaces CycleGAN's cycle-consistency loss with a **patchwise contrastive (NCE) objective** computed on the generator's own encoder features, giving comparable quality with a single generator (no `F: B → A`, no cycle).

## The objective

Two losses are combined in `compute_G_loss` (`models/cut_model.py:174`):

1. **Adversarial loss** — `G(x)` should fool `D` on the target domain (LSGAN by default).
2. **PatchNCE loss** — for each spatial location in `G(x)`, the corresponding location in `x` should be more similar to it (in encoder feature space) than to any other location of `x`. This is what enforces structural correspondence in lieu of a cycle.

```
L = λ_GAN · L_GAN(G, D)  +  λ_NCE · L_NCE(G, F; X)
                        (+ λ_NCE · L_NCE(G, F; Y)   if --nce_idt)
```

Variants resolved inside `modify_commandline_options` (`cut_model.py:43-50`):

| Mode       | `nce_idt` | `λ_NCE` | Other                                       |
|------------|-----------|---------|---------------------------------------------|
| **CUT**    | True      | 1.0     | identity NCE on real-B passed through G     |
| **FastCUT**| False     | 10.0    | flip-equivariance, 150+50 epochs, no idt    |
| **SinCUT** | (sincut)  | —       | single-image; uses StyleGAN2 disc/gen blocks |

## How PatchNCE works (the distinctive piece)

`calculate_NCE_loss` (`cut_model.py:198`) does the following per minibatch:

1. **Encode source twice through the same encoder.** Run `netG(real_A, nce_layers, encode_only=True)` to get features `feat_k` (the "keys"), and `netG(fake_B, nce_layers, encode_only=True)` to get `feat_q` ("queries"). The encoder portion of the ResNet generator is reused — there is no separate feature network.
2. **Sample patches via `netF` (`PatchSampleF`, `networks.py:531`).** At each layer in `--nce_layers` (default `0,4,8,12,16`), it flattens the spatial dims, picks `--num_patches` (256) random spatial positions, and projects each through a 2-layer MLP to a 256-d vector that's L2-normalized. Crucially, the same `patch_ids` are used for queries and keys, so a "positive pair" is *the same spatial location in source vs. translation*.
3. **Cross-entropy contrastive loss (`models/patchnce.py`).** For each query `q_i` and its positive key `k_i⁺` (and `N-1` other patches as negatives) at temperature `τ = 0.07`:

   ```
   L_NCE = -log [ exp(q·k⁺/τ) / (exp(q·k⁺/τ) + Σ_n exp(q·kₙ⁻/τ)) ]
   ```

   The diagonal of the negative-similarity matrix is masked with `-10` so a patch isn't its own negative. Negatives come from *the same image* unless `--nce_includes_all_negatives_from_minibatch` (used by SinCUT, where the "minibatch" is crops of one image).

4. **Identity NCE (CUT only, `--nce_idt`).** Also pass `real_B` through `G` (yielding `idt_B`) and apply NCE between `real_B` and `idt_B`. This regularizes `G` toward an identity on the target domain.

5. **Flip-equivariance (FastCUT, `--flip_equivariance`).** With prob 0.5, horizontally flip the input; if the input was flipped, flip `feat_q` back before NCE so the loss is computed in a canonical orientation.

### Why `data_dependent_initialize` exists

`netF`'s MLPs aren't built at construction time — their input dims depend on the encoder feature shapes at each `nce_layer`. So `train.py` runs one forward pass on the first minibatch (`cut_model.py:94`) before `model.setup()` / `parallelize()`. After that pass, `PatchSampleF.create_mlp` instantiates the MLPs and the optimizer for `netF` is appended (`cut_model.py:109-111`). Don't reorder this, or DataParallel wrap, before that first pass.

## Networks

- **netG** — default `resnet_9blocks` ResNet generator. The integers in `--nce_layers` index *positions inside the generator's module list* (input conv, downsamples, residual blocks). Changing `--netG` changes what those indices mean.
- **netD** — default `basic` (PatchGAN, 70×70).
- **netF** — `PatchSampleF`, modes `sample` / `reshape` / `mlp_sample` (default `mlp_sample`).
- For `--model sincut`, the generator/discriminator switch to StyleGAN2 blocks from `models/stylegan_networks.py`.

## Datasets

### Layout — `UnalignedDataset` (`data/unaligned_dataset.py`)

This is the dataset used for CUT/FastCUT. It expects:

```
<dataroot>/
  trainA/   trainB/    # two unpaired collections
  testA/    testB/     # (or valA/valB, auto-detected at test time)
```

Each `__getitem__` returns one A image and a *random* B image (paired only by index when `--serial_batches`). `__len__` is `max(|A|, |B|)`. Use `--direction BtoA` to swap roles.

Pre-packaged sets via `datasets/download_cut_dataset.sh <name>` — these are the same CycleGAN datasets, hosted at `efrosgans.eecs.berkeley.edu/cyclegan/datasets/`:

```
apple2orange, summer2winter_yosemite, horse2zebra,
monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo,
maps, cityscapes, facades, iphone2dslr_flower,
ae_photos, grumpifycat, mini, mini_pix2pix, mini_colorization
```

Cityscapes can't be redistributed — download from cityscapes-dataset.com and run `datasets/prepare_cityscapes_dataset.py`.

### `SingleImageDataset` (for SinCUT)

Takes one image in each of `trainA/` and `trainB/` and pretends it's 100,000 samples. To make NCE meaningful on a single image, it precomputes per-batch random zoom factors and a non-repeating permutation of patch indices, so negatives within a batch come from disjoint crop locations of the same source image. This is why SinCUT sets `--nce_includes_all_negatives_from_minibatch True`.

### `SingleDataset` and `template_dataset.py`

`SingleDataset` is for inference on one folder (no domain B). `template_dataset.py` is a scaffold for new dataset types.

## Preprocessing (`data/base_dataset.py:get_transform`)

Pipeline assembled by string-matching on `--preprocess`. Building blocks (in this order):

| Stage      | Trigger                              | Effect |
|------------|--------------------------------------|--------|
| Grayscale  | `grayscale=True`                     | 1-channel conversion |
| Fixsize    | `'fixsize' in preprocess`            | Resize to `params["size"]` |
| Resize     | `'resize' in preprocess`             | To `[load_size, load_size]` (gta2cityscapes halves H) |
| Scale-width| `'scale_width' in preprocess`        | Width → `load_size`, keep aspect, height ≥ `crop_size` |
| Scale-shortside | `'scale_shortside' in preprocess` | Shorter side → `load_size` if smaller |
| Zoom       | `'zoom' in preprocess`               | Random zoom in `[0.8, 1.0]` (or fixed factor) |
| Crop       | `'crop' in preprocess`               | Random `crop_size×crop_size` (or fixed pos) |
| Patch      | `'patch' in preprocess`              | Tile into a grid, take patch `params['patch_index']` |
| Trim       | `'trim' in preprocess`               | Crop to at most `crop_size` per side |
| **Power-of-4** | **always**                       | Round H, W to nearest multiple of 4 |
| Flip       | `not no_flip`                        | `RandomHorizontalFlip` |
| ToTensor + Normalize | `convert=True`             | Range → `[-1, 1]` (`mean=std=0.5`) |

Two important defaults:

- **The power-of-4 round happens unconditionally** because the ResNet generator's `2×` downsamples don't preserve arbitrary spatial sizes. Even `--preprocess none` images are silently resized — a one-time warning is printed (`__print_size_warning`).
- Defaults from `base_options.py` are `--load_size 286 --crop_size 256`, with `--preprocess resize_and_crop` (the CycleGAN-style "resize to 286, random-crop to 256").

### "Finetuning" trick in `UnalignedDataset`

Once `current_epoch > n_epochs` (i.e. learning rate is decaying), the dataset overrides `load_size = crop_size` (`unaligned_dataset.py:64-65`) so no random crop padding remains. This disables resize-crop augmentation during finetuning — empirically helpful for CUT.

## Training-loop wiring (why `BaseModel` matters)

Every model in this repo, including `CUTModel`, must declare:

- `self.loss_names`, `self.model_names`, `self.visual_names`, `self.optimizers`
- `set_input`, `forward`, `optimize_parameters`
- Optionally `data_dependent_initialize(data)` — called by `train.py` on the first batch *before* `model.setup()` / `model.parallelize()`. CUT uses this to materialize `netF`.

`optimize_parameters` (`cut_model.py:113`) does the standard alternating step: D update on detached fakes, then G+F update sharing the same forward.

## Quick mental model

- CycleGAN: "the translation, when sent back, should reproduce the original" → cycle loss with two generators.
- CUT: "patches in source and translation should align in the encoder's own feature space, against negatives drawn from elsewhere in the same image" → one generator, contrastive loss, faster + lower memory.
