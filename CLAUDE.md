# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PyTorch implementation of **Contrastive Unpaired Translation (CUT)** (Park et al., ECCV 2020) — unpaired image-to-image translation using patchwise contrastive learning plus adversarial loss. Three related models live in this repo:

- **CUT** — full model (`--CUT_mode CUT`): identity NCE loss, `lambda_NCE=1.0`.
- **FastCUT** — lighter variant (`--CUT_mode FastCUT`): no identity, `lambda_NCE=10.0`, flip-equivariance, 150+50 epochs.
- **SinCUT** — single-image training (`--model sincut`): uses StyleGAN2 components from `models/stylegan_networks.py`.

Also ships a CycleGAN reimplementation (`--model cycle_gan`). Codebase is forked from `pytorch-CycleGAN-and-pix2pix` and keeps its plugin-style architecture.

## Common commands

Install (either works):
```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml
```

Visdom (required for training plots at http://localhost:8097):
```bash
python -m visdom.server
```

Download a dataset:
```bash
bash ./datasets/download_cut_dataset.sh grumpifycat
```

Train / test directly:
```bash
python train.py --dataroot ./datasets/grumpifycat --name grumpycat_CUT --CUT_mode CUT
python test.py  --dataroot ./datasets/grumpifycat --name grumpycat_CUT --CUT_mode CUT --phase train
```

Train / test via the experiment launcher (see `experiments/<name>_launcher.py`):
```bash
python -m experiments grumpifycat train 0     # runs commands()[0]
python -m experiments grumpifycat run_test 0  # runs test_commands()[0]
```
Launcher subcommands parsed in `experiments/__main__.py`: `run`/`train`, `run_test`/`test`, `launch`, `launch_test`, `relaunch`, `close`, `stop`, `dry`, `print_names`, `print_test_names`, `create_comparison_html`.

Pretrained models:
```bash
wget http://efrosgans.eecs.berkeley.edu/CUT/pretrained_models.tar && tar -xf pretrained_models.tar
python -m experiments pretrained run_test [0-5]
```

FID evaluation (via external `pytorch-fid`):
```bash
python -m pytorch_fid <real_images_dir> <generated_images_dir>
```

There is no test suite, no linter config, and no build step — this is a research training codebase.

## Architecture

The codebase is built around four plugin registries driven by option strings. Each registry dynamically imports a module by name and locates a class by convention, which is why adding new components is a matter of dropping in a correctly-named file.

### Plugin registries (string → class)

| Option | Resolves via | Module pattern | Class pattern | Base class |
|---|---|---|---|---|
| `--model <name>` | `models/__init__.py:find_model_using_name` | `models/<name>_model.py` | `<Name>Model` | `BaseModel` (`models/base_model.py`) |
| `--dataset_mode <name>` | `data/__init__.py:find_dataset_using_name` | `data/<name>_dataset.py` | `<Name>Dataset` | `BaseDataset` (`data/base_dataset.py`) |
| `--netG` / `--netD` / `--netF` | `models/networks.py:define_G/D/F` | (switch in `networks.py`) | — | — |

Class name match is case-insensitive and strips underscores (e.g. `cycle_gan` → `CycleGanModel`, `single_image` → `SingleImageDataset`).

### Option system

Options are assembled in three layers, each contributing flags via `modify_commandline_options(parser, is_train)`:

1. `options/base_options.py` — shared flags.
2. `options/train_options.py` or `options/test_options.py` — phase flags.
3. The model class and dataset class selected by `--model` / `--dataset_mode` inject their own flags.

Because of layer 3, running `python train.py --help` without a model/dataset gives an incomplete view; pass `--model <x> --dataset_mode <y>` to see the full set. Model defaults can also be overridden inside `modify_commandline_options` (e.g. FastCUT resets `n_epochs`, `flip_equivariance`, and `lambda_NCE` from inside `cut_model.py`).

### Training loop contract (`train.py`)

Every `BaseModel` subclass must implement: `set_input`, `forward`, `optimize_parameters`, and declare `self.loss_names`, `self.model_names`, `self.visual_names`, `self.optimizers`. Models with data-shape-dependent submodules (notably `netF`, the patch sampling MLP) implement `data_dependent_initialize(data)` — `train.py` calls it on the **first** minibatch before `model.setup()` / `model.parallelize()`. Do not break this ordering; `netF`'s layers are materialized from the first forward pass's feature shapes (see `cut_model.py` and `networks.PatchSampleF.create_mlp`).

### The NCE / PatchNCE core

The distinctive piece of this repo versus CycleGAN is the contrastive objective:

- `models/patchnce.py` — the cross-entropy contrastive loss itself.
- `models/networks.py:PatchSampleF` — the `netF` feature projector/sampler (modes: `sample`, `reshape`, `mlp_sample`). It extracts patch features at layers specified by `--nce_layers` (default `0,4,8,12,16` of the ResNet generator) and projects each with a small MLP.
- `models/cut_model.py` — wires NCE against the generator's *own* encoder features of real vs. translated images. With `--nce_idt`, an identity NCE term on `G(Y)` is added.

Layer indices in `--nce_layers` refer to positions inside the `resnet_*` generator; changing `--netG` changes what those indices mean.

### StyleGAN2 components

`models/stylegan_networks.py` ports rosinality's StyleGAN2. It is used when `--netG stylegan2/smallstylegan2` or `--netD stylegan2/tilestylegan2` is selected, and by default for `--model sincut`.

### Preprocessing

Controlled by `--preprocess {resize_and_crop | crop | scale_width | scale_width_and_crop | scale_shortside_and_crop | none}` + `--load_size` + `--crop_size`, implemented in `data/base_dataset.py:get_transform`. With `--preprocess none`, images are still rounded to the nearest multiple of 4 because the ResNet generator cannot preserve arbitrary spatial sizes.

### Outputs and state

- Checkpoints: `./checkpoints/<name>/` (`latest_net_G.pth`, `latest_net_D.pth`, `latest_net_F.pth`, per-epoch copies).
- Training HTML + images: `./checkpoints/<name>/web/`.
- Test results: `./results/<name>/<phase>_<epoch>/index.html`.
- Visdom server on port 8097 is assumed; set `--display_id 0` (or `None`) to disable.

## Conventions worth knowing

- `--gpu_ids 0,1,2` enables DataParallel via `model.parallelize()`; `--gpu_ids -1` forces CPU.
- `batch_size=1` is the tested default; larger batches work but NCE includes the `--nce_includes_all_negatives_from_minibatch` flag which changes the negative-sampling semantics.
- `phase=train` at test time is intentional for datasets without a test split (e.g. grumpifycat) — see `experiments/grumpifycat_launcher.py`.
- `tox.ini` exists but is not wired to a test runner; ignore unless you are adding one.
