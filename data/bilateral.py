"""Bilateral mammography dataset: paired left/right CC views, no-finding studies."""

import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BilateralDataset(Dataset):
    """Paired left/right CC mammogram dataset built from VinDr-Mammo annotations.

    Filters ``finding_annotations.csv`` to studies where **every** image is
    labelled ``'No Finding'``, then pairs the L CC and R CC images for each
    such study.

    Expected PNG layout::

        data_root/{study_id}/{image_id}.png

    Args:
        data_root:       Root directory containing converted PNG images.
        annotations_csv: Path to ``finding_annotations.csv``.
        split:           ``'training'``, ``'test'``, or ``None`` for all rows.
        img_size:        Output image size. Either a single int for a square
                         (e.g. ``512``) or a ``(height, width)`` tuple
                         (e.g. ``(512, 384)``). Default ``512``.
        flip_right:      Flip right-breast images horizontally so the nipple
                         faces left, matching the left-breast orientation.
                         Set to ``False`` if images were already flipped during
                         DICOM conversion (``--annotations`` flag in dicom_io).
    """

    def __init__(
        self,
        data_root: str,
        annotations_csv: str,
        split: str | None = None,
        img_size: int | tuple[int, int] = 512,
        flip_right: bool = False,
    ):
        self.data_root = Path(data_root)
        self.img_size: tuple[int, int] = (
            (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)  # type: ignore[arg-type]
        )
        self.flip_right = flip_right

        self.pairs = self._build_pairs(Path(annotations_csv), split)

    # ------------------------------------------------------------------
    # Building the pair list
    # ------------------------------------------------------------------

    def _build_pairs(
        self, annotations_csv: Path, split: str | None
    ) -> list[tuple[Path, Path]]:
        # --- pass 1: collect all rows, optionally filtered by split ----------
        # study_id → {(laterality, view_position) → image_id}
        studies: dict[str, dict[tuple[str, str], str]] = {}
        # study_id → set of finding categories seen
        findings: dict[str, set[str]] = {}

        with open(annotations_csv, newline="") as f:
            for row in csv.DictReader(f):
                if split is not None and row["split"] != split:
                    continue
                sid = row["study_id"]
                findings.setdefault(sid, set()).add(row["finding_categories"])
                key = (row["laterality"], row["view_position"])
                studies.setdefault(sid, {})[key] = row["image_id"]

        # --- pass 2: keep only no-finding studies with both L CC and R CC ----
        pairs: list[tuple[Path, Path]] = []
        skipped: list[tuple[str, str]] = []  # (study_id, reason)

        for sid, images in studies.items():
            # Every row for this study must be "No Finding"
            if findings[sid] != {"['No Finding']"}:
                continue

            l_id = images.get(("L", "CC"))
            r_id = images.get(("R", "CC"))

            if l_id is None or r_id is None:
                missing = "L CC" if l_id is None else "R CC"
                skipped.append((sid, f"missing {missing}"))
                continue

            l_path = self.data_root / sid / f"{l_id}.png"
            r_path = self.data_root / sid / f"{r_id}.png"

            missing_files = [
                str(p) for p in (l_path, r_path) if not p.exists()
            ]
            if missing_files:
                skipped.append((sid, f"PNG not found: {', '.join(missing_files)}"))
                continue

            pairs.append((l_path, r_path))

        if skipped:
            print(f"[BilateralDataset] Skipped {len(skipped)} studies:")
            for sid, reason in skipped:
                print(f"  {sid}: {reason}")

        print(
            f"[BilateralDataset] Loaded {len(pairs)} paired studies"
            + (f" (split={split})" if split else "")
        )
        return pairs

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        l_path, r_path = self.pairs[idx]
        img_l = self._load(l_path, flip=False)
        img_r = self._load(r_path, flip=self.flip_right)
        return img_l, img_r

    def _load(self, path: Path, flip: bool) -> torch.Tensor:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        target_h, target_w = self.img_size
        if img.shape[0] != target_h or img.shape[1] != target_w:
            img = cv2.resize(img, (target_w, target_h),  # cv2 takes (width, height)
                             interpolation=cv2.INTER_LINEAR)
        if flip:
            img = cv2.flip(img, 1)
        # [0, 255] uint8 → [-1, 1] float32, shape [1, H, W]
        tensor = torch.from_numpy(img.astype(np.float32)) / 127.5 - 1.0
        return tensor.unsqueeze(0)


class UnpairedBilateralDataset(Dataset):
    """Unpaired left/right CC mammogram dataset for CUT/CycleGAN training.

    Builds separate pools of left and right CC images from no-finding studies.
    Each ``__getitem__`` returns a randomly sampled right image independently of
    the left image at that index, so the model sees every L×R combination over
    training rather than fixed pairs.

    The dataset length is ``len(left_images)``.  Right images are sampled
    uniformly at random, which matches the unpaired assumption of CUT/CycleGAN.

    Args:
        data_root:       Root directory containing converted PNG images.
        annotations_csv: Path to ``finding_annotations.csv``.
        split:           ``'training'``, ``'test'``, or ``None`` for all rows.
        img_size:        Output image size. Either a single int for a square
                         (e.g. ``512``) or a ``(height, width)`` tuple
                         (e.g. ``(512, 384)``). Default ``512``.
        flip_right:      Flip right-breast images horizontally so the nipple
                         faces left, matching the left-breast orientation.
    """

    def __init__(
        self,
        data_root: str,
        annotations_csv: str,
        split: str | None = None,
        img_size: int | tuple[int, int] = 512,
        flip_right: bool = False,
    ):
        self.data_root = Path(data_root)
        self.img_size: tuple[int, int] = (
            (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)  # type: ignore[arg-type]
        )
        self.flip_right = flip_right

        self.left_images, self.right_images = self._build_pools(
            Path(annotations_csv), split
        )

    def _build_pools(
        self, annotations_csv: Path, split: str | None
    ) -> tuple[list[Path], list[Path]]:
        studies: dict[str, dict[tuple[str, str], str]] = {}
        findings: dict[str, set[str]] = {}

        with open(annotations_csv, newline="") as f:
            for row in csv.DictReader(f):
                if split is not None and row["split"] != split:
                    continue
                sid = row["study_id"]
                findings.setdefault(sid, set()).add(row["finding_categories"])
                key = (row["laterality"], row["view_position"])
                studies.setdefault(sid, {})[key] = row["image_id"]

        left_images: list[Path] = []
        right_images: list[Path] = []
        skipped: list[tuple[str, str]] = []

        for sid, images in studies.items():
            if findings[sid] != {"['No Finding']"}:
                continue

            l_id = images.get(("L", "CC"))
            r_id = images.get(("R", "CC"))

            if l_id is None and r_id is None:
                skipped.append((sid, "missing both L CC and R CC"))
                continue

            if l_id is not None:
                l_path = self.data_root / sid / f"{l_id}.png"
                if l_path.exists():
                    left_images.append(l_path)
                else:
                    skipped.append((sid, f"PNG not found: {l_path}"))

            if r_id is not None:
                r_path = self.data_root / sid / f"{r_id}.png"
                if r_path.exists():
                    right_images.append(r_path)
                else:
                    skipped.append((sid, f"PNG not found: {r_path}"))

        if skipped:
            print(f"[UnpairedBilateralDataset] Skipped {len(skipped)} entries:")
            for sid, reason in skipped:
                print(f"  {sid}: {reason}")

        print(
            f"[UnpairedBilateralDataset] {len(left_images)} left, "
            f"{len(right_images)} right images"
            + (f" (split={split})" if split else "")
        )
        return left_images, right_images

    def __len__(self) -> int:
        return len(self.left_images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        l_path = self.left_images[idx]
        r_path = random.choice(self.right_images)
        img_l = self._load(l_path, flip=False)
        img_r = self._load(r_path, flip=self.flip_right)
        return img_l, img_r

    def _load(self, path: Path, flip: bool) -> torch.Tensor:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        target_h, target_w = self.img_size
        if img.shape[0] != target_h or img.shape[1] != target_w:
            img = cv2.resize(img, (target_w, target_h),  # cv2 takes (width, height)
                             interpolation=cv2.INTER_LINEAR)
        if flip:
            img = cv2.flip(img, 1)
        tensor = torch.from_numpy(img.astype(np.float32)) / 127.5 - 1.0
        return tensor.unsqueeze(0)


class ScheduledBilateralDataset(Dataset):
    """Bilateral dataset with a trainer-controlled random-pair probability.

    Built from paired no-finding L CC / R CC studies (same filter as
    :class:`BilateralDataset`).  Each ``__getitem__`` returns the indexed
    left image together with either the true paired right image or a
    randomly sampled right image from another pair.  The trainer controls
    the random-sampling probability per epoch via
    :meth:`set_epoch_state`, enabling curriculum schedules such as
    ``[(20, 0.5), (20, 0.3), (20, 0.1), (20, 0.0)]``.

    Sampling is seeded by ``(seed, epoch, idx)`` so that, for a given
    epoch, the dataset is deterministic regardless of DataLoader workers.

    Args:
        data_root:       Root directory containing converted PNG images.
        annotations_csv: Path to ``finding_annotations.csv``.
        split:           ``'training'``, ``'test'``, or ``None`` for all rows.
        img_size:        Output image size. Either a single int for a square
                         (e.g. ``512``) or a ``(height, width)`` tuple.
        flip_right:      Flip right-breast images horizontally.
        seed:            Base seed for per-sample RNG.
    """

    def __init__(
        self,
        data_root: str,
        annotations_csv: str,
        split: str | None = None,
        img_size: int | tuple[int, int] = 512,
        flip_right: bool = False,
        seed: int = 0,
    ):
        self.data_root = Path(data_root)
        self.img_size: tuple[int, int] = (
            (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)  # type: ignore[arg-type]
        )
        self.flip_right = flip_right
        self.seed = seed
        self.epoch: int = 0
        self.p: float = 0.0

        self.pairs = self._build_pairs(Path(annotations_csv), split)

    def set_epoch_state(self, epoch: int, p: float) -> None:
        """Set current epoch and random-pair probability for this epoch."""
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.epoch = epoch
        self.p = p

    def _build_pairs(
        self, annotations_csv: Path, split: str | None
    ) -> list[tuple[Path, Path]]:
        studies: dict[str, dict[tuple[str, str], str]] = {}
        findings: dict[str, set[str]] = {}

        with open(annotations_csv, newline="") as f:
            for row in csv.DictReader(f):
                if split is not None and row["split"] != split:
                    continue
                sid = row["study_id"]
                findings.setdefault(sid, set()).add(row["finding_categories"])
                key = (row["laterality"], row["view_position"])
                studies.setdefault(sid, {})[key] = row["image_id"]

        pairs: list[tuple[Path, Path]] = []
        skipped: list[tuple[str, str]] = []

        for sid, images in studies.items():
            if findings[sid] != {"['No Finding']"}:
                continue

            l_id = images.get(("L", "CC"))
            r_id = images.get(("R", "CC"))

            if l_id is None or r_id is None:
                missing = "L CC" if l_id is None else "R CC"
                skipped.append((sid, f"missing {missing}"))
                continue

            l_path = self.data_root / sid / f"{l_id}.png"
            r_path = self.data_root / sid / f"{r_id}.png"

            missing_files = [
                str(p) for p in (l_path, r_path) if not p.exists()
            ]
            if missing_files:
                skipped.append((sid, f"PNG not found: {', '.join(missing_files)}"))
                continue

            pairs.append((l_path, r_path))

        if skipped:
            print(f"[ScheduledBilateralDataset] Skipped {len(skipped)} studies:")
            for sid, reason in skipped:
                print(f"  {sid}: {reason}")

        print(
            f"[ScheduledBilateralDataset] Loaded {len(pairs)} paired studies"
            + (f" (split={split})" if split else "")
        )
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        l_path, r_path = self.pairs[idx]

        rng = random.Random(f"{self.seed}:{self.epoch}:{idx}")
        if self.p > 0.0 and len(self.pairs) > 1 and rng.random() < self.p:
            j = rng.randrange(len(self.pairs) - 1)
            if j >= idx:
                j += 1
            r_path = self.pairs[j][1]

        img_l = self._load(l_path, flip=False)
        img_r = self._load(r_path, flip=self.flip_right)
        return img_l, img_r

    def _load(self, path: Path, flip: bool) -> torch.Tensor:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        target_h, target_w = self.img_size
        if img.shape[0] != target_h or img.shape[1] != target_w:
            img = cv2.resize(img, (target_w, target_h),  # cv2 takes (width, height)
                             interpolation=cv2.INTER_LINEAR)
        if flip:
            img = cv2.flip(img, 1)
        tensor = torch.from_numpy(img.astype(np.float32)) / 127.5 - 1.0
        return tensor.unsqueeze(0)
