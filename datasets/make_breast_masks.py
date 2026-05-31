#!/usr/bin/env python
"""Precompute breast-foreground masks for mammography PNGs.

For every ``*.png`` under ``--data_root`` (recursively, skipping files that are
already ``*_mask.png``), segment the breast by intensity thresholding (the
background is 0): every pixel ``> --threshold`` is foreground. The largest
connected component is kept (dropping burnt-in labels / specks), then
**interior holes are filled** so dark (0-valued) pixels *enclosed by* the breast
stay foreground — only genuine exterior background is masked out. The result is
written as ``{stem}_mask.png`` (uint8 0/255) next to the source image at its
**native resolution**.

The dataset loader (``data/bilateral.py:_load_mask``) resizes/flips/crops the
mask identically to the image, so masks must be saved at the image's native
size. The script is idempotent: existing ``*_mask.png`` are skipped unless
``--overwrite`` is given.

Usage::

    python datasets/make_breast_masks.py --data_root ./datasets/<vindr-png-root>
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill background regions fully enclosed by the foreground.

    Corner-independent: a 1-px background border is added so the exterior
    background always connects to ``(0, 0)`` of the padded image, then flood
    filled. Whatever the flood does not reach is an interior hole and is set to
    foreground. This keeps dark pixels *inside* the breast while leaving the real
    exterior background masked out, regardless of which edge/corner the breast
    touches.
    """
    h, w = mask.shape
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(padded, flood, (0, 0), 255)        # fill exterior background
    exterior = padded[1:-1, 1:-1]
    holes = cv2.bitwise_not(exterior)                # interior holes only
    return cv2.bitwise_or(mask, holes)


def breast_mask(gray: np.ndarray, threshold: int = 0, min_area_frac: float = 0.02) -> np.ndarray:
    """Return a uint8 ``{0, 255}`` foreground mask for a grayscale mammogram.

    Foreground is ``gray > threshold`` (the background is 0). The largest
    connected component is kept and its interior holes filled. Returns an
    all-zero mask when segmentation looks like it failed (no components, or the
    largest component is smaller than ``min_area_frac`` of the image) so the
    loader can treat it as "no mask".
    """
    fg = (gray > threshold).astype(np.uint8) * 255

    # Keep the largest connected component (the breast); drops burnt-in labels
    # and stray specks that are disconnected from the breast.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return np.zeros_like(gray)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    if mask.sum() < min_area_frac * mask.size * 255:
        return np.zeros_like(gray)

    return _fill_holes(mask)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data_root", required=True, type=Path,
                    help="Root directory of converted PNG images (searched recursively)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute and overwrite existing *_mask.png")
    ap.add_argument("--threshold", type=int, default=0,
                    help="Foreground = pixel intensity > threshold; the background "
                         "is 0 (default 0). Interior dark pixels enclosed by the "
                         "breast are kept via hole filling.")
    ap.add_argument("--min_area_frac", type=float, default=0.02,
                    help="Reject masks whose breast component is smaller than this "
                         "fraction of the image (default 0.02)")
    args = ap.parse_args()

    pngs = sorted(
        p for p in args.data_root.rglob("*.png") if not p.stem.endswith("_mask")
    )
    written = skipped = empty = unreadable = 0
    samples: list[Path] = []

    for p in tqdm(pngs, desc="masking", unit="img"):
        out = _mask_out_path(p)
        if out.exists() and not args.overwrite:
            skipped += 1
            continue
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  WARN unreadable: {p}")
            unreadable += 1
            continue
        mask = breast_mask(gray, args.threshold, args.min_area_frac)
        if int(mask.sum()) == 0:
            empty += 1
            print(f"  WARN empty mask (segmentation failed?): {p}")
        cv2.imwrite(str(out), mask)
        written += 1
        if len(samples) < 5:
            samples.append(out)

    print(
        f"[make_breast_masks] wrote {written}, skipped {skipped} existing, "
        f"{empty} empty, {unreadable} unreadable, of {len(pngs)} source PNGs"
    )
    for s in samples:
        print(f"  sample: {s}")


def _mask_out_path(image_path: Path) -> Path:
    """Mirror ``data/bilateral.py:_mask_path`` so train/test read what we write."""
    return image_path.with_name(f"{image_path.stem}_mask.png")


if __name__ == "__main__":
    main()
