"""Crop zero-padding from the right side of preprocessed mammogram PNGs.

All images are 512x512 with the breast left-aligned and zero-padded on the right.
This script crops every image to a fixed width (default 384) that covers 100% of
the dataset content, saving ~25% of pixels.

Usage:
    python utils/crop_width.py \
        --images_dir data/vindr/images \
        --output_dir data/vindr/images_384 \
        [--width 384]
"""

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

DEFAULT_WIDTH = 384  # minimum width covering 100% of VinDr-Mammo images; divisible by 4


def crop_images(images_dir: Path, output_dir: Path, width: int) -> None:
    assert width % 4 == 0, f"width must be divisible by 4, got {width}"

    png_files = list(images_dir.rglob("*.png"))
    print(f"Found {len(png_files):,} images. Cropping to width={width}.")

    skipped = 0
    for src in tqdm(png_files, desc="Cropping"):
        dst = output_dir / src.relative_to(images_dir)

        if dst.exists():
            skipped += 1
            continue

        img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: could not read {src}, skipping.")
            continue

        h, w = img.shape
        if w < width:
            print(f"Warning: {src.name} is only {w}px wide, padding to {width}.")
            import numpy as np
            img = cv2.copyMakeBorder(img, 0, 0, 0, width - w, cv2.BORDER_CONSTANT, value=0)
        else:
            img = img[:, :width]

        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), img)

    if skipped:
        print(f"Skipped {skipped:,} already-existing files.")
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop right-side zero-padding from mammogram PNGs to a fixed width."
    )
    parser.add_argument(
        "--images_dir",
        required=True,
        help="Root directory of PNG images (study_id/image_id.png structure)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Destination directory (same sub-structure as images_dir)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Output width in pixels, must be divisible by 4 (default: {DEFAULT_WIDTH})",
    )
    args = parser.parse_args()

    if args.width % 4 != 0:
        raise ValueError(f"--width must be divisible by 4, got {args.width}")

    crop_images(
        images_dir=Path(args.images_dir),
        output_dir=Path(args.output_dir),
        width=args.width,
    )


if __name__ == "__main__":
    main()
