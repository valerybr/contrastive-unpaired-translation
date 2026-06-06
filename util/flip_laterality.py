"""Flip right-laterality (R) mammogram images horizontally.

Reads breast-level_annotations.csv or finding_annotations.csv to determine
laterality per image, then flips R images so all breasts face left.

Usage:
    python utils/flip_laterality.py \
        --annotations data/vindr/breast-level_annotations.csv \
        --images_dir  data/vindr/images \
        --output_dir  data/vindr/images_flipped

    # or with finding annotations:
    python utils/flip_laterality.py \
        --annotations data/vindr/finding_annotations.csv \
        --images_dir  data/vindr/images \
        --output_dir  data/vindr/images_flipped
"""

import argparse
import csv
from pathlib import Path

import cv2
from tqdm import tqdm


def build_laterality_map(annotations_csv: str) -> dict[str, str]:
    """Return {image_id: laterality} from either annotation CSV.

    Both breast-level_annotations.csv and finding_annotations.csv share the
    same column names (image_id, laterality), so one parser handles both.
    For finding_annotations an image may appear multiple times (one row per
    finding); the laterality is the same across rows for the same image_id,
    so the last value wins — which is fine.
    """
    laterality: dict[str, str] = {}
    with open(annotations_csv, newline="") as f:
        for row in csv.DictReader(f):
            laterality[row["image_id"]] = row["laterality"]
    return laterality


def flip_images(
    images_dir: Path,
    output_dir: Path,
    laterality: dict[str, str],
    skip_existing: bool = True,
) -> None:
    png_files = list(images_dir.rglob("*.png"))
    for src in tqdm(png_files, desc="Processing"):
        image_id = src.stem
        lat = laterality.get(image_id)

        dst = output_dir / src.relative_to(images_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)

        if skip_existing and dst.exists():
            continue

        img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: could not read {src}, skipping.")
            continue

        if lat == "R":
            img = cv2.flip(img, 1)

        cv2.imwrite(str(dst), img)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flip right-laterality mammogram PNGs horizontally."
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to breast-level_annotations.csv or finding_annotations.csv",
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
        "--no_skip",
        action="store_true",
        help="Overwrite existing output files instead of skipping them",
    )
    args = parser.parse_args()

    laterality = build_laterality_map(args.annotations)
    print(f"Loaded laterality for {len(laterality):,} images.")
    n_right = sum(1 for v in laterality.values() if v == "R")
    print(f"  L: {len(laterality) - n_right:,}  R: {n_right:,}")

    flip_images(
        images_dir=Path(args.images_dir),
        output_dir=Path(args.output_dir),
        laterality=laterality,
        skip_existing=not args.no_skip,
    )
    print("Done.")


if __name__ == "__main__":
    main()
