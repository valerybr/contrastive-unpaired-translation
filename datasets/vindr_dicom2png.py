import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pydicom
from pydicom.pixels import apply_voi_lut  # type: ignore
from tqdm import tqdm
from typing import Sequence


@dataclass
class Transform:
    """Geometry mapping original DICOM pixel coords -> converted PNG coords.

    Pipeline: crop breast (origin x0,y0, size w_crop x h_crop) -> scale by
    ``scale`` (height-anchored) -> place on an (out_h x out_w) canvas with the
    breast's left edge at ``x_off`` (chest wall kept flush to a canvas edge).
    """

    x0: int
    y0: int
    w_crop: int
    h_crop: int
    scale: float
    x_off: int
    out_w: int
    out_h: int


def _resize_to(
    img: np.ndarray, out_w: int, out_h: int = 512, align_right: bool = False
) -> tuple[np.ndarray, float, int]:
    """Height-anchored resize onto an (out_h x out_w) canvas, chest wall flush.

    Scales so the breast crop height becomes ``out_h`` (aspect preserved), then
    places it on a zero canvas of width ``out_w`` flush to one side: left when
    ``align_right`` is False, right when True. Padding therefore lands on the
    nipple side; if the scaled breast is wider than the canvas the nipple side
    is cropped instead.

    Returns the canvas, the applied scale, and ``x_off`` (the x of the breast's
    left edge on the canvas; may be negative when the nipple side is cropped).
    """
    h, w = img.shape
    scale = out_h / h
    new_w = round(w * scale)
    resized = cv2.resize(img, (new_w, out_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((out_h, out_w), dtype=np.uint8)
    if new_w >= out_w:
        print(f"Warning: scaled width {new_w} exceeds out_w {out_w}; cropping nipple side.")
        if align_right:
            canvas[:, :] = resized[:, new_w - out_w:]  # keep right (chest wall) part
        else:
            canvas[:, :] = resized[:, :out_w]  # keep left (chest wall) part
        x_off = out_w - new_w if align_right else 0
    else:
        x_off = out_w - new_w if align_right else 0
        canvas[:, x_off : x_off + new_w] = resized
    return canvas, scale, x_off


def count_up_continuing_ones(b_arr):
    # indice continuing zeros from left side.
    # ex: [0,1,1,0,1,0,0,1,1,1,0] -> [0,0,0,3,3,5,6,6,6,6,10]
    left = np.arange(len(b_arr))
    left[b_arr > 0] = 0
    left = np.maximum.accumulate(left)

    # from right side.
    # ex: [0,1,1,0,1,0,0,1,1,1,0] -> [0,3,3,3,5,5,6,10,10,10,10]
    rev_arr = b_arr[::-1]
    right = np.arange(len(rev_arr))
    right[rev_arr > 0] = 0
    right = np.maximum.accumulate(right)
    right = len(rev_arr) - 1 - right[::-1]

    return right - left - 1


def extract_breast(img):
    """Crop to the breast bounding box (Mammo-CLIP preprocessing).

    Returns the cropped image and its bounding box ``(x0, y0, w, h)`` in the
    coordinate space of the input image. The kept column/row indices are the
    longest contiguous run of non-constant lines, so the crop is a plain slice.
    """
    img_copy = img.copy()
    img = np.where(img <= 40, 0, img)  # To detect backgrounds easily
    height, _ = img.shape

    # whether each col is non-constant or not
    y_a = height // 2 + int(height * 0.4)
    y_b = height // 2 - int(height * 0.4)
    b_arr = img[y_b:y_a].std(axis=0) != 0
    continuing_ones = count_up_continuing_ones(b_arr)
    # longest should be the breast
    col_ind = np.where(continuing_ones == continuing_ones.max())[0]
    img = img[:, col_ind]

    # whether each row is non-constant or not
    _, width = img.shape
    x_a = width // 2 + int(width * 0.4)
    x_b = width // 2 - int(width * 0.4)
    b_arr = img[:, x_b:x_a].std(axis=1) != 0
    continuing_ones = count_up_continuing_ones(b_arr)
    # longest should be the breast
    row_ind = np.where(continuing_ones == continuing_ones.max())[0]

    box = (int(col_ind[0]), int(row_ind[0]), int(len(col_ind)), int(len(row_ind)))
    return img_copy[row_ind][:, col_ind], box


def _decode_and_crop(path: str) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    """Read a DICOM, normalise to uint8, and crop to the breast bounding box.

    Returns the cropped uint8 image, its bounding box ``(x0, y0, w, h)`` in the
    original DICOM pixel space (which matches the annotation coordinates), and
    the original image width (used to locate the chest-wall side).
    """
    ds = pydicom.dcmread(path)
    pixels = ds.pixel_array  # keep integer dtype for apply_voi_lut

    # Apply VOI LUT: uses LUT table if present, falls back to WindowCenter/Width
    img = apply_voi_lut(pixels, ds).astype(np.float32)

    # MONOCHROME1: 0 = white (air), max = black (tissue) — invert to MONOCHROME2 convention
    if ds.PhotometricInterpretation == "MONOCHROME1":
        img = img.max() - img

    # Normalize to [0, 255]
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo) * 255.0
    img = img.astype(np.uint8)

    cropped, box = extract_breast(img)
    return cropped, box, img.shape[1]


def load_mammogram(
    path: str, out_w: int, out_h: int = 512
) -> tuple[np.ndarray, Transform]:
    """Read a DICOM mammography file, remove background, crop breast, resize.

    Steps:
      1. Read pixel array and apply VOI LUT (window/level or LUT table)
      2. Invert MONOCHROME1 images so tissue is always bright
      3. Normalize to uint8
      4. Remove black background and crop to the breast bounding box
      5. Height-anchored scale to ``out_h`` and place on an (out_h x out_w)
         canvas with the chest wall flush to the edge it touches (padding on the
         nipple side). Orientation is preserved (no horizontal flip).

    Args:
        path: Path to a .dicom / .dcm file.
        out_w: Output width in pixels (see ``measure_width``).
        out_h: Output height in pixels (default 512).

    Returns:
        Tuple of (uint8 array of shape (out_h, out_w), Transform).
    """
    cropped, (x0, y0, w_crop, h_crop), orig_w = _decode_and_crop(path)

    # The chest wall is the border the breast touches (the smaller gap); keep it
    # flush against the canvas edge so padding falls on the nipple side.
    left_gap, right_gap = x0, orig_w - (x0 + w_crop)
    align_right = right_gap < left_gap

    img, scale, x_off = _resize_to(cropped, out_w, out_h, align_right=align_right)
    transform = Transform(
        x0=x0, y0=y0, w_crop=w_crop, h_crop=h_crop,
        scale=scale, x_off=x_off, out_w=out_w, out_h=out_h,
    )
    return img, transform


def transform_bbox(
    x1: float, y1: float, x2: float, y2: float, t: Transform
) -> tuple[float, float, float, float] | None:
    """Map a bounding box from original DICOM coords into converted-PNG coords.

    Applies the same crop -> scale -> place pipeline used for the image, then
    clamps to the canvas. Returns ``None`` if the box ends up with no area
    inside the canvas.
    """
    # 1. translate by crop origin
    xa, xb = x1 - t.x0, x2 - t.x0
    ya, yb = y1 - t.y0, y2 - t.y0

    # 2. height-anchored scale, then shift to the breast's placement on the canvas
    nx1, nx2 = xa * t.scale + t.x_off, xb * t.scale + t.x_off
    ny1, ny2 = ya * t.scale, yb * t.scale

    # 3. clamp to the canvas
    nx1, nx2 = max(0.0, min(nx1, t.out_w)), max(0.0, min(nx2, t.out_w))
    ny1, ny2 = max(0.0, min(ny1, t.out_h)), max(0.0, min(ny2, t.out_h))

    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return (round(nx1, 4), round(ny1, 4), round(nx2, 4), round(ny2, 4))


def measure_width(files: list[Path], out_h: int = 512) -> int:
    """Find the minimum canvas width that fits every breast, rounded up to /4.

    Decodes each DICOM, crops the breast, and computes its height-anchored
    scaled width (``w_crop * out_h / h_crop``). Returns the maximum across all
    images rounded up to the nearest multiple of 4.
    """
    max_w = 0.0
    for src in tqdm(files, desc="Measuring width"):
        try:
            _, (_, _, w_crop, h_crop), _ = _decode_and_crop(str(src))
        except Exception as e:  # noqa: BLE001
            print(f"Warning: could not measure {src}: {e}")
            continue
        max_w = max(max_w, w_crop * out_h / h_crop)

    out_w = math.ceil(max_w / 4) * 4
    print(f"Max scaled breast width = {max_w:.1f} px -> out_w = {out_w} (multiple of 4)")
    return out_w


BBOX_COLS = ("xmin", "ymin", "xmax", "ymax")


def _list_dicoms(dicom_path: Path) -> list[Path]:
    return [p for p in dicom_path.rglob("*") if p.suffix.lower() in {".dcm", ".dicom"}]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert VinDr DICOMs to (out_h x out_w) PNGs and rewrite "
        "finding annotations into the converted image space."
    )
    parser.add_argument("--dicom_path", help="path to dicom images", type=str, required=True)
    parser.add_argument("--output_path", help="path to output PNG images", type=str, required=True)
    parser.add_argument("--out_height", help="output image height (default 512)", type=int, default=512)
    parser.add_argument(
        "--out_width",
        help="output image width; if omitted it is measured across the dataset "
        "(min width fitting every breast, rounded up to a multiple of 4)",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--annotations",
        help="path to finding_annotations.csv; provides the bounding boxes to transform",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--out_annotations",
        help="path to write the transformed annotations CSV",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    dicom_path = Path(args.dicom_path)
    output_path = Path(args.output_path)
    out_h = args.out_height

    files = _list_dicoms(dicom_path)
    print(f"Found {len(files):,} DICOM files under {dicom_path}.")

    # image_id -> list of annotation rows
    fieldnames: Sequence[str] | None = None
    rows_by_image: dict[str, list[dict[str, str]]] = {}
    if args.annotations:
        with open(args.annotations, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                rows_by_image.setdefault(row["image_id"], []).append(row)

    # Determine output width (measure if not provided).
    out_w = args.out_width if args.out_width is not None else measure_width(files, out_h)

    out_rows: list[dict[str, str]] = []
    n_converted = n_skipped = n_dropped = n_missing_disk = 0

    for src in tqdm(files, desc="Converting"):
        dst = output_path / src.relative_to(dicom_path).with_suffix(".png")
        image_id = src.stem

        if dst.exists():
            n_skipped += 1
            # We still need geometry to transform this image's annotations.
            if image_id not in rows_by_image:
                continue
            try:
                _, transform = load_mammogram(str(src), out_w=out_w, out_h=out_h)
            except Exception as e:  # noqa: BLE001
                print(f"Warning: could not process {src}: {e}")
                continue
        else:
            try:
                img, transform = load_mammogram(str(src), out_w=out_w, out_h=out_h)
            except Exception as e:  # noqa: BLE001
                print(f"Warning: could not process {src}: {e}")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), img)
            n_converted += 1

        for row in rows_by_image.get(image_id, []):
            new_row = dict(row)
            new_row["height"] = str(out_h)
            new_row["width"] = str(out_w)
            if all(str(row.get(c, "")).strip() for c in BBOX_COLS):
                box = transform_bbox(
                    float(row["xmin"]), float(row["ymin"]),
                    float(row["xmax"]), float(row["ymax"]), transform,
                )
                if box is None:
                    n_dropped += 1
                    new_row["xmin"] = new_row["ymin"] = new_row["xmax"] = new_row["ymax"] = ""
                else:
                    new_row["xmin"], new_row["ymin"], new_row["xmax"], new_row["ymax"] = (
                        str(v) for v in box
                    )
            out_rows.append(new_row)

    # Annotation rows whose image was not found on disk.
    seen_ids = {p.stem for p in files}
    for image_id, rows in rows_by_image.items():
        if image_id not in seen_ids:
            n_missing_disk += len(rows)

    if args.out_annotations and fieldnames:
        with open(args.out_annotations, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"Wrote {len(out_rows):,} annotation rows to {args.out_annotations}.")

    print(
        f"Done. converted={n_converted:,} skipped_existing={n_skipped:,} "
        f"boxes_dropped={n_dropped:,} rows_missing_on_disk={n_missing_disk:,} "
        f"out_size={out_w}x{out_h}."
    )


if __name__ == "__main__":
    main()
