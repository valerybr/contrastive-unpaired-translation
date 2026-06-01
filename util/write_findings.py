#!/usr/bin/env python
"""Write a ``findings.json`` sidecar of finding bounding boxes for a results dir.

After ``test.py`` saves ``real_A`` / ``real_B`` / ``fake_B`` PNGs for a bilateral
findings run, this script emits the finding bounding boxes (from the VinDr
``finding_annotations.csv``) transformed into the *saved-image* coordinate frame,
so the overlay viewer (``util/overlay.html``) can draw them on the results.

It reuses ``data.bilateral.BilateralDataset`` to rebuild the exact same L/R
pairing the test run used (deterministic — the findings test uses the paired
``bilateral`` adapter with serial batches), and ``data.bilateral._transform_boxes``
to apply the same flip + crop geometry. The CSV boxes are assumed to already be
at ``--bilateral_size`` scale (no resize is applied).

Mapping (see ``cut_model.set_input``): the saved image filename is the stem of
the model's ``image_paths``, i.e. the A-domain image. ``fake_B = G(real_A)`` so
it carries the same boxes as ``real_A``; ``real_B`` carries its own.

    AtoB (default):  real_A = left CC (no flip),  real_B = right CC (flip_right),  file = left id
    BtoA:            real_A = right CC (flip_right), real_B = left CC (no flip),  file = right id

Output ``<results_dir>/<name>/<phase>_<epoch>/findings.json``::

    { "<id>.png": { "real_A": [{"box":[x0,y0,x1,y1], "label":"Mass"}],
                    "real_B": [...] } }

with box coords normalized to ``[0, 1]`` against the saved (cropped) image.

Run as a module (so ``util/`` doesn't shadow the stdlib ``html`` package),
mirroring the test invocation::

    python -m util.write_findings \
        --dataroot ... --annotations_csv ... \
        --results_dir ... --name ... --phase test --epoch latest \
        --finding_filter right_finding --split test \
        --bilateral_size 512 384 --crop_width 360 --direction AtoB
"""

import argparse
import ast
import csv
import json
from pathlib import Path

from data.bilateral import BilateralDataset, _transform_boxes


def _parse_label(finding_categories: str) -> str:
    """Turn a VinDr ``finding_categories`` cell into a short readable label.

    The cell is a stringified list, e.g. ``"['Mass']"`` or
    ``"['Suspicious Calcification', 'Mass']"``. Multiple categories are joined
    with ``", "``. Falls back to the raw string if it isn't a parseable list.
    """
    try:
        cats = ast.literal_eval(finding_categories)
    except (ValueError, SyntaxError):
        return finding_categories.strip()
    if isinstance(cats, (list, tuple)):
        return ", ".join(str(c) for c in cats)
    return str(cats)


def _boxes_by_image_id(
    annotations_csv: Path, split: str | None
) -> dict[str, list[tuple[float, float, float, float, str]]]:
    """Map ``image_id`` → list of ``(xmin, ymin, xmax, ymax, label)``.

    Only rows with a non-empty bounding box are kept (no-finding rows have empty
    coordinate cells). An image may appear in several rows (multiple findings).
    """
    out: dict[str, list[tuple[float, float, float, float, str]]] = {}
    with open(annotations_csv, newline="") as f:
        for row in csv.DictReader(f):
            if split is not None and row.get("split") != split:
                continue
            cells = (row.get("xmin"), row.get("ymin"),
                     row.get("xmax"), row.get("ymax"))
            if any(c is None or c == "" for c in cells):
                continue
            try:
                xmin, ymin, xmax, ymax = (float(c) for c in cells)
            except ValueError:
                continue
            label = _parse_label(row.get("finding_categories", ""))
            out.setdefault(row["image_id"], []).append(
                (xmin, ymin, xmax, ymax, label)
            )
    return out


def _side_boxes(
    image_path: Path,
    flip: bool,
    boxes_by_id: dict[str, list[tuple[float, float, float, float, str]]],
    img_size: tuple[int, int],
    crop_width: int | None,
) -> list[dict]:
    """Transformed, labeled boxes for one image (empty list if none survive)."""
    out: list[dict] = []
    for xmin, ymin, xmax, ymax, label in boxes_by_id.get(image_path.stem, []):
        # Transform one box at a time so labels stay aligned even when the crop
        # drops some boxes.
        transformed = _transform_boxes([(xmin, ymin, xmax, ymax)],
                                       img_size, flip, crop_width)
        if transformed:
            out.append({"box": transformed[0], "label": label})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--annotations_csv", required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--phase", default="test")
    ap.add_argument("--epoch", default="latest")
    ap.add_argument("--split", default="test")
    ap.add_argument("--finding_filter", default="right_finding",
                    choices=["no_finding", "left_finding",
                             "right_finding", "either_finding"])
    ap.add_argument("--bilateral_size", type=int, nargs=2, default=(512, 384),
                    metavar=("H", "W"))
    ap.add_argument("--crop_width", type=int, default=360)
    ap.add_argument("--flip_right", action="store_true")
    ap.add_argument("--direction", default="AtoB", choices=["AtoB", "BtoA"])
    args = ap.parse_args()

    img_size = (args.bilateral_size[0], args.bilateral_size[1])
    crop_width = args.crop_width or None

    boxes_by_id = _boxes_by_image_id(Path(args.annotations_csv), args.split)

    # Rebuild the exact same L/R pairing the test run used.
    dataset = BilateralDataset(
        data_root=args.dataroot,
        annotations_csv=args.annotations_csv,
        split=args.split,
        img_size=img_size,
        flip_right=args.flip_right,
        crop_width=crop_width,
        finding_filter=args.finding_filter,
    )

    atob = args.direction == "AtoB"
    findings: dict[str, dict[str, list[dict]]] = {}
    for l_path, r_path in dataset.pairs:
        if atob:
            (a_path, a_flip), (b_path, b_flip) = (l_path, False), (r_path, args.flip_right)
        else:
            (a_path, a_flip), (b_path, b_flip) = (r_path, args.flip_right), (l_path, False)
        filename = f"{a_path.stem}.png"  # = basename of model.image_paths

        a_boxes = _side_boxes(a_path, a_flip, boxes_by_id, img_size, crop_width)
        b_boxes = _side_boxes(b_path, b_flip, boxes_by_id, img_size, crop_width)
        entry: dict[str, list[dict]] = {}
        if a_boxes:
            entry["real_A"] = a_boxes
        if b_boxes:
            entry["real_B"] = b_boxes
        if entry:
            findings[filename] = entry

    web_dir = Path(args.results_dir) / args.name / f"{args.phase}_{args.epoch}"
    web_dir.mkdir(parents=True, exist_ok=True)
    out_path = web_dir / "findings.json"
    with open(out_path, "w") as f:
        json.dump(findings, f, indent=1)

    n_boxes = sum(len(v) for e in findings.values() for v in e.values())
    print(f"[write_findings] wrote {out_path} "
          f"({len(findings)} images, {n_boxes} boxes)")


if __name__ == "__main__":
    main()
