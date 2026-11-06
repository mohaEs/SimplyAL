#!/usr/bin/env python
"""Build a long-tailed EuroSAT subset for simulate_al.py.

EuroSAT (RGB) has 27,000 Sentinel-2 satellite images (64 x 64 px) in 10
land-use classes. This script draws a fixed, reproducible subset whose class
sizes fall off exponentially from the largest to the smallest class
(imbalance ratio = largest / smallest), copies those images into --out and
writes the CSV that simulate_al.py reads (columns file_path, annotation).

Default: about 2,000 images, ratio 10, class sizes
489, 379, 293, 227, 176, 136, 105, 82, 63, 49.
Which class gets which size is drawn at random with --seed and written to
class_counts.csv.

Examples
  # download through torchvision (about 90 MB) and build the subset
  python prepare_eurosat.py --root ~/data/eurosat_raw --out ~/data/eurosat_lt

  # if the download fails: download EuroSAT_RGB yourself, unzip it, and point
  # --image-dir at the folder that holds AnnualCrop/, Forest/, ... :
  python prepare_eurosat.py --image-dir ~/Downloads/EuroSAT_RGB --out ~/data/eurosat_lt

Then run the simulator, for example:
  python backend/simulate_al.py --csv ~/data/eurosat_lt/eurosat_lt.csv \\
      --label-column annotation --path-column file_path \\
      --models resnet18 --strategies least_confidence,random \\
      --lr 1e-5 --seed-size 16 --seeds 0,1,2 --max-frac 0.3 --out runs/eurosat
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np

CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
           "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", type=Path, help="download EuroSAT here with torchvision")
    src.add_argument("--image-dir", type=Path,
                     help="already unzipped EuroSAT RGB folder (contains AnnualCrop/, Forest/, ...)")
    p.add_argument("--out", type=Path, required=True, help="output folder for images and CSV")
    p.add_argument("--total", type=int, default=2000, help="approximate number of images")
    p.add_argument("--imbalance", type=float, default=10.0, help="largest / smallest class size")
    p.add_argument("--seed", type=int, default=0, help="fixes class order and image choice")
    return p.parse_args()


def find_class_root(base):
    """Folder under base whose subfolders are the 10 EuroSAT classes."""
    base = base.expanduser().resolve()
    for d in [base, *sorted(p for p in base.rglob("*") if p.is_dir())]:
        if all((d / c).is_dir() for c in CLASSES):
            return d
    return None


def download(root):
    try:
        from torchvision.datasets import EuroSAT
    except ImportError:
        sys.exit("torchvision is not installed; use --image-dir with a manual download instead")
    root.mkdir(parents=True, exist_ok=True)
    try:
        EuroSAT(root=str(root), download=True)
    except Exception as exc:  # broken mirror, no network, ...
        sys.exit(f"torchvision could not download EuroSAT ({exc}).\n"
                 "Download the RGB version (EuroSAT_RGB.zip) from https://github.com/phelber/EuroSAT, "
                 "unzip it, and rerun with --image-dir pointing at the unzipped folder.")
    return root


def class_sizes(total, ratio, k):
    """Exponential profile n_i = n_max * ratio^(-i/(k-1)), scaled to about `total`."""
    profile = ratio ** (-np.arange(k) / (k - 1))
    return np.maximum(1, np.round(total * profile / profile.sum())).astype(int)


def main():
    args = parse_args()
    base = download(args.root) if args.root else args.image_dir
    class_root = find_class_root(base)
    if class_root is None:
        sys.exit(f"no folder with the 10 EuroSAT class folders ({', '.join(CLASSES)}) under {base}")
    files = {c: sorted(f for f in (class_root / c).iterdir() if f.suffix.lower() in IMAGE_EXT)
             for c in CLASSES}
    sizes = class_sizes(args.total, args.imbalance, len(CLASSES))
    rng = np.random.RandomState(args.seed)
    order = [CLASSES[i] for i in rng.permutation(len(CLASSES))]  # which class gets which size
    for c, n in zip(order, sizes):
        if len(files[c]) < n:
            sys.exit(f"class {c} has only {len(files[c])} images, {n} needed")

    out = args.out.expanduser().resolve()
    img_root = out / "images"
    if img_root.exists():
        shutil.rmtree(img_root)  # rebuild from scratch so reruns stay reproducible
    rows, counts = [], []
    for c, n in zip(order, sizes):
        chosen = sorted(rng.choice(len(files[c]), size=n, replace=False))
        (img_root / c).mkdir(parents=True, exist_ok=True)
        for i in chosen:
            src = files[c][i]
            dst = img_root / c / src.name
            shutil.copy2(src, dst)
            rows.append((str(dst), c))
        counts.append((c, n))

    csv_path = out / "eurosat_lt.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file_path", "annotation"])
        w.writerows(rows)
    with open(out / "class_counts.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "n_images", "rank"])
        for rank, (c, n) in enumerate(counts, 1):
            w.writerow([c, n, rank])

    print(f"wrote {len(rows)} images in {len(counts)} classes to {img_root}")
    print(f"CSV: {csv_path}")
    print(f"imbalance ratio {counts[0][1] / counts[-1][1]:.1f} "
          f"(largest {counts[0][0]} {counts[0][1]}, smallest {counts[-1][0]} {counts[-1][1]})")
    for c, n in counts:
        print(f"  {c:22s} {n:4d}")


if __name__ == "__main__":
    main()
