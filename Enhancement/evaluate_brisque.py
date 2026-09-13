r"""Evaluate BRISQUE for one method on multiple unpaired datasets.

Expected directory layout for one method:

method_root/
  DICM/
  LIME/
  MEF/
  NPE/
  VV/

PowerShell example:
python Enhancement/evaluate_brisque.py `
  --pred_root results\Retinexformer `
  --method_name Retinexformer `
  --subdirs DICM LIME MEF NPE VV `
  --device cuda:0 `
  --save_csv results\Retinexformer_brisque.csv

Install dependency first if needed:
pip install pyiqa
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_SUBDIRS = ["DICM", "LIME", "MEF", "NPE", "VV"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate BRISQUE for one method across multiple unpaired datasets."
    )
    parser.add_argument(
        "--pred_root",
        type=str,
        required=True,
        help="Root directory containing one method's enhanced results for multiple datasets.",
    )
    parser.add_argument(
        "--method_name",
        type=str,
        default="",
        help="Optional method name written to CSV/log. Defaults to pred_root folder name.",
    )
    parser.add_argument(
        "--subdirs",
        nargs="*",
        default=DEFAULT_SUBDIRS,
        help="Dataset subdirectories under pred_root. Default: DICM LIME MEF NPE VV.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device used by pyiqa, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default="",
        help="Optional path to save summary CSV.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of images per dataset. 0 means all images.",
    )
    return parser.parse_args()


def load_brisque_metric(device):
    try:
        import pyiqa
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'pyiqa'. Install it with: pip install pyiqa"
        ) from exc

    return pyiqa.create_metric("brisque", device=device)


def collect_image_paths(folder):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {folder}")

    image_paths = [
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    ]
    return sorted(image_paths)


def read_image_tensor(image_path, device):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return tensor


def evaluate_dataset(metric, dataset_name, pred_dir, device, limit):
    image_paths = collect_image_paths(pred_dir)
    if limit > 0:
        image_paths = image_paths[:limit]

    if not image_paths:
        raise FileNotFoundError(f"No image files found in: {pred_dir}")

    scores = []
    for image_path in tqdm(image_paths, desc=f"Evaluating {dataset_name}", total=len(image_paths)):
        image_tensor = read_image_tensor(image_path, device)
        with torch.inference_mode():
            score = float(metric(image_tensor).item())
        scores.append(score)

    return {
        "dataset": dataset_name,
        "num_images": len(scores),
        "brisque": float(np.mean(scores)),
        "pred_dir": str(Path(pred_dir)),
    }


def save_csv(csv_path, rows, method_name):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["method", "dataset", "num_images", "brisque", "pred_dir"],
        )
        writer.writeheader()
        for row in rows:
            row_with_method = dict(row)
            row_with_method["method"] = method_name
            writer.writerow(row_with_method)


def main():
    args = parse_args()
    pred_root = Path(args.pred_root)
    method_name = args.method_name if args.method_name else pred_root.name
    metric = load_brisque_metric(args.device)

    summaries = []
    for subdir in args.subdirs:
        pred_dir = pred_root / subdir
        summary = evaluate_dataset(metric, subdir, pred_dir, args.device, args.limit)
        summaries.append(summary)
        print(
            f"{method_name} | {summary['dataset']}: "
            f"BRISQUE={summary['brisque']:.4f}, "
            f"images={summary['num_images']}"
        )

    if len(summaries) > 1:
        total_images = int(sum(item["num_images"] for item in summaries))
        overall = {
            "dataset": "Average",
            "num_images": total_images,
            "brisque": float(
                sum(item["brisque"] * item["num_images"] for item in summaries) / total_images
            ),
            "pred_dir": str(pred_root),
        }
        summaries.append(overall)
        print(
            f"{method_name} | {overall['dataset']}: "
            f"BRISQUE={overall['brisque']:.4f}, "
            f"images={overall['num_images']}"
        )

    if args.save_csv:
        save_csv(args.save_csv, summaries, method_name)
        print(f"Saved CSV to: {args.save_csv}")


if __name__ == "__main__":
    main()
