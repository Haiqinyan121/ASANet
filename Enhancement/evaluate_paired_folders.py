"""Evaluate paired output and ground-truth folders with the paper's conventions."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage import img_as_ubyte

from Enhancement import utils


SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    return parser.parse_args()


def image_map(folder):
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(root)
    return {
        path.relative_to(root).with_suffix("").as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUFFIXES
    }


def load_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main():
    args = parse_args()
    preds = image_map(args.pred_dir)
    targets = image_map(args.gt_dir)
    common = sorted(set(preds) & set(targets))
    if not common:
        raise RuntimeError("No paired images with matching relative names were found.")
    missing_pred = sorted(set(targets) - set(preds))
    missing_gt = sorted(set(preds) - set(targets))
    if missing_pred or missing_gt:
        raise RuntimeError(
            f"Folder mismatch: {len(missing_pred)} missing predictions, "
            f"{len(missing_gt)} predictions without ground truth."
        )

    psnrs, ssims, rmses = [], [], []
    for key in common:
        pred = load_rgb(preds[key])
        target = load_rgb(targets[key])
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch for {key}: {pred.shape} vs {target.shape}")

        pred_float = pred.astype(np.float32) / 255.0
        target_float = target.astype(np.float32) / 255.0
        psnrs.append(utils.PSNR(target_float, pred_float))
        ssims.append(utils.calculate_ssim(img_as_ubyte(target_float), img_as_ubyte(pred_float)))

        pred_gray = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY).astype(np.float64)
        target_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY).astype(np.float64)
        rmses.append(float(np.sqrt(np.mean((pred_gray - target_gray) ** 2))))

    print(f"Images: {len(common)}")
    print(f"PSNR (RGB): {np.mean(psnrs):.6f}")
    print(f"SSIM (RGB): {np.mean(ssims):.6f}")
    print(f"RMSE (8-bit grayscale): {np.mean(rmses):.6f}")


if __name__ == "__main__":
    main()

