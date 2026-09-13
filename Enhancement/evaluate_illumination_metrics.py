"""Evaluate illumination-map smoothness/leakage metrics for multiple models.

Metrics implemented on the estimator output M:
- TV(M): mean absolute finite-difference total variation.
- Laplacian Energy: mean squared Laplacian response of M.
- Edge Gradient Mean: mean illumination-gradient magnitude on the strongest
  input-image edges.
- Dark-region Variance: variance of M inside the darkest input-image pixels.

Example (Linux/bash):
python Enhancement/evaluate_illumination_metrics.py \
  --model "baseline::configs/evaluation/lolv2_real_test.yml::/path/to/baseline.pth" \
  --model "EIE-only::configs/evaluation/lolv2_real_test.yml::/path/to/eie_only.pth" \
  --model "ASANet::configs/evaluation/lolv2_real_test.yml::/path/to/asanet.pth" \
  --dataset "LOLv1::data/LOLv1/Test/input" \
  --dataset "LOLv2-real::data/LOLv2/Real_captured/Test/Low" \
  --dataset "LOLv2-syn::data/LOLv2/Synthetic/Test/Low" \
  --dataset "DICM::data/DICM" \
  --dataset "LIME::data/LIME" \
  --dataset "MEF::data/MEF" \
  --dataset "NPE::data/NPE" \
  --dataset "VV::data/VV" \
  --save_dir illumination_metrics --save_maps
"""

import argparse
import contextlib
import csv
import importlib
import importlib.util
import io
import os
import re
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from basicsr.utils.options import parse  # noqa: E402


VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
METRIC_NAMES = (
    "tv_m",
    "laplacian_energy",
    "edge_gradient_mean",
    "dark_region_variance",
)


def suppress_import(module_name):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        return importlib.import_module(module_name)


def load_module_from_arch_file(arch_path):
    arch_path = Path(arch_path).resolve()
    if not arch_path.exists():
        raise FileNotFoundError(f"Missing arch file: {arch_path}")

    suppress_import("basicsr.models.archs")
    safe_stem = re.sub(r"\W+", "_", arch_path.stem)
    module_name = f"basicsr.models.archs._custom_{safe_stem}"

    spec = importlib.util.spec_from_file_location(module_name, str(arch_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load arch module from: {arch_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_model_spec(raw):
    parts = [item.strip() for item in raw.split("::")]
    if len(parts) not in (3, 4):
        raise ValueError(
            f"Invalid --model spec: {raw}\n"
            "Expected format: name::opt_path::weights_path "
            "or name::opt_path::weights_path::arch_file"
        )
    name, opt_path, weights_path = parts[:3]
    spec = {
        "name": name,
        "opt_path": Path(opt_path),
        "weights_path": Path(weights_path),
    }
    spec["arch_path"] = Path(parts[3]) if len(parts) == 4 else None
    return spec


def parse_dataset_spec(raw):
    parts = [item.strip() for item in raw.split("::")]
    if len(parts) != 2:
        raise ValueError(
            f"Invalid --dataset spec: {raw}\n"
            "Expected format: name::input_dir"
        )
    name, input_dir = parts
    return {
        "name": name,
        "input_dir": Path(input_dir),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Quantitatively evaluate illumination maps exported from ASANet-style "
            "estimators. Metrics are computed on the estimator output M: "
            "TV(M), Laplacian Energy, Edge Gradient Mean, and Dark-region Variance."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help=(
            "Repeated model spec: name::opt_path::weights_path "
            "or name::opt_path::weights_path::arch_file"
        ),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Repeated dataset spec: name::input_dir",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="illumination_metrics",
        help="Directory used to save per-image and summary CSV files.",
    )
    parser.add_argument(
        "--save_maps",
        action="store_true",
        help="Also save raw illumination maps (.npy) and per-image normalized grayscale PNGs.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="GPU devices, e.g. 0 or 0,1. Use -1 for CPU.",
    )
    parser.add_argument(
        "--param_key",
        type=str,
        default="",
        help="Preferred checkpoint key, e.g. params or params_ema. Default auto-detects.",
    )
    parser.add_argument(
        "--no_strict",
        action="store_true",
        help="Load checkpoint weights with strict=False.",
    )
    parser.add_argument(
        "--stage_index",
        type=int,
        default=0,
        help="Stage index used to fetch the estimator when the network has multiple stages.",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=4,
        help="Pad images so H/W are divisible by this factor before inference.",
    )
    parser.add_argument(
        "--crop_border",
        type=int,
        default=0,
        help="Crop border before metric computation.",
    )
    parser.add_argument(
        "--edge_percentile",
        type=float,
        default=90.0,
        help="Top-x percentile of input gradient magnitude used as the strong-edge mask.",
    )
    parser.add_argument(
        "--dark_percentile",
        type=float,
        default=25.0,
        help="Bottom-x percentile of input grayscale intensity used as the dark-region mask.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of images per dataset. 0 means all images.",
    )
    return parser.parse_args()


def validate_percentile(value, name):
    if value < 0 or value > 100:
        raise ValueError(f"{name} must be in [0, 100], but got {value}.")


def collect_image_paths(folder):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {folder}")

    def is_hidden_path(path):
        try:
            relative_parts = path.relative_to(folder).parts
        except ValueError:
            relative_parts = path.parts
        return any(part.startswith(".") for part in relative_parts if part not in ("", "."))

    image_paths = [
        path for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VALID_SUFFIXES
        and not is_hidden_path(path)
    ]
    return sorted(image_paths)


def read_rgb_float(image_path):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        return np.asarray(image, dtype=np.float32) / 255.0


def image_to_tensor(img_rgb, device):
    tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor.to(device)


def pad_to_factor(x, factor):
    _, _, h, w = x.shape
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, (pad_h, pad_w)


def crop_back(x, pad_hw):
    pad_h, pad_w = pad_hw
    if pad_h > 0:
        x = x[:, :, :-pad_h, :]
    if pad_w > 0:
        x = x[:, :, :, :-pad_w]
    return x


def tensor_to_gray_map(tensor):
    array = tensor.detach().float().cpu().squeeze(0)
    if array.dim() == 3:
        if array.shape[0] == 1:
            array = array[0]
        else:
            array = array.mean(dim=0)
    return array.numpy().astype(np.float32)


def rgb_to_gray(image_rgb):
    coeffs = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    return np.tensordot(image_rgb.astype(np.float32), coeffs, axes=([-1], [0]))


def crop_border_array(array, crop_border):
    if crop_border <= 0:
        return array
    h, w = array.shape[:2]
    if h <= 2 * crop_border or w <= 2 * crop_border:
        raise ValueError(
            f"crop_border={crop_border} is too large for shape {array.shape}."
        )
    return array[crop_border:h - crop_border, crop_border:w - crop_border]


def filter2d(image, kernel):
    image = image.astype(np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")

    output = np.zeros_like(image, dtype=np.float32)
    height, width = image.shape
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            output += kernel[i, j] * padded[i:i + height, j:j + width]
    return output


def sobel_magnitude(image):
    sobel_x = np.asarray(
        [[-1.0, 0.0, 1.0],
         [-2.0, 0.0, 2.0],
         [-1.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    sobel_y = np.asarray(
        [[-1.0, -2.0, -1.0],
         [0.0, 0.0, 0.0],
         [1.0, 2.0, 1.0]],
        dtype=np.float32,
    )
    grad_x = filter2d(image, sobel_x)
    grad_y = filter2d(image, sobel_y)
    return np.sqrt(grad_x * grad_x + grad_y * grad_y)


def laplacian_response(image):
    lap_kernel = np.asarray(
        [[0.0, 1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    return filter2d(image, lap_kernel)


def safe_percentile_mask(values, percentile, low_is_selected):
    flat = values.reshape(-1)
    if flat.size == 0:
        return np.zeros_like(values, dtype=bool)

    threshold = float(np.percentile(flat, percentile))
    if low_is_selected:
        mask = values <= threshold
    else:
        mask = values >= threshold

    if int(mask.sum()) == 0:
        mask = np.zeros_like(values, dtype=bool)
        index = int(np.argmin(flat) if low_is_selected else np.argmax(flat))
        mask.reshape(-1)[index] = True
    return mask


def compute_metrics(illum_map, input_gray, crop_border, edge_percentile, dark_percentile):
    illum_map = crop_border_array(illum_map, crop_border)
    input_gray = crop_border_array(input_gray, crop_border)

    dx = np.abs(np.diff(illum_map, axis=1))
    dy = np.abs(np.diff(illum_map, axis=0))
    tv_m = 0.5 * (float(dx.mean()) + float(dy.mean()))

    lap = laplacian_response(illum_map)
    laplacian_energy = float(np.mean(np.square(lap)))

    illum_grad = sobel_magnitude(illum_map)
    input_grad = sobel_magnitude(input_gray)
    edge_mask = safe_percentile_mask(input_grad, edge_percentile, low_is_selected=False)

    edge_gradient_mean = float(illum_grad[edge_mask].mean())

    dark_mask = safe_percentile_mask(input_gray, dark_percentile, low_is_selected=True)
    dark_region_variance = float(np.var(illum_map[dark_mask]))

    return {
        "tv_m": tv_m,
        "laplacian_energy": laplacian_energy,
        "edge_gradient_mean": edge_gradient_mean,
        "dark_region_variance": dark_region_variance,
        "edge_pixels": int(edge_mask.sum()),
        "dark_pixels": int(dark_mask.sum()),
        "illum_min": float(illum_map.min()),
        "illum_max": float(illum_map.max()),
        "illum_mean": float(illum_map.mean()),
        "illum_std": float(illum_map.std()),
    }


def find_param_key(payload, preferred_key):
    if preferred_key and preferred_key in payload:
        return preferred_key
    for key in ("params_ema", "params", "state_dict"):
        if key in payload:
            return key
    return None


def load_weights(model, ckpt_path, preferred_key="", strict=True, device="cpu"):
    payload = torch.load(str(ckpt_path), map_location=device)
    if isinstance(payload, dict):
        param_key = find_param_key(payload, preferred_key)
        state_dict = payload[param_key] if param_key is not None else payload
    else:
        state_dict = payload

    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key[7:]] = value
        else:
            cleaned[key] = value
    model.load_state_dict(cleaned, strict=strict)


def build_network_from_opt(
    opt_path,
    weights_path,
    device,
    preferred_key="",
    strict=True,
    arch_path=None,
):
    opt = parse(str(opt_path), is_train=False)
    network_opt = deepcopy(opt["network_g"])
    network_type = network_opt.pop("type")

    if arch_path is None:
        module = suppress_import("basicsr.models.archs.ASANet_arch")
    else:
        module = load_module_from_arch_file(arch_path)
    model_cls = getattr(module, network_type)
    model = model_cls(**network_opt).to(device)
    load_weights(
        model,
        weights_path,
        preferred_key=preferred_key,
        strict=strict,
        device=device,
    )
    model.eval()
    return model


def get_stage_module(model, stage_index):
    if hasattr(model, "body"):
        num_stages = len(model.body)
        index = stage_index if stage_index >= 0 else num_stages + stage_index
        if index < 0 or index >= num_stages:
            raise IndexError(
                f"Invalid stage_index={stage_index} for network with {num_stages} stages."
            )
        stage = model.body[index]
    else:
        if stage_index not in (0, -1):
            raise IndexError("stage_index must be 0 or -1 when the model has no body.")
        stage = model

    if not hasattr(stage, "estimator"):
        raise AttributeError("Selected stage does not expose an estimator module.")
    return stage


def export_map(output_root, model_name, dataset_name, image_path, dataset_root, illum_map):
    relative_path = image_path.relative_to(dataset_root).with_suffix("")
    raw_path = output_root / "maps" / model_name / dataset_name / relative_path
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(str(raw_path) + ".npy", illum_map.astype(np.float32))

    vis = illum_map.astype(np.float32)
    vis_min = float(vis.min())
    vis_max = float(vis.max())
    scale = max(vis_max - vis_min, 1e-8)
    vis = np.clip((vis - vis_min) / scale, 0.0, 1.0)
    vis_u8 = (vis * 255.0).round().astype(np.uint8)
    Image.fromarray(vis_u8, mode="L").save(str(raw_path) + ".png")


def evaluate_one_image(model, stage_index, image_path, device, factor):
    img_rgb = read_rgb_float(image_path)
    input_gray = rgb_to_gray(img_rgb)

    x = image_to_tensor(img_rgb, device)
    x_pad, pad_hw = pad_to_factor(x, factor)

    with torch.inference_mode():
        stage = get_stage_module(model, stage_index)
        _, illum_map = stage.estimator(x_pad)

    illum_map = crop_back(illum_map, pad_hw)
    illum_gray = tensor_to_gray_map(illum_map)
    return illum_gray, input_gray


def aggregate_rows(rows, scope, model_name, dataset_name):
    summary = {
        "scope": scope,
        "model": model_name,
        "dataset": dataset_name,
        "num_images": len(rows),
    }
    for metric in METRIC_NAMES:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std())
    return summary


def save_csv(csv_path, fieldnames, rows):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    validate_percentile(args.edge_percentile, "edge_percentile")
    validate_percentile(args.dark_percentile, "dark_percentile")

    if args.gpus != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    use_cuda = args.gpus != "-1" and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    model_specs = [parse_model_spec(item) for item in args.model]
    dataset_specs = [parse_dataset_spec(item) for item in args.dataset]

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    models = {}
    for spec in model_specs:
        if not spec["opt_path"].exists():
            raise FileNotFoundError(f"Missing opt file: {spec['opt_path']}")
        if not spec["weights_path"].exists():
            raise FileNotFoundError(f"Missing weights file: {spec['weights_path']}")
        models[spec["name"]] = build_network_from_opt(
            opt_path=spec["opt_path"],
            weights_path=spec["weights_path"],
            device=device,
            preferred_key=args.param_key,
            strict=not args.no_strict,
            arch_path=spec["arch_path"],
        )
        print(
            f"  {spec['name']}: "
            f"opt={spec['opt_path']} | weights={spec['weights_path']}"
            + (f" | arch={spec['arch_path']}" if spec["arch_path"] is not None else "")
        )

    per_image_rows = []

    for dataset_spec in dataset_specs:
        dataset_name = dataset_spec["name"]
        dataset_root = dataset_spec["input_dir"]
        image_paths = collect_image_paths(dataset_root)
        if args.limit > 0:
            image_paths = image_paths[:args.limit]
        if not image_paths:
            raise FileNotFoundError(f"No images found under: {dataset_root}")

        for model_name, model in models.items():
            desc = f"{model_name} | {dataset_name}"
            for image_path in tqdm(image_paths, desc=desc, ncols=100):
                illum_map, input_gray = evaluate_one_image(
                    model=model,
                    stage_index=args.stage_index,
                    image_path=image_path,
                    device=device,
                    factor=args.factor,
                )
                metrics = compute_metrics(
                    illum_map=illum_map,
                    input_gray=input_gray,
                    crop_border=args.crop_border,
                    edge_percentile=args.edge_percentile,
                    dark_percentile=args.dark_percentile,
                )

                row = {
                    "model": model_name,
                    "dataset": dataset_name,
                    "image": image_path.relative_to(dataset_root).as_posix(),
                    "input_path": str(image_path),
                }
                row.update(metrics)
                per_image_rows.append(row)

                if args.save_maps:
                    export_map(
                        output_root=save_dir,
                        model_name=model_name,
                        dataset_name=dataset_name,
                        image_path=image_path,
                        dataset_root=dataset_root,
                        illum_map=illum_map,
                    )

    summary_rows = []

    for model_name in models:
        model_rows = [row for row in per_image_rows if row["model"] == model_name]
        for dataset_spec in dataset_specs:
            dataset_name = dataset_spec["name"]
            rows = [
                row for row in model_rows
                if row["dataset"] == dataset_name
            ]
            if rows:
                summary_rows.append(
                    aggregate_rows(
                        rows=rows,
                        scope="dataset",
                        model_name=model_name,
                        dataset_name=dataset_name,
                    )
                )
        if model_rows:
            summary_rows.append(
                aggregate_rows(
                    rows=model_rows,
                    scope="overall",
                    model_name=model_name,
                    dataset_name="Overall",
                )
            )

    per_image_fields = [
        "model",
        "dataset",
        "image",
        "input_path",
        "tv_m",
        "laplacian_energy",
        "edge_gradient_mean",
        "dark_region_variance",
        "edge_pixels",
        "dark_pixels",
        "illum_min",
        "illum_max",
        "illum_mean",
        "illum_std",
    ]
    summary_fields = [
        "scope",
        "model",
        "dataset",
        "num_images",
        "tv_m_mean",
        "tv_m_std",
        "laplacian_energy_mean",
        "laplacian_energy_std",
        "edge_gradient_mean_mean",
        "edge_gradient_mean_std",
        "dark_region_variance_mean",
        "dark_region_variance_std",
    ]

    per_image_csv = save_dir / "per_image.csv"
    summary_csv = save_dir / "summary.csv"
    save_csv(per_image_csv, per_image_fields, per_image_rows)
    save_csv(summary_csv, summary_fields, summary_rows)

    print("\nSummary")
    for row in summary_rows:
        print(
            f"[{row['scope']}] {row['model']} | {row['dataset']} | n={row['num_images']} | "
            f"TV={row['tv_m_mean']:.6f} | "
            f"Lap={row['laplacian_energy_mean']:.6f} | "
            f"EGM={row['edge_gradient_mean_mean']:.6f} | "
            f"DRV={row['dark_region_variance_mean']:.6f}"
        )

    print(f"\nSaved per-image CSV to: {per_image_csv}")
    print(f"Saved summary CSV to: {summary_csv}")


if __name__ == "__main__":
    main()
