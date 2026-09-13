import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from basicsr.metrics import calculate_loe, calculate_niqe


VALID_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def normalize_result_component(value):
    path = Path(value)
    if path.suffix.lower() in {'.yml', '.yaml', '.pth', '.pt', '.ckpt'}:
        return path.stem
    return path.name


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate NIQE and LOE on unpaired low-light datasets.')
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input dataset root or a single low-light dataset directory.')
    parser.add_argument(
        '--pred_dir',
        type=str,
        default='',
        help='Prediction directory for a single dataset.')
    parser.add_argument(
        '--result_root',
        type=str,
        default='',
        help='Result root used with --subdirs, e.g. results.')
    parser.add_argument(
        '--subdirs',
        nargs='*',
        default=None,
        help='Optional dataset subdirectories under input_dir, e.g. DICM LIME MEF NPE.')
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='',
        help='Dataset name when evaluating a single dataset.')
    parser.add_argument(
        '--config_name',
        type=str,
        default='',
        help='Config folder name under result_root/<dataset>, e.g. asanet_lolv2_real.')
    parser.add_argument(
        '--checkpoint_name',
        type=str,
        default='',
        help='Checkpoint folder name under result_root/<dataset>/<config_name>, e.g. LOL_v2_real.')
    parser.add_argument(
        '--crop_border', type=int, default=0, help='Crop border for NIQE/LOE.')
    parser.add_argument(
        '--niqe_convert_to',
        type=str,
        default='y',
        choices=['y', 'gray'],
        help='Color conversion used by NIQE.')
    parser.add_argument(
        '--loe_downsample_to',
        type=int,
        default=50,
        help='Resize shorter side to this value before LOE.')
    parser.add_argument(
        '--save_csv',
        type=str,
        default='',
        help='Optional path to save summary CSV.')
    return parser.parse_args()


def collect_image_paths(folder):
    folder = Path(folder)
    return sorted(
        path for path in folder.rglob('*')
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES)


def build_key_map(root_dir):
    root_dir = Path(root_dir)
    key_map = {}
    for path in collect_image_paths(root_dir):
        key = path.relative_to(root_dir).with_suffix('').as_posix()
        key_map[key] = path
    return key_map


def read_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'Failed to read image: {image_path}')
    return image


def evaluate_dataset(dataset_name,
                     input_dir,
                     pred_dir,
                     crop_border,
                     niqe_convert_to,
                     loe_downsample_to):
    input_map = build_key_map(input_dir)
    pred_map = build_key_map(pred_dir)
    common_keys = sorted(set(input_map.keys()) & set(pred_map.keys()))

    if not common_keys:
        raise FileNotFoundError(
            f'No matched image pairs found between {input_dir} and {pred_dir}.')

    niqe_scores = []
    loe_scores = []

    for key in tqdm(common_keys, desc=f'Evaluating {dataset_name}', total=len(common_keys)):
        input_img = read_image(input_map[key])
        pred_img = read_image(pred_map[key])

        niqe_value = calculate_niqe(
            pred_img,
            crop_border=crop_border,
            input_order='HWC',
            convert_to=niqe_convert_to)
        loe_value = calculate_loe(
            input_img,
            pred_img,
            crop_border=crop_border,
            input_order='HWC',
            downsample_to=loe_downsample_to)

        niqe_scores.append(float(niqe_value))
        loe_scores.append(float(loe_value))

    return {
        'dataset': dataset_name,
        'num_images': len(common_keys),
        'niqe': float(np.mean(niqe_scores)),
        'loe': float(np.mean(loe_scores)),
        'input_dir': str(Path(input_dir)),
        'pred_dir': str(Path(pred_dir)),
    }


def resolve_jobs(args):
    input_root = Path(args.input_dir)

    if args.subdirs:
        if not args.result_root or not args.config_name or not args.checkpoint_name:
            raise ValueError(
                '--result_root, --config_name and --checkpoint_name are required when using --subdirs.')

        config_name = normalize_result_component(args.config_name)
        checkpoint_name = normalize_result_component(args.checkpoint_name)
        jobs = []
        for subdir in args.subdirs:
            jobs.append({
                'dataset_name': subdir,
                'input_dir': input_root / subdir,
                'pred_dir': Path(args.result_root) / subdir / config_name / checkpoint_name,
            })
        return jobs

    if not args.pred_dir:
        raise ValueError('--pred_dir is required when evaluating a single dataset.')

    dataset_name = args.dataset_name if args.dataset_name else input_root.name
    return [{
        'dataset_name': dataset_name,
        'input_dir': input_root,
        'pred_dir': Path(args.pred_dir),
    }]


def save_csv(csv_path, rows):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['dataset', 'num_images', 'niqe', 'loe', 'input_dir', 'pred_dir'])
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    jobs = resolve_jobs(args)

    summaries = []
    for job in jobs:
        summary = evaluate_dataset(
            dataset_name=job['dataset_name'],
            input_dir=job['input_dir'],
            pred_dir=job['pred_dir'],
            crop_border=args.crop_border,
            niqe_convert_to=args.niqe_convert_to,
            loe_downsample_to=args.loe_downsample_to)
        summaries.append(summary)
        print(
            f"{summary['dataset']}: "
            f"NIQE={summary['niqe']:.4f}, "
            f"LOE={summary['loe']:.4f}, "
            f"images={summary['num_images']}")

    if len(summaries) > 1:
        total_images = int(sum(item['num_images'] for item in summaries))
        overall = {
            'dataset': 'Average',
            'num_images': total_images,
            'niqe': float(sum(item['niqe'] * item['num_images'] for item in summaries) / total_images),
            'loe': float(sum(item['loe'] * item['num_images'] for item in summaries) / total_images),
            'input_dir': '',
            'pred_dir': '',
        }
        summaries.append(overall)
        print(
            f"{overall['dataset']}: "
            f"NIQE={overall['niqe']:.4f}, "
            f"LOE={overall['loe']:.4f}, "
            f"images={overall['num_images']}")

    if args.save_csv:
        save_csv(args.save_csv, summaries)
        print(f'Saved CSV to: {args.save_csv}')


if __name__ == '__main__':
    main()
