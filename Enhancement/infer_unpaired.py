import argparse
import gc
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from natsort import natsorted
from skimage import img_as_ubyte
from tqdm import tqdm

import utils
from basicsr.models import create_model
from basicsr.utils.options import parse


VALID_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def self_ensemble(x, model):
    def forward_transformed(inp, hflip, vflip, rotate):
        if hflip:
            inp = torch.flip(inp, (-2,))
        if vflip:
            inp = torch.flip(inp, (-1,))
        if rotate:
            inp = torch.rot90(inp, dims=(-2, -1))
        out = model(inp)
        if isinstance(out, list):
            out = out[-1]
        if rotate:
            out = torch.rot90(out, dims=(-2, -1), k=3)
        if vflip:
            out = torch.flip(out, (-1,))
        if hflip:
            out = torch.flip(out, (-2,))
        return out

    outputs = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rotate in [False, True]:
                outputs.append(forward_transformed(x, hflip, vflip, rotate))
    return torch.mean(torch.stack(outputs), dim=0)


def forward_model(model, tensor, self_ensemble_enabled):
    if self_ensemble_enabled:
        restored = self_ensemble(tensor, model)
    else:
        restored = model(tensor)
        if isinstance(restored, list):
            restored = restored[-1]
    return restored


def is_oom_error(err):
    message = str(err).lower()
    return 'out of memory' in message or 'cuda error: out of memory' in message


def get_tile_starts(length, tile_size, tile_overlap):
    tile_size = min(tile_size, length)
    if tile_size >= length:
        return [0]

    stride = max(1, tile_size - tile_overlap)
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def build_blend_weight(tile_h, tile_w, tile_overlap):
    weight = torch.ones((1, 1, tile_h, tile_w), dtype=torch.float32)
    overlap_h = min(tile_overlap, tile_h - 1)
    overlap_w = min(tile_overlap, tile_w - 1)

    if overlap_h > 0:
        ramp_h = torch.linspace(0.0, 1.0, steps=overlap_h + 2, dtype=torch.float32)[1:-1]
        weight[:, :, :overlap_h, :] *= ramp_h.view(1, 1, overlap_h, 1)
        weight[:, :, -overlap_h:, :] *= torch.flip(ramp_h, dims=[0]).view(1, 1, overlap_h, 1)

    if overlap_w > 0:
        ramp_w = torch.linspace(0.0, 1.0, steps=overlap_w + 2, dtype=torch.float32)[1:-1]
        weight[:, :, :, :overlap_w] *= ramp_w.view(1, 1, 1, overlap_w)
        weight[:, :, :, -overlap_w:] *= torch.flip(ramp_w, dims=[0]).view(1, 1, 1, overlap_w)

    return weight


def infer_with_overlap_tiles(model,
                             tensor,
                             self_ensemble_enabled,
                             tile_size,
                             tile_overlap,
                             min_tile_size):
    _, channels, height, width = tensor.shape
    tile_size = min(tile_size, max(height, width))
    tile_overlap = min(tile_overlap, max(0, tile_size - 1))

    try:
        tile_h = min(tile_size, height)
        tile_w = min(tile_size, width)
        y_starts = get_tile_starts(height, tile_h, tile_overlap)
        x_starts = get_tile_starts(width, tile_w, tile_overlap)

        output = torch.zeros((1, channels, height, width), dtype=torch.float32)
        weight = torch.zeros((1, 1, height, width), dtype=torch.float32)
        blend_weight = build_blend_weight(tile_h, tile_w, tile_overlap)

        for top in y_starts:
            for left in x_starts:
                patch = tensor[:, :, top:top + tile_h, left:left + tile_w]
                restored = forward_model(model, patch, self_ensemble_enabled).detach().float().cpu()
                output[:, :, top:top + tile_h, left:left + tile_w] += restored * blend_weight
                weight[:, :, top:top + tile_h, left:left + tile_w] += blend_weight

                del patch
                del restored
                if tensor.is_cuda:
                    torch.cuda.empty_cache()

        return output / weight.clamp_min(1e-6)
    except RuntimeError as err:
        if not tensor.is_cuda or not is_oom_error(err):
            raise
        if tile_size <= min_tile_size:
            raise

        smaller_tile = max(min_tile_size, tile_size // 2)
        if smaller_tile >= tile_size:
            raise

        print(f'CUDA OOM with tile_size={tile_size}, retrying with tile_size={smaller_tile}')
        gc.collect()
        torch.cuda.empty_cache()
        return infer_with_overlap_tiles(
            model,
            tensor,
            self_ensemble_enabled,
            tile_size=smaller_tile,
            tile_overlap=tile_overlap,
            min_tile_size=min_tile_size)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inference on unpaired low-light datasets using ASANet weights.')
    parser.add_argument(
        '--opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--weights', type=str, required=True, help='Path to pretrained weights.')
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input root directory or a single dataset directory.')
    parser.add_argument(
        '--subdirs',
        nargs='*',
        default=None,
        help='Optional dataset subdirectories under input_dir, e.g. DICM LIME MEF NPE.')
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='',
        help='Dataset name when input_dir points to a single dataset directory.')
    parser.add_argument(
        '--result_dir',
        type=str,
        default='./results',
        help='Root directory for inference results.')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='',
        help='Optional custom output directory. When subdirs are used, outputs are saved under output_dir/<dataset_name>.')
    parser.add_argument(
        '--gpus', type=str, default='0', help='GPU devices, e.g. 0 or 0,1. Use -1 for CPU.')
    parser.add_argument(
        '--self_ensemble', action='store_true', help='Use self-ensemble for inference.')
    parser.add_argument(
        '--tile_size',
        type=int,
        default=512,
        help='Overlap-tile size. Set to 0 to disable tiled inference.')
    parser.add_argument(
        '--tile_overlap',
        type=int,
        default=32,
        help='Overlap size used by tiled inference.')
    parser.add_argument(
        '--min_tile_size',
        type=int,
        default=128,
        help='Minimum tile size used by automatic OOM fallback.')
    return parser.parse_args()


def build_model(opt_path, weights_path, use_cuda):
    opt = parse(opt_path, is_train=False)
    opt['dist'] = False

    model_restoration = create_model(opt).net_g
    checkpoint = torch.load(weights_path, map_location='cuda' if use_cuda else 'cpu')

    try:
        model_restoration.load_state_dict(checkpoint['params'])
    except Exception:
        new_checkpoint = {}
        for key in checkpoint['params']:
            new_checkpoint['module.' + key] = checkpoint['params'][key]
        model_restoration.load_state_dict(new_checkpoint)

    if use_cuda:
        model_restoration = model_restoration.cuda()
        model_restoration = nn.DataParallel(model_restoration)

    model_restoration.eval()
    return model_restoration


def collect_image_paths(folder):
    folder = Path(folder)
    image_paths = [
        path for path in folder.rglob('*')
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    ]
    return natsorted(image_paths)


def infer_image(model,
                img_path,
                use_cuda,
                self_ensemble_enabled,
                tile_size,
                tile_overlap,
                min_tile_size,
                factor=4):
    img = utils.load_img(str(img_path)).astype('float32') / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    if use_cuda:
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        tensor = tensor.cuda(non_blocking=True)

    _, _, h, w = tensor.shape
    H = ((h + factor) // factor) * factor
    W = ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    tensor = F.pad(tensor, (0, padw, 0, padh), 'reflect')

    with torch.inference_mode():
        if tile_size > 0:
            restored = infer_with_overlap_tiles(
                model,
                tensor,
                self_ensemble_enabled,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                min_tile_size=min_tile_size)
        else:
            restored = forward_model(model, tensor, self_ensemble_enabled).detach().float().cpu()

    restored = restored[:, :, :h, :w]
    restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
    if use_cuda:
        gc.collect()
        torch.cuda.empty_cache()
    return img_as_ubyte(restored)


def resolve_jobs(input_dir, subdirs, dataset_name):
    input_root = Path(input_dir)
    if subdirs:
        jobs = []
        for subdir in subdirs:
            dataset_dir = input_root / subdir
            jobs.append((subdir, dataset_dir))
        return jobs

    inferred_name = dataset_name if dataset_name else input_root.name
    return [(inferred_name, input_root)]


def resolve_output_root(result_dir, output_dir, dataset_name, opt_path, weights_path, multi_dataset):
    if output_dir:
        if multi_dataset:
            return Path(output_dir) / dataset_name
        return Path(output_dir)

    config = Path(opt_path).stem
    checkpoint_name = Path(weights_path).stem
    return Path(result_dir) / dataset_name / config / checkpoint_name


def main():
    args = parse_args()

    use_cuda = torch.cuda.is_available() and args.gpus != '-1'
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        print('export CUDA_VISIBLE_DEVICES=' + args.gpus)
    else:
        print('Running on CPU.')

    model = build_model(args.opt, args.weights, use_cuda)
    jobs = resolve_jobs(args.input_dir, args.subdirs, args.dataset_name)
    multi_dataset = len(jobs) > 1

    for dataset_name, dataset_dir in jobs:
        dataset_dir = Path(dataset_dir)
        if not dataset_dir.exists():
            raise FileNotFoundError(f'Dataset directory not found: {dataset_dir}')

        input_paths = collect_image_paths(dataset_dir)
        if not input_paths:
            raise FileNotFoundError(f'No image files found in: {dataset_dir}')

        output_root = resolve_output_root(
            args.result_dir, args.output_dir, dataset_name, args.opt, args.weights, multi_dataset)
        output_root.mkdir(parents=True, exist_ok=True)

        print(f'Inferencing {dataset_name}: {len(input_paths)} images')
        for img_path in tqdm(input_paths, total=len(input_paths)):
            restored = infer_image(
                model,
                img_path,
                use_cuda,
                args.self_ensemble,
                args.tile_size,
                args.tile_overlap,
                args.min_tile_size)
            rel_path = img_path.relative_to(dataset_dir)
            save_path = output_root / rel_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            utils.save_img(str(save_path), restored)

        print(f'Saved results to: {output_root}')


if __name__ == '__main__':
    main()
