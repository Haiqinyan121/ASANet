import argparse
import copy
import contextlib
import csv
import importlib
import importlib.util
import io
import json
import os
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml


SCRIPT_ROOT = Path(__file__).resolve().parent


def build_builtin_asanet_config() -> Dict[str, Any]:
    return copy.deepcopy({
        "runtime": {
            "device": "auto",
            "cuda_visible_devices": "0",
            "seed": 123,
            "cudnn_benchmark": True,
            "allow_tf32": False,
            "benchmark": {
                "backend": "auto",
                "warmup": 50,
                "runs": 200,
                "memory_runs": 20,
            },
            "input": {
                "batch_size": 1,
                "channels": 3,
                "height": 128,
                "width": 128,
                "dtype": "float32",
            },
        },
        "output": {
            "csv": "benchmark_outputs/asanet_efficiency.csv",
            "markdown": "benchmark_outputs/asanet_efficiency.md",
        },
        "methods": [
            {
                "name": "baseline",
                "repo_root": ".",
                "silence_stdout": True,
                "builder": {
                    "type": "class",
                    "file": "basicsr/models/archs/ASANet_arch.py",
                    "package": "basicsr.models.archs",
                    "class_name": "ASANet",
                    "kwargs": {
                        "in_channels": 3,
                        "out_channels": 3,
                        "n_feat": 40,
                        "stage": 1,
                        "num_blocks": [1, 2, 2],
                        "d_state": 16,
                        "illu_use_a": False,
                        "illu_use_b": False,
                        "ffn_use_sem": False,
                    },
                },
                "forward": {"output_mode": "tensor"},
            },
            {
                "name": "estimator_only",
                "repo_root": ".",
                "silence_stdout": True,
                "builder": {
                    "type": "class",
                    "file": "basicsr/models/archs/ASANet_arch.py",
                    "package": "basicsr.models.archs",
                    "class_name": "ASANet",
                    "kwargs": {
                        "in_channels": 3,
                        "out_channels": 3,
                        "n_feat": 40,
                        "stage": 1,
                        "num_blocks": [1, 2, 2],
                        "d_state": 16,
                        "illu_use_a": True,
                        "illu_use_b": True,
                        "ffn_use_sem": False,
                    },
                },
                "forward": {"output_mode": "tensor"},
            },
            {
                "name": "sefn_only",
                "repo_root": ".",
                "silence_stdout": True,
                "builder": {
                    "type": "class",
                    "file": "basicsr/models/archs/ASANet_arch.py",
                    "package": "basicsr.models.archs",
                    "class_name": "ASANet",
                    "kwargs": {
                        "in_channels": 3,
                        "out_channels": 3,
                        "n_feat": 40,
                        "stage": 1,
                        "num_blocks": [1, 2, 2],
                        "d_state": 16,
                        "illu_use_a": False,
                        "illu_use_b": False,
                        "ffn_use_sem": True,
                    },
                },
                "forward": {"output_mode": "tensor"},
            },
            {
                "name": "full",
                "repo_root": ".",
                "silence_stdout": True,
                "builder": {
                    "type": "class",
                    "file": "basicsr/models/archs/ASANet_arch.py",
                    "package": "basicsr.models.archs",
                    "class_name": "ASANet",
                    "kwargs": {
                        "in_channels": 3,
                        "out_channels": 3,
                        "n_feat": 40,
                        "stage": 1,
                        "num_blocks": [1, 2, 2],
                        "d_state": 16,
                        "illu_use_a": True,
                        "illu_use_b": True,
                        "ffn_use_sem": True,
                    },
                },
                "forward": {"output_mode": "tensor"},
            },
            {
                "name": "cwnet",
                "repo_root": "D:/code/CWNet-Causal-Wavelet-Network",
                "silence_stdout": True,
                "builder": {
                    "type": "class",
                    "file": "models/archs/CWNet.py",
                    "package": "models.archs",
                    "class_name": "CWNet",
                    "kwargs": {
                        "nc": 16,
                        "n_l_blocks": [1, 3, 4, 3, 1],
                        "n_h_blocks": [1, 2, 2, 2, 1],
                    },
                },
                "forward": {"output_mode": "tensor"},
            },
        ],
    })


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def resolve_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sanitize_name(name: str) -> str:
    cleaned = []
    for char in name:
        if char.isalnum() or char in ("-", "_"):
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "method"


@contextlib.contextmanager
def pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextlib.contextmanager
def prepend_sys_path(path: Path):
    inserted = False
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        inserted = True
    try:
        yield
    finally:
        if inserted and sys.path and sys.path[0] == path_str:
            sys.path.pop(0)


def maybe_redirect_stdout(enabled: bool):
    if enabled:
        return contextlib.redirect_stdout(io.StringIO())
    return contextlib.nullcontext()


def infer_module_name_from_file(abs_file: Path,
                                repo_root: Path,
                                explicit_package: str = "") -> Tuple[str, List[Path]]:
    if explicit_package:
        package_parts = [part for part in explicit_package.split(".") if part]
        return ".".join(package_parts + [abs_file.stem]), list(abs_file.parents)[:-1]

    try:
        relative_no_suffix = abs_file.relative_to(repo_root).with_suffix("")
        module_parts = list(relative_no_suffix.parts)
        package_dirs = [repo_root / Path(*module_parts[:idx + 1]) for idx in range(len(module_parts) - 1)]
        return ".".join(module_parts), package_dirs
    except ValueError:
        unique_name = f"_efficiency_benchmark_{sanitize_name(abs_file.stem)}_{abs(hash(str(abs_file))) & 0xffffffff:x}"
        return unique_name, []


def register_package_stubs(full_module_name: str, abs_file: Path, package_dirs: List[Path]) -> None:
    package_parts = full_module_name.split(".")[:-1]
    if not package_parts:
        return

    if package_dirs and len(package_dirs) == len(package_parts):
        package_paths = package_dirs
    else:
        reversed_paths = list(reversed(list(abs_file.parents)[:len(package_parts)]))
        package_paths = reversed_paths

    for idx, package_name in enumerate(".".join(package_parts[:end]) for end in range(1, len(package_parts) + 1)):
        if package_name in sys.modules:
            continue
        package_module = types.ModuleType(package_name)
        package_module.__package__ = package_name
        package_module.__path__ = [str(package_paths[idx])]
        sys.modules[package_name] = package_module


def resolve_device(runtime_cfg: Dict[str, Any]) -> torch.device:
    device_name = str(runtime_cfg.get("device", "auto")).strip()
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def setup_runtime(runtime_cfg: Dict[str, Any], device: torch.device) -> None:
    seed = runtime_cfg.get("seed")
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    num_threads = runtime_cfg.get("num_threads")
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(runtime_cfg.get("cudnn_benchmark", True))
        allow_tf32 = bool(runtime_cfg.get("allow_tf32", False))
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = allow_tf32


def parse_input_shape(input_cfg: Dict[str, Any]) -> Tuple[int, int, int, int]:
    if "shape" in input_cfg:
        shape = input_cfg["shape"]
        if not isinstance(shape, list) or len(shape) != 4:
            raise ValueError("input.shape must be a list of 4 integers, e.g. [1, 3, 256, 256]")
        return tuple(int(item) for item in shape)  # type: ignore[return-value]

    batch_size = int(input_cfg.get("batch_size", 1))
    channels = int(input_cfg.get("channels", 3))
    height = int(input_cfg.get("height", 256))
    width = int(input_cfg.get("width", 256))
    return batch_size, channels, height, width


def build_dummy_input(input_cfg: Dict[str, Any], device: torch.device) -> torch.Tensor:
    batch_size, channels, height, width = parse_input_shape(input_cfg)
    dtype_name = str(input_cfg.get("dtype", "float32")).lower()
    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported input dtype: {dtype_name}")

    return torch.randn(batch_size, channels, height, width, device=device, dtype=dtype_map[dtype_name])


def apply_attr_path(obj: Any, attr_path: str) -> Any:
    current = obj
    for attr in attr_path.split("."):
        current = getattr(current, attr)
    return current


def load_module_from_spec(module_name: Optional[str],
                          file_path: Optional[str],
                          repo_root: Path,
                          silence_stdout: bool,
                          package_name: str = ""):
    with maybe_redirect_stdout(silence_stdout):
        if module_name:
            return importlib.import_module(module_name)

        if not file_path:
            raise ValueError("Either builder.module or builder.file must be provided.")

        abs_file = resolve_path(file_path, repo_root)
        full_module_name, package_dirs = infer_module_name_from_file(abs_file, repo_root, package_name)
        register_package_stubs(full_module_name, abs_file, package_dirs)
        spec = importlib.util.spec_from_file_location(full_module_name, abs_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load module from file: {abs_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_module_name] = module
        spec.loader.exec_module(module)
        return module


def build_model(method_cfg: Dict[str, Any], repo_root: Path):
    builder_cfg = method_cfg.get("builder", {})
    if not isinstance(builder_cfg, dict):
        raise ValueError(f"builder must be a mapping for method: {method_cfg.get('name', '<unnamed>')}")

    builder_type = str(builder_cfg.get("type", "class")).strip().lower()
    silence_stdout = bool(method_cfg.get("silence_stdout", True))
    model_attr = str(builder_cfg.get("model_attr", "")).strip()
    package_name = str(builder_cfg.get("package", "")).strip()

    if builder_type == "class":
        module = load_module_from_spec(
            module_name=builder_cfg.get("module"),
            file_path=builder_cfg.get("file"),
            repo_root=repo_root,
            silence_stdout=silence_stdout,
            package_name=package_name)
        class_name = builder_cfg.get("class_name")
        if not class_name:
            raise ValueError("builder.class_name is required when builder.type == 'class'")
        cls = getattr(module, class_name)
        kwargs = builder_cfg.get("kwargs", {}) or {}
        with maybe_redirect_stdout(silence_stdout):
            model = cls(**kwargs)
        return apply_attr_path(model, model_attr) if model_attr else model

    if builder_type == "function":
        module = load_module_from_spec(
            module_name=builder_cfg.get("module"),
            file_path=builder_cfg.get("file"),
            repo_root=repo_root,
            silence_stdout=silence_stdout,
            package_name=package_name)
        function_name = builder_cfg.get("function_name")
        if not function_name:
            raise ValueError("builder.function_name is required when builder.type == 'function'")
        fn = getattr(module, function_name)
        kwargs = builder_cfg.get("kwargs", {}) or {}
        with maybe_redirect_stdout(silence_stdout):
            model = fn(**kwargs)
        return apply_attr_path(model, model_attr) if model_attr else model

    if builder_type == "basicsr_option":
        opt_path_raw = builder_cfg.get("opt")
        if not opt_path_raw:
            raise ValueError("builder.opt is required when builder.type == 'basicsr_option'")

        options_module = load_module_from_spec(
            module_name="basicsr.utils.options",
            file_path=None,
            repo_root=repo_root,
            silence_stdout=silence_stdout)
        models_module = load_module_from_spec(
            module_name="basicsr.models",
            file_path=None,
            repo_root=repo_root,
            silence_stdout=silence_stdout)

        parse_opt = getattr(options_module, "parse")
        create_model = getattr(models_module, "create_model")
        opt_path = resolve_path(str(opt_path_raw), repo_root)

        with maybe_redirect_stdout(silence_stdout):
            opt = parse_opt(str(opt_path), is_train=False)
            opt["dist"] = False
            built = create_model(opt)

        default_attr = builder_cfg.get("model_attr", "net_g")
        return apply_attr_path(built, default_attr) if default_attr else built

    raise ValueError(f"Unsupported builder.type: {builder_type}")


def extract_state_dict(checkpoint: Any, checkpoint_cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    preferred_key = str(checkpoint_cfg.get("key", "")).strip()
    candidate_keys: List[str] = []
    if preferred_key:
        candidate_keys.append(preferred_key)
    candidate_keys.extend(["params", "params_ema", "state_dict", "model", "net", "generator"])

    if isinstance(checkpoint, dict):
        for key in candidate_keys:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value

        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint

    raise ValueError(
        "Unable to locate a state_dict in checkpoint. Specify checkpoint.key if the weights are nested.")


def strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if prefix and all(key.startswith(prefix) for key in state_dict.keys()):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def add_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not prefix:
        return state_dict
    return {prefix + key: value for key, value in state_dict.items()}


def candidate_state_dicts(state_dict: Dict[str, torch.Tensor], checkpoint_cfg: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
    strip_prefixes = checkpoint_cfg.get("strip_prefixes", []) or []
    if isinstance(strip_prefixes, str):
        strip_prefixes = [strip_prefixes]

    add_prefix_value = str(checkpoint_cfg.get("add_prefix", "")).strip()

    candidates: List[Dict[str, torch.Tensor]] = [state_dict]
    for prefix in strip_prefixes:
        candidates.append(strip_prefix_if_present(state_dict, str(prefix)))
    candidates.append(strip_prefix_if_present(state_dict, "module."))
    if add_prefix_value:
        candidates.append(add_prefix(state_dict, add_prefix_value))
    candidates.append(add_prefix(state_dict, "module."))

    unique: List[Dict[str, torch.Tensor]] = []
    seen_keys = set()
    for item in candidates:
        key_signature = tuple(item.keys())
        if key_signature not in seen_keys:
            seen_keys.add(key_signature)
            unique.append(item)
    return unique


def load_checkpoint_if_needed(model: torch.nn.Module,
                              method_cfg: Dict[str, Any],
                              repo_root: Path,
                              map_location: str = "cpu") -> None:
    checkpoint_cfg = method_cfg.get("checkpoint")
    if not checkpoint_cfg:
        return

    if not isinstance(checkpoint_cfg, dict):
        raise ValueError("checkpoint must be a mapping when provided.")

    checkpoint_path = resolve_path(str(checkpoint_cfg["path"]), repo_root)
    checkpoint = torch.load(str(checkpoint_path), map_location=map_location)
    state_dict = extract_state_dict(checkpoint, checkpoint_cfg)
    strict = bool(checkpoint_cfg.get("strict", True))

    errors = []
    for candidate in candidate_state_dicts(state_dict, checkpoint_cfg):
        try:
            model.load_state_dict(candidate, strict=strict)
            return
        except Exception as exc:
            errors.append(str(exc))

    raise RuntimeError(
        f"Failed to load checkpoint for method '{method_cfg.get('name', '<unnamed>')}'.\n"
        + "\n".join(errors)
    )


def extract_model_output(output: Any, forward_cfg: Dict[str, Any]) -> torch.Tensor:
    output_mode = str(forward_cfg.get("output_mode", "tensor")).strip().lower()

    if output_mode == "tensor":
        if not torch.is_tensor(output):
            raise TypeError(
                "Model output is not a tensor. Set forward.output_mode to first/last/dict "
                "or provide a wrapper factory for this method.")
        return output

    if output_mode == "first":
        return output[0]

    if output_mode == "last":
        return output[-1]

    if output_mode == "dict":
        output_key = str(forward_cfg.get("key", "")).strip()
        if not output_key:
            raise ValueError("forward.key is required when forward.output_mode == 'dict'")
        return output[output_key]

    raise ValueError(f"Unsupported forward.output_mode: {output_mode}")


class ForwardAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, forward_cfg: Dict[str, Any]):
        super().__init__()
        self.model = model
        self.forward_cfg = forward_cfg

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.model(inputs)
        return extract_model_output(outputs, self.forward_cfg)


def count_with_fvcore(model: torch.nn.Module, inputs: torch.Tensor) -> Tuple[float, float, str]:
    from fvcore.nn import FlopCountAnalysis

    with torch.no_grad():
        _ = model(inputs)

    analysis = FlopCountAnalysis(model, inputs)
    if hasattr(analysis, "unsupported_ops_warnings"):
        analysis.unsupported_ops_warnings(False)
    if hasattr(analysis, "uncalled_modules_warnings"):
        analysis.uncalled_modules_warnings(False)
    if hasattr(analysis, "tracer_warnings"):
        analysis.tracer_warnings("none")

    macs = float(analysis.total())
    params = float(sum(p.numel() for p in model.parameters()))
    return params, macs, "fvcore"


def count_with_thop(model: torch.nn.Module, inputs: torch.Tensor) -> Tuple[float, float, str]:
    from thop import profile

    with torch.no_grad():
        macs, params = profile(model, inputs=(inputs,), verbose=False)
    return float(params), float(macs), "thop"


def count_complexity(model: torch.nn.Module,
                     inputs: torch.Tensor,
                     backend: str) -> Tuple[float, float, str]:
    if backend == "fvcore":
        return count_with_fvcore(model, inputs)
    if backend == "thop":
        return count_with_thop(model, inputs)

    errors = []
    for candidate in ("fvcore", "thop"):
        try:
            if candidate == "fvcore":
                return count_with_fvcore(model, inputs)
            return count_with_thop(model, inputs)
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    raise RuntimeError(
        "Failed to count complexity with both fvcore and thop.\n"
        + "\n".join(errors)
    )


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_latency(model: torch.nn.Module,
                      inputs: torch.Tensor,
                      device: torch.device,
                      warmup: int,
                      runs: int) -> Tuple[float, float, float]:
    timings_ms: List[float] = []
    batch_size = max(1, int(inputs.shape[0]))

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(inputs)
        synchronize_if_needed(device)

        for _ in range(runs):
            synchronize_if_needed(device)
            start = time.perf_counter()
            _ = model(inputs)
            synchronize_if_needed(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0 / batch_size
            timings_ms.append(elapsed_ms)

    if not timings_ms:
        raise ValueError("benchmark.runs must be >= 1")

    mean_ms = statistics.mean(timings_ms)
    median_ms = statistics.median(timings_ms)
    std_ms = statistics.pstdev(timings_ms) if len(timings_ms) > 1 else 0.0
    return mean_ms, median_ms, std_ms


def benchmark_peak_memory(model: torch.nn.Module,
                          inputs: torch.Tensor,
                          device: torch.device,
                          warmup: int,
                          runs: int) -> Optional[float]:
    if device.type != "cuda":
        return None

    with torch.inference_mode():
        for _ in range(max(1, warmup)):
            _ = model(inputs)
        synchronize_if_needed(device)

        peaks_gb: List[float] = []
        for _ in range(max(1, runs)):
            base_memory = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            _ = model(inputs)
            synchronize_if_needed(device)
            peak_memory = max(base_memory, torch.cuda.max_memory_allocated(device))
            peaks_gb.append(peak_memory / (1024 ** 3))

    return max(peaks_gb) if peaks_gb else None


def build_result_row(method_name: str,
                     device: torch.device,
                     input_cfg: Dict[str, Any],
                     params: float,
                     macs: float,
                     backend: str,
                     latency_mean_ms: float,
                     latency_median_ms: float,
                     latency_std_ms: float,
                     peak_memory_gb: Optional[float]) -> Dict[str, Any]:
    batch_size, channels, height, width = parse_input_shape(input_cfg)
    row: Dict[str, Any] = {
        "method": method_name,
        "params": int(params),
        "params_m": params / 1e6,
        "macs": macs,
        "gmacs": macs / 1e9,
        "backend": backend,
        "latency_ms_per_image": latency_mean_ms,
        "latency_median_ms_per_image": latency_median_ms,
        "latency_std_ms_per_image": latency_std_ms,
        "peak_gpu_memory_gb": peak_memory_gb,
        "device": str(device),
        "batch_size": batch_size,
        "channels": channels,
        "height": height,
        "width": width,
    }
    return row


def format_number(value: Optional[float], digits: int) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def print_paper_table(rows: List[Dict[str, Any]]) -> None:
    headers = [
        "Method",
        "Params (M)",
        "GMACs",
        "Latency (ms/image)",
        "Peak GPU Memory (GB)",
    ]
    table = [headers]
    for row in rows:
        table.append([
            str(row["method"]),
            format_number(row["params_m"], 4),
            format_number(row["gmacs"], 4),
            format_number(row["latency_ms_per_image"], 3),
            format_number(row["peak_gpu_memory_gb"], 3),
        ])

    widths = [max(len(str(line[idx])) for line in table) for idx in range(len(headers))]
    for idx, line in enumerate(table):
        rendered = "  ".join(str(cell).ljust(widths[col]) for col, cell in enumerate(line))
        print(rendered)
        if idx == 0:
            print("  ".join("-" * width for width in widths))


def save_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    ensure_parent(output_path)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "params",
                "params_m",
                "macs",
                "gmacs",
                "backend",
                "latency_ms_per_image",
                "latency_median_ms_per_image",
                "latency_std_ms_per_image",
                "peak_gpu_memory_gb",
                "device",
                "batch_size",
                "channels",
                "height",
                "width",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_markdown(rows: List[Dict[str, Any]], output_path: Path) -> None:
    ensure_parent(output_path)
    lines = [
        "| Method | Params (M) | GMACs | Latency (ms/image) | Peak GPU Memory (GB) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {params_m} | {gmacs} | {latency} | {memory} |".format(
                method=row["method"],
                params_m=format_number(row["params_m"], 4),
                gmacs=format_number(row["gmacs"], 4),
                latency=format_number(row["latency_ms_per_image"], 3),
                memory=format_number(row["peak_gpu_memory_gb"], 3),
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_method_by_name(config: Dict[str, Any], method_name: str) -> Dict[str, Any]:
    methods = config.get("methods", [])
    if not isinstance(methods, list) or not methods:
        raise ValueError("Config must contain a non-empty methods list.")

    for method_cfg in methods:
        if method_cfg.get("name") == method_name:
            return method_cfg
    raise ValueError(f"Unknown method name: {method_name}")


def list_method_names(config: Dict[str, Any]) -> List[str]:
    methods = config.get("methods", [])
    if not isinstance(methods, list):
        raise ValueError("Config must contain a methods list.")
    return [str(method_cfg.get("name", "")).strip() for method_cfg in methods if str(method_cfg.get("name", "")).strip()]


def apply_method_filter(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    requested_methods = args.methods
    if not requested_methods and not args.config:
        requested_methods = ["baseline", "full"]

    if not requested_methods:
        return config

    wanted = [item.strip() for item in requested_methods if str(item).strip()]
    available = list_method_names(config)
    unknown = [item for item in wanted if item not in available]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Available: {available}")

    config["methods"] = [method_cfg for method_cfg in config.get("methods", []) if method_cfg.get("name") in wanted]
    return config


def apply_runtime_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    runtime_cfg = config.setdefault("runtime", {})
    if not isinstance(runtime_cfg, dict):
        raise ValueError("runtime must be a mapping in config.")

    input_cfg = runtime_cfg.setdefault("input", {})
    if not isinstance(input_cfg, dict):
        raise ValueError("runtime.input must be a mapping in config.")

    if int(args.batch_size or 0) > 0:
        input_cfg["batch_size"] = int(args.batch_size)
    if int(args.height or 0) > 0:
        input_cfg["height"] = int(args.height)
    if int(args.width or 0) > 0:
        input_cfg["width"] = int(args.width)
    return config


def load_base_config_from_args(args: argparse.Namespace) -> Tuple[Dict[str, Any], Path]:
    if args.config:
        config_path = Path(args.config).resolve()
        config = load_yaml(config_path)
        config_dir = config_path.parent.resolve()
    else:
        if args.preset != "asanet":
            raise ValueError(f"Unsupported preset: {args.preset}")
        config = build_builtin_asanet_config()
        config_dir = SCRIPT_ROOT

    return config, config_dir


def load_config_from_args(args: argparse.Namespace) -> Tuple[Dict[str, Any], Path]:
    config, config_dir = load_base_config_from_args(args)
    config = apply_method_filter(config, args)
    config = apply_runtime_overrides(config, args)
    return config, config_dir


def print_available_methods(args: argparse.Namespace) -> None:
    config, _ = load_base_config_from_args(args)
    for name in list_method_names(config):
        print(name)


def benchmark_single_method(config_path: Path, method_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    if args.config:
        config = load_yaml(config_path)
        config_dir = config_path.parent.resolve()
    else:
        config = build_builtin_asanet_config()
        config_dir = SCRIPT_ROOT
    config = apply_runtime_overrides(config, args)
    runtime_cfg = config.get("runtime", {}) or {}
    benchmark_cfg = runtime_cfg.get("benchmark", {}) or {}
    input_cfg = runtime_cfg.get("input", {}) or {}
    method_cfg = get_method_by_name(config, method_name)

    repo_root = resolve_path(str(method_cfg.get("repo_root", ".")), config_dir)
    device = resolve_device(runtime_cfg)
    setup_runtime(runtime_cfg, device)

    with pushd(repo_root), prepend_sys_path(repo_root):
        model = build_model(method_cfg, repo_root)
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        load_checkpoint_if_needed(model, method_cfg, repo_root, map_location="cpu")

        forward_cfg = method_cfg.get("forward", {}) or {}
        wrapped_model = ForwardAdapter(model, forward_cfg).to(device)
        wrapped_model.eval()

        inputs = build_dummy_input(input_cfg, device)
        backend = str(benchmark_cfg.get("backend", "auto")).strip().lower()
        warmup = int(benchmark_cfg.get("warmup", 50))
        runs = int(benchmark_cfg.get("runs", 200))
        memory_runs = int(benchmark_cfg.get("memory_runs", 20))

        params, macs, backend_name = count_complexity(wrapped_model, inputs, backend)
        if device.type == "cuda":
            torch.cuda.empty_cache()

        latency_mean_ms, latency_median_ms, latency_std_ms = benchmark_latency(
            wrapped_model, inputs, device, warmup, runs)
        peak_memory_gb = benchmark_peak_memory(
            wrapped_model, inputs, device, warmup=max(1, warmup // 2), runs=memory_runs)

    return build_result_row(
        method_name=method_name,
        device=device,
        input_cfg=input_cfg,
        params=params,
        macs=macs,
        backend=backend_name,
        latency_mean_ms=latency_mean_ms,
        latency_median_ms=latency_median_ms,
        latency_std_ms=latency_std_ms,
        peak_memory_gb=peak_memory_gb,
    )


def run_worker(args: argparse.Namespace) -> None:
    if not args.method_name:
        raise ValueError("--method-name is required in worker mode.")
    if not args.worker_output:
        raise ValueError("--worker-output is required in worker mode.")

    config_path = Path(args.config).resolve() if args.config else SCRIPT_ROOT
    row = benchmark_single_method(config_path, args.method_name, args)
    output_path = Path(args.worker_output).resolve()
    ensure_parent(output_path)
    output_path.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_output_path(raw_path: Optional[str], config_dir: Path, fallback_name: str) -> Path:
    if raw_path:
        return resolve_path(raw_path, config_dir)
    return (config_dir / "benchmark_outputs" / fallback_name).resolve()


def build_worker_command(script_path: Path,
                         python_executable: str,
                         config_path: Path,
                         method_name: str,
                         worker_output: Path,
                         args: argparse.Namespace) -> List[str]:
    command = [
        python_executable,
        str(script_path),
        "--worker",
        "--method-name",
        method_name,
        "--worker-output",
        str(worker_output),
    ]
    if args.config:
        command.extend(["--config", str(config_path)])
    else:
        command.extend(["--preset", args.preset])
    if int(args.height or 0) > 0:
        command.extend(["--height", str(args.height)])
    if int(args.width or 0) > 0:
        command.extend(["--width", str(args.width)])
    if int(args.batch_size or 0) > 0:
        command.extend(["--batch-size", str(args.batch_size)])
    return command


def run_controller(args: argparse.Namespace) -> None:
    config, config_dir = load_config_from_args(args)
    config_path = Path(args.config).resolve() if args.config else SCRIPT_ROOT
    methods = config.get("methods", [])
    if not isinstance(methods, list) or not methods:
        raise ValueError("Config must contain a non-empty methods list.")

    output_cfg = config.get("output", {}) or {}
    csv_path = resolve_output_path(args.output_csv or output_cfg.get("csv"), config_dir, "efficiency_results.csv")
    markdown_path = resolve_output_path(
        args.output_markdown or output_cfg.get("markdown"),
        config_dir,
        "efficiency_results.md")

    worker_dir = (csv_path.parent / ".worker_cache").resolve()
    worker_dir.mkdir(parents=True, exist_ok=True)

    runtime_cfg = config.get("runtime", {}) or {}
    env = os.environ.copy()
    cuda_visible_devices = runtime_cfg.get("cuda_visible_devices")
    if cuda_visible_devices not in (None, ""):
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    rows: List[Dict[str, Any]] = []
    script_path = Path(__file__).resolve()
    total = len(methods)
    for index, method_cfg in enumerate(methods, start=1):
        method_name = str(method_cfg.get("name", "")).strip()
        if not method_name:
            raise ValueError("Each method entry must include a non-empty name.")

        python_executable = str(
            resolve_path(str(method_cfg.get("python_executable", sys.executable)), config_dir)
        ) if method_cfg.get("python_executable") else sys.executable
        worker_output = worker_dir / f"{index:02d}_{sanitize_name(method_name)}.json"
        command = build_worker_command(
            script_path,
            python_executable,
            config_path,
            method_name,
            worker_output,
            args)
        print(f"[{index}/{total}] Benchmarking {method_name}")
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(config_dir),
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Benchmark failed for method '{method_name}'.\n"
                f"STDOUT:\n{completed.stdout}\n"
                f"STDERR:\n{completed.stderr}"
            )

        row = json.loads(worker_output.read_text(encoding="utf-8"))
        rows.append(row)
        try:
            worker_output.unlink()
        except OSError:
            pass

    print("")
    print_paper_table(rows)
    save_csv(rows, csv_path)
    save_markdown(rows, markdown_path)

    try:
        worker_dir.rmdir()
    except OSError:
        pass

    print("")
    print(f"Saved CSV to: {csv_path}")
    print(f"Saved Markdown table to: {markdown_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified efficiency benchmark for image enhancement/restoration models. "
            "Params and GMACs follow the same fvcore/thop counting logic as the existing "
            "count_complexity.py script."
        )
    )
    parser.add_argument("--config", default="", help="Optional YAML benchmark config. Omit it to use the built-in ASANet preset.")
    parser.add_argument("--preset", default="asanet", help="Built-in preset name used when --config is omitted.")
    parser.add_argument("--methods", "--variants", nargs="+", default=None, help="Methods to benchmark, e.g. baseline full.")
    parser.add_argument("--list-methods", action="store_true", help="Print available method names and exit.")
    parser.add_argument("--output-csv", default="", help="Optional CSV output path.")
    parser.add_argument("--output-markdown", default="", help="Optional Markdown table output path.")
    parser.add_argument("--height", type=int, default=0, help="Optional override for input height.")
    parser.add_argument("--width", type=int, default=0, help="Optional override for input width.")
    parser.add_argument("--batch-size", type=int, default=0, help="Optional override for input batch size.")
    parser.add_argument("--worker", action="store_true", help="Internal flag used for per-method subprocess benchmarking.")
    parser.add_argument("--method-name", default="", help="Method name used in worker mode.")
    parser.add_argument("--worker-output", default="", help="JSON output path used in worker mode.")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.worker:
        run_worker(args)
        return

    if args.list_methods:
        print_available_methods(args)
        return

    run_controller(args)


if __name__ == "__main__":
    main()
