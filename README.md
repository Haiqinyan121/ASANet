# ASANet

Official implementation and reproducibility package for **ASANet: Adaptive-Scale Illumination Assembly for Low-Light Image Enhancement**.

Zihan Yin, Yun Wei, Lecheng Lin, Bo Pang — University of Shanghai for Science and Technology; Henan Institute of Technology.

ASANet keeps a state-space illumination-guided restoration backbone (RetinexMamba) and redesigns both the illumination-representation constructor and the site where that representation is spent. The **Explicit Illumination Encoder (EIE)** contains an **Adaptive Illumination Prior Module (AIPM)** that mixes pooled candidates of several extents under a per-pixel softmax, so the effective smoothing extent is selected by content rather than fixed by architecture, and a **Boundary-Aware Illumination Refinement (BAIR)** module that adds local receptive fields together with a separable directional gate. The resulting representation is then re-injected inside the gated nonlinear update by an **Illumination-Guided Feed-Forward Network (IGFFN)**.

---

## 1. Test

### 1.1 Test PSNR, SSIM and RMSE

Trained weights for the reported configurations will be released in this repository once the paper is accepted. Until then, train a checkpoint with [§3](#3-train) or use your own, place it under `pretrained_weights/`, and generate the outputs on the Test split:

```bash
python Enhancement/test_from_dataset.py \
  --opt configs/evaluation/lolv1_test.yml \
  --weights pretrained_weights/LOLv1.pth \
  --dataset LOLv1 \
  --result_dir results/generated \
  --gpus 0
```

`configs/evaluation/` provides one configuration per dataset: `lolv1_test.yml`, `lolv2_real_test.yml`, `lolv2_synthetic_test.yml`, and `lolv2_synthetic_eie_test.yml` (the last for the EIE-only transplant row). `test_from_dataset.py` reads their `datasets.val` entry after forcing `phase = 'test'`; the `datasets.train` entry they also carry is not touched during evaluation.

Score the saved images with the reported paired-image conventions:

```bash
python Enhancement/evaluate_paired_folders.py \
  --pred_dir results/generated/LOLv1 \
  --gt_dir data/LOLv1/Test/target
```

Do **not** add `--GT_mean`: the paper reports results without GT-mean adjustment.

In every released training configuration the Test subset is also assigned to BasicSR's `datasets.val` entry, so periodic evaluation runs on the Test set and the checkpoint with the highest Test PSNR is retained. The full-protocol configurations use a 250-iteration evaluation interval. Under the reduced protocol, LOLv1 uses a 1,000-iteration interval, while LOLv2-real and LOLv2-synthetic use a 250-iteration interval. No separate validation split is used. Within each dataset and regime, the compared configurations use the identical evaluation frequency and checkpoint-selection rule.

## 2. Model parameters and FLOPs evaluation

```bash
python benchmark_efficiency.py --config configs/efficiency.yml --output-csv results/efficiency.csv
python benchmark_efficiency.py --list-methods
```

`configs/efficiency.yml` enumerates the six ablation variants; `--list-methods` prints the available names.

## 3. Train

```bash
python basicsr/train.py --opt configs/full_300k/asanet_lolv1.yml
```

`configs/full_300k/` holds the full-protocol (300,000-iteration) configurations for all three LOL datasets, for ASANet (`asanet_*.yml`) and for the matched-protocol RetinexMamba baseline (`retinexmamba_controlled_*.yml`); `configs/ablation_compressed/` holds the reduced-protocol configurations for all six ablation variants. The `retinexmamba_controlled_*` configurations set all three component switches to `false`, which leaves the backbone with its original estimator and feed-forward path. Like every other configuration they name the class `ASANet`, so switches-off `ASANet` reproduces the RetinexMamba backbone.

## Acknowledgments

This implementation is built on RetinexMamba and BasicSR. Their original licenses and citations are retained; see `LICENSE`. Please retain them when reusing the corresponding code.

---

## Reference

### Component switches

The unified architecture uses three explicit switches:

| Variant | `illu_use_a` | `illu_use_b` | `ffn_use_sem` |
|---|---:|---:|---:|
| Controlled RetinexMamba | false | false | false |
| AIPM-only | true | false | false |
| BAIR-only | false | true | false |
| EIE | true | true | false |
| IGFFN-only | false | false | true |
| ASANet | true | true | true |

### Reported results

| File | Content | Manuscript |
|---|---|---|
| `results/main_controlled_comparison.csv` | ASANet vs the matched-protocol RetinexMamba baseline, full protocol | Table 2, controlled block |
| `results/ablation_results.csv` | Six-variant component ablation with parameters and GMACs | Table 3, plus Table 4 for the complexity columns |
| `results/two_path_interface_results.csv` | 24-run Two-Path interface matrix | Table 7 |
| `results/illumination_metrics.csv` | Map-level diagnostics (TV, Laplacian energy, edge-gradient mean, dark-region variance) over 343 images | Table 9 |
| `results/exdark_detection.csv` | ExDark detection transfer, mAP@50 and mAP@50:95 | Table 11 |

Datasets are not redistributed here. The protocols behind every reported number are specified in the manuscript and implemented by the released configurations in `configs/`.
