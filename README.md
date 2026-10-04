# GenConViT-TFKD

A lightweight **GenConViT-based** deepfake video detector that combines a generative (AE/VAE) branch with
**T**emporal modelling, **F**requency analysis, spatial-**A**rtifact analysis and **K**nowledge **D**istillation (TFKD).
Everything (data handling, model, training, evaluation) lives in a single script, `GenConViT-TFKD.py`, designed to run as a
one-cell Kaggle notebook.

> **Note.** The backbone is a *lightweight GenConViT-based design*, **not** an exact reproduction of the original
> ~695M-parameter GenConViT. Differences: the AE/VAE branches are trained jointly end-to-end (with an optional reconstruction-only
> pre-training stage) rather than independently pretrained, and one ConvNeXt/Swin weight set is shared across inputs.

## Architecture

| Component | Description |
|---|---|
| **GenConViT-based backbone** | AE and/or VAE generative branches (`GENCONVIT_VARIANT`: `ed` / `vae` / `both`) + pretrained **ConvNeXt-Tiny** and **Swin-Tiny** feature extractors. With the default `GEN_FEED="residual_cnn"`, the reconstruction residual (`x - rec`) is encoded by a small CNN; with `"recon"`, reconstructions are fed through the backbones. |
| **Spatial-artifact branch** (`A`) | Fixed high-pass residuals (Laplacian, SRM-style KV, 2nd-order) on RGB, then a small CNN with attention pooling. Targets blending boundaries and local texture inconsistencies. |
| **Frequency branch** (`F`) | Global log-magnitude FFT of Hann-windowed luminance (`FREQ_TYPE="fft"`), or 8x8 block DCT (`"dct"`), followed by a CNN encoder. |
| **Gated fusion** | Per-channel softmax gate over the active streams (spatial / artifact / frequency). |
| **Temporal branch** (`T`) | Transformer encoder (default 2 layers, 4 heads) over per-frame embeddings with a CLS token. Frame-level and video-level heads. |
| **Knowledge distillation** (`KD`) | Frozen **dual-path ViT-B/16 teacher** (RGB + Laplacian streams). Frame- and video-level KD with confidence / correct-only / entropy weighting. |
| **Other** | EMA weights, deep supervision on per-branch heads, adjacent-frame temporal-consistency loss, augmentation curriculum (blur, JPEG, downscale, colour, noise). |

Each run is identified by an ablation tag `T{0/1}F{0/1}A{0/1}KD{0/1}` (for example `T1F1A1KD0`).
Toggle components with `USE_TEMPORAL`, `USE_FREQUENCY`, `USE_SPATIAL_ARTIFACT` and `USE_KD`.

## Datasets

Frames are expected to be **pre-extracted face crops**, organised in folders (see `FACES_ROOT` in the script):

```
<FACES_ROOT>/
├── ffpp_faces/   # FaceForensics++: one top-level folder per manipulation + original/real
└── cdf_faces/    # Celeb-DF
```

| Dataset | Used for | Config key |
|---|---|---|
| FaceForensics++ | training / validation / in-domain test (official 720/140/140 split) | `FFPP_PATH` |
| Celeb-DF | cross-dataset evaluation | `CELEBDF_PATH` |
| DFDC (faces) | cross-dataset evaluation | `DFDC_PATH` |
| Diffusion images | optional cross-dataset (image-level) evaluation | `DIFFUSION_PATH` (`""` to skip) |

- The official FF++ `train/val/test.json` splits are downloaded from the FaceForensics repository (Internet required), or loaded from `FFPP_SPLIT_DIR`.
  The script stops if the real-video counts are not exactly 720/140/140, and runs identity- and video-level leak checks.
- Folders whose names contain `real`, `original`, `pristine`, `youtube`, `genuine` or `authentic` are labelled real (0); everything else is fake (1).
- Frame-to-video grouping is configurable (`*_GROUPING_MODE`: `auto` / `video_dirs` / `filename_prefix` / `single_image`).

None of the datasets are included in this repository. Please obtain them from their official sources and respect their licences.

## Installation

```bash
git clone https://github.com/sabbir-just/GenConViT.git
cd GenConViT
pip install -r requirements.txt
```

A **CUDA GPU is required** (the script asserts this). Pretrained ConvNeXt/Swin weights are downloaded through `timm`, so Internet access is needed
unless you are only evaluating from a checkpoint.

## Usage

The script has **no command-line arguments**. All settings live in the `CONFIG` dictionary near the top of `GenConViT-TFKD.py`;
edit it and run the file (or paste it into a single Kaggle notebook cell).

```bash
python GenConViT-TFKD.py
```

### Train

```python
CONFIG = {
    "MODE": "train",
    "EXP_NAME": "auto",           # -> genconvit_T1F1A1KD1, etc.
    "USE_SPATIAL_ARTIFACT": True,
    "USE_FREQUENCY": True,
    "USE_TEMPORAL": True,
    "USE_KD": True,               # requires TEACHER_CHECKPOINT
    "TEACHER_CHECKPOINT": "/path/to/dpvit_faceforensics_final.pth",
    ...
}
```

Before training starts, the script runs a series of **pre-run checks** (split validation, grouping checks, shortcut probe, temporal-sampling checks,
reconstruction sanity, forward/backward dry-run, batch-size probe, checkpoint round-trip and resume self-test). Any failure aborts the run.

**Resuming:** training is deterministic and resumable, including mid-epoch. With `AUTO_RESUME=True` the script picks up
`<OUTPUT_DIR>/<EXP_NAME>/checkpoints/last.pth`. You can also set `RESUME_CHECKPOINT` to a `session_end_epochXXX.pth`.
Training stops automatically before `MAX_SESSION_MINUTES` (default 645, tuned for Kaggle sessions) and writes a resumable checkpoint.

### Evaluate

```python
CONFIG = {
    "MODE": "evaluate",
    "BEST_CHECKPOINT": "/path/to/checkpoints/best_auc.pth",
    "EVAL_WEIGHTS": "best",       # "best" | "raw" | "ema"
    "VIDEO_AGGREGATION": "mean",  # mean | median | max | topk_mean
    ...
}
```

The architecture is read from the checkpoint's stored config, so ablation switches in `CONFIG` do not need to be re-set for evaluation.
Decision thresholds (Youden's J) are derived on **FF++ validation only** and applied to every test set.
Evaluation covers FF++ test, Celeb-DF, DFDC and (optionally) the diffusion set, at video level (primary) and frame level (supporting).

## Teacher model (for KD)

The default teacher is a dual-path ViT: ViT-B/16 on RGB plus ViT-B/16 on the Laplacian (high-frequency) map, concatenated (1536-d) and passed through an MLP to a single logit.
Its architecture is inferred from the checkpoint's state-dict keys, and loading is strict. **Teacher weights are not included**; set `TEACHER_CHECKPOINT` to your own.
To use a different teacher, edit `build_teacher_model()` and `teacher_forward()`.

## Outputs

All outputs are written to `<OUTPUT_DIR>/<EXP_NAME>/`:

```
checkpoints/   last.pth, best_auc.pth, session_end_epochXXX.pth, ...
configs/       experiment_config.json, ffpp_split.csv
logs/          timestamped run logs
metrics/       training_history.csv, validation_metrics.csv, final_metrics_summary.csv,
               bootstrap_confidence_intervals.csv, confusion_matrix_values.csv,
               fusion_gate_statistics.csv, model_complexity.csv, validation_thresholds.csv, ...
predictions/   frame- and video-level prediction CSVs per dataset
figures/       ROC / PR curves, confusion matrices, calibration, training curves (PNG + PDF)
diagnostics/   session_diagnostic.csv
```

Reported metrics include Acc, balanced Acc, Precision, Recall, Specificity, F1, MCC, ROC-AUC, PR-AUC, EER, NLL, Brier score and ECE,
with 95% bootstrap confidence intervals (videos resampled).

## Repository structure

```
GenConViT/
├── GenConViT-TFKD.py   # full pipeline: data, model, train, evaluate
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Notes and limitations

- Paths in `CONFIG` and `FACES_ROOT` currently point to Kaggle input locations; change them to run elsewhere.
- Setting `ALLOW_DEBUG_SPLIT` or `ALLOW_DEBUG_FALLBACK` to `True` is for debugging only and must not be used for reported results.
- This is research code under active development; expect to adapt it to your own data layout.

## License

Released under the [MIT License](LICENSE).

## Citation

If you use this code, please cite the associated paper (details to be added on publication).
