# Rethinking KWS Depth

Code and saved results for block deletion, whole-block reuse, low-cost recovery and block-output moment regularization in **BC-ResNet-8, TC-ResNet-14, KWT-1 and KW-MLP**.

Reference and Regularized use moment-loss coefficients 0 and 0.1. Both retain the source recipe's other regularization, including weight decay.

## Layout

```text
parent/
├── datasets/speech_commands_v0.02/  # extracted audio, outside Git
└── rethinking-kws-depth/
    ├── run.py                      # common command entry
    ├── configs/                    # fixed splits, recipes, source versions, weight index
    ├── environment/                # PyTorch and TensorFlow dependencies
    ├── train/run.py                # source-training dispatcher
    ├── experiments/
    │   ├── train/                  # author-loop adaptations and KWT cache preparation
    │   ├── export.py               # native features and checkpoint conversion
    │   ├── evaluate.py             # deletion, reuse and recovery
    │   ├── budget.py               # recovery budgets, computation and timing
    │   └── inference.py            # single-sample network computation
    ├── shared/                     # models, data, structure, recovery, metrics, costs
    ├── third_party/                # author implementations and original licenses
    ├── paper/                      # per-figure entries, shared plotting and tables
    ├── results/                    # saved paper records and budget trials
    ├── checkpoints/                # local paper weights, excluded from Git
    └── outputs/                    # new features, training, experiments and figures
```

Use distinct run IDs and output directories for new experiments. Commands write to the specified destination. Only the fixed paper models and protocols are maintained.

## Environment

Recorded hardware: **NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB)**, Linux and Python 3.12.3. Details are in `configs/environment.json`.

Use separate PyTorch and TensorFlow environments because their NumPy versions differ.

**PyTorch:** BC8/KW-MLP training, all structural evaluation, recovery, cost measurement and plotting.

```bash
python3.12 -m venv .venv-torch
source .venv-torch/bin/activate
python -m pip install -r requirements.txt
```

The root requirements include `environment/torch-requirements.txt`, with PyTorch 2.8.0 and CUDA 12.8.

**TensorFlow:** TC14/KWT training, KWT training-cache preparation and native feature/weight export. Use NVIDIA's `nvcr.io/nvidia/tensorflow:25.02-tf2-py3` environment, then:

```bash
python -m pip install -r environment/tensorflow-requirements.txt
```

This uses the recorded TensorFlow `2.17.0+nv25.2` build. Mount both the repository and audio directory into the container. The author code retains the TF1 graph/Keras adaptations required by this environment.

## Data and checkpoints

Download the [Speech Commands v0.02 main archive](https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz). Extract word directories and `_background_noise_` directly into `../datasets/speech_commands_v0.02/`, or set `KWS_DATA_ROOT` to the extracted directory.

Committed manifests fix the audio splits and class order:
- BC8, TC14 and KWT use GSC12: ten commands, unknown speech and silence.
- KW-MLP uses GSC35.
- Evaluation uses the main archive's test split; recovery uses training entries.

The eight paper weights are indexed in `configs/checkpoints.json`:
```text
checkpoints/MODEL/reference/model.pth
checkpoints/MODEL/regularized/model.pth
```
MODEL is `bc8`, `tc14`, `kwt1` or `kwmlp`. TC14/KWT also contain native weights and frontend configurations under `native/`.

The local project includes the weights. They are excluded from Git; a public download location has not yet been assigned. Plotting the saved results requires neither weights nor audio.

## Generate paper figures and tables

Run from the repository root in the PyTorch environment:

```bash
python run.py paper
python run.py paper --figure 2
python run.py paper --figure 8 --output outputs/figure8
python -m paper.tables.generate
```

Outputs: `outputs/paper/fig/` (PDF/SVG/PNG), `table/` (TeX), and `doc/` (plot data). Existing publication assets remain in `paper/published/`. The manuscript is not edited or compiled.

| Item | Content | Entry/source |
|---|---|---|
| Fig.1 | Method overview | paper/fig01_method |
| Fig.2 | Single deletion | paper/fig02_single_deletion |
| Fig.3 | Single-position reuse | paper/fig03_single_reuse |
| Fig.4 | Multiple deletion | paper/fig04_multi_deletion |
| Fig.5 | Multiple-position reuse | paper/fig05_multi_reuse |
| Fig.6 | Normalization recovery | paper/fig06_normalization_recovery |
| Fig.7 | Reference/Regularized deletion | paper/fig07_regularized_deletion |
| Fig.8 | Paired normalization/inserted-affine recovery | paper/fig08_regularized_recovery |
| Tables 1–4 | Authored related work, rules, models and objectives | paper/tables/authored |
| Tables 5–12 | Counts, examples, recovery, budgets, costs and comparisons | paper/tables/generate.py |

Individual figure entries run as modules, e.g. `python -m paper.fig04_multi_deletion.generate`. Fig.1 is an authored vector diagram; after editing its SVG, `python paper/fig01_method/export.py` exports it with `rsvg-convert`.

Shared statistics are in `paper/build.py`; styles are in `structure_style.py` and `paired_style.py`. Generated tables contain the numerical results and captions. Manually formatted publication tables are also saved in `paper/published/tables/`.

To plot a new experiment, update the result/budget paths in `configs/checkpoints.json` and the corresponding `results/inference/` summaries. Keep all compared outputs tied to the intended checkpoint.

## Prepare clean features

In PyTorch:

```bash
python run.py prepare --model bc8 --output outputs/features/bc8
python run.py prepare --model kwmlp --output outputs/features/kwmlp
```

For TC14/KWT, first run native feature/weight export in TensorFlow:

```bash
python run.py prepare --model tc14 --stage native --checkpoint checkpoints/tc14/reference/native/model --config checkpoints/tc14/reference/native/config.json --output outputs/features/tc14
python run.py prepare --model kwt1 --stage native --checkpoint checkpoints/kwt1/reference/native/best_weights --config checkpoints/kwt1/reference/native/config.json --output outputs/features/kwt1
```

Then switch to PyTorch:

```bash
python run.py prepare --model tc14 --stage convert --output outputs/features/tc14
python run.py prepare --model kwt1 --stage convert --output outputs/features/kwt1
```

Each prepared directory holds ordered train/validation/test features and labels. The clean features can be shared between conditions of the same architecture. Conversion also writes `model.pth` for the native checkpoint supplied; use the separately indexed weights for the archived Reference/Regularized comparisons.

## Source training

Run in the model's native environment. Both conditions start from scratch:

```bash
python run.py train --model bc8 --lambda-max 0 --run-id paper_pair --workers 4
python run.py train --model bc8 --lambda-max 0.1 --run-id paper_pair --workers 4
```

Replace `bc8` with `tc14`, `kwt1` or `kwmlp`. Before training KWT, prepare its cache in TensorFlow:

```bash
python run.py cache-kwt --workers 4
```

KWT uses one clean and three waveform-augmented MFCC views, with online author SpecAugment. The default cache is `outputs/cache/kwt1_author/`; override it with `KWS_KWT_CACHE`.

| Model | Backend | Budget | Batch |
|---|---|---:|---:|
| BC8 | PyTorch | 200 epochs | 100 |
| TC14 | TensorFlow | 30,000 updates | 100 |
| KWT-1 | TensorFlow | 23,438 updates | 512 |
| KW-MLP | PyTorch | 140 epochs | 256 |

The moment coefficient stays zero for five epochs, then ramps to its target over fifteen epochs. Both conditions compute the moment objective. `configs/models/` documents the settings; the author-loop adapters implement the fixed recipes.

Status and metrics: `outputs/train/RUN_ID/MODEL/ARM/full/`, where ARM is `lambda_0` or `lambda_0p1`.
Weights: `weights/retrained/RUN_ID/MODEL/ARM/full/` for BC8/KW-MLP; `weights/RUN_ID/MODEL/ARM/full/` for TC14/KWT.
Best checkpoints are selected by validation accuracy. New TensorFlow checkpoints use the same native/convert export procedure, with their new checkpoint and training config paths.

## Deletion, reuse and recovery

All evaluation uses PyTorch. For example:

```bash
python run.py evaluate --model bc8 --arm lambda_0 --checkpoint checkpoints/bc8/reference/model.pth --prepared-dir outputs/features/bc8 --output outputs/eval/bc8/reference --families drop reuse norm affine --max-blocks 4 --scope suffix --device cuda
```

For other models, change the model and paths. For Regularized use `--arm lambda_0p1` and its checkpoint.

| Family | Operation |
|---|---|
| drop | Direct deletion with retained weights/states fixed |
| reuse | Neighbor whole-block reuse, retaining every source |
| norm | S/A/S+A for BN; A for LN |
| affine | Inserted channel-wise affine recovery |
| reuse_norm | Normalization recovery at reused target positions |

Use `--max-blocks 1` for single-position experiments. TC14 has three removable blocks and no compatible adjacent reuse pair. Recovery uses 100 updates, batch 128, Adam LR 0.001, gradient clipping 1 and BN momentum 0.1.

`--scope suffix` selects normalization after the first deleted block, including head normalization where present. `--scope all` selects all retained normalization. For Table 12's full candidate pool, also evaluate `norm --scope all` into a separate directory and append its non-baseline rows to the model's result ledger.

Each run writes `results.jsonl`, one row per configuration. Positions are one-based; accuracy change is modified minus complete-model accuracy, in percentage points.

## Recovery budgets and inference costs

Run budget jobs serially with the GPU free of other workloads:

```bash
python run.py budget --model bc8 --arm lambda_0 --checkpoint checkpoints/bc8/reference/model.pth --prepared-dir outputs/features/bc8 --output outputs/budget/bc8/reference --device cuda
```

This runs three repeats of:
1. One batch with five training examples per class: 60 for GSC12, 175 for GSC35.
2. 100 updates with batch 128, including Full with the same budget.
3. Full-parameter fine-tuning for one epoch.

Fixed deletion sets: BC8 {4,7,8,12}; TC14 {2,4}; KWT/KW-MLP {2,5,8,12}. One-batch BN calibration uses momentum 1; other budgets use 0.1. S updates use inputs without labels; labels construct the balanced one-batch subset.

Time covers synchronized updates including indexing and optimizer work. Construction, warm-up, cost tracing and evaluation are outside the timer. FLOPs/MACs cover the entire update schedule. Core FLOPs equal twice contraction MACs; non-core FLOPs are arithmetic estimates.

Single-sample inference costs:

```bash
python run.py inference --model bc8 --checkpoint checkpoints/bc8/reference/model.pth --prepared-dir outputs/features/bc8 --output outputs/inference/bc8
```

This compares the complete and fixed-deletion models from prepared features. Audio preprocessing and latency are excluded.

## Results and terminology

- F: no recovery. Full: fine-tuning all retained parameters.
- S: BN statistics; A: normalization affine parameters; S+A: both.
- Ins. A: inserted affine layers.
- Box centers are medians. Recovery tables report means over compatible deletion sets.
- Paired recovery differences are Regularized minus Reference for identical deletion sets/methods.
- Tables 6 and 12 are retrospectively selected test-sweep examples; selection is implemented in `paper/build.py`.
- `results/reference/` and `regularized/` hold per-configuration records. `results/budgets/` retains individual trials and summaries; `results/inference/` holds forward costs.

## Sources

| Architecture | Author implementation |
|---|---|
| BC-ResNet | [Qualcomm AI Research](https://github.com/Qualcomm-AI-research/bcresnet) |
| TC-ResNet | [Hyperconnect](https://github.com/hyperconnect/TC-ResNet) |
| KWT | [Arm](https://github.com/ARM-software/keyword-transformer) |
| KW-MLP | [AI Research BD](https://github.com/AI-Research-BD/Keyword-MLP) |

Versions are recorded in `configs/sources.json`. Original licenses are retained beside the upstream code. Dataset: [Speech Commands](https://www.tensorflow.org/datasets/catalog/speech_commands).

This release copy removes original run audits, synthetic tests, cloud queue controllers and historical compatibility branches. It has not been executed end to end after refactoring. Bundled paper results are the existing completed experiment records, not new results from this refactored implementation.

