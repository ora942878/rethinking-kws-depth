# Rethinking KWS Depth

| Model | Condition | Moment coefficient | Dataset | Training | Batch | Requirements |
|---|---|---:|---|---|---:|---|
| BC-ResNet-8 | Reference | 0 | GSC12 | 200 epochs | 100 | PyTorch |
| BC-ResNet-8 | Regularized | 0.1 | GSC12 | 200 epochs | 100 | PyTorch |
| TC-ResNet-14 | Reference | 0 | GSC12 | 30,000 steps | 100 | TensorFlow |
| TC-ResNet-14 | Regularized | 0.1 | GSC12 | 30,000 steps | 100 | TensorFlow |
| KWT-1 | Reference | 0 | GSC12 | 23,438 steps | 512 | TensorFlow |
| KWT-1 | Regularized | 0.1 | GSC12 | 23,438 steps | 512 | TensorFlow |
| KW-MLP | Reference | 0 | GSC35 | 140 epochs | 256 | PyTorch |
| KW-MLP | Regularized | 0.1 | GSC35 | 140 epochs | 256 | PyTorch |

- **PyTorch:** `environment/torch-requirements.txt`; training BC8/KW-MLP, all evaluation and plotting.
- **TensorFlow:** `environment/tensorflow-requirements.txt`; training TC14/KWT and native export. Container setup is included in its comments.
- **Hardware:** NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB), Linux, Python 3.12.3.

## Data to figures

Run from the repository root: `python run.py COMMAND --help`.

| Stage | Files | Command |
|---|---|---|
| Data | Extract [Speech Commands v0.02](https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz) to `../datasets/speech_commands_v0.02/`; fixed splits: `configs/manifests/` | Set `KWS_DATA_ROOT` for another location |
| KWT cache | `experiments/train/prepare_kwt_cache.py` | `cache-kwt` |
| Train | `experiments/train/{torch_pair,tc14_pair,kwt_pair}.py`; recipes: `configs/models/` | `train` |
| Features / weight conversion | `experiments/export.py` | `prepare` |
| Deletion / reuse / recovery | `experiments/evaluate.py` | `evaluate` |
| Recovery budgets / timing | `experiments/budget.py` | `budget` |
| Inference FLOPs / MACs | `experiments/inference.py` | `inference` |
| Figures / tables | `paper/fig*/`, `paper/build.py`, `paper/tables/` | `paper` |

Both conditions train from scratch; use `--lambda-max 0` or `0.1`. TC14/KWT export uses TensorFlow (`--stage native`), then PyTorch (`--stage convert`). Evaluation uses the main dataset's test split; recovery uses training samples. Run timing jobs alone on the GPU.

## Saved results

`configs/checkpoints.json` indexes the eight weights and result files. [Download the eight checkpoints](https://drive.google.com/file/d/1qr14btgaV-_N04X3Yy0VQwRcPYGFGn-_/view?usp=sharing). Extract the archive and place its four model folders (`bc8`, `tc14`, `kwt1`, `kwmlp`) inside the repository's `checkpoints/` directory, so each weight is at `checkpoints/MODEL/{reference,regularized}/model.pth`. New outputs go to `outputs/`; published figures and tables remain in `paper/published/`.

Generate saved-result figures and tables without audio or weights:

```bash
python run.py paper
python run.py paper --figure 2
```

Figs.2–5: deletion/reuse; Fig.6: normalization recovery; Figs.7–8: Reference/Regularized comparisons. Fig.1 and Tables 1–4 are authored assets; Tables 5–12 are generated from results.

Box centers are medians; recovery tables report means. Tables 6 and 12 show retrospectively selected test-sweep examples.

## Sources

[BC-ResNet](https://github.com/Qualcomm-AI-research/bcresnet) · [TC-ResNet](https://github.com/hyperconnect/TC-ResNet) · [KWT](https://github.com/ARM-software/keyword-transformer) · [KW-MLP](https://github.com/AI-Research-BD/Keyword-MLP)

Source versions: `configs/sources.json`. Original project code is MIT-licensed (see LICENSE); third-party code and adaptations retain their upstream licenses. Saved results are from completed experiments; this refactored code has not yet been run end to end.


