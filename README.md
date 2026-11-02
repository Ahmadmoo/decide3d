# decide3d

RGB-only 3D decision model. One image and typed 3D questions with text options in; one calibrated probability per option out, in one pass. The answer does not change when the options are reordered.

- Backbone: Qwen3-VL-2B-Instruct + frozen MoGe-2 geometry
- Data: CA-VQA (binary, multichoice, cardinality, regression)
- Target: Jetson Thor, ~10 Hz, single image

Details: [docs/design.md](docs/design.md) (network, training), [docs/dataset.md](docs/dataset.md) (data pipeline).

## Layout

```
preprocessing/
  get_cavqa.sh        download -> check -> convert, one task at a time
  cavqa_convert.py    CA-VQA -> typed questions (yes_no / choice / score)
  tfrecord_lite.py    TFRecord reader, no tensorflow needed
  build_frames.py     group questions by frame, split val into dev / test by video
  moge_geometry.py    MoGe-2 geometry per frame (run once)
decide3d/
  data.py             prompt layout, segments, rope positions, augmentation, packing
  model.py            geometry adapter, scene token, segment mask, Yes-No readout, set head, loss
  train.py            stages A / B, DDP, resume
  evaluate.py         metrics, temperature fit, order robustness
configs/train.yaml
slurm/                TGI RAILS jobs: build_frames, moge, train, evaluate
tests/test_model.py   tiny random model on CPU, no download
tests/test_preprocessing.py   converter + frame builder on fake data in the real format
```

## Use

```bash
conda create -n decide3d python=3.12 pip
conda env config vars set -n decide3d PYTHONNOUSERSITE=1    # keep ~/.local out of this env
pip install -e ".[dev,preprocess]"

# data on TGI RAILS (docs/dataset.md), paths in configs/train.yaml and slurm/
RAW=/projects/beqw/ahmadmoo/cavqa/raw OUT=/projects/beqw/ahmadmoo/cavqa/out bash preprocessing/get_cavqa.sh
sbatch slurm/build_frames.slurm
sbatch slurm/moge.slurm

# train; a stopped job resumes from <out_dir>/<stage>/last.pt
sbatch slurm/train.slurm A
sbatch slurm/train.slurm B

# temperatures on val-dev, report on val-test -> report.json, final.pt
sbatch slurm/evaluate.slurm

pytest -q
```

## Status

| Part | State |
|---|---|
| Data pipeline | done on the real CA-VQA data: 625M train questions on 241,451 frames, 92,214 val questions on 2,559 frames |
| Model, training, evaluation | done, tested on CPU with a tiny random Qwen3-VL (torch 2.15, transformers 5.22): one pass = separate runs, order invariance, flex = dense mask, DDP with 2 processes |
| Real run on 2× H100 | 60-step test with the real weights and data: stages A, B, resume, evaluate all work. 8.6 s / step (A), 9.8 s / step (B), 43 GB per GPU. Option order in fp32: 0 flips, max \|Δp\| 3.5e-6 |
| Full training | not run yet |
| Thor inference (KV cache, batched options) | planned |

CA-VQA is CC-BY-NC-ND (non-commercial, no derivatives): keep the converted records private.
