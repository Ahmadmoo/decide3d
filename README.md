# decide3d

RGB-only 3D decision model. One image and typed 3D questions with text options in; one calibrated probability per option out, in one pass. The answer does not change when the options are reordered.

- Backbone: Qwen3-VL-2B-Instruct + frozen MoGe-2 geometry
- Data: CA-VQA (binary, multichoice, cardinality, regression)
- Target: Jetson Thor, ~10 Hz, single image

Details: [docs/design.md](docs/design.md) (network, training), [docs/dataset.md](docs/dataset.md) (data pipeline).

## Layout

```
preprocessing/
  get_cavqa.sh        download -> convert -> delete, one task at a time
  cavqa_convert.py    CA-VQA -> typed questions (yes_no / choice / score)
  build_frames.py     group questions by frame, split val into dev / test
  moge_geometry.py    MoGe-2 geometry per frame (run once)
decide3d/
  data.py             prompt layout, segments, rope positions, augmentation, packing
  model.py            geometry adapter, scene token, segment mask, Yes-No readout, set head, loss
  train.py            stages A / B, DDP, resume
  evaluate.py         metrics, temperature fit, order robustness
configs/train.yaml
tests/test_model.py   tiny random model on CPU, no download
```

## Use

```bash
pip install -e ".[dev]"                 # + ".[preprocess]" on the data machine

# data (docs/dataset.md)
RAW=/scratch/cavqa_raw OUT=/scratch/cavqa_records bash preprocessing/get_cavqa.sh
python preprocessing/build_frames.py /scratch/cavqa_records /scratch/cavqa_frames
python preprocessing/moge_geometry.py /scratch/cavqa_frames /scratch/cavqa_geom

# train (paths in configs/train.yaml); a stopped job resumes from <out_dir>/<stage>/last.pt
torchrun --nproc_per_node=2 -m decide3d.train configs/train.yaml A
torchrun --nproc_per_node=2 -m decide3d.train configs/train.yaml B

# temperatures on val-dev, report on val-test -> report.json, final.pt
python -m decide3d.evaluate configs/train.yaml runs/main/B/last.pt

pytest -q
```

## Status

| Part | State |
|---|---|
| Data pipeline | done, tested on fake data; real CA-VQA wording still to check (`--peek`, `--templates`) |
| Model, training, evaluation | done, tested on a tiny random Qwen3-VL on CPU |
| Full run on H100 | not run yet |
| Thor inference (KV cache, batched options) | planned |

CA-VQA is CC-BY-NC-ND (non-commercial, no derivatives): keep the converted records private.
