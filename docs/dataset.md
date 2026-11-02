# Dataset: CA-VQA → frame records + MoGe-2 geometry

Four CA-VQA tasks (binary, multichoice, regression, cardinality) plus val, turned into typed decision questions grouped by frame, plus one small geometry file per frame.

---

## 1. Storage

| Data | Size | Frames |
|---|---|---|
| binary | 2.28 TB | 241,451 |
| multichoice | 2.09 TB | 204,595 |
| regression | 2.25 TB | 237,282 |
| cardinality | 0.42 TB | 43,912 |
| val (`cavqa_val.tar.gz`) | 31.7 GB | — |
| **Total download** | **~7.1 TB** | |

grounding2d / grounding3d are not used.

Measured on the real data:

| Data | Disk |
|---|---|
| Reference images (241,451, one per frame, shared by all tasks) | 310 GB |
| GT depth | 98 GB |
| Records (JSONL) | 394 GB |
| Frames (records grouped by frame) | 189 GB |
| MoGe geometry | ~31 GB (130 KB per frame) |

Raw files stay unless `DELETE_RAW=1`. Keep `raw/cavqa_val` in any case: val records point to its images.

| Task | Questions | Frames with this task |
|---|---|---|
| binary | 53,698,921 | 241,451 |
| multichoice | 546,986,568 | 204,595 |
| regression | 21,904,953 | 237,282 |
| cardinality | 577,617 | 43,912 |
| val | 92,214 | 2,559 |

Counts are after removing questions repeated inside a frame. A frame has ~2,600 questions on average (median 356, max 1.25M), about 88% multichoice, so training samples 32 per visit (`decide3d/data.py: sample`).

---

## 2. Steps

```bash
pip install -e ".[preprocess]"
export WORK=/projects/beqw/ahmadmoo/cavqa

# 1. download + convert (val, then one task at a time): ~9 h download at 250 MB/s, ~1.5 h convert with 32 processes
RAW=$WORK/raw OUT=$WORK/out NP=32 bash preprocessing/get_cavqa.sh

# 2. group records by frame, split val into dev / test
sbatch slurm/build_frames.slurm
python preprocessing/build_frames.py $WORK/out $WORK/frames --templates     # look at the question wording

# 3. MoGe-2 geometry, one GPU per array task
sbatch slurm/moge.slurm
```

`get_cavqa.sh` can be stopped and run again: downloads resume, and finished steps are skipped (`$OUT/.<task>_done`). It stops if a shard is incomplete, a convert part fails, or under 95% of a task is kept.

---

## 3. What each script does

| Script | Input | Output |
|---|---|---|
| `cavqa_convert.py` | CA-VQA val folder (Arrow) / train TFRecord folders | `<split>_<task>[_partXXXofYYY].jsonl`, reference images, GT depth |
| `build_frames.py` | Converted records | `<split>_XX.jsonl` (one frame per line), `<split>_index.npy` (per frame: shard, byte offset, questions, subset), `<split>_shards.txt`, `<split>_images.txt`, `<split>_stats.json` |
| `moge_geometry.py` | Frames | `<geom>/<fid[:2]>/<fid>.npz` |
| `get_cavqa.sh` | Apple's link lists | Runs the download → verify → convert → delete loop |

### 3.1 Converter flags

| Flag | Effect |
|---|---|
| `--tasks=a,b` | Only these tasks |
| `--part=i/n` | Train only: shards i, i + n, ... for parallel runs |
| `--depth` | Also save GT depth (needed later for answerability labels) |
| `--support` | Also save the 0–4 support frames (~4× disk, for multi-view later) |
| `--peek` | Print 5 raw Q/A per task and stop |

### 3.2 Task → question type

| Task | Type | Options | Target |
|---|---|---|---|
| binary | yes_no | no, yes | one-hot |
| multichoice | choice | parsed from the question text | one-hot |
| cardinality | score | 0 … 20 | one-hot |
| regression | score | 22 fixed log bins, `<0.05 m` … `>10 m` | Gaussian in log space, σ = 10% |

### 3.3 Frame record

```json
{"fid": "0ca30061bb6340c5", "image": ".../images/e8/e8de....png", "intrinsics": [[...]], "depth_gt": ".../depth_gt/22/....png",
 "subset": "dev",
 "questions": [{"id": "...", "task": "binary", "type": "yes_no", "question": "Is the chair to the left of the table?",
                "options": ["no", "yes"], "target": [0.0, 1.0]}, ...]}
```

`subset` exists for val only. Val is split by video id (the start of the record id, `<video>_<frame>_<n>`), so dev and test share no videos: 126 + 139 videos, 1,237 + 1,322 frames. `id` is prefixed with the task, because val ids repeat across tasks. Repeated questions of a frame are kept once. `fid` comes from the image file name, so geometry stays valid if the data moves.

### 3.4 Geometry file

| Array | Shape | Content |
|---|---|---|
| `grid` | (6, H/16, W/16) fp16 | log-depth (median of valid pixels), normal xyz (mean), valid fraction, depth spread (IQR / median) |
| `scene` | (6,) | log-depth p5 / p50 / p95, valid fraction, fov_x, fov_y |
| `K` | (3, 3) | MoGe intrinsics, normalized by image size |

Geometry is always MoGe's, never GT depth, because MoGe is what the model gets on the robot. `--num-tokens` (default 1200) should match the robot setting.

---

## 4. Checks

| Check | Where | Action if wrong |
|---|---|---|
| Raw question format | `--peek` output in the `get_cavqa.sh` log | Fix the parser in `cavqa_convert.py` |
| Kept ratio ≥ 95% per task | `kept … / …` lines | Look at `*_unparsed.jsonl` (reason per failure) |
| Questions per frame, options per type | `<split>_stats.json` | Sets `max_questions` and memory in `configs/train.yaml` |
| Question wording | `build_frames.py --templates` | Add rewrites to `paraphrase:` in `configs/train.yaml` |

---

## 5. Known gaps

| Item | Status |
|---|---|
| "Not answerable" option | Not added: needs visible-vs-full labels from CA-1M boxes |
| Video id for train frames | Not in the TFRecords. Train frames are grouped by image only |
| Multichoice mix | ~75% is one template ("Estimate the real-world distances ... closer to"). `sample` is uniform inside a task |
| Box coordinates | Multichoice questions hold boxes in full-resolution pixels; the robot must send the same scale |

CA-VQA and CA-1M are CC-BY-NC-ND: fine for research, but do not share the converted records. Share the scripts instead.
