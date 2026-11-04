# Design v2: network and training

One RGB image in. Typed 3D questions in. Out: one calibrated probability per option (yes/no, choice), or a calibrated distribution over counts or distances. Everything in one forward pass, with the same answer whatever the option order.

---

## 1. Changes from v1

| Part | v1 | v2 | Why |
|---|---|---|---|
| Backbone | Qwen3-VL-2B, full fine-tune | **Qwen3-VL-4B, LoRA r=64** | 4B ≈ 8B on CausalSpatial at half the cost; LoRA memorizes scans less and keeps pretrained knowledge |
| Geometry | Added into visual tokens (`x + α·g·e`) | **Separate geometry tokens** (second vision block) | Additive fusion became ~79% constant bias; removing it changed nothing |
| Counts | 21 option rows | **Count head**, one readout token | No number-as-text comparison; 1 row instead of 21 |
| Distances | 22 option rows (5 cm–10 m) | **Metric head**: 119 log bins (×1.06) from 2 cm to 20 m + "< 2 cm" + "> 20 m" | Continuous estimate, manipulation-scale resolution, 1 row instead of 22 |
| Box coordinates in text | Source pixels, always | Qwen3-VL 0–1000 scale, dropped for 50% of training questions | Uses Qwen's grounding format; the model also works with names only |
| Calibration | 3 temperatures | 4 (yes_no, choice, count, metric); metric estimate = mode, unchanged by temperature | v1's mean estimate got worse after calibration |
| Questions per frame visit | 32 | 128 | More signal per image; number questions are now cheap |
| Schedule | 3 epochs, last checkpoint | Steps-based (A 500, B 3000), EMA, best checkpoint by calibrated dev NLL | v1 overfit after ~1–2k steps |
| Loss | CE + RPS | + label smoothing 0.05 (yes_no, choice); task weight power −0.25 | Overconfidence; cardinality overfit first |
| Augmentation | — | + horizontal flip (image, geometry, boxes, left ↔ right) | Appearance diversity |
| Data | CA-VQA | `sources:` list with shares, scenes sampled uniformly | Ready for Omni3D + GraspNet-1B |

---

## 2. Network

```
image ──► Qwen3-VL ViT (frozen) ──► merger (trained) ──► visual tokens (1 per 32×32 px) + DeepStack
  └──► MoGe-2 (offline) ──► 16-px cells ──► 4×4 cells per 64×64 px ──► MLP ──► geometry tokens (1 per 2×2 visual tokens)
                                     └──► scene stats ──► scene token

[system] [image block] [geometry block] [scene] [question 1] [option 1.1] [option 1.2] ... [count question ▸] [metric question ▸] ...
        └────────────── Qwen3-VL-4B LLM + LoRA, segment mask + shared positions ──────────────┘
option rows   ──► s = h · (W_yes − W_no) ──► set head ──► softmax / T_type
count  ▸ token ──► count head (21)          ──► softmax / T_count
metric ▸ token ──► metric head (121 bins)   ──► softmax / T_metric
```

### 2.1 Parts

| Part | Design |
|---|---|
| Backbone | Qwen3-VL-4B-Instruct (36 layers, full attention). ViT frozen; mergers trained (fp32, LR 1e-5); LoRA r=64, α=128 on q, k, v, o, gate, up, down of every LLM layer; base weights frozen in bf16 |
| Geometry tokens | Per 2×2 visual tokens: the 4×4 MoGe cells under it (log-depth, normal xyz, valid fraction, spread; zero where invalid) + ray of its center + Fourier of its mean log-depth → MLP (115 → 1024 → d) + a learned modality embedding. ~80 tokens at 0.3 MP |
| Geometry positions | A second vision block after the image: token (r, c) at rows / cols 2r, 2c of its own block, so it has the same offset to every visual token it covers |
| Geometry dropout | 15% of training frames have no geometry block at all, so the model also runs without MoGe |
| Scene token | `\n` embedding + zero-init MLP of 6 scene stats (log-depth p5/p50/p95, valid fraction, fov x/y); off when there is no geometry |
| Option score | `s_k = h_k · (W_yes − W_no)` from the LM head rows, + zero-init set head (2 layers, width 512, no positions) |
| Number heads | LayerNorm → 512 → GELU → 21 (count) or 121 (metric); zero-init output. Read from the last token of `Question … Answer with a number.<assistant prefix>` |
| Appearance masking | 30% of frames: 30–60% of visual tokens → learned mask token, DeepStack zeroed there (geometry kept) |

### 2.2 Order invariance (exact)

| Rule | Effect |
|---|---|
| Segment mask: context sees context; a question sees context + itself; an option sees context + its question + itself | No option sees another option; no question sees another question |
| All questions start at the same position; all options of a question start at the same position | An option's input is identical whatever its place |
| Set head has no positional encoding; number heads have no options | Equivariant / no order at all |

`tests/test_model.py` checks that one packed pass (two frames, with and without geometry, option and number questions) equals running every option and every number question alone with the plain causal mask.

### 2.3 Question types

| Type | Task | Output | Target | Extra loss |
|---|---|---|---|---|
| yes_no | binary | 2 option rows | one-hot, smoothed 0.05 | — |
| choice | multichoice | option rows (object names) | one-hot, smoothed 0.05 | — |
| count | cardinality | count head, 0 … 20 | one-hot | RPS |
| metric | regression | metric head, 121 bins | Gaussian in log space (σ = log 1.05); one-hot in the edge bins | RPS |

Records with `type: score` (CA-VQA) become count or metric by task. New sources can write `count` / `metric` directly.

---

## 3. Training

| Stage | Steps | Trained | LR |
|---|---|---|---|
| A | 500 | new modules: geometry tokens, scene token, mask token, Yes−No row, set head, number heads | 2e-4 |
| B | 3000 | + LoRA, + mergers | LoRA 1e-4, mergers 1e-5, new 1e-4 |
| C | — | one temperature per type, fit on val-dev (`evaluate.py`) | grid search on NLL |

| Setting | Value |
|---|---|
| Batch | 64 frames per step (1 frame per sequence per GPU at 4B, gradient accumulation) |
| Questions | 128 per frame visit, shared evenly over the tasks of the frame |
| Loss | soft-target CE + 0.5 · RPS (count, metric); task weight ∝ (share of questions)^-0.25, mean 1 |
| Optimizer | AdamW β = (0.9, 0.95), weight decay 0.1 (none on norms, biases, vectors, embeddings, LoRA), grad clip 1.0 |
| Schedule | WSD: 3% warmup, flat, linear decay over the last 20% |
| EMA | 0.999 (with warm-up); dev eval and `best.pt` use the EMA weights |
| Best checkpoint | lowest dev NLL after a temperature fit on the same dev questions |
| Precision | bf16 frozen base; fp32 LoRA, mergers and new modules; bf16 autocast; fp32 readout |
| Speed | FlexAttention block mask, gradient checkpointing, DDP (`find_unused_parameters`: a micro-batch may have no count / metric / geometry) |
| Full fine-tune | `net.lora_rank: 0` → fp32 LLM layers as in v1 |

### 3.1 Data

| Setting | Value |
|---|---|
| Sources | `sources:` in `configs/train.yaml`: frames_dir, geom_dir, share |
| Sampling | source by share, then scenes uniform (`<split>_scenes.npy`, one id per frame; without it each frame is its own scene), then frames uniform |
| Val | each source's `val` split; dev / test from the index subset column (a source without subsets uses all of val for both) |
| Resolution | 0.25 / 0.5 / 0.8 MP per frame (eval and Thor: 0.3 MP) |
| Text | paraphrase rules, relation inversion with flipped label, box coordinates → 0–1000 or dropped (50%) |
| Flip | 50%: image, MoGe grid (normal x negated, cx mirrored), box x-coordinates, left ↔ right words; labels unchanged |

---

## 4. Evaluation (`decide3d/evaluate.py`)

| Metric | For |
|---|---|
| Accuracy, NLL, Brier, ECE (15 bins) | all types, per task, per source |
| RPS | count, metric |
| acc@10%, acc@25%, MRA, median AbsRel | metric (estimate = mode, refined between neighbour bins) |
| Same metrics with geometry off | value of the geometry tokens |
| Flip rate, max \|Δp\| under shuffled options | order robustness (yes_no, choice) |

Outputs `report.json` and `final.pt` (model + temperatures) next to the checkpoint.

---

## 5. Not in v2

Dense depth head, sensor depth input, robot state inputs, action-conditioned questions, Thor KV-cache inference.
