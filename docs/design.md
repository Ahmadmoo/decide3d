# Design: network and training

One RGB image in. Typed 3D questions with text options in. One calibrated probability per option out, in one forward pass, and the same answer whatever the option order.

---

## 1. Network

```
image ──► Qwen3-VL ViT (frozen) ──► merger (trained) ──► visual tokens ──┐
  │                         └──► DeepStack features (layers 1-3) ───────┤
  └──► MoGe-2 ViT-S (frozen, precomputed) ──► per-token geometry ─► gated add ─┘
                                      └──► scene stats ─► scene token

[system | image | scene] [question 1] [option 1.1] [option 1.2] ... [question 2] [option 2.1] ...
                     └──────── Qwen3-VL-2B LLM, segment mask + shared positions ────────┘
last token of every option ──► s = h · (W_yes − W_no) ──► set head (per question) ──► softmax / T_type
```

| Part | Choice | Why |
|---|---|---|
| Backbone | Qwen3-VL-2B-Instruct | Full attention, so a custom mask is simple. Qwen3.5 is hybrid (Gated DeltaNet) and cannot take it. ~35% faster on Thor than Qwen3.5-2B (28.0 vs 43.3 ms at 2k tokens) |
| Vision | ViT frozen, merger + DeepStack mergers trained | Keeps pretrained vision; the mergers adapt it to geometry questions |
| Geometry | MoGe-2 ViT-S, frozen, run once offline (`preprocessing/moge_geometry.py`) | Metric depth + normals + intrinsics from one RGB image |
| Geometry adapter | Per visual token: log-depth (8 Fourier freqs, sin + cos), ray (3), normal (3), valid fraction, depth spread = 24 dims → MLP 24→512→2048 → `x + α ⊙ σ(W[LN x; LN e]) ⊙ e`, α = 0 at start | Zero start = the pretrained model is unchanged at step 0. The gate lets the model ignore bad geometry |
| Scene token | `\n` embedding + zero-init MLP of 6 scene stats (log-depth p5/p50/p95, valid fraction, fov x/y) | Global scale and field of view |
| Option score | `s_k = h_k · (W_yes − W_no)`, from the LM head rows | A real Yes/No answer from step 0, no new head to learn first |
| Set head | 2 layers, width 512, no positions, zero-init output: `z_k = s_k + Δ_k` | Options of a question see each other, still order-free |
| Output | softmax over the options of a question, one temperature per type (yes_no, choice, score) | Calibrated probabilities |

### 1.1 Order invariance (exact)

| Rule | Effect |
|---|---|
| Segment mask: context sees context; a question sees context + itself; an option sees context + its question + itself | No option can see another option |
| All questions start at the same position; all options of a question start at the same position | An option's input is identical whatever its place |
| Set head has no positional encoding | Equivariant over options |

So the output for an option does not depend on where it is: exact in math, equal up to float rounding in practice. Reordering cannot change the answer (Jev-style models lose 23–64% of first-try successes under reordering).

The mask also makes questions independent, so many questions share one image encoding in one pass. `tests/test_model.py` checks that one pass equals running every option alone.

### 1.2 Question types

| Type | Tasks | Options | Target | Extra loss |
|---|---|---|---|---|
| yes_no | binary | no, yes | one-hot | — |
| choice | multichoice | object names from the question | one-hot | — |
| score | cardinality | 0 … 20 | one-hot | RPS |
| score | regression | 22 log-spaced distance bins | Gaussian in log space, σ = log 1.1 | RPS |

---

## 2. Training

| Stage | Steps | Trained | LR |
|---|---|---|---|
| A | 1000 | new modules only (adapter, scene token, mask token, Yes−No row, set head) | 2e-4 |
| B | 3 epochs | + LLM layers, vision mergers | LLM 1e-5, new 1e-4 |
| C | — | one temperature per type, fit on val-dev | grid search on NLL |

| Setting | Value |
|---|---|
| Loss | soft-target cross entropy + 0.5 · RPS for score questions |
| Questions | 32 per frame visit, shared evenly over the tasks of the frame (a frame has ~2,600, ~88% multichoice) |
| Task weights | ∝ (frames with task)^-0.5, mean weight 1 per sampled question |
| Optimizer | AdamW, β = (0.9, 0.95), weight decay 0.1 (none on norms, biases, vectors, embeddings), grad clip 1.0 |
| Schedule | WSD: 3% warmup, flat, linear decay over the last 20% |
| Batch | 64 frames per step (2 frames packed per sequence per GPU, gradient accumulation) |
| Precision | fp32 master weights, bf16 autocast, fp32 readout and softmax |
| Speed | FlexAttention block mask (skips masked blocks), packed frames, gradient checkpointing, DDP |
| Frozen | ViT, token embeddings / LM head |

Muon is not used: fine-tuning an Adam-pretrained model with Muon does worse.

### 2.1 Augmentation

| Augmentation | Setting |
|---|---|
| Resolution | 0.25 / 0.5 / 0.8 MP per frame (eval and Thor: 0.3 MP) |
| Paraphrase | regex rewrites from `configs/train.yaml` |
| Relation inversion | closer ↔ farther, left ↔ right, … with the label flipped (yes/no and 2-option questions with exactly one relation phrase; left, right, behind inside object names block it) |
| Geometry dropout | 15% of frames get no geometry |
| Appearance masking | 30% of frames: 30–60% of visual tokens replaced by a learned mask token, geometry kept, DeepStack zeroed there |

### 2.2 Cost

Measured on 2× H100 with 64 frames per step: 8.6 s / step in A, 9.8 s / step in B, 43 GB per GPU. A (1000 steps) + B (3 epochs = 11,317 steps) is about 34 h and 8,200 SU, evals included (estimate from those speeds).

---

## 3. Evaluation (`decide3d/evaluate.py`)

| Metric | For |
|---|---|
| Accuracy, NLL, Brier, ECE (15 bins) | all types |
| RPS | score questions |
| acc@10%, acc@25%, MRA | regression (estimate = probability-weighted geometric mean of the bins) |
| Flip rate, max \|Δp\| under shuffled options | order robustness |

Val is split by video into dev (eval during training, temperature fit) and test (final numbers).

---

## 4. Deployment on Jetson Thor (planned)

| Step | Work |
|---|---|
| 1 | MoGe-2 ViT-S at 1200 tokens + Qwen3-VL ViT on the image |
| 2 | Prefix [system, image, scene] once → KV cache |
| 3 | Questions as batched rows on the prefix cache, then options as batched rows on the question cache |

This gives the same numbers as the training mask, with no custom mask kernel. Target: ~10 Hz at 0.3 MP.
