import json, math, random, re, pathlib
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

SYSTEM = "You answer questions about the 3D scene in the image."
QUESTION = "\nQuestion: {}\nIs the candidate answer below correct? Answer Yes or No."
NUMBER = "\nQuestion: {}\nAnswer with a number."
OPTION = "\nCandidate: {}"
ASSISTANT = "<|im_end|>\n<|im_start|>assistant\n"
TYPES = {"yes_no": 0, "choice": 1, "count": 2, "metric": 3}
TASKS = {"binary": 0, "multichoice": 1, "regression": 2, "cardinality": 3}
SUBSETS = [None, "dev", "test"]
PX = 32  # pixels per visual token: 16 px patches, 2x2 merge
GEO = 2  # one geometry token per GEO x GEO visual tokens
GEO_FEATURES = (2 * GEO) ** 2 * 6 + 3  # 16-px cells under a geometry token x (log-depth, normal xyz, valid, spread) + ray of its center
COUNTS = 21  # count head: 0 ... 20
EDGES = np.geomspace(0.02, 20.0, 120)  # metric head: 119 log bins (x1.06) from 2 cm to 20 m, plus "< 2 cm" and "> 20 m"
CENTERS = np.r_[EDGES[0] / math.sqrt(2), np.sqrt(EDGES[:-1] * EDGES[1:]), EDGES[-1] * math.sqrt(2)]
METRIC = len(CENTERS)
SIGMA = math.log(1.05)

INVERSE = {"to the left of": "to the right of", "to the right of": "to the left of", "infront of": "behind of", "behind of": "infront of",
           "closer to": "farther from", "farther from": "closer to", "further from": "closer to", "nearer to": "farther from",
           "closest": "farthest", "farthest": "closest", "nearest": "farthest",
           "higher than": "lower than", "lower than": "higher than",
           "larger": "smaller", "bigger": "smaller", "smaller": "larger", "largest": "smallest", "biggest": "smallest", "smallest": "largest",
           "taller": "shorter", "shorter": "taller", "tallest": "shortest", "shortest": "tallest",
           "wider": "narrower", "narrower": "wider", "widest": "narrowest", "narrowest": "widest"}
BLOCK = ["left", "right", "in front of", "behind", "longer than"]  # no safe inverse, or part of an object name: leave the question alone
INVERSE_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, [*INVERSE, *BLOCK]), key=len, reverse=True)) + r")\b", re.I)
SIDE = {"left": "right", "right": "left", "leftmost": "rightmost", "rightmost": "leftmost"}
SIDE_RE = re.compile(r"\b(leftmost|rightmost|left|right)\b", re.I)
BOX_RE = re.compile(r"\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]")


class Tok:
    def __init__(self, tokenizer):
        self.t = tokenizer
        self.pre = self.enc(f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n<|vision_start|>")
        self.vis_start, self.vis_end = self.enc("<|vision_start|>"), self.enc("<|vision_end|>")
        self.image_pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.assistant = self.enc(ASSISTANT)
        (self.scene,), (self.yes,), (self.no,) = self.enc("\n"), self.enc("Yes"), self.enc("No")

    def enc(self, s):
        return self.t.encode(s, add_special_tokens=False)


def qtype(q):
    return q["type"] if q["type"] != "score" else "count" if q["task"] == "cardinality" else "metric"


def build(tok, gh, gw, questions, geo=True):
    """Token ids, 3D rope positions, token kind (0 text, 1 visual, 2 geometry) and segments for one frame.
    Context (system, image, geometry block, scene token) is segment 0. The geometry block is a second vision block on the
    image grid: geometry token (r, c) sits at rows/cols 2r, 2c, so it keeps the same offset to the visual tokens it covers.
    Each question and each option is its own segment; `par` points an option to its question. All questions start at the
    same position, and so do all options of a question, so with the segment mask everything is computed as if alone.
    Count and metric questions end with the assistant prefix: their last token is read by a number head."""
    ids, kind, pos, nxt = [], [], [], 0

    def text(t):
        nonlocal nxt
        ids.extend(t); kind.extend([0] * len(t)); pos.extend([[nxt + i] * 3 for i in range(len(t))]); nxt += len(t)

    def grid(h, w, step, k):
        nonlocal nxt
        ids.extend([tok.image_pad] * (h * w)); kind.extend([k] * (h * w))
        pos.extend([nxt, nxt + step * r, nxt + step * c] for r in range(h) for c in range(w)); nxt += step * max(h, w)

    text(tok.pre); grid(gh, gw, 1, 1); text(tok.vis_end)
    if geo: text(tok.vis_start); grid(-(-gh // GEO), -(-gw // GEO), GEO, 2); text(tok.vis_end)
    text([tok.scene])
    c0, scene_pos = nxt, len(ids) - 1
    seg, par, q_end, opt_end, s = [0] * len(ids), [0] * len(ids), [], [], 1
    for q in questions:
        number = qtype(q) in ("count", "metric")
        qi = tok.enc((NUMBER if number else QUESTION).format(q["question"])) + (tok.assistant if number else [])
        ids += qi; kind += [0] * len(qi); seg += [s] * len(qi); par += [s] * len(qi); pos += [[c0 + i] * 3 for i in range(len(qi))]
        q_end.append(len(ids) - 1); qs, s, ends = s, s + 1, []
        for o in [] if number else q["options"]:
            oi = tok.enc(OPTION.format(o)) + tok.assistant
            ids += oi; kind += [0] * len(oi); seg += [s] * len(oi); par += [qs] * len(oi); pos += [[c0 + len(qi) + i] * 3 for i in range(len(oi))]
            ends.append(len(ids) - 1); s += 1
        opt_end.append(ends)
    return dict(ids=ids, pos=pos, kind=kind, seg=seg, par=par, q_end=q_end, opt_end=opt_end, scene_pos=scene_pos)


def geometry_tokens(geo, gh, gw, flip=False):
    """MoGe grid (16 px cells of the source image) -> one vector per GEO x GEO visual tokens: the 4 x 4 cells under it
    (log-depth, normal xyz, valid fraction, spread; zero where invalid) + the ray of its center. (tokens, GEO_FEATURES)"""
    g = torch.from_numpy(np.asarray(geo["grid"], dtype=np.float32))
    if flip:
        g = g.flip(-1)
        g[1] = -g[1]
    n, hh, ww = 2 * GEO, -(-gh // GEO), -(-gw // GEO)
    v = F.adaptive_avg_pool2d(g[4:5][None], (2 * gh, 2 * gw))[0]
    x = F.adaptive_avg_pool2d((g[[0, 1, 2, 3, 5]] * g[4:5])[None], (2 * gh, 2 * gw))[0] / v.clamp_min(1e-6)
    ok = (v > 0).float()
    cells = torch.cat([x[:1] * ok, F.normalize(x[1:4], dim=0) * ok, v, x[4:5] * ok])
    cells = F.pad(cells, (0, n * ww - 2 * gw, 0, n * hh - 2 * gh)).view(6, hh, n, ww, n).permute(1, 3, 2, 4, 0).reshape(hh * ww, -1)
    K = np.asarray(geo["K"], dtype=np.float32)
    cx = 1 - K[0, 2] if flip else K[0, 2]
    u, t = (torch.arange(ww) + 0.5) * GEO / gw, (torch.arange(hh) + 0.5) * GEO / gh
    ray = torch.stack([((u[None] - cx) / K[0, 0]).expand(hh, ww), ((t[:, None] - K[1, 2]) / K[1, 1]).expand(hh, ww), torch.ones(hh, ww)])
    return torch.cat([cells, F.normalize(ray, dim=0).flatten(1).T], 1)


def metric_target(v):
    """Gaussian in log space over the metric bins; one-hot in the "< 2 cm" / "> 20 m" bins."""
    if v < EDGES[0] or v >= EDGES[-1]: return np.eye(METRIC)[0 if v < EDGES[0] else -1]
    t = np.exp(-0.5 * ((np.log(CENTERS) - math.log(v)) / SIGMA) ** 2)
    return t / t.sum()


def edit_text(s, w0, h0, boxes, flip):
    """Boxes [x1, y1, x2, y2] in source pixels -> Qwen3-VL's 0-1000 scale, or removed. flip mirrors boxes and swaps left / right."""
    def box(m):
        if not boxes: return ""
        x1, y1, x2, y2 = map(float, m.groups())
        if flip: x1, x2 = w0 - x2, w0 - x1
        return " [" + ", ".join(str(min(1000, max(0, round(1000 * a / d)))) for a, d in ((x1, w0), (y1, h0), (x2, w0), (y2, h0))) + "]"

    def side(m):
        w = SIDE[m.group(1).lower()]
        return w.capitalize() if m.group(1)[0].isupper() else w

    s = BOX_RE.sub(box, s)
    return SIDE_RE.sub(side, s) if flip else s


def edit(q, w0, h0, boxes, flip):
    return dict(q, question=edit_text(q["question"], w0, h0, boxes, flip), options=[edit_text(o, w0, h0, boxes, flip) for o in q["options"]], qtype=qtype(q))


def answer(q):
    """Target over the question's outputs: option rows, count values 0...20 or metric bins."""
    if q["qtype"] == "metric": return metric_target(q["value_m"])
    if q["qtype"] == "count": return np.eye(COUNTS)[int(q["options"][int(np.argmax(q["target"]))])]
    return q["target"]


def fit_size(w0, h0, pixels):
    r = w0 / h0
    return max(PX, round(math.sqrt(pixels * r) / PX) * PX), max(PX, round(math.sqrt(pixels / r) / PX) * PX)


def invert(q):
    """'closer' <-> 'farther' etc. with the label flipped; only for yes/no and 2-option questions with exactly one such word."""
    m = INVERSE_RE.findall(q["question"])
    if len(m) != 1 or len(q["options"]) != 2 or q["type"] == "score" or m[0].lower() not in INVERSE: return q
    w = m[0]
    inv = INVERSE[w.lower()]
    text = INVERSE_RE.sub(inv[0].upper() + inv[1:] if w[0].isupper() else inv, q["question"], count=1)
    return dict(q, question=text, target=q["target"][::-1])


def paraphrase(q, rules):
    text = q["question"]
    for pat, alts in rules:
        if re.search(pat, text) and random.random() < 0.5: text = re.sub(pat, random.choice(alts), text, count=1)
    return dict(q, question=text)


def shuffle(q):
    p = random.sample(range(len(q["options"])), len(q["options"]))
    return dict(q, options=[q["options"][k] for k in p], target=[q["target"][k] for k in p], perm=p)


def sample(qs, n):
    """One visit of a frame: n random questions, shared evenly over its tasks (a frame has thousands, mostly multichoice)."""
    by = {}
    for q in qs: by.setdefault(q["task"], []).append(q)
    for v in by.values(): random.shuffle(v)
    return [v[i] for i in range(n) for v in by.values() if i < len(v)][:n]


class Frames(torch.utils.data.Dataset):
    """Frames of one source (`src`: frames_dir, geom_dir). Train: one item = one visit of a frame with `max_questions`
    sampled questions. Eval: every question once; a frame with more than `max_questions` gives several items.
    Questions never see each other (segment mask), so how they are grouped does not change any answer.
    Val frames carry a dev / test subset; a source without subsets uses all of val for both."""

    def __init__(self, cfg, src, split, tok, train, subset=None, shuffle_options=False, geometry=True, sid=0):
        root = pathlib.Path(src["frames_dir"])
        self.c, self.tok, self.train, self.shuffle_options, self.geometry, self.sid = cfg["data"], tok, train, shuffle_options, geometry, sid
        self.geom, self.rules = pathlib.Path(src["geom_dir"]), list(self.c.get("paraphrase", {}).items())
        self.shards = [root / n for n in (root / f"{split}_shards.txt").read_text().split()]
        index = np.load(root / f"{split}_index.npy")
        scene = np.load(root / f"{split}_scenes.npy") if (root / f"{split}_scenes.npy").exists() else np.arange(len(index))
        keep = index[:, 3] == SUBSETS.index(subset) if subset and (index[:, 3] > 0).any() else np.ones(len(index), bool)
        index, scene = index[keep], scene[keep]
        k = np.ones_like(index[:, 2]) if train else -(-index[:, 2] // self.c["max_questions"])
        start = np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)
        self.items, self.scene = np.column_stack([np.repeat(index[:, :2], k, 0), start * self.c["max_questions"]]), np.repeat(scene, k)

    def __len__(self):
        return len(self.items)

    def weights(self):
        """Sampling weight per item: scenes uniform, items uniform inside a scene; sums to 1."""
        _, inv, cnt = np.unique(self.scene, return_inverse=True, return_counts=True)
        return torch.from_numpy(1.0 / (len(cnt) * cnt[inv]))

    def __getitem__(self, i):
        s, off, j = self.items[i].tolist()
        with open(self.shards[s], "rb") as f:
            f.seek(off)
            fr = json.loads(f.readline())
        c, train = self.c, self.train
        qs = sample(fr["questions"], c["max_questions"]) if train else fr["questions"][j:j + c["max_questions"]]
        if train: qs = [paraphrase(invert(q) if random.random() < c["invert_prob"] else q, self.rules) for q in qs]
        if self.shuffle_options: qs = [shuffle(q) if qtype(q) in ("yes_no", "choice") else q for q in qs]
        img = Image.open(fr["image"]).convert("RGB")
        w0, h0 = img.size
        flip, box_drop = train and random.random() < c["flip_prob"], c["box_drop"] if train else c["eval_box_drop"]
        qs = [edit(q, w0, h0, random.random() >= box_drop, flip) for q in qs]
        if flip: img = img.transpose(Image.FLIP_LEFT_RIGHT)
        w, h = fit_size(w0, h0, random.choice(c["train_pixels"]) if train else c["eval_pixels"])
        gh, gw = h // PX, w // PX
        geo, use_geo = np.load(self.geom / fr["fid"][:2] / f"{fr['fid']}.npz"), self.geometry and not (train and random.random() < c["geo_drop"])
        drop = torch.zeros(gh * gw, dtype=torch.bool)
        if train and random.random() < c["mask_prob"]:
            drop[torch.randperm(gh * gw)[: int(random.uniform(*c["mask_ratio"]) * gh * gw)]] = True
        return dict(build(self.tok, gh, gw, qs, use_geo), image=img.resize((w, h), Image.BICUBIC), questions=qs, source=self.sid, drop=drop,
                    geometry=geometry_tokens(geo, gh, gw, flip) if use_geo else torch.zeros(0, GEO_FEATURES),
                    scene=torch.from_numpy(np.asarray(geo["scene"], dtype=np.float32)), scene_off=not use_geo)


class Collate:
    """Packs the frames of a micro-batch into one sequence (row 0); `doc` keeps them apart in the mask.
    Outputs per question: option rows (yes_no, choice), 21 count values or METRIC distance bins; `out_mask` marks the valid ones."""

    def __init__(self, image_processor, tok):
        self.ip, self.tok = image_processor, tok

    def __call__(self, items):
        px = self.ip(images=[it["image"] for it in items], do_resize=False, return_tensors="pt")
        grid = px["image_grid_thw"]
        ids, pos, kind, doc, seg, par, q_end, opt_end, scene_pos, off = [], [], [], [], [], [], [], [], [], 0
        for d, it in enumerate(items):
            assert grid[d, 1] * grid[d, 2] == 4 * it["kind"].count(1), (grid[d], it["kind"].count(1))
            ids += it["ids"]; pos += it["pos"]; kind += it["kind"]; seg += it["seg"]; par += it["par"]; doc += [d] * len(it["ids"])
            q_end += [off + e for e in it["q_end"]]
            opt_end += [[off + e for e in es] for es in it["opt_end"]]
            scene_pos.append(off + it["scene_pos"])
            off += len(it["ids"])
        qs = [q for it in items for q in it["questions"]]
        by = {t: [k for k, q in enumerate(qs) if q["qtype"] == t] for t in TYPES}
        opt_q = by["yes_no"] + by["choice"]
        Ko = max([len(opt_end[k]) for k in opt_q] + [1])
        K = max([Ko] + [COUNTS] * bool(by["count"]) + [METRIC] * bool(by["metric"]))
        target, perm, opt_index = torch.zeros(len(qs), K), torch.full((len(qs), K), -1), torch.full((len(opt_q), Ko), -1)
        for r, k in enumerate(opt_q): opt_index[r, :len(opt_end[k])] = torch.tensor(opt_end[k])
        n_out = []
        for k, q in enumerate(qs):
            t = answer(q)
            target[k, :len(t)] = torch.as_tensor(t, dtype=torch.float32)
            n_out.append(len(t))
            if "perm" in q: perm[k, :len(q["perm"])] = torch.tensor(q["perm"])
        ids, kind = torch.tensor(ids), torch.tensor(kind)
        return dict(input_ids=ids, position_ids=torch.tensor(pos).T[:, None], doc=torch.tensor(doc), seg=torch.tensor(seg), par=torch.tensor(par),
                    pixel_values=px["pixel_values"], image_grid_thw=grid, vis_pos=kind == 1, geo_pos=kind == 2,
                    geometry=torch.cat([it["geometry"] for it in items]), drop=torch.cat([it["drop"] for it in items]),
                    scene=torch.stack([it["scene"] for it in items]), scene_off=torch.tensor([it["scene_off"] for it in items]),
                    scene_pos=torch.tensor(scene_pos), q_end=torch.tensor(q_end), opt_q=torch.tensor(opt_q, dtype=torch.long), opt_index=opt_index,
                    opt_mask=opt_index >= 0, cnt_q=torch.tensor(by["count"], dtype=torch.long), met_q=torch.tensor(by["metric"], dtype=torch.long),
                    target=target, out_mask=torch.arange(K)[None] < torch.tensor(n_out)[:, None], perm=perm,
                    qtype=torch.tensor([TYPES[q["qtype"]] for q in qs]), task=torch.tensor([TASKS[q["task"]] for q in qs]),
                    value=torch.tensor([q.get("value_m", math.nan) for q in qs]), src=torch.tensor([it["source"] for it in items for _ in it["questions"]]),
                    qid=[q["id"] for q in qs])
