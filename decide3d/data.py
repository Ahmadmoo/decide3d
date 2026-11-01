import json, math, random, re, pathlib
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

SYSTEM = "You answer questions about the 3D scene in the image."
QUESTION = "\nQuestion: {}\nIs the candidate answer below correct? Answer Yes or No."
OPTION = "\nCandidate: {}"
ASSISTANT = "<|im_end|>\n<|im_start|>assistant\n"
TYPES = {"yes_no": 0, "choice": 1, "score": 2}
TASKS = {"binary": 0, "multichoice": 1, "regression": 2, "cardinality": 3}
SUBSETS = [None, "dev", "test"]
PX = 32  # pixels per visual token: 16 px patches, 2x2 merge

INVERSE = {"left": "right", "right": "left", "in front of": "behind", "behind": "in front of",
           "closer to": "farther from", "farther from": "closer to", "further from": "closer to", "nearer to": "farther from",
           "closest": "farthest", "farthest": "closest", "nearest": "farthest",
           "larger": "smaller", "bigger": "smaller", "smaller": "larger", "largest": "smallest", "biggest": "smallest", "smallest": "largest",
           "taller": "shorter", "shorter": "taller", "tallest": "shortest", "shortest": "tallest",
           "wider": "narrower", "narrower": "wider", "widest": "narrowest", "narrowest": "widest"}
INVERSE_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, INVERSE), key=len, reverse=True)) + r")\b", re.I)


class Tok:
    def __init__(self, tokenizer):
        self.t = tokenizer
        self.pre = self.enc(f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n<|vision_start|>")
        self.vis_end = self.enc("<|vision_end|>")
        self.image_pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.assistant = self.enc(ASSISTANT)
        (self.scene,), (self.yes,), (self.no,) = self.enc("\n"), self.enc("Yes"), self.enc("No")

    def enc(self, s):
        return self.t.encode(s, add_special_tokens=False)


def build(tok, gh, gw, questions):
    """Token ids, 3D rope positions and segments for one frame.
    Context (system, image, scene token) is segment 0. Each question and each option is its own segment;
    `par` points an option to its question. All questions start at the same position, and so do all options of a question,
    so with the segment mask every option is computed exactly as if it were alone."""
    ids = tok.pre + [tok.image_pad] * (gh * gw) + tok.vis_end + [tok.scene]
    p0 = len(tok.pre)
    pos = [[i] * 3 for i in range(p0)] + [[p0, p0 + r, p0 + c] for r in range(gh) for c in range(gw)]
    pos += [[p0 + max(gh, gw) + i] * 3 for i in range(len(tok.vis_end) + 1)]
    c0, scene_pos = pos[-1][0] + 1, len(ids) - 1
    seg, par, q_end, opt_end, s = [0] * len(ids), [0] * len(ids), [], [], 1
    for q in questions:
        qi = tok.enc(QUESTION.format(q["question"]))
        ids += qi; seg += [s] * len(qi); par += [s] * len(qi); pos += [[c0 + i] * 3 for i in range(len(qi))]
        q_end.append(len(ids) - 1); qs, s, ends = s, s + 1, []
        for o in q["options"]:
            oi = tok.enc(OPTION.format(o)) + tok.assistant
            ids += oi; seg += [s] * len(oi); par += [qs] * len(oi); pos += [[c0 + len(qi) + i] * 3 for i in range(len(oi))]
            ends.append(len(ids) - 1); s += 1
        opt_end.append(ends)
    return dict(ids=ids, pos=pos, seg=seg, par=par, q_end=q_end, opt_end=opt_end, scene_pos=scene_pos)


def geometry_tokens(geo, gh, gw):
    """MoGe grid (16 px cells) -> one 9-dim vector per visual token: log-depth, ray xyz, normal xyz, valid fraction, spread."""
    g = torch.from_numpy(np.asarray(geo["grid"], dtype=np.float32))
    v = g[4:5]
    w = F.adaptive_avg_pool2d(v[None], (gh, gw))[0]
    x = F.adaptive_avg_pool2d((g[[0, 1, 2, 3, 5]] * v)[None], (gh, gw))[0] / w.clamp_min(1e-6)
    K = np.asarray(geo["K"], dtype=np.float32)
    u, t = (torch.arange(gw) + 0.5) / gw, (torch.arange(gh) + 0.5) / gh
    ray = torch.stack([((u[None] - K[0, 2]) / K[0, 0]).expand(gh, gw), ((t[:, None] - K[1, 2]) / K[1, 1]).expand(gh, gw), torch.ones(gh, gw)])
    ok = (w > 0).float()
    out = torch.cat([x[:1] * ok, F.normalize(ray, dim=0), F.normalize(x[1:4], dim=0) * ok, w, x[4:5] * ok])
    return out.flatten(1).T.contiguous()


def fit_size(w0, h0, pixels):
    r = w0 / h0
    return max(PX, round(math.sqrt(pixels * r) / PX) * PX), max(PX, round(math.sqrt(pixels / r) / PX) * PX)


def invert(q):
    """'closer' <-> 'farther' etc. with the label flipped; only for yes/no and 2-option questions with exactly one such word."""
    m = INVERSE_RE.findall(q["question"])
    if len(m) != 1 or len(q["options"]) != 2 or q["type"] == "score": return q
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


class Frames(torch.utils.data.Dataset):
    """One item = one frame with up to `max_questions` of its questions; a frame with more questions gives several items.
    Questions never see each other (segment mask), so how they are grouped does not change any answer."""

    def __init__(self, cfg, split, tok, train, subset=None, shuffle_options=False):
        root = pathlib.Path(cfg["frames_dir"])
        self.c, self.tok, self.train, self.shuffle_options = cfg["data"], tok, train, shuffle_options
        self.geom, self.rules = pathlib.Path(cfg["geom_dir"]), list(self.c.get("paraphrase", {}).items())
        self.shards = [root / n for n in (root / f"{split}_shards.txt").read_text().split()]
        index = np.load(root / f"{split}_index.npy")
        index = index[index[:, 3] == SUBSETS.index(subset)] if subset else index
        k = -(-index[:, 2] // self.c["max_questions"])
        start = np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)
        self.items = np.column_stack([np.repeat(index[:, :2], k, 0), start * self.c["max_questions"]])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        s, off, j = self.items[i].tolist()
        with open(self.shards[s], "rb") as f:
            f.seek(off)
            fr = json.loads(f.readline())
        c = self.c
        qs = fr["questions"][j:j + c["max_questions"]]
        if self.train: qs = [paraphrase(invert(q) if random.random() < c["invert_prob"] else q, self.rules) for q in qs]
        if self.shuffle_options: qs = [shuffle(q) for q in qs]
        img = Image.open(fr["image"]).convert("RGB")
        w, h = fit_size(*img.size, random.choice(c["train_pixels"]) if self.train else c["eval_pixels"])
        gh, gw = h // PX, w // PX
        geo = np.load(self.geom / fr["fid"][:2] / f"{fr['fid']}.npz")
        drop = torch.zeros(gh * gw, dtype=torch.bool)
        if self.train and random.random() < c["mask_prob"]:
            drop[torch.randperm(gh * gw)[: int(random.uniform(*c["mask_ratio"]) * gh * gw)]] = True
        return dict(build(self.tok, gh, gw, qs), image=img.resize((w, h), Image.BICUBIC), geometry=geometry_tokens(geo, gh, gw),
                    scene=torch.from_numpy(np.asarray(geo["scene"], dtype=np.float32)), drop=drop,
                    geo_off=self.train and random.random() < c["geo_drop"], questions=qs)


class Collate:
    """Packs the frames of a micro-batch into one sequence (row 0); `doc` keeps them apart in the mask."""

    def __init__(self, image_processor, tok):
        self.ip, self.tok = image_processor, tok

    def __call__(self, items):
        px = self.ip(images=[it["image"] for it in items], do_resize=False, return_tensors="pt")
        grid = px["image_grid_thw"]
        for it, g in zip(items, grid):
            assert g[1] * g[2] == 4 * len(it["geometry"]), (g, it["geometry"].shape)
        ids, pos, doc, seg, par, q_end, opt_end, scene_pos, off = [], [], [], [], [], [], [], [], 0
        for d, it in enumerate(items):
            ids += it["ids"]; pos += it["pos"]; seg += it["seg"]; par += it["par"]; doc += [d] * len(it["ids"])
            q_end += [off + e for e in it["q_end"]]
            opt_end += [[off + e for e in es] for es in it["opt_end"]]
            scene_pos.append(off + it["scene_pos"])
            off += len(it["ids"])
        qs = [q for it in items for q in it["questions"]]
        K = max(len(q["options"]) for q in qs)
        target, opt_index, perm = torch.zeros(len(qs), K), torch.full((len(qs), K), -1), torch.full((len(qs), K), -1)
        for k, (q, es) in enumerate(zip(qs, opt_end)):
            target[k, :len(es)], opt_index[k, :len(es)] = torch.tensor(q["target"]), torch.tensor(es)
            if "perm" in q: perm[k, :len(es)] = torch.tensor(q["perm"])
        ids = torch.tensor(ids)
        return dict(input_ids=ids, position_ids=torch.tensor(pos).T[:, None], doc=torch.tensor(doc), seg=torch.tensor(seg), par=torch.tensor(par),
                    pixel_values=px["pixel_values"], image_grid_thw=grid, vis_pos=ids == self.tok.image_pad,
                    geometry=torch.cat([it["geometry"] for it in items]), drop=torch.cat([it["drop"] for it in items]),
                    geo_off=torch.cat([torch.full((len(it["geometry"]),), it["geo_off"]) for it in items]),
                    scene=torch.stack([it["scene"] for it in items]), scene_off=torch.tensor([it["geo_off"] for it in items]),
                    scene_pos=torch.tensor(scene_pos), q_end=torch.tensor(q_end), opt_index=opt_index, opt_mask=opt_index >= 0,
                    target=target, perm=perm, qtype=torch.tensor([TYPES[q["type"]] for q in qs]), task=torch.tensor([TASKS[q["task"]] for q in qs]),
                    value=torch.tensor([q.get("value_m", math.nan) for q in qs]), qid=[q["id"] for q in qs])
