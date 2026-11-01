import sys, json, re, hashlib, pathlib, collections
import numpy as np

# Groups converted CA-VQA records (cavqa_convert.py output) by image, so all questions of one frame form one training sequence.
#   python preprocessing/build_frames.py <records_dir> <frames_dir>
#   python preprocessing/build_frames.py <records_dir> <frames_dir> --templates     # print common question shapes per task
# Output per split: <split>_XX.jsonl (one frame per line), <split>_shards.txt, <split>_stats.json,
#   <split>_index.npy: one row per frame (shard, byte offset, questions, subset 0 none / 1 dev / 2 test)
# Val frames get subset "dev" or "test" by video id (dev: eval during training + temperature fit, test: final numbers).

KEEP = ("id", "task", "type", "question", "options", "target", "value_m")
SUBSETS = [None, "dev", "test"]
VIDEO_RE = re.compile(r"(?<!\d)(\d{8})(?!\d)")


def fid_of(image):
    return hashlib.sha1(image.encode()).hexdigest()[:16]


def subset_of(image, fid):
    m = VIDEO_RE.search(image)
    key = m.group(1) if m else fid
    return "dev" if int(hashlib.sha1(key.encode()).hexdigest(), 16) % 2 == 0 else "test"


def build(split, files, out):
    tmp = out / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    buckets = {}
    for f in files:
        for line in open(f):
            b = fid_of(json.loads(line)["image"])[:2]
            if b not in buckets: buckets[b] = open(tmp / f"{split}_{b}.jsonl", "w")
            buckets[b].write(line)
    for h in buckets.values(): h.close()

    index, stats = [], collections.Counter()
    q_per_frame, opts = [], collections.defaultdict(list)
    shard_names = sorted(buckets)
    for si, b in enumerate(shard_names):
        frames = {}
        for line in open(tmp / f"{split}_{b}.jsonl"):
            r = json.loads(line)
            fr = frames.setdefault(r["image"], dict(fid=fid_of(r["image"]), image=r["image"], intrinsics=r.get("intrinsics"),
                                                   depth_gt=r.get("depth_gt"), questions=[]))
            fr["questions"].append({k: r[k] for k in KEEP if k in r})
            stats[r["task"]] += 1
            opts[r["type"]].append(len(r["options"]))
        with open(out / f"{split}_{b}.jsonl", "wb") as f:
            for fr in frames.values():
                if split == "val": fr["subset"] = subset_of(fr["image"], fr["fid"])
                index.append((si, f.tell(), len(fr["questions"]), SUBSETS.index(fr.get("subset"))))
                f.write((json.dumps(fr) + "\n").encode())
                q_per_frame.append(len(fr["questions"]))
        (tmp / f"{split}_{b}.jsonl").unlink()
    tmp.rmdir()

    np.save(out / f"{split}_index.npy", np.array(index, dtype=np.int64))
    (out / f"{split}_shards.txt").write_text("\n".join(f"{split}_{b}.jsonl" for b in shard_names) + "\n")
    q = np.array(q_per_frame)
    s = dict(frames=len(q), questions=int(q.sum()), per_task=dict(stats),
             questions_per_frame=dict(mean=float(q.mean()), p50=float(np.median(q)), p95=float(np.percentile(q, 95)), max=int(q.max())),
             options_per_question={t: float(np.mean(v)) for t, v in opts.items()})
    (out / f"{split}_stats.json").write_text(json.dumps(s, indent=1))
    print(split, json.dumps(s))


def templates(files, top=15):
    by_task = collections.defaultdict(list)
    for f in files:
        for line in open(f):
            r = json.loads(line)
            by_task[r["task"]].append(r["question"])
    for task, qs in sorted(by_task.items()):
        words = collections.Counter(w for q in qs for w in re.findall(r"[a-z']+", q.lower()))
        vocab = {w for w, _ in words.most_common(120)}
        shapes = collections.Counter()
        for q in qs:
            toks = [t if t in vocab else "*" for t in re.findall(r"[a-z']+|\d+(?:\.\d+)?|[^\sa-z\d]", q.lower())]
            shapes[re.sub(r"(\* )+\*", "*", " ".join("#" if re.match(r"\d", t) else t for t in toks))] += 1
        print(f"\n== {task}: {len(qs)} questions")
        for s, n in shapes.most_common(top): print(f"{n:>9d}  {s}")


src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
files = {sp: sorted(f for f in src.glob(f"{sp}_*.jsonl") if not f.name.endswith("_unparsed.jsonl")) for sp in ("train", "val")}
if "--templates" in sys.argv:
    templates(files["train"] or files["val"])
    sys.exit()
for sp, fs in files.items():
    if fs: build(sp, fs, out)
