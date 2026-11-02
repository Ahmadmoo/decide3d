import sys, os, json, re, shutil, hashlib, pathlib, itertools, collections, multiprocessing
import numpy as np

# Groups converted CA-VQA records (cavqa_convert.py output) by image, so all questions of one frame form one training sequence.
#   python preprocessing/build_frames.py <records_dir> <frames_dir> [--workers=N]
#   python preprocessing/build_frames.py <records_dir> <frames_dir> --templates     # print common question shapes per task
# Output per split: <split>_XX.jsonl (one frame per line, questions unique and sorted by id), <split>_shards.txt, <split>_stats.json,
#   <split>_index.npy: one row per frame (shard, byte offset, questions, subset 0 none / 1 dev / 2 test)
#   <split>_images.txt: fid <tab> image, one frame per line, for moge_geometry.py
# Val frames get subset "dev" or "test" by video id (dev: eval during training + temperature fit, test: final numbers).

KEEP = ("id", "task", "type", "question", "options", "target", "value_m")
SUBSETS = [None, "dev", "test"]
VIDEO_RE = re.compile(r"^(\d+)_")  # val ids are <video>_<frame>_<n>
HEAD = ("fid", "image", "intrinsics", "depth_gt")


def fid_of(image):
    """From the file name, not the path: geometry files stay valid when the data moves."""
    return hashlib.sha1(image.rsplit("/", 1)[-1].encode()).hexdigest()[:16]


def subset_of(rid, fid):
    m = VIDEO_RE.search(rid)
    key = m.group(1) if m else fid
    return "dev" if int(hashlib.sha1(key.encode()).hexdigest(), 16) % 2 == 0 else "test"


def scatter(job):
    """Copy the lines of one record file into 256 buckets by frame. Each worker has its own bucket files."""
    f, tmp, split = job
    buckets = {}
    for line in open(f):
        b = fid_of(json.loads(line)["image"])[:2]
        if b not in buckets: buckets[b] = open(tmp / f"{split}_{b}.{os.getpid()}", "a", buffering=1 << 20)
        buckets[b].write(line)
    for h in buckets.values(): h.close()


def gather(job):
    """One bucket -> one shard of frames. Questions stay JSON strings: a bucket holds millions of them."""
    si, b, tmp, out, split = job
    frames, stats, opts = {}, collections.Counter(), collections.defaultdict(lambda: [0, 0])
    for part in tmp.glob(f"{split}_{b}.*"):
        for line in open(part):
            r = json.loads(line)
            fr = frames.setdefault(r["image"], dict(fid=fid_of(r["image"]), image=r["image"], intrinsics=r.get("intrinsics"),
                                                   depth_gt=r.get("depth_gt"), questions={}, video=r["id"]))
            rid = f"{r['task']}_{r['id']}"  # val ids repeat across tasks
            if rid in fr["questions"]: continue
            fr["questions"][rid] = json.dumps({k: r[k] for k in KEEP if k in r} | dict(id=rid))
            stats[r["task"]] += 1
            stats["frames_" + r["task"]] += r["task"] not in fr.setdefault("tasks", set())
            fr["tasks"].add(r["task"])
            opts[r["type"]][0] += 1; opts[r["type"]][1] += len(r["options"])
    rows, images = [], []
    with open(out / f"{split}_{b}.jsonl", "wb") as f:
        for fr in sorted(frames.values(), key=lambda x: x["fid"]):
            head = {k: fr[k] for k in HEAD}
            if split == "val": head["subset"] = subset_of(fr["video"], fr["fid"])
            rows.append((si, f.tell(), len(fr["questions"]), SUBSETS.index(head.get("subset"))))
            images.append(f"{fr['fid']}\t{fr['image']}\n")
            qs = ", ".join(fr["questions"][k] for k in sorted(fr["questions"]))
            f.write((json.dumps(head)[:-1] + f', "questions": [{qs}]}}\n').encode())
    for part in tmp.glob(f"{split}_{b}.*"): part.unlink()
    return rows, images, stats, dict(opts)


def build(split, files, out, pool):
    tmp = out / "tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for _ in pool.imap_unordered(scatter, [(f, tmp, split) for f in files]): pass
    names = sorted({p.name.split(".")[0].rsplit("_", 1)[1] for p in tmp.glob(f"{split}_*")})

    index, images, stats, opts = [], [], collections.Counter(), collections.defaultdict(lambda: [0, 0])
    for rows, im, st, op in pool.imap(gather, [(si, b, tmp, out, split) for si, b in enumerate(names)]):
        index += rows; images += im; stats += st
        for t, (n, k) in op.items(): opts[t][0] += n; opts[t][1] += k
    tmp.rmdir()

    index = np.array(index, dtype=np.int64)
    np.save(out / f"{split}_index.npy", index)
    (out / f"{split}_shards.txt").write_text("\n".join(f"{split}_{b}.jsonl" for b in names) + "\n")
    (out / f"{split}_images.txt").write_text("".join(images))
    q = index[:, 2]
    s = dict(frames=len(q), questions=int(q.sum()), per_task={k: n for k, n in stats.items() if not k.startswith("frames_")},
             per_task_frames={k[7:]: n for k, n in stats.items() if k.startswith("frames_")},
             questions_per_frame=dict(mean=float(q.mean()), p50=float(np.median(q)), p95=float(np.percentile(q, 95)), max=int(q.max())),
             options_per_question={t: n[1] / n[0] for t, n in opts.items()})
    if split == "val": s["subsets"] = {SUBSETS[i]: int((index[:, 3] == i).sum()) for i in (1, 2)}
    (out / f"{split}_stats.json").write_text(json.dumps(s, indent=1))
    print(split, json.dumps(s), flush=True)


def templates(files, top=15, per_file=50000):
    by_task = collections.defaultdict(list)
    for f in files:
        for line in itertools.islice(open(f), per_file):
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


if __name__ == "__main__":
    src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    files = {sp: sorted(f for f in src.glob(f"{sp}_*.jsonl") if not f.name.endswith("_unparsed.jsonl")) for sp in ("train", "val")}
    if "--templates" in sys.argv:
        templates(files["train"] or files["val"])
        sys.exit()
    workers = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--workers=")), len(os.sched_getaffinity(0)))
    with multiprocessing.Pool(workers) as pool:
        for sp, fs in files.items():
            if fs: build(sp, fs, out, pool)
