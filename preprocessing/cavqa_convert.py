import sys, os, json, re, math, hashlib, pathlib, collections
import numpy as np
from tfrecord_lite import records, parse_example

# usage:
#   python cavqa_convert.py val   <cavqa_val_dir>    <out_dir> [--peek] [--depth]
#   python cavqa_convert.py train <cavqa_train_root> <out_dir> [--peek] [--depth]
#   add --tasks=binary,multichoice to convert only some tasks; each task writes its own <split>_<task>.jsonl
#   train saves only the reference frame; add --support to also save the 0-4 support frames (about 4x more disk)
#   add --part=i/n (train only) to run n processes in parallel, i = 0..n-1; each writes its own part files
# train root holds cavqa_<task>/1.0.0 folders (TFRecords, read by tfrecord_lite.py); val dir is the unpacked cavqa_val.tar.gz
# images are named by content hash, so later runs into the same out_dir reuse frames already saved

TASKS = ["binary", "cardinality", "multichoice", "regression"]
LETTERS = "ABCDEFGH"
NUM_WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
UNIT = {"m": 1, "meter": 1, "meters": 1, "metre": 1, "metres": 1, "cm": .01, "centimeter": .01, "centimeters": .01,
        "centimetre": .01, "centimetres": .01, "mm": .001, "millimeter": .001, "millimeters": .001,
        "ft": .3048, "feet": .3048, "foot": .3048, "in": .0254, "inch": .0254, "inches": .0254}
NUM_RE = re.compile(r"(-?\d*\.?\d+)\s*(" + "|".join(sorted(UNIT, key=len, reverse=True)) + r")?\b", re.I)
INSTR_RE = re.compile(r"(?:(?<=[.?!\n])|^)\s*(please\s+)?(answer|respond|reply|choose|select|pick)\b[^\n?]*\b(letter|option|choice|yes or no|single word|one word|number|integer|meters?|unit)\b[^\n]*", re.I)
MARK_RE = re.compile(r"(?:\(([A-H])\)|(?<![A-Za-z0-9'])([A-H])[\.\):])\s+")
LIST_RE = re.compile(r"(?:options|choices)\s*(?:are)?\s*[:\-]\s*(.+)$", re.I | re.S)

COUNT_LEVELS = [str(i) for i in range(21)]
EDGES = np.geomspace(0.05, 10.0, 21)
REPS = np.concatenate([[0.025], np.sqrt(EDGES[:-1] * EDGES[1:]), [14.1]])
SCORE_LEVELS = ["<0.05 m"] + [f"{r:.2f} m" if r < 1 else f"{r:.1f} m" for r in REPS[1:-1]] + [">10 m"]
SIGMA = math.log(1.1)


def clean(q):
    s = re.sub(r"\s+", " ", INSTR_RE.sub("", q)).strip()
    return s or re.sub(r"\s+", " ", q).strip()


def parse_options(q):
    ms = [m for m in MARK_RE.finditer(q)]
    seq, want = [], 0
    for m in ms:
        if (m.group(1) or m.group(2)) == LETTERS[want]:
            seq.append(m); want += 1
            if want == len(LETTERS): break
    if len(seq) >= 2:
        opts = [q[a.end():(seq[i + 1].start() if i + 1 < len(seq) else len(q))] for i, a in enumerate(seq)]
        opts[-1] = opts[-1].split("\n")[0]
        opts = [re.sub(INSTR_RE, "", o).strip(" ,;\n") for o in opts]
        return clean(q[:seq[0].start()]), opts
    m = LIST_RE.search(q)
    if m:
        body = m.group(1).split("\n")[0]
        opts = [o.strip(" .") for o in re.split(r"[;,]|\bor\b", INSTR_RE.sub("", body)) if o.strip(" .")]
        if len(opts) >= 2:
            return clean(q[:m.start()]), opts
    return None, None


def norm(s):
    return re.sub(r"[^a-z0-9.]+", " ", s.lower()).strip()


def mc_index(ans, opts):
    hits = [i for i, o in enumerate(opts) if norm(o) == norm(ans)]
    if len(hits) == 1: return hits[0]
    m = re.match(r"^\s*\(?([A-H])\)?\s*(?:[\.\):]|$)", ans)
    if m and LETTERS.index(m.group(1)) < len(opts):
        return LETTERS.index(m.group(1))
    return None


def to_meters(s):
    m = NUM_RE.search(s.replace(",", ""))
    if not m: return None, None
    unit = (m.group(2) or "").lower()
    return float(m.group(1)) * UNIT.get(unit, 1), bool(unit)


def soft_bins(v):
    d = np.log(REPS) - math.log(min(max(v, 0.01), 30.0))
    t = np.exp(-0.5 * (d / SIGMA) ** 2)
    if v < 0.05: t[0] = max(t[0], 1.0)
    if v > 10.0: t[-1] = max(t[-1], 1.0)
    return (t / t.sum()).round(5).tolist()


def convert(task, q, a):
    a = a.strip()
    if task == "binary":
        y = re.sub(r"[^a-z]", "", norm(a).split(" ")[0])
        if y in ("yes", "true"): t = [0.0, 1.0]
        elif y in ("no", "false"): t = [1.0, 0.0]
        else: return None, "binary_answer"
        return dict(type="yes_no", question=clean(q), options=["no", "yes"], target=t), None
    if task == "multichoice":
        stem, opts = parse_options(q)
        if not opts: return None, "mc_options"
        i = mc_index(a, opts)
        if i is None: return None, "mc_answer"
        return dict(type="choice", question=stem, options=opts, target=[float(j == i) for j in range(len(opts))]), None
    if task == "cardinality":
        m = re.search(r"\d+", a)
        n = int(m.group()) if m else NUM_WORDS.get(norm(a).split(" ")[0].strip("."))
        if n is None: return None, "count_answer"
        if n > 20: return None, "count_over_20"
        return dict(type="score", question=clean(q), options=COUNT_LEVELS, target=[float(j == n) for j in range(21)]), None
    if task == "regression":
        v, has_unit = to_meters(a)
        if v is None or v < 0: return None, "metric_answer"
        if not has_unit: return None, "metric_no_unit"
        return dict(type="score", question=clean(q), options=SCORE_LEVELS, target=soft_bins(v), value_m=round(v, 4), unit_given=has_unit), None
    return None, "task"


def save_blob(b, sub):
    h = hashlib.sha1(b).hexdigest()[:20]
    p = OUT / sub / h[:2] / (h + ".png")
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".tmp{os.getpid()}")
        tmp.write_bytes(b); os.replace(tmp, p)
    return str(p)


def val_rows(root):
    import pyarrow as pa
    for task in TASKS:
        for f in sorted((root / task).glob("*.arrow")):
            with pa.memory_map(str(f)) as m:
                for batch in pa.ipc.open_stream(m):
                    for r in batch.to_pylist():
                        ref = r["reference_frame"]
                        sup = [r[f"support_frame_{k}"]["image"]["path"] for k in (4, 3, 2, 1) if r.get(f"support_frame_{k}") and r[f"support_frame_{k}"].get("image")]
                        img = str(root / ref["image"]["path"])
                        depth = str(root / ref["depth_map_ground_truth"]["path"]) if DEPTH and ref.get("depth_map_ground_truth") else None
                        yield task, r["id"], r["question"], r["answer"], img, [str(root / s) for s in sup] if SUPPORT else [], ref.get("intrinsics"), depth


def train_rows(root):
    for task in TASKS:
        d = root / f"cavqa_{task}" / "1.0.0"
        if not d.exists(): continue
        for shard in sorted(d.glob("*.tfrecord-*"))[IP::NP]:
            for rec in records(shard):
                ex = parse_example(rec)
                qa = [(bytes(q).decode(), bytes(a).decode()) for q, a in zip(ex["questions"], ex["answers"])]
                if PEEK:
                    for q, a in qa: yield task, "", q, a, "", [], None, None
                    continue
                img = save_blob(bytes(ex["images"][-1]), "images")
                sup = [save_blob(bytes(s), "images") for s in ex["images"][:-1]] if SUPPORT else []
                depth = save_blob(bytes(ex["depth_maps_ground_truth"][-1]), "depth_gt") if DEPTH else None
                K = ex["intrinsics"].reshape(-1, 3, 3)[-1].astype(float).round(3).tolist()
                frame = pathlib.Path(img).stem
                for q, a in qa:
                    yield task, hashlib.sha1((frame + q).encode()).hexdigest()[:16], q, a, img, sup, K, depth


split, SRC, OUT = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
PEEK, DEPTH, SUPPORT = "--peek" in sys.argv, "--depth" in sys.argv, "--support" in sys.argv
TASKS = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--tasks=")), TASKS)
assert all(t in ("binary", "cardinality", "multichoice", "regression") for t in TASKS), TASKS
IP, NP = map(int, next((a.split("=", 1)[1].split("/") for a in sys.argv if a.startswith("--part=")), ("0", "1")))
SUF = f"_part{IP:03d}of{NP:03d}" if NP > 1 else ""
OUT.mkdir(parents=True, exist_ok=True)
rows = val_rows(SRC) if split == "val" else train_rows(SRC)

if PEEK:
    seen = collections.Counter()
    for task, rid, q, a, *_ in rows:
        if seen[task] < 5:
            print(f"[{task}] Q: {q!r}\n         A: {a!r}\n"); seen[task] += 1
        if len(seen) >= len(TASKS) and min(seen.values()) >= 5: break
    sys.exit()

kept, skipped, bad, outs = collections.Counter(), collections.Counter(), collections.defaultdict(list), {}
for task, rid, q, a, img, sup, K, depth in rows:
    if task not in TASKS: continue
    rec, why = convert(task, q, a)
    if rec is None:
        skipped[(task, why)] += 1
        if len(bad[task]) < 50: bad[task].append(dict(id=rid, question=q, answer=a, reason=why))
        continue
    if task not in outs: outs[task] = open(OUT / f"{split}_{task}{SUF}.jsonl", "w")
    rec = dict(id=rid, source="cavqa", split=split, task=task, frame=pathlib.Path(img).stem, image=img, support_images=sup, intrinsics=K, depth_gt=depth, answer_raw=a, **rec)
    outs[task].write(json.dumps(rec) + "\n"); kept[task] += 1
    if kept[task] % 5000000 == 0: print(task, "kept", kept[task], flush=True)

for f in outs.values(): f.close()
for t, v in bad.items():
    with open(OUT / f"{split}_{t}{SUF}_unparsed.jsonl", "w") as f:
        for r in v: f.write(json.dumps(r) + "\n")

for t in sorted(set(kept) | {k[0] for k in skipped}):
    s = {k[1]: n for k, n in skipped.items() if k[0] == t}
    tot = kept[t] + sum(s.values())
    print(f"{t:12s} kept {kept[t]:>9d} / {tot:>9d}  skipped {s}")
