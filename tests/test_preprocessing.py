import io, json, pathlib, struct, subprocess, sys
import numpy as np, pytest
from PIL import Image

# Converter + frame builder on small fake data in the real CA-VQA format. No torch, no download.
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "preprocessing"))
from tfrecord_lite import parse_example, records


def varint(n):
    out = bytearray()
    while n > 127:
        out.append(n & 127 | 128); n >>= 7
    return bytes(out + bytes([n]))


def field(no, b):
    return varint(no << 3 | 2) + varint(len(b)) + b


def example(features):
    """tf.train.Example bytes: {key: list of bytes | numpy float32 array}."""
    entries = b""
    for k, v in features.items():
        kind = field(2, field(1, v.astype("<f4").tobytes())) if isinstance(v, np.ndarray) else field(1, b"".join(field(1, x) for x in v))
        entries += field(1, field(1, k.encode()) + field(2, kind))
    return field(1, entries)


def write_tfrecord(path, examples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for e in examples:
            f.write(struct.pack("<Q", len(e)) + bytes(4) + e + bytes(4))


def png(value, mode="RGB", size=(64, 48)):
    b = io.BytesIO()
    Image.new(mode, size, value).save(b, format="PNG")
    return b.getvalue()


def frame(i, questions, answers):
    K = np.tile(np.array([100.0 + i, 0, 32, 0, 100, 24, 0, 0, 1], dtype=np.float32), 2)
    return example(dict(images=[png((1, 2, 3)), png((i, 9, 9))], depth_maps_ground_truth=[png(1, "I;16"), png(i + 2, "I;16")], intrinsics=K,
                        questions=[q.encode() for q in questions], answers=[a.encode() for a in answers]))


MC = "Which object is closer to the ottoman [0, 1, 2, 3], the sofa [4, 5, 6, 7] or the table [8, 9, 1, 2]?\nA. sofa\nB. table\nAnswer with the option's letter from the given choices directly."


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    write_tfrecord(d / "cavqa_binary/1.0.0/cavqa_binary-train.tfrecord-00000-of-00002", [frame(0, ["Is the plate behind of the bench?", "Is there a cup in the image?"], ["Yes", "No"])])
    write_tfrecord(d / "cavqa_binary/1.0.0/cavqa_binary-train.tfrecord-00001-of-00002", [frame(1, ["Is the cup wider than the plate?"], ["No"])])
    write_tfrecord(d / "cavqa_multichoice/1.0.0/cavqa_multichoice-train.tfrecord-00000-of-00001", [frame(0, [MC, MC, "What is the height of the door?\nA. A door\nB. 2\nAnswer with the option's letter from the given choices directly."], ["B", "B", "A"])])
    write_tfrecord(d / "cavqa_regression/1.0.0/cavqa_regression-train.tfrecord-00000-of-00001", [frame(1, ["How long is the sofa?", "How wide is the rug?", "How far away is the door?"], ["74cm", "120", "1.67m"])])
    write_tfrecord(d / "cavqa_cardinality/1.0.0/cavqa_cardinality-train.tfrecord-00000-of-00001", [frame(0, ["How many chairs are in the image?", "How many toys are in the image?"], ["2", "21"])])
    return d


@pytest.fixture(scope="module")
def records_dir(raw, tmp_path_factory):
    out = tmp_path_factory.mktemp("records")
    for part in range(2):
        subprocess.run([sys.executable, ROOT / "preprocessing/cavqa_convert.py", "train", raw, out, f"--part={part}/2", "--depth"], check=True, capture_output=True)
    return out


def read(path):
    return [json.loads(l) for l in open(path)]


def test_tfrecord_lite(raw):
    ex = [parse_example(r) for r in records(raw / "cavqa_regression/1.0.0/cavqa_regression-train.tfrecord-00000-of-00001")]
    assert len(ex) == 1 and [bytes(q) for q in ex[0]["questions"]][0] == b"How long is the sofa?"
    assert len(ex[0]["images"]) == 2 and bytes(ex[0]["images"][-1])[:4] == b"\x89PNG" and ex[0]["intrinsics"].reshape(-1, 3, 3).shape == (2, 3, 3)


def test_truncated_tfrecord(tmp_path):
    write_tfrecord(tmp_path / "x", [example(dict(a=[b"x" * 100]))])
    (tmp_path / "y").write_bytes((tmp_path / "x").read_bytes()[:50])
    with pytest.raises(IOError): list(records(tmp_path / "y"))


def test_convert(records_dir):
    rs = {r["question"]: r for f in records_dir.glob("train_*_part*.jsonl") if "unparsed" not in f.name for r in read(f)}
    assert rs["Is the plate behind of the bench?"]["target"] == [0.0, 1.0]
    assert rs["How long is the sofa?"]["value_m"] == 0.74 and "How wide is the rug?" not in rs  # no unit in the answer: dropped
    assert not any("toys" in q for q in rs)
    q = "What is the height of the door?"
    assert rs[q]["options"] == ["A door", "2"] and rs[q]["target"] == [1.0, 0.0]
    m = rs[MC.split("\n")[0]]
    assert m["options"] == ["sofa", "table"] and m["target"] == [0.0, 1.0] and "Answer" not in m["question"]
    im, d = Image.open(m["image"]), Image.open(m["depth_gt"])
    assert im.getpixel((0, 0))[0] == 0 and im.format == "PNG" and d.mode.startswith("I")  # frame 0, last image = reference
    assert m["intrinsics"][0][0] == 100.0


@pytest.fixture(scope="module")
def frames_dir(records_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("frames")
    vals = [("11111111", 0), ("11111111", 50), ("22222222", 0), ("33333333", 0), ("44444444", 0), ("55555555", 0), ("66666666", 0), ("77777777", 0)]
    with open(records_dir / "val_binary.jsonl", "w") as f:
        for i, (video, n) in enumerate(vals):
            f.write(json.dumps(dict(id=f"{video}_{n:05d}_1", task="binary", type="yes_no", question=f"Is there a {i}?", options=["no", "yes"], target=[1.0, 0.0],
                                    image=f"/data/images/{i:032x}", intrinsics=None, depth_gt=None)) + "\n")
    with open(records_dir / "val_cardinality.jsonl", "w") as f:  # same id as a binary question of the same frame, as in real val
        f.write(json.dumps(dict(id="11111111_00000_1", task="cardinality", type="score", question="How many cups are in the image?", options=["0", "1"], target=[0.0, 1.0],
                                image=f"/data/images/{0:032x}", intrinsics=None, depth_gt=None)) + "\n")
    subprocess.run([sys.executable, ROOT / "preprocessing/build_frames.py", records_dir, out, "--workers=2"], check=True, capture_output=True)
    return out


def test_frames(frames_dir):
    st = json.loads((frames_dir / "train_stats.json").read_text())
    assert st["frames"] == 2 and st["per_task"] == {"binary": 3, "multichoice": 2, "regression": 2, "cardinality": 1}  # the repeated multichoice question counts once
    assert st["per_task_frames"] == {"binary": 2, "multichoice": 1, "regression": 1, "cardinality": 1}
    index = np.load(frames_dir / "train_index.npy")
    shards = (frames_dir / "train_shards.txt").read_text().split()
    assert index[:, 2].sum() == st["questions"] == 8 and not list(frames_dir.glob("tmp"))
    for s, off, n, _ in index.tolist():
        with open(frames_dir / shards[s], "rb") as f:
            f.seek(off)
            fr = json.loads(f.readline())
        assert len(fr["questions"]) == n and [q["id"] for q in fr["questions"]] == sorted(q["id"] for q in fr["questions"])
    lines = (frames_dir / "train_images.txt").read_text().splitlines()
    assert len(lines) == 2 and all(pathlib.Path(l.split("\t")[1]).exists() for l in lines)


def test_val_split_by_video(frames_dir):
    fr = [json.loads(l) for n in (frames_dir / "val_shards.txt").read_text().split() for l in open(frames_dir / n)]
    assert len(fr) == 8 and {f["subset"] for f in fr} == {"dev", "test"}
    assert sum(len(f["questions"]) for f in fr) == 9 and len({q["id"] for f in fr for q in f["questions"]}) == 9
    video = {f["fid"]: f["questions"][0]["id"].split("_")[1] for f in fr}
    dev, test = ({video[f["fid"]] for f in fr if f["subset"] == s} for s in ("dev", "test"))
    assert not dev & test and len(dev | test) == 7


def test_fid_follows_the_file_name():
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_frames", ROOT / "preprocessing/build_frames.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.fid_of("/a/images/x.png") == m.fid_of("/tmp/job/images/x.png")
