import copy, importlib.util, json, pathlib, subprocess, sys
import numpy as np, pytest, torch, yaml
from PIL import Image
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
from decide3d.data import PX, Collate, Frames, Tok, build, invert
from decide3d.evaluate import REPS
from decide3d.model import Decide3D, loss_fn

# Everything runs on CPU with a tiny random Qwen3-VL and a byte-level tokenizer, so no download is needed.
ROOT = pathlib.Path(__file__).resolve().parents[1]
NET = {"geo_hidden": 32, "set_dim": 32, "set_layers": 2}
QUESTIONS = [
    {"id": "b", "task": "binary", "type": "yes_no", "question": "Is the chair closer to the camera than the table?", "options": ["no", "yes"], "target": [0.0, 1.0]},
    {"id": "m", "task": "multichoice", "type": "choice", "question": "Which is taller?", "options": ["lamp", "sofa", "door"], "target": [0.0, 0.0, 1.0]},
    {"id": "c", "task": "cardinality", "type": "score", "question": "How many chairs are there?", "options": [str(i) for i in range(6)], "target": [0, 0, 1.0, 0, 0, 0]},
    {"id": "r", "task": "regression", "type": "score", "question": "How far is the lamp from the door?", "options": [f"{v:.2f} m" for v in REPS],
     "target": [float(i == 12) for i in range(len(REPS))], "value_m": float(REPS[12])},
]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "preprocessing" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def parts():
    t = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(pre_tokenizers.ByteLevel.alphabet())}, merges=[]))
    t.pre_tokenizer, t.decoder = pre_tokenizers.ByteLevel(add_prefix_space=False), decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=t)
    tokenizer.add_special_tokens({"extra_special_tokens": ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "Yes", "No"]})
    torch.manual_seed(0)
    cfg = Qwen3VLConfig(text_config=dict(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                                         head_dim=16, rope_parameters=dict(rope_type="default", rope_theta=10000.0, mrope_section=[4, 2, 2])),
                        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, num_position_embeddings=64,
                                           deepstack_visual_indexes=[0, 1]), tie_word_embeddings=True)
    cfg._attn_implementation = "sdpa"
    base = Qwen3VLForConditionalGeneration(cfg).eval()
    ip = Qwen2VLImageProcessorPil(patch_size=16, temporal_patch_size=2, merge_size=2, image_mean=[0.5] * 3, image_std=[0.5] * 3)
    return Tok(tokenizer), base, ip


def item(tok, questions, gh=3, gw=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    img = Image.fromarray((torch.rand(gh * PX, gw * PX, 3, generator=g) * 255).byte().numpy())
    return dict(build(tok, gh, gw, questions), image=img, geometry=torch.rand(gh * gw, 9, generator=g), scene=torch.rand(6, generator=g),
                drop=torch.zeros(gh * gw, dtype=torch.bool), geo_off=False, questions=questions)


def test_positions_match_hf(parts):
    tok, base, ip = parts
    b = Collate(ip, tok)([item(tok, [dict(QUESTIONS[0], options=["yes"])])])  # one question, one option = a plain sequence
    ids = b["input_ids"][None]
    hf, _ = base.model.get_rope_index(ids, (ids == tok.image_pad).int(), b["image_grid_thw"])
    assert torch.equal(hf, b["position_ids"])


def test_one_pass_equals_separate_runs(parts):
    """Two packed frames, all questions and options in one pass == every option run alone with the plain causal mask."""
    tok, base, ip = parts
    net = Decide3D(base, tok, NET, attn="sdpa").eval()
    frames = [(QUESTIONS[:2], 3, 4, 0), (QUESTIONS[2:3], 2, 3, 1)]
    with torch.no_grad():
        b = Collate(ip, tok)([item(tok, qs, gh, gw, s) for qs, gh, gw, s in frames])
        h = net.hidden(b)
        got = h[b["opt_index"].clamp_min(0)]
        net.mask = lambda b, L: None
        k = 0
        for qs, gh, gw, s in frames:
            for q in qs:
                for j, o in enumerate(q["options"]):
                    b1 = Collate(ip, tok)([item(tok, [dict(q, options=[o])], gh, gw, s)])
                    torch.testing.assert_close(got[k, j], net.hidden(b1)[b1["opt_index"][0, 0]], atol=1e-4, rtol=1e-4)
                k += 1


def test_option_order_invariance(parts):
    tok, base, ip = parts
    net = Decide3D(base, tok, NET, attn="sdpa").eval()
    torch.nn.init.normal_(net.set.out.weight), torch.nn.init.normal_(net.geometry.alpha)
    for q in QUESTIONS[1:3]:
        p = torch.randperm(len(q["options"]), generator=torch.Generator().manual_seed(1)).tolist()
        q2 = dict(q, options=[q["options"][i] for i in p], target=[q["target"][i] for i in p])
        with torch.no_grad():
            z1, z2 = (net(Collate(ip, tok)([item(tok, [x])])) for x in (q, q2))
        torch.testing.assert_close(z1[0, p], z2[0], atol=1e-5, rtol=1e-5)


def test_training_step(parts):
    tok, base, ip = parts
    net = Decide3D(copy.deepcopy(base), tok, NET, attn="sdpa").train_llm(True).train()
    net.base.model.language_model.gradient_checkpointing_enable()
    frozen = {n: p.clone() for n, p in net.base.named_parameters() if not p.requires_grad}
    assert frozen and any(".blocks." in n for n in frozen) and any("embed_tokens" in n for n in frozen)
    b = Collate(ip, tok)([item(tok, QUESTIONS, seed=3)])
    b["drop"][:4] = True
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-3)
    losses = []
    for i in range(10):
        loss = loss_fn(net(b), b, torch.ones(4), 0.5)
        loss.backward()
        if i == 0:
            for p in (net.geometry.alpha, net.set.out.weight, net.yes_no, net.mask_token, net.scene[-1].weight):
                assert p.grad.abs().sum() > 0
        opt.step(), opt.zero_grad()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0], losses
    assert all(torch.equal(p, frozen[n]) for n, p in net.base.named_parameters() if n in frozen)


def test_invert():
    q = invert(QUESTIONS[0])
    assert q["question"] == "Is the chair farther from the camera than the table?" and q["target"] == [1.0, 0.0]
    assert invert(QUESTIONS[1]) is QUESTIONS[1]


def test_moge_pool():
    pool = load_script("moge_geometry").pool
    depth, normal, mask = torch.full((64, 96), 2.0), torch.zeros(64, 96, 3), torch.ones(64, 96, dtype=torch.bool)
    normal[..., 2], mask[:16, :16] = 1.0, False
    grid, scene = pool(depth, normal, mask)
    assert grid.shape == (6, 4, 6) and grid[:, 0, 0].abs().sum() == 0
    torch.testing.assert_close(grid[:, 1, 1], torch.tensor([np.log(2.0), 0, 0, 1, 1, 0], dtype=torch.float32))
    torch.testing.assert_close(scene, torch.tensor([np.log(2.0)] * 3 + [1 - 256 / (64 * 96)], dtype=torch.float32))


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """Fake converted records -> build_frames.py -> frames; fake MoGe output -> pool -> geometry files."""
    d, rng = tmp_path_factory.mktemp("data"), np.random.default_rng(0)
    (d / "records").mkdir()
    for split, n in (("train", 6), ("val", 8)):
        with open(d / "records" / f"{split}_all.jsonl", "w") as f:
            for i in range(n):
                p = d / "images" / f"{40000000 + 7919 * i}" / f"{split}{i}.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rng.integers(0, 255, (96, 128, 3), dtype=np.uint8)).save(p)
                for j, q in enumerate(QUESTIONS + QUESTIONS[:2] * (i % 2)):  # odd frames have 6 questions -> 2 items at max_questions 4
                    f.write(json.dumps(dict(q, id=f"{split}{i}_{j}", image=str(p), intrinsics=None, depth_gt=None)) + "\n")
    subprocess.run([sys.executable, ROOT / "preprocessing/build_frames.py", d / "records", d / "frames"], check=True, capture_output=True)
    pool = load_script("moge_geometry").pool
    for split in ("train", "val"):
        for name in (d / "frames" / f"{split}_shards.txt").read_text().split():
            for line in open(d / "frames" / name):
                fr = json.loads(line)
                depth = torch.from_numpy(rng.uniform(0.5, 6.0, (96, 128))).float()
                grid, scene = pool(depth, torch.nn.functional.normalize(torch.randn(96, 128, 3), dim=-1), depth < 5.5)
                out = d / "geom" / fr["fid"][:2] / f"{fr['fid']}.npz"
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez(out, grid=grid.half().numpy(), scene=np.r_[scene.numpy(), 1.2, 1.0].astype(np.float32),
                         K=np.array([[0.7, 0, 0.5], [0, 0.93, 0.5], [0, 0, 1]], dtype=np.float32))
    return d


def test_frames(dataset, parts):
    tok, _, ip = parts
    cfg = yaml.safe_load(open(ROOT / "configs/train.yaml")) | dict(frames_dir=str(dataset / "frames"), geom_dir=str(dataset / "geom"))
    cfg["data"] |= dict(max_questions=4, train_pixels=[20000], eval_pixels=30000)
    train = Frames(cfg, "train", tok, train=True)
    assert len(train) == 9 and sorted(train.items[:, 2].tolist()) == [0] * 6 + [4] * 3
    dev, test = (Frames(cfg, "val", tok, train=False, subset=s) for s in ("dev", "test"))
    assert len(dev) and len(test) and len(dev) + len(test) == 12
    b = Collate(ip, tok)([train[0], train[1]])
    L, nvis = len(b["input_ids"]), int(b["vis_pos"].sum())
    assert b["position_ids"].shape == (3, 1, L) and b["doc"].shape == (L,)
    assert nvis == len(b["geometry"]) == int((b["image_grid_thw"].prod(1) // 4).sum())
    assert torch.isfinite(b["geometry"]).all() and (b["opt_mask"].sum(1) >= 2).all()


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory, parts):
    tok, base, ip = parts
    d = tmp_path_factory.mktemp("tiny")
    base.save_pretrained(d), tok.t.save_pretrained(d), ip.save_pretrained(d)
    return d


def test_scripts(dataset, tiny_dir, tmp_path):
    """train.py stage A -> stage B (init from A) -> resume is a no-op -> evaluate.py."""
    cfg = yaml.safe_load(open(ROOT / "configs/train.yaml"))
    run = tmp_path / "run"
    cfg |= dict(model=str(tiny_dir), frames_dir=str(dataset / "frames"), geom_dir=str(dataset / "geom"), out_dir=str(run), attn="sdpa",
                net=NET, batch_frames=4, eval_every=2, eval_questions=10, save_every=2)
    cfg["data"] |= dict(max_questions=4, train_pixels=[20000, 40000], eval_pixels=30000, workers=0)
    cfg["stages"] = {"A": dict(steps=3, lr_llm=0.0, lr_new=1e-3), "B": dict(epochs=1, lr_llm=1e-4, lr_new=1e-3, init=str(run / "A/last.pt"))}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    sh = lambda *a: subprocess.run([sys.executable, "-m", *a], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    sh("decide3d.train", path, "A"), sh("decide3d.train", path, "B")
    assert '"step"' not in sh("decide3d.train", path, "B")
    log = [json.loads(l) for l in open(run / "B/log.jsonl")]
    assert log[0]["steps"] == 2 and any("dev" in l for l in log)
    sh("decide3d.evaluate", path, run / "B/last.pt")
    report = json.loads((run / "B/report.json").read_text())
    assert report["order"]["flip"] == 0 and report["order"]["max_dp"] < 1e-2
    assert set(report["test"]) >= {"all", "yes_no", "choice", "score", "regression"} and "mra" in report["test"]["regression"]
    assert (run / "B/final.pt").exists()
