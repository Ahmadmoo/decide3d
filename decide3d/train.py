import os, sys, json, time, datetime, pathlib, contextlib, yaml, torch, torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoTokenizer
from .data import Frames, Collate, Tok, TASKS
from .model import load, loss_fn
from .evaluate import evaluate, metrics

# torchrun --nproc_per_node=2 -m decide3d.train configs/train.yaml A     (stage A, then B)
# A stopped job resumes from <out_dir>/<stage>/last.pt when started again.

cfg, stage = yaml.safe_load(open(sys.argv[1])), sys.argv[2]
st, dc, oc = cfg["stages"][stage], cfg["data"], cfg["optim"]
ddp = "LOCAL_RANK" in os.environ
if ddp: dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", timeout=datetime.timedelta(minutes=30))  # rank 0 evaluates and saves alone
rank, world = (dist.get_rank(), dist.get_world_size()) if ddp else (0, 1)
dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if torch.cuda.is_available() else torch.device("cpu")
if dev.type == "cuda": torch.cuda.set_device(dev)
torch.manual_seed(rank)
out = pathlib.Path(cfg["out_dir"]) / stage
out.mkdir(parents=True, exist_ok=True)

tok = Tok(AutoTokenizer.from_pretrained(cfg["model"]))
collate = Collate(AutoImageProcessor.from_pretrained(cfg["model"]), tok)
train, val = Frames(cfg, "train", tok, train=True), Frames(cfg, "val", tok, train=False, subset="dev")
val_order = torch.randperm(len(val), generator=torch.Generator().manual_seed(0)).tolist()
n = json.loads((pathlib.Path(cfg["frames_dir"]) / "train_stats.json").read_text())["per_task_frames"]  # sampled questions per task ~ frames with that task
w = {t: c ** cfg["loss"]["task_power"] for t, c in n.items()}
task_w = torch.tensor([w.get(t, 0.0) for t in TASKS], device=dev) * sum(n.values()) / sum(n[t] * w[t] for t in n)  # mean weight 1 per question

net = load(cfg, tok, st.get("init")).to(dev).train_llm(st["lr_llm"] > 0).train()
net.base.model.language_model.gradient_checkpointing_enable()
groups = {}
for name, p in net.named_parameters():
    if p.requires_grad:
        key = ("lr_llm" if name.startswith("base.") else "lr_new", p.ndim >= 2 and "embed" not in name and ".kind." not in name)
        groups.setdefault(key, []).append(p)
params = [p for ps in groups.values() for p in ps]
opt = torch.optim.AdamW([dict(params=ps, peak=st[k], weight_decay=oc["weight_decay"] * decay) for (k, decay), ps in groups.items()],
                        lr=0.0, betas=tuple(oc["betas"]), fused=dev.type == "cuda")
fpm = dc["frames_per_micro"]
accum = max(1, cfg["batch_frames"] // (fpm * world))
n_micro = len(train) // (fpm * world)
total = st.get("steps") or int(st["epochs"] * n_micro / accum)
step = 0
if (out / "last.pt").exists():
    ck = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
    net.load_state_dict(ck["model"], strict=False)
    opt.load_state_dict(ck["opt"])
    step = ck["step"]
model = torch.nn.parallel.DistributedDataParallel(net, device_ids=[dev.index] if dev.type == "cuda" else None) if ddp else net


def batches(m):
    """Endless micro-batches from micro-batch m on; epoch e is one fixed permutation split over ranks."""
    while True:
        e, k = divmod(m, n_micro)
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(e))[rank::world][:n_micro * fpm].tolist()
        yield from DataLoader(train, fpm, sampler=order[k * fpm:], collate_fn=collate, num_workers=dc["workers"], pin_memory=dev.type == "cuda")
        m = (e + 1) * n_micro


def to_dev(b):
    return {k: v.to(dev, non_blocking=True) if torch.is_tensor(v) else v for k, v in b.items()}


def log(**kv):
    print(json.dumps(kv), flush=True)
    with open(out / "log.jsonl", "a") as f: f.write(json.dumps(kv) + "\n")


def save():
    keep = {name for name, p in net.named_parameters() if p.requires_grad}
    sd = {k: v.cpu() for k, v in net.state_dict().items() if k in keep or not k.startswith("base.")}
    torch.save(dict(model=sd, opt=opt.state_dict(), step=step), out / "last.tmp")
    os.replace(out / "last.tmp", out / "last.pt")


if rank == 0: log(stage=stage, steps=total, accum=accum, world=world, train_items=len(train), trainable=sum(p.numel() for p in params))
it = batches(step * accum)
while step < total:
    t0 = time.time()
    scale = min(1.0, (step + 1) / max(1.0, oc["warmup"] * total), (total - step) / max(1.0, oc["decay"] * total))
    for g in opt.param_groups: g["lr"] = g["peak"] * scale
    acc = torch.zeros((), device=dev)
    for i in range(accum):
        b = to_dev(next(it))
        with model.no_sync() if ddp and i < accum - 1 else contextlib.nullcontext():
            with torch.autocast(dev.type, torch.bfloat16):
                loss = loss_fn(model(b), b, task_w, cfg["loss"]["rps_weight"]) / accum
            loss.backward()
        acc += loss.detach()
    gn = torch.nn.utils.clip_grad_norm_(params, oc["grad_clip"])
    opt.step()
    opt.zero_grad(set_to_none=True)
    step += 1
    if ddp: dist.all_reduce(acc, op=dist.ReduceOp.AVG)
    if rank == 0 and (step % 10 == 0 or step == total):
        log(step=step, loss=round(acc.item(), 4), grad_norm=round(gn.item(), 3), lr_scale=round(scale, 4), sec_per_step=round(time.time() - t0, 2))
    if rank == 0 and (step % cfg["eval_every"] == 0 or step == total):
        m = metrics(evaluate(net, DataLoader(val, fpm, sampler=val_order, collate_fn=collate, num_workers=dc["workers"]), cfg["eval_questions"]))
        log(step=step, dev={k: {c: round(x, 4) for c, x in v.items()} for k, v in m.items()})
    if rank == 0 and (step % cfg["save_every"] == 0 or step == total): save()
if ddp:
    dist.barrier()
    dist.destroy_process_group()
