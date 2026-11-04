import os, sys, json, time, datetime, pathlib, contextlib, yaml, numpy as np, torch, torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoImageProcessor, AutoTokenizer
from .data import Frames, Collate, Tok, TASKS
from .model import load, loss_fn
from .evaluate import evaluate, metrics, fit_temperature, concat

# torchrun --nproc_per_node=4 -m decide3d.train configs/train.yaml A     (stage A, then B)
# A stopped job resumes from <out_dir>/<stage>/last.pt. best.pt = EMA weights with the lowest calibrated dev NLL.

cfg, stage = yaml.safe_load(open(sys.argv[1])), sys.argv[2]
st, dc, oc, srcs = cfg["stages"][stage], cfg["data"], cfg["optim"], cfg["sources"]
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
train = ConcatDataset([Frames(cfg, s, "train", tok, train=True, sid=i) for i, s in enumerate(srcs)])
weights = torch.cat([d.weights() * s["share"] for d, s in zip(train.datasets, srcs)])  # source share, then scenes uniform
vals = [Frames(cfg, s, "val", tok, train=False, subset="dev", sid=i) for i, s in enumerate(srcs)]
val_orders = [torch.randperm(len(v), generator=torch.Generator().manual_seed(0)).tolist() for v in vals]
frac = {}  # share of sampled questions per task ~ share of frames with that task, mixed over sources
for s in srcs:
    n = json.loads((pathlib.Path(s["frames_dir"]) / "train_stats.json").read_text())["per_task_frames"]
    for t, c in n.items(): frac[t] = frac.get(t, 0.0) + s["share"] * c / sum(n.values())
w = {t: f ** cfg["loss"]["task_power"] for t, f in frac.items()}
task_w = torch.tensor([w.get(t, 0.0) for t in TASKS], device=dev) / sum(frac[t] * w[t] for t in frac)  # mean weight 1 per question

net = load(cfg, tok, st.get("init"), train=True).to(dev).train_llm(st["lr_llm"] > 0).train()
net.base.model.language_model.gradient_checkpointing_enable()
groups = {}
for name, p in net.named_parameters():
    if p.requires_grad:
        lr = "lr_merger" if "merger" in name else "lr_llm" if name.startswith("base.") else "lr_new"
        groups.setdefault((lr, p.ndim >= 2 and not name.endswith((".a", ".b")) and ".kind." not in name), []).append(p)
params = [p for ps in groups.values() for p in ps]
opt = torch.optim.AdamW([dict(params=ps, peak=st.get(lr, st["lr_llm"]), weight_decay=oc["weight_decay"] * decay) for (lr, decay), ps in groups.items()],
                        lr=0.0, betas=tuple(oc["betas"]), fused=dev.type == "cuda")
ema = {n: p.detach().clone() for n, p in net.named_parameters() if p.requires_grad}
fpm, total, step, best = dc["frames_per_micro"], st["steps"], 0, float("inf")
accum = max(1, cfg["batch_frames"] // (fpm * world))
if (out / "last.pt").exists():
    ck = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
    net.load_state_dict(ck["model"], strict=False)
    opt.load_state_dict(ck["opt"])
    for n, v in ck["ema"].items(): ema[n].copy_(v)
    step, best = ck["step"], ck["best"]
model = torch.nn.parallel.DistributedDataParallel(net, device_ids=[dev.index] if dev.type == "cuda" else None, find_unused_parameters=True) if ddp else net


def batches(step):
    """Endless micro-batches: frames drawn by `weights`, a different stream per rank, restarted from the step on resume."""
    g = torch.Generator().manual_seed(100003 * rank + step)
    while True:
        idx = torch.multinomial(weights, 1 << 16, replacement=True, generator=g).tolist()
        yield from DataLoader(train, fpm, sampler=idx, collate_fn=collate, num_workers=dc["workers"], pin_memory=dev.type == "cuda", drop_last=True)


def to_dev(b):
    return {k: v.to(dev, non_blocking=True) if torch.is_tensor(v) else v for k, v in b.items()}


def log(**kv):
    print(json.dumps(kv), flush=True)
    with open(out / "log.jsonl", "a") as f: f.write(json.dumps(kv) + "\n")


@torch.no_grad()
def swap():
    """Exchange the live weights with the EMA weights (calling it twice restores them)."""
    for n, p in net.named_parameters():
        if n in ema:
            t = p.detach().clone()
            p.copy_(ema[n])
            ema[n].copy_(t)


def state():
    keep = {n for n, p in net.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in net.state_dict().items() if k in keep or not k.startswith("base.")}


def save(name, **extra):
    torch.save(extra, out / f"{name}.tmp")
    os.replace(out / f"{name}.tmp", out / f"{name}.pt")


if rank == 0: log(stage=stage, steps=total, accum=accum, world=world, train_items=len(train), trainable=sum(p.numel() for p in params))
it = batches(step)
while step < total:
    t0 = time.time()
    scale = min(1.0, (step + 1) / max(1.0, oc["warmup"] * total), (total - step) / max(1.0, oc["decay"] * total))
    for g in opt.param_groups: g["lr"] = g["peak"] * scale
    acc = torch.zeros((), device=dev)
    for i in range(accum):
        b = to_dev(next(it))
        with model.no_sync() if ddp and i < accum - 1 else contextlib.nullcontext():
            with torch.autocast(dev.type, torch.bfloat16):
                loss = loss_fn(model(b), b, task_w, cfg["loss"]["rps_weight"], cfg["loss"]["smooth"]) / accum
            loss.backward()
        acc += loss.detach()
    gn = torch.nn.utils.clip_grad_norm_(params, oc["grad_clip"])
    opt.step()
    opt.zero_grad(set_to_none=True)
    step += 1
    with torch.no_grad():
        d = min(oc["ema"], (1 + step) / (10 + step))
        for n, p in net.named_parameters():
            if n in ema: ema[n].lerp_(p.detach(), 1 - d)
    if ddp: dist.all_reduce(acc, op=dist.ReduceOp.AVG)
    if rank == 0 and (step % 10 == 0 or step == total):
        log(step=step, loss=round(acc.item(), 4), grad_norm=round(gn.item(), 3), lr_scale=round(scale, 4), sec_per_step=round(time.time() - t0, 2))
    if rank == 0 and (step % cfg["eval_every"] == 0 or step == total):
        swap()
        r = concat([evaluate(net, DataLoader(v, fpm, sampler=o, collate_fn=collate, num_workers=dc["workers"]), cfg["eval_questions"]) for v, o in zip(vals, val_orders)])
        T = fit_temperature(r)
        m, nll = metrics(r, names=[s["name"] for s in srcs]), metrics(r, T)["all"]["nll"]
        if nll < best:
            best = nll
            save("best", model=state() | {"temp": torch.tensor(T)}, step=step, nll=nll)
        swap()
        log(step=step, dev_nll_calibrated=round(nll, 4), best=round(best, 4), dev={k: {c: round(x, 4) for c, x in v.items()} for k, v in m.items()})
    if rank == 0 and (step % cfg["save_every"] == 0 or step == total):
        save("last", model=state(), ema={n: v.cpu() for n, v in ema.items()}, opt=opt.state_dict(), step=step, best=best)
if ddp:
    dist.barrier()
    dist.destroy_process_group()
