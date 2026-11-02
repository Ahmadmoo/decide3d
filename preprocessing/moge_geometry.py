import os, sys, pathlib
from multiprocessing.pool import ThreadPool
import numpy as np, torch
from PIL import Image

# Runs frozen MoGe-2 on every frame image once and stores small geometry grids for the model.
#   pip install git+https://github.com/microsoft/MoGe.git
#   python preprocessing/moge_geometry.py <frames_dir> <geom_dir> [--split=train,val] [--part=i/n] [--num-tokens=1200] [--batch=16] [--threads=8] [--limit=N]
# Per frame: <geom_dir>/<fid[:2]>/<fid>.npz with
#   grid  (6, H/16, W/16) fp16: log-depth (median of valid), normal xyz (mean), valid fraction, depth spread (IQR / median)
#   scene (6,) fp32: log-depth p5, p50, p95, valid fraction, fov_x, fov_y (radians)
#   K     (3, 3) fp32: MoGe intrinsics, normalized by image width / height
# Keep --num-tokens equal to what the robot will run, so the model learns how reliable MoGe is at that setting.
# Stopped runs continue: finished frames are skipped.

CELL = 16


def cells(x, gh, gw):
    x = x[: gh * CELL, : gw * CELL]
    return x.reshape(gh, CELL, gw, CELL, *x.shape[2:]).transpose(1, 2).reshape(gh, gw, CELL * CELL, *x.shape[2:])


def pool(depth, normal, mask):
    H, W = mask.shape
    gh, gw = H // CELL, W // CELL
    m = mask & torch.isfinite(depth) & (depth > 0)
    d = cells(torch.where(m, depth, torch.nan), gh, gw)
    mc = cells(m, gh, gw)
    valid = mc.float().mean(-1)
    ok = valid > 0
    med = d.nanmedian(-1).values.nan_to_num(1.0).clamp_min(1e-3)
    q25, q75 = torch.nanquantile(d, torch.tensor([0.25, 0.75], device=d.device), dim=-1).nan_to_num(0.0)
    n = torch.nn.functional.normalize((cells(normal, gh, gw) * mc[..., None]).sum(-2), dim=-1)
    grid = torch.stack([med.log(), n[..., 0], n[..., 1], n[..., 2], valid, (q75 - q25) / med])
    grid = torch.where(ok, grid, torch.zeros_like(grid))
    grid[4] = valid
    ld = depth[m].log()
    if ld.numel() > 1_000_000: ld = ld[torch.randperm(ld.numel(), device=ld.device)[:1_000_000]]
    p = torch.quantile(ld, torch.tensor([0.05, 0.5, 0.95], device=ld.device)) if ld.numel() else torch.zeros(3, device=depth.device)
    return grid, torch.cat([p, m.float().mean()[None]])


def frames(frames_dir, splits, ip, n_parts):
    i = 0
    for sp in splits:
        for line in open(frames_dir / f"{sp}_images.txt"):
            if i % n_parts == ip: yield line.rstrip("\n").split("\t")
            i += 1


def decode(path):
    return torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1)


def run(batch, model, out, num_tokens, threads):
    imgs = torch.stack(threads.map(decode, [p for _, p in batch])).cuda().float() / 255
    with torch.no_grad():
        o = model.infer(imgs, num_tokens=num_tokens)
    for b, (fid, _) in enumerate(batch):
        grid, scene = pool(o["depth"][b], o["normal"][b], o["mask"][b])
        K = o["intrinsics"][b].float()
        fov = 2 * torch.atan(0.5 / K[[0, 1], [0, 1]])
        path = out / fid[:2] / f"{fid}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}.npz")
        np.savez(tmp, grid=grid.half().cpu().numpy(), scene=torch.cat([scene, fov]).cpu().numpy().astype(np.float32), K=K.cpu().numpy())
        os.replace(tmp, path)


if __name__ == "__main__":
    from moge.model.v2 import MoGeModel
    args = dict(a[2:].split("=", 1) for a in sys.argv[3:] if a.startswith("--"))
    frames_dir, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    ip, n_parts = map(int, args.get("part", "0/1").split("/"))
    num_tokens, bs = int(args.get("num-tokens", 1200)), int(args.get("batch", 16))
    limit = int(args.get("limit", 0))
    model = MoGeModel.from_pretrained(args.get("model", "Ruicheng/moge-2-vits-normal")).cuda().eval()
    threads = ThreadPool(int(args.get("threads", 8)))
    waiting, done = {}, 0  # landscape and portrait frames batch separately
    for fid, image in frames(frames_dir, args.get("split", "train,val").split(","), ip, n_parts):
        if (out / fid[:2] / f"{fid}.npz").exists(): continue
        batch = waiting.setdefault(Image.open(image).size, [])
        batch.append((fid, image))
        if len(batch) == bs:
            run(batch, model, out, num_tokens, threads); done += bs; batch.clear()
            if done % 5000 < bs: print(f"{done} frames", flush=True)
            if limit and done >= limit: break
    for batch in waiting.values():
        if batch: run(batch, model, out, num_tokens, threads); done += len(batch)
    print(f"done: {done} frames written to {out}")
