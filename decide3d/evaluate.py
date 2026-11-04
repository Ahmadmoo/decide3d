import sys, json, math, pathlib, yaml, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoTokenizer
from .data import Frames, Collate, Tok, TYPES, TASKS, METRIC, CENTERS, EDGES
from .model import load

# python -m decide3d.evaluate configs/train.yaml runs/v2/B/best.pt
#   dev: fit one temperature per question type; test: raw + calibrated metrics, geometry off, and order robustness.
#   Writes report.json and final.pt (model + temperatures, no optimizer) next to the checkpoint.

KMAX = 128  # >= METRIC
STEP = math.log(EDGES[1] / EDGES[0])


@torch.no_grad()
def evaluate(net, loader, limit=None):
    """Logits at temperature 1 for the questions in `loader` (stops after `limit` questions). Invalid outputs are -inf."""
    was, dev, temp = net.training, net.yes_no.device, net.temp.clone()
    net.eval()
    net.temp.fill_(1.0)
    out, qid = {k: [] for k in ("z", "target", "perm", "qtype", "task", "value", "src")}, []
    for b in loader:
        b = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in b.items()}
        with torch.autocast(dev.type, torch.bfloat16):
            z = net(b)
        pad = (0, KMAX - z.shape[1])
        out["z"].append(F.pad(z.float(), pad, value=-math.inf))
        out["target"].append(F.pad(b["target"], pad))
        out["perm"].append(F.pad(b["perm"], pad, value=-1))
        for k in ("qtype", "task", "value", "src"): out[k].append(b[k])
        qid += b["qid"]
        if limit and len(qid) >= limit: break
    net.temp.copy_(temp)
    net.train(was)
    return dict({k: torch.cat(v).cpu().numpy() for k, v in out.items()}, qid=np.array(qid))


def concat(rs):
    return {k: np.concatenate([r[k] for r in rs]) for k in rs[0]}


def log_softmax(z):
    m = z.max(1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(1, keepdims=True))


def softmax(z):
    return np.exp(log_softmax(z))


def nll(z, t):
    return -(t * np.where(np.isfinite(z), log_softmax(z), 0)).sum(1)


def ece(conf, hit, bins=15):
    i = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum(abs(hit[i == k].mean() - conf[i == k].mean()) * (i == k).mean() for k in range(bins) if (i == k).any()))


def metric_estimate(z):
    """Mode of the metric bins, refined between its neighbours (parabola on the logits): the same at any temperature."""
    k = z.argmax(1)
    kk = np.clip(k, 2, METRIC - 3)
    l, c, r = (np.take_along_axis(z, (kk + d)[:, None], 1)[:, 0] for d in (-1, 0, 1))
    den = l - 2 * c + r
    off = np.where((k == kk) & (den < 0), np.clip(0.5 * (l - r) / np.where(den < 0, den, -1.0), -0.5, 0.5), 0.0)
    return np.exp(np.log(CENTERS)[k] + off * STEP)


def metrics(r, temp=(1.0,) * len(TYPES), names=None):
    z = r["z"] / np.asarray(temp)[r["qtype"]][:, None]
    p, t = softmax(z), r["target"]
    hit, loss = p.argmax(1) == t.argmax(1), nll(z, t)
    rps = ((p.cumsum(1) - t.cumsum(1)) ** 2).sum(1) / np.maximum(np.isfinite(r["z"]).sum(1) - 1, 1)
    groups = {"all": np.ones(len(z), bool)} | {k: r["qtype"] == v for k, v in TYPES.items()} | {k: r["task"] == v for k, v in TASKS.items()}
    if names and len(names) > 1: groups |= {n: r["src"] == i for i, n in enumerate(names)}
    out = {}
    for name, s in groups.items():
        if not s.any(): continue
        m = dict(n=int(s.sum()), acc=float(hit[s].mean()), nll=float(loss[s].mean()), brier=float(((p[s] - t[s]) ** 2).sum(1).mean()), ece=ece(p[s].max(1), hit[s]))
        if name in ("count", "metric", "regression", "cardinality"): m["rps"] = float(rps[s].mean())
        s = s & (r["qtype"] == TYPES["metric"]) & np.isfinite(r["value"])
        if s.any() and name != "all":
            rel = np.abs(metric_estimate(r["z"][s][:, :METRIC]) / np.maximum(r["value"][s], 1e-3) - 1)
            m |= {"acc@10%": float((rel <= 0.1).mean()), "acc@25%": float((rel <= 0.25).mean()), "absrel": float(np.median(rel)),
                  "mra": float(np.mean([(rel < 1 - th).mean() for th in np.arange(0.5, 1.0, 0.05)]))}
        out[name] = m
    return out


def fit_temperature(r):
    grid = np.exp(np.linspace(math.log(0.05), math.log(20.0), 241))
    T = []
    for ty in TYPES.values():
        s = r["qtype"] == ty
        T.append(float(grid[np.argmin([nll(r["z"][s] / g, r["target"][s]).mean() for g in grid])]) if s.any() else 1.0)
    return T


def order_gap(r, r2):
    """r2 = same questions with shuffled options (option questions only). Exact invariance: flip = 0, |dp| at bf16 noise level."""
    s = (r2["perm"] >= 0).any(1)
    p, p2, ok = softmax(r["z"]), softmax(r2["z"][s]), r2["perm"][s] >= 0
    back = np.zeros_like(p2)
    np.put_along_axis(back, np.where(ok, r2["perm"][s], KMAX - 1), np.where(ok, p2, 0), 1)
    pos = {q: i for i, q in enumerate(r["qid"])}
    p = p[[pos[q] for q in r2["qid"][s]]]
    return dict(n=len(back), flip=float((p.argmax(1) != back.argmax(1)).mean()), max_dp=float(np.abs(p - back).max()), mean_dp=float(np.abs(p - back).sum(1).mean() / 2))


def table(m):
    cols = ["n", "acc", "nll", "ece", "brier", "rps", "acc@10%", "mra"]
    lines = ["".join(f"{c:>12}" for c in ["", *cols])]
    for name, v in m.items():
        lines.append(f"{name:>12}" + "".join(f"{v[c]:>12d}" if c == "n" else f"{v[c]:>12.4f}" if c in v else f"{'':>12}" for c in cols))
    return "\n".join(lines)


if __name__ == "__main__":
    cfg, ckpt = yaml.safe_load(open(sys.argv[1])), pathlib.Path(sys.argv[2])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tok(AutoTokenizer.from_pretrained(cfg["model"]))
    collate = Collate(AutoImageProcessor.from_pretrained(cfg["model"]), tok)
    net, srcs = load(cfg, tok, ckpt).to(dev), cfg["sources"]
    names = [s["name"] for s in srcs]
    run = lambda subset, **kw: concat([evaluate(net, DataLoader(Frames(cfg, s, "val", tok, False, subset, sid=i, **kw), cfg["data"]["frames_per_micro"],
                                                                collate_fn=collate, num_workers=cfg["data"]["workers"])) for i, s in enumerate(srcs)])
    d, t = run("dev"), run("test")
    T = fit_temperature(d)
    report = dict(temperature=dict(zip(TYPES, T)), dev=metrics(d, T, names), test_raw=metrics(t, names=names), test=metrics(t, T, names),
                  test_no_geometry=metrics(run("test", geometry=False), T, names), order=order_gap(t, run("test", shuffle_options=True)))
    (ckpt.parent / "report.json").write_text(json.dumps(report, indent=1))
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    torch.save(dict(model=sd.get("model", sd) | {"temp": torch.tensor(T)}), ckpt.parent / "final.pt")
    print("temperature", report["temperature"], "\norder", report["order"])
    for k in ("test_raw", "test", "test_no_geometry"): print(f"\n== {k}\n{table(report[k])}")
