import sys, json, math, pathlib, yaml, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoTokenizer
from .data import Frames, Collate, Tok, TYPES, TASKS
from .model import load

# python -m decide3d.evaluate configs/train.yaml runs/main/B/last.pt
#   dev: fit one temperature per question type; test: report raw + calibrated metrics and order robustness.
#   Writes report.json and final.pt (model + temperatures, no optimizer) next to the checkpoint.

KMAX = 24
EDGES = np.geomspace(0.05, 10.0, 21)  # same distance bins as preprocessing/cavqa_convert.py
REPS = np.concatenate([[0.025], np.sqrt(EDGES[:-1] * EDGES[1:]), [14.1]])


@torch.no_grad()
def evaluate(net, loader, limit=None):
    """Logits at temperature 1 for the questions in `loader` (stops after `limit` questions). Padded options are -inf."""
    was, dev, temp = net.training, net.yes_no.device, net.temp.clone()
    net.eval()
    net.temp.fill_(1.0)
    out, qid = {k: [] for k in ("z", "target", "perm", "qtype", "task", "value")}, []
    for b in loader:
        b = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in b.items()}
        with torch.autocast(dev.type, torch.bfloat16):
            z = net(b)
        pad = (0, KMAX - z.shape[1])
        out["z"].append(F.pad(z.float(), pad, value=-math.inf))
        out["target"].append(F.pad(b["target"], pad))
        out["perm"].append(F.pad(b["perm"], pad, value=-1))
        for k in ("qtype", "task", "value"): out[k].append(b[k])
        qid += b["qid"]
        if limit and len(qid) >= limit: break
    net.temp.copy_(temp)
    net.train(was)
    return dict({k: torch.cat(v).cpu().numpy() for k, v in out.items()}, qid=np.array(qid))


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


def metrics(r, temp=(1.0, 1.0, 1.0)):
    z = r["z"] / np.asarray(temp)[r["qtype"]][:, None]
    p, t = softmax(z), r["target"]
    hit, loss = p.argmax(1) == t.argmax(1), nll(z, t)
    rps = ((p.cumsum(1) - t.cumsum(1)) ** 2).sum(1) / np.maximum(np.isfinite(r["z"]).sum(1) - 1, 1)
    groups = {"all": np.ones(len(z), bool)} | {k: r["qtype"] == v for k, v in TYPES.items()} | {k: r["task"] == v for k, v in TASKS.items()}
    out = {}
    for name, s in groups.items():
        if not s.any(): continue
        m = dict(n=int(s.sum()), acc=float(hit[s].mean()), nll=float(loss[s].mean()), brier=float(((p[s] - t[s]) ** 2).sum(1).mean()), ece=ece(p[s].max(1), hit[s]))
        if name in ("score", "regression", "cardinality"): m["rps"] = float(rps[s].mean())
        out[name] = m
    s = (r["task"] == TASKS["regression"]) & np.isfinite(r["value"])
    if s.any():
        est = np.exp((p[s][:, :len(REPS)] * np.log(REPS)).sum(1))
        rel = np.abs(est / np.maximum(r["value"][s], 1e-3) - 1)
        out["regression"] |= {"acc@10%": float((rel <= 0.1).mean()), "acc@25%": float((rel <= 0.25).mean()),
                              "mra": float(np.mean([(rel < 1 - th).mean() for th in np.arange(0.5, 1.0, 0.05)]))}
    return out


def fit_temperature(r):
    grid = np.exp(np.linspace(math.log(0.25), math.log(4.0), 161))
    T = []
    for ty in TYPES.values():
        s = r["qtype"] == ty
        T.append(float(grid[np.argmin([nll(r["z"][s] / g, r["target"][s]).mean() for g in grid])]) if s.any() else 1.0)
    return T


def order_gap(r, r2):
    """r2 = same questions with shuffled options. Exact invariance means flip = 0 and |dp| at bf16 noise level."""
    p, p2, ok = softmax(r["z"]), softmax(r2["z"]), r2["perm"] >= 0
    back = np.zeros_like(p2)
    np.put_along_axis(back, np.where(ok, r2["perm"], KMAX - 1), np.where(ok, p2, 0), 1)
    pos = {q: i for i, q in enumerate(r["qid"])}
    p = p[[pos[q] for q in r2["qid"]]]
    return dict(n=len(back), flip=float((p.argmax(1) != back.argmax(1)).mean()), max_dp=float(np.abs(p - back).max()), mean_dp=float(np.abs(p - back).sum(1).mean() / 2))


def table(m):
    cols = ["n", "acc", "nll", "ece", "brier", "rps", "acc@10%", "mra"]
    lines = ["".join(f"{c:>10}" for c in ["", *cols])]
    for name, v in m.items():
        lines.append(f"{name:>10}" + "".join(f"{v[c]:>10d}" if c == "n" else f"{v[c]:>10.4f}" if c in v else f"{'':>10}" for c in cols))
    return "\n".join(lines)


if __name__ == "__main__":
    cfg, ckpt = yaml.safe_load(open(sys.argv[1])), pathlib.Path(sys.argv[2])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tok(AutoTokenizer.from_pretrained(cfg["model"]))
    collate = Collate(AutoImageProcessor.from_pretrained(cfg["model"]), tok)
    net = load(cfg, tok, ckpt).to(dev)
    run = lambda subset, shuffle=False: evaluate(net, DataLoader(Frames(cfg, "val", tok, False, subset, shuffle), cfg["data"]["frames_per_micro"],
                                                                 collate_fn=collate, num_workers=cfg["data"]["workers"]))
    d, t = run("dev"), run("test")
    T = fit_temperature(d)
    report = dict(temperature=dict(zip(TYPES, T)), dev=metrics(d, T), test_raw=metrics(t), test=metrics(t, T), order=order_gap(t, run("test", True)))
    (ckpt.parent / "report.json").write_text(json.dumps(report, indent=1))
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    torch.save(dict(model=sd.get("model", sd) | {"temp": torch.tensor(T)}), ckpt.parent / "final.pt")
    print("temperature", report["temperature"], "\norder", report["order"])
    for k in ("test_raw", "test"): print(f"\n== {k}\n{table(report[k])}")
