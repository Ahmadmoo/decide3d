import math, torch, torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask
from transformers import AttentionInterface, Qwen3VLForConditionalGeneration
from transformers.integrations.flex_attention import flex_attention_forward
from .data import TYPES


def _flex(module, q, k, v, mask, **kw):  # under autocast q, k leave the fp32 RMSNorms as fp32 while v is bf16
    return flex_attention_forward(module, q.to(v.dtype), k.to(v.dtype), v, mask, **kw)


AttentionInterface.register("decide3d_flex", _flex)


class Geometry(nn.Module):
    """MoGe features of a visual token -> gated residual: x + alpha * sigmoid(W [LN x; LN e]) * e, alpha = 0 at start."""

    def __init__(self, d, hidden, freqs=8):
        super().__init__()
        self.register_buffer("freq", 2 * math.pi * 2.0 ** torch.arange(freqs) / 8, persistent=False)  # log-depth periods 8 ... 1/16
        self.mlp = nn.Sequential(nn.Linear(2 * freqs + 8, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.ln_x, self.ln_e, self.gate = nn.LayerNorm(d), nn.LayerNorm(d), nn.Linear(2 * d, d)
        self.alpha = nn.Parameter(torch.zeros(d))

    def forward(self, x, g, off):
        a = g[:, :1] * self.freq
        e = self.mlp(torch.cat([a.sin(), a.cos(), g[:, 1:]], -1)) * ~off[:, None]
        return x + self.alpha * torch.sigmoid(self.gate(torch.cat([self.ln_x(x), self.ln_e(e)], -1))) * e


class SetHead(nn.Module):
    """Options of a question attend to each other with no positions (permutation equivariant); zero-init correction."""

    def __init__(self, d, w, layers, heads=8):
        super().__init__()
        self.opt, self.ctx = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, w)), nn.Sequential(nn.LayerNorm(d), nn.Linear(d, w))
        self.score, self.type = nn.Linear(1, w), nn.Embedding(len(TYPES), w)
        self.blocks = nn.ModuleList(nn.TransformerEncoderLayer(w, heads, 4 * w, 0.0, "gelu", batch_first=True, norm_first=True) for _ in range(layers))
        self.out = nn.Linear(w, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, ho, hq, s, qtype, ok):
        x = self.opt(ho) + self.score(s[..., None]) + (self.ctx(hq) + self.type(qtype))[:, None]
        for blk in self.blocks: x = blk(x, src_key_padding_mask=~ok)
        return self.out(x)[..., 0]


class Decide3D(nn.Module):
    def __init__(self, base, tok, net, attn="flex"):
        super().__init__()
        self.base, self.attn = base, attn
        d = base.config.text_config.hidden_size
        self.geometry = Geometry(d, net["geo_hidden"])
        self.scene = nn.Sequential(nn.Linear(6, net["geo_hidden"]), nn.GELU(), nn.Linear(net["geo_hidden"], d))
        nn.init.zeros_(self.scene[-1].weight)
        nn.init.zeros_(self.scene[-1].bias)
        self.mask_token = nn.Parameter(torch.zeros(d))
        w = base.get_output_embeddings().weight.detach().float()
        self.yes_no = nn.Parameter(w[tok.yes] - w[tok.no])  # score = logit(Yes) - logit(No): a real answer from step 0
        self.set = SetHead(d, net["set_dim"], net["set_layers"])
        self.register_buffer("temp", torch.ones(len(TYPES)))  # per question type, fit after training (evaluate.py)

    def train_llm(self, on):
        """LLM layers + vision mergers train, the ViT and the token embeddings stay frozen."""
        for n, p in self.base.named_parameters():
            p.requires_grad = on and any(k in n for k in ("language_model.layers.", "language_model.norm.", "visual.merger.", "visual.deepstack_merger_list."))
        return self

    def mask(self, b, L):
        doc, seg, par = b["doc"], b["seg"], b["par"]

        def rule(_b, _h, q, k):  # same frame, causal, and key is context / same segment / the query's question
            return (doc[k] == doc[q]) & (k <= q) & ((seg[k] == 0) | (seg[k] == seg[q]) | (seg[k] == par[q]))

        if self.attn == "flex": return create_block_mask(rule, None, None, L, L, device=doc.device)
        i = torch.arange(L, device=doc.device)
        return rule(None, None, i[:, None], i[None])[None, None]

    def hidden(self, b):
        """Last hidden state of every token of the packed sequence: (L, d)."""
        m, drop = self.base.model, b["drop"][:, None]
        v = m.get_image_features(b["pixel_values"], b["image_grid_thw"])
        img = torch.where(drop, self.mask_token, torch.cat(v.pooler_output).float())
        img = self.geometry(img, b["geometry"], b["geo_off"])
        deep = [torch.cat(f).masked_fill(drop, 0) for f in v.deepstack_features]
        lm = m.language_model
        x = lm.embed_tokens(b["input_ids"])
        x = x.masked_scatter(b["vis_pos"][:, None], img.to(x.dtype))
        x = x.index_add(0, b["scene_pos"], (self.scene(b["scene"]) * ~b["scene_off"][:, None]).to(x.dtype))
        return lm(inputs_embeds=x[None], position_ids=b["position_ids"], attention_mask=self.mask(b, len(x)), use_cache=False,
                  visual_pos_masks=b["vis_pos"][None], deepstack_visual_embeds=deep).last_hidden_state[0]

    def forward(self, b):
        """Logits (questions, options) at the fitted temperatures; padded options are -inf."""
        h = self.hidden(b)
        with torch.autocast(h.device.type, enabled=False):  # readout in fp32
            h, ok = h.float(), b["opt_mask"]
            ho = h[b["opt_index"].clamp_min(0)]
            s = ho @ self.yes_no
            z = s + self.set(ho, h[b["q_end"]], s, b["qtype"], ok)
            return (z / self.temp[b["qtype"]][:, None]).masked_fill(~ok, -math.inf)


def loss_fn(z, b, task_weight, rps_weight):
    """Soft-target cross entropy + ranked probability score for ordered answers (counts, distance bins)."""
    ok, t = b["opt_mask"], b["target"]
    logp = z.log_softmax(-1).masked_fill(~ok, 0)
    ce = -(t * logp).sum(-1)
    rps = ((logp.exp() * ok).cumsum(-1) - t.cumsum(-1)).pow(2).sum(-1) / (ok.sum(-1) - 1).clamp_min(1)
    return (task_weight[b["task"]] * (ce + rps_weight * rps * (b["qtype"] == TYPES["score"]))).mean()


def load(cfg, tok, ckpt=None):
    base = Qwen3VLForConditionalGeneration.from_pretrained(cfg["model"], dtype=torch.float32, attn_implementation="sdpa")
    if cfg["attn"] == "flex": base.set_attn_implementation({"text_config": "decide3d_flex"})
    net = Decide3D(base, tok, cfg["net"], cfg["attn"])
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        missing, unexpected = net.load_state_dict(sd.get("model", sd), strict=False)
        assert not unexpected and all(k.startswith("base.") for k in missing), (unexpected, [k for k in missing if not k.startswith("base.")])
    return net
