import math, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask
from transformers import AttentionInterface, Qwen3VLForConditionalGeneration
from transformers.integrations.flex_attention import flex_attention_forward
from .data import TYPES, COUNTS, METRIC, GEO_FEATURES

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _flex(module, q, k, v, mask, **kw):
    """HF flex attention under our own name (Qwen3-VL does not declare flex support).
    Under autocast q, k leave the fp32 RMSNorms as fp32 while v is bf16; older torch has no autocast rule for flex."""
    return flex_attention_forward(module, q.to(v.dtype), k.to(v.dtype), v, mask, **kw)


AttentionInterface.register("decide3d_flex", _flex)


class LoRA(nn.Module):
    """y = W x + (alpha / r) B A x. W stays frozen; B = 0 at start, so the model starts unchanged."""

    def __init__(self, base, r, alpha):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.a = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a.to(x.dtype)), self.b.to(x.dtype)) * self.scale


class GeometryTokens(nn.Module):
    """MoGe cells under each 2 x 2 group of visual tokens -> one LLM token, + a learned modality embedding."""

    def __init__(self, d, hidden, freqs=8):
        super().__init__()
        self.register_buffer("freq", 2 * math.pi * 2.0 ** torch.arange(freqs) / 8, persistent=False)  # log-depth periods 8 ... 1/16
        self.mlp = nn.Sequential(nn.Linear(GEO_FEATURES + 2 * freqs, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.kind = nn.Parameter(torch.randn(d) * 0.02)

    def forward(self, g):
        cells = g[:, :-3].unflatten(1, (-1, 6))
        v = cells[..., 4]
        a = ((cells[..., 0] * v).sum(1) / v.sum(1).clamp_min(1e-6))[:, None] * self.freq  # mean log-depth of the token, Fourier
        return self.mlp(torch.cat([g, a.sin(), a.cos()], -1)) + self.kind


class SetHead(nn.Module):
    """Options of a question attend to each other with no positions (permutation equivariant); zero-init correction."""

    def __init__(self, d, w, layers, heads=8):
        super().__init__()
        self.opt, self.ctx = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, w)), nn.Sequential(nn.LayerNorm(d), nn.Linear(d, w))
        self.score, self.kind = nn.Linear(1, w), nn.Embedding(len(TYPES), w)
        self.blocks = nn.ModuleList(nn.TransformerEncoderLayer(w, heads, 4 * w, 0.0, "gelu", batch_first=True, norm_first=True) for _ in range(layers))
        self.out = nn.Linear(w, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, ho, hq, s, qtype, ok):
        x = self.opt(ho) + self.score(s[..., None]) + (self.ctx(hq) + self.kind(qtype))[:, None]
        for blk in self.blocks: x = blk(x, src_key_padding_mask=~ok)
        return self.out(x)[..., 0]


def number_head(d, w, k):
    """Readout token -> k logits; zero-init output, so it starts uniform."""
    h = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, w), nn.GELU(), nn.Linear(w, k))
    nn.init.zeros_(h[-1].weight)
    nn.init.zeros_(h[-1].bias)
    return h


class Decide3D(nn.Module):
    def __init__(self, base, tok, net, attn="flex"):
        super().__init__()
        self.base, self.attn, self.lora = base, attn, net.get("lora_rank", 0)
        d = base.config.text_config.hidden_size
        if self.lora:
            for layer in base.model.language_model.layers:
                for parent in (layer.self_attn, layer.mlp):
                    for name in LORA_TARGETS:
                        if hasattr(parent, name): setattr(parent, name, LoRA(getattr(parent, name), self.lora, net["lora_alpha"]))
        self.geometry = GeometryTokens(d, net["geo_hidden"])
        self.scene = nn.Sequential(nn.Linear(6, net["geo_hidden"]), nn.GELU(), nn.Linear(net["geo_hidden"], d))
        nn.init.zeros_(self.scene[-1].weight)
        nn.init.zeros_(self.scene[-1].bias)
        self.mask_token = nn.Parameter(torch.zeros(d))
        w = base.get_output_embeddings().weight.detach().float()
        self.yes_no = nn.Parameter(w[tok.yes] - w[tok.no])  # option score = logit(Yes) - logit(No): a real answer from step 0
        self.set = SetHead(d, net["set_dim"], net["set_layers"])
        self.count, self.metric = number_head(d, net["head_dim"], COUNTS), number_head(d, net["head_dim"], METRIC)
        self.register_buffer("temp", torch.ones(len(TYPES)))  # per question type, fit after training (evaluate.py)

    def train_llm(self, on):
        """LoRA (or all LLM layers without LoRA) + vision mergers train; the ViT and the token embeddings stay frozen."""
        for n, p in self.base.named_parameters():
            llm = n.endswith((".a", ".b")) if self.lora else n.startswith(("model.language_model.layers.", "model.language_model.norm."))
            p.requires_grad = on and (llm or "visual.merger." in n or "visual.deepstack_merger_list." in n)
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
        deep = [torch.cat(f).masked_fill(drop, 0) for f in v.deepstack_features]
        lm = m.language_model
        x = lm.embed_tokens(b["input_ids"])
        x = x.masked_scatter(b["vis_pos"][:, None], img.to(x.dtype))
        if len(b["geometry"]): x = x.masked_scatter(b["geo_pos"][:, None], self.geometry(b["geometry"]).to(x.dtype))
        x = x.index_add(0, b["scene_pos"], (self.scene(b["scene"]) * ~b["scene_off"][:, None]).to(x.dtype))
        return lm(inputs_embeds=x[None], position_ids=b["position_ids"], attention_mask=self.mask(b, len(x)), use_cache=False,
                  visual_pos_masks=b["vis_pos"][None], deepstack_visual_embeds=deep).last_hidden_state[0]

    def forward(self, b):
        """Logits (questions, outputs) at the fitted temperatures: option rows, count values or metric bins; invalid = -inf."""
        h = self.hidden(b)
        with torch.autocast(h.device.type, enabled=False):  # readout in fp32
            h = h.float()
            hq, z = h[b["q_end"]], torch.full(b["out_mask"].shape, -math.inf, device=h.device)
            if len(b["opt_q"]):
                q, ok = b["opt_q"], b["opt_mask"]
                ho = h[b["opt_index"].clamp_min(0)]
                s = ho @ self.yes_no
                z[q, :ok.shape[1]] = (s + self.set(ho, hq[q], s, b["qtype"][q], ok)).masked_fill(~ok, -math.inf)
            if len(b["cnt_q"]): z[b["cnt_q"], :COUNTS] = self.count(hq[b["cnt_q"]])
            if len(b["met_q"]): z[b["met_q"], :METRIC] = self.metric(hq[b["met_q"]])
            return (z / self.temp[b["qtype"]][:, None]).masked_fill(~b["out_mask"], -math.inf)


def loss_fn(z, b, task_weight, rps_weight, smooth):
    """Soft-target cross entropy (label smoothing on yes_no / choice) + ranked probability score for counts and metric bins."""
    ok, t, qt = b["out_mask"], b["target"], b["qtype"]
    t = torch.where((qt <= TYPES["choice"])[:, None], (1 - smooth) * t + smooth * ok / ok.sum(-1, keepdim=True), t)
    logp = z.log_softmax(-1).masked_fill(~ok, 0)
    ce = -(t * logp).sum(-1)
    rps = ((logp.exp() * ok).cumsum(-1) - t.cumsum(-1)).pow(2).sum(-1) / (ok.sum(-1) - 1).clamp_min(1)
    return (task_weight[b["task"]] * (ce + rps_weight * rps * (qt >= TYPES["count"]))).mean()


def load(cfg, tok, ckpt=None, train=False):
    """bf16 base for LoRA training and for inference; fp32 base only for full fine-tuning (fp32 master weights).
    Trained parts (LoRA, mergers, new modules) are fp32 during training."""
    full = train and not cfg["net"].get("lora_rank", 0)
    base = Qwen3VLForConditionalGeneration.from_pretrained(cfg["model"], dtype=torch.float32 if full else torch.bfloat16, attn_implementation="sdpa")
    if train:
        base.model.visual.merger.float()
        base.model.visual.deepstack_merger_list.float()
    if cfg["attn"] == "flex": base.set_attn_implementation({"text_config": "decide3d_flex"})
    net = Decide3D(base, tok, cfg["net"], cfg["attn"])
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        missing, unexpected = net.load_state_dict(sd.get("model", sd), strict=False)
        assert not unexpected and all(k.startswith("base.") for k in missing), (unexpected, [k for k in missing if not k.startswith("base.")])
    return net
