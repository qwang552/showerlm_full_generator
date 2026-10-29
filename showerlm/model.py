"""ShowerLM: an autoregressive transformer over stroke tokens.

Sequence = [prefix: BOS, QTOT, NTOK, QCUM, NCUM, TSTART] + stroke tokens (+ EOS).  No positional encoding; order comes
from the token times.  Each stroke's attributes are decoded one after another by a small attribute decoder.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

PREFIX = ["bos", "qtot", "ntok", "qcum", "ncum", "tstart"]
N_PREFIX = len(PREFIX)
# SHOWERLM_ATTR_KV=1: the samplers' attribute decoder (sample.decode_one_rules, detail.decode_detail) keeps a KV
# cache and runs only the new slot per attribute instead of the whole prefix again (about 10x less work for 20 attributes).
# The same function up to float rounding (a sampled value can change where two bins are within rounding); default off.
ATTR_KV = os.environ.get("SHOWERLM_ATTR_KV", "0") == "1"
# rows per attention call of the KV path.  The CUDA attention kernel puts rows x heads on one grid axis (limit
# 65535); one call over ~8e4 rows (A40, 32 showers per detail batch) failed with "invalid configuration argument" (REAL,
# v064b ev_tz_a40kv, 10-28).  Rows are independent, so the chunks give the same values.
ATTR_KV_CHUNK = int(os.environ.get("SHOWERLM_ATTR_KV_CHUNK", "8192"))


def attr_kv_step(model, x_new, caches):
    """One new slot of the attribute decoder: x_new (N, 1, d_attr) = its input + slot embedding; caches: a dict (empty at
    the first slot) that keeps, per chunk of ATTR_KV_CHUNK rows, one KV dict per attr block (filled by Block.forward), so the
    slot attends to itself and every earlier slot, as the causal mask does."""
    N = x_new.shape[0]
    C = ATTR_KV_CHUNK if ATTR_KV_CHUNK > 0 else N
    out = []
    for c0 in range(0, N, C):
        cc = caches.setdefault(c0, [dict() for _ in model.attr_blocks])
        x = x_new[c0: c0 + C]
        for blk, c in zip(model.attr_blocks, cc):
            x = blk(x, None, c)
        out.append(x[:, 0])
    return model.attr_ln(out[0] if len(out) == 1 else torch.cat(out, 0))


def banded_causal_mask(L, window, n_prefix, device):
    """(L, L) bool, True = may attend.  Token i attends to the prefix and to j in [i - window + 1, i]."""
    i = torch.arange(L, device=device)[:, None]
    j = torch.arange(L, device=device)[None, :]
    m = (j <= i) & (j > i - window)
    m |= (j < n_prefix) & (j <= i)
    return m


def landmark_mask(L, window, n_prefix, lm):
    """(B, 1, L, L) bool: the banded causal mask plus, per sample, the landmark strokes (lm (B, L - n_prefix) bool),
    which stay visible to every later token however far back they are (0.3.0: the bright trunk never leaves the window)."""
    B = lm.shape[0]
    m = banded_causal_mask(L, window, n_prefix, lm.device)[None].expand(B, L, L).clone()
    i = torch.arange(L, device=lm.device)
    col = torch.zeros(B, L, dtype=torch.bool, device=lm.device)
    col[:, n_prefix:] = lm
    m |= col[:, None, :] & (i[None, :] <= i[:, None])[None]
    return m[:, None]


class Block(nn.Module):
    def __init__(self, d, n_heads, dropout=0.0):
        super().__init__()
        self.h = n_heads
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)
        self.dropout = dropout

    def forward(self, x, mask=None, cache=None):
        B, L, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q = q.view(B, L, self.h, -1).transpose(1, 2); k = k.view(B, L, self.h, -1).transpose(1, 2); v = v.view(B, L, self.h, -1).transpose(1, 2)
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], 2); v = torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
            # new queries see the whole cache; a first pass of several positions without a mask (the 6 prefix positions
            # in sample.generate) is causal, as in training (0.5 fix: 0.4.0 let the prefix positions see each other both ways)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None and k.shape[2] == L and L > 1)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).reshape(B, L, d)
        x = x + self.drop(self.proj(y))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class ShowerLM(nn.Module):
    def __init__(self, vocab_sizes, attr_names, n_t, n_qtot=64, n_ntok=64, n_cum=33, d=512, n_layers=8, n_heads=8,
                 d_attr=256, n_attr_layers=2, n_attr_heads=4, window=2048, w_ptr=2048, dropout=0.0, attr_order=None, causal_ptr=False,
                 budget=False, n_bud=40, landmarks=False, t_window=0, t_rule_table=None, q_weight_pow=0.0, t_rel=False, n_gfeat=0, n_gbins=32,
                 n_zslice=0, n_zbins=48, n_tslice=0, n_tbins=48):
        super().__init__()
        self.sizes = list(vocab_sizes); self.names = list(attr_names); self.A = len(self.sizes)
        self.ai = {n: i for i, n in enumerate(self.names)}
        # options (all off = 0.2.0 behaviour)
        self.cont = self.sizes[0] >= 4                                            # vocabulary has kind CONT (and attribute lt)
        self.budget, self.landmarks, self.q_weight_pow = bool(budget), bool(landmarks), float(q_weight_pow)
        # 0.5: exact rules applied in the loss and in sampling (showerlm.rules.Rules, set by train.build_model; None = off)
        # and t_rel: the t head predicts the offset above the rule floor (previous chain head / previous detail stroke)
        self.rules, self.t_rel = None, bool(t_rel)
        # CONT time window: int w -> [rule - w, rule + w]; pair [lo, hi] -> [rule - lo, rule + hi] (the data tail is on the late side)
        lo, hi = (int(t_window), int(t_window)) if isinstance(t_window, (int, float)) else (int(t_window[0]), int(t_window[1]))
        self.t_win_lo, self.t_win_hi = max(lo, 0), max(hi, 0)
        self.t_window = max(self.t_win_lo, self.t_win_hi)                        # > 0: the window is applied
        if self.budget:
            self.bud_n = nn.Embedding(n_bud, d); self.bud_q = nn.Embedding(n_bud, d)
        # CONT time rule: t of a CONT = t of the previous token + light travel of the previous piece (table over len bins)
        self.register_buffer("t_rule", torch.as_tensor(list(t_rule_table) if t_rule_table is not None else [0] * self.sizes[self.ai["len"]], dtype=torch.long), persistent=False)
        # decoding order of the attributes inside one stroke (default: vocabulary order).  order[p] = attribute decoded p-th.
        names = list(attr_order) if attr_order else list(self.names)
        assert sorted(names) == sorted(self.names) and names[0] == "kind", "attr_order must be a permutation of the attributes starting with kind"
        self.order = [self.ai[n] for n in names]
        self.causal_ptr = bool(causal_ptr)                                        # parents later than the child are masked
        self.d, self.window, self.w_ptr = d, window, w_ptr
        self.prefix_sizes = dict(bos=1, qtot=n_qtot, ntok=n_ntok, qcum=n_cum, ncum=n_cum, tstart=n_t)
        self.emb = nn.ModuleList([nn.Embedding(n, d) for n in self.sizes])
        self.prefix_emb = nn.ModuleDict({k: nn.Embedding(n, d) for k, n in self.prefix_sizes.items()})
        self.type_emb = nn.Embedding(2, d)
        # gplan: physical global variables G of the shower (showerlm.gfeat), one bin each, carried in the prefix
        # tensor after the 6 prefix tokens; their summed embedding is added to every position (zero-initialised below, so a
        # model loaded from a checkpoint without G starts out exactly as that checkpoint)
        self.n_gfeat, self.n_gbins = int(n_gfeat), int(n_gbins)
        if self.n_gfeat:
            self.g_emb = nn.ModuleList([nn.Embedding(self.n_gbins, d) for _ in range(self.n_gfeat)])
        # zplan: skeleton light per depth slice (showerlm.zplan).  The planned light of every slice rides in the prefix
        # tensor after G (zp_emb, added like G to every position); the light still to come in every slice after a token rides
        # in the budget tensor after the two 0.3 budgets (zr_emb, added to that token).  Both zero-initialised.
        self.n_zslice, self.n_zbins = int(n_zslice), int(n_zbins)
        if self.n_zslice:
            self.zp_emb = nn.ModuleList([nn.Embedding(self.n_zbins, d) for _ in range(self.n_zslice)])
            self.zr_emb = nn.ModuleList([nn.Embedding(self.n_zbins, d) for _ in range(self.n_zslice)])
        # tplan: skeleton light per lateness slice (showerlm.tplan), the same mechanism as zplan: the planned light per
        # slice rides in the prefix after the zplan columns (tp_emb), the light still to come per slice after a token in
        # the budget tensor after the zplan columns (tr_emb).  Both zero-initialised (a 0.6.1z checkpoint loads unchanged).
        self.n_tslice, self.n_tbins = int(n_tslice), int(n_tbins)
        if self.n_tslice:
            self.tp_emb = nn.ModuleList([nn.Embedding(self.n_tbins, d) for _ in range(self.n_tslice)])
            self.tr_emb = nn.ModuleList([nn.Embedding(self.n_tbins, d) for _ in range(self.n_tslice)])
        self.blocks = nn.ModuleList([Block(d, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head_qtot = nn.Linear(d, n_qtot); self.head_ntok = nn.Linear(d, n_ntok)
        # attribute decoder
        self.h_proj = nn.Linear(d, d_attr)
        self.attr_in = nn.ModuleList([nn.Embedding(n, d_attr) for n in self.sizes])
        self.slot = nn.Embedding(self.A, d_attr)
        self.attr_blocks = nn.ModuleList([Block(d_attr, n_attr_heads, dropout) for _ in range(n_attr_layers)])
        self.attr_ln = nn.LayerNorm(d_attr)
        self.heads = nn.ModuleList([nn.Linear(d_attr, n) for n in self.sizes])
        self.ptr_q = nn.Linear(d_attr, d); self.ptr_k = nn.Linear(d, d)
        self.ptr_none = nn.Parameter(torch.zeros(d))
        self.apply(self._init)
        if self.n_gfeat:
            for e in self.g_emb:
                nn.init.zeros_(e.weight)
        if self.n_zslice:
            for e in list(self.zp_emb) + list(self.zr_emb):
                nn.init.zeros_(e.weight)
        if self.n_tslice:
            for e in list(self.tp_emb) + list(self.tr_emb):
                nn.init.zeros_(e.weight)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    # ---- embeddings
    def embed_strokes(self, tok, bud=None, g=None):
        """tok (B, L, A) int -> (B, L, d); bud (B, L, 2) budget bins (strokes / photons still to come) when the model uses them;
        g (B, d) the embedding of the global variables (0.6.1g, embed_global) or None.
        bud (B, L, 2 + n_zslice), columns 2.. = bins of the light still to come per depth slice"""
        x = sum(self.emb[a](tok[..., a]) for a in range(self.A))
        if self.budget and bud is not None:
            x = x + self.bud_n(bud[..., 0]) + self.bud_q(bud[..., 1])
            if self.n_zslice and bud.shape[-1] >= 2 + self.n_zslice:
                zb = bud[..., 2: 2 + self.n_zslice].clamp(0, self.n_zbins - 1)
                x = x + sum(self.zr_emb[k](zb[..., k]) for k in range(self.n_zslice))
            o = 2 + self.n_zslice
            if self.n_tslice and bud.shape[-1] >= o + self.n_tslice:                 # light still to come per lateness slice
                tb = bud[..., o: o + self.n_tslice].clamp(0, self.n_tbins - 1)
                x = x + sum(self.tr_emb[k](tb[..., k]) for k in range(self.n_tslice))
        if g is not None:
            x = x + g[:, None, :]
        return x + self.type_emb.weight[1]

    def embed_global(self, prefix):
        """prefix (B, 6 + n_gfeat) -> (B, d) summed embeddings of the G bins; None without gplan (or a 6-column prefix).
        prefix (B, 6 + n_gfeat + n_zslice), the planned light per depth slice after G, its embeddings added too."""
        out = None
        if self.n_gfeat and prefix.shape[1] >= N_PREFIX + self.n_gfeat:
            g = prefix[:, N_PREFIX: N_PREFIX + self.n_gfeat].clamp(0, self.n_gbins - 1)
            out = sum(self.g_emb[k](g[:, k]) for k in range(self.n_gfeat))
        o = N_PREFIX + self.n_gfeat
        if self.n_zslice and prefix.shape[1] >= o + self.n_zslice:
            z = prefix[:, o: o + self.n_zslice].clamp(0, self.n_zbins - 1)
            zs = sum(self.zp_emb[k](z[:, k]) for k in range(self.n_zslice))
            out = zs if out is None else out + zs
        o += self.n_zslice
        if self.n_tslice and prefix.shape[1] >= o + self.n_tslice:                # planned light per lateness slice
            tt = prefix[:, o: o + self.n_tslice].clamp(0, self.n_tbins - 1)
            ts = sum(self.tp_emb[k](tt[:, k]) for k in range(self.n_tslice))
            out = ts if out is None else out + ts
        return out

    def cont_t_mask(self, t_logits, t_rule):
        """Restrict the t logits of CONT targets to [rule - t_win_lo, rule + t_win_hi]: t_logits (N, n_t), t_rule (N,)."""
        b = torch.arange(t_logits.shape[1], device=t_logits.device)[None, :]
        d = b - t_rule[:, None]
        inside = (d >= -self.t_win_lo) & (d <= self.t_win_hi)
        inside |= b == t_rule.clamp(0, t_logits.shape[1] - 1)[:, None]            # the rule bin itself is always allowed
        return t_logits.masked_fill(~inside, float("-inf"))

    def cont_t_target(self, t_target, t_rule):
        """Training target of a CONT's t, clipped into the window (a residual outside it, ~2% of the pieces, is clipped)."""
        return torch.minimum(torch.maximum(t_target, t_rule - self.t_win_lo), t_rule + self.t_win_hi).clamp(0, self.sizes[self.ai["t"]] - 1)

    def rule_t(self, prev_tok):
        """t bin of a CONT that follows prev_tok (..., A): previous t + light travel of the previous piece."""
        return prev_tok[..., self.ai["t"]] + self.t_rule[prev_tok[..., self.ai["len"]]]

    def embed_prefix(self, prefix):
        """prefix (B, 6) int -> (B, 6, d) (0.6.1g: (B, 6 + n_gfeat), the G embedding added to the 6 positions)"""
        xs = [self.prefix_emb[k](prefix[:, i]) for i, k in enumerate(PREFIX)]
        x = torch.stack(xs, 1) + self.type_emb.weight[0]
        ge = self.embed_global(prefix)
        return x if ge is None else x + ge[:, None, :]

    def trunk(self, x, mask=None, caches=None):
        for i, blk in enumerate(self.blocks):
            x = blk(x, mask, None if caches is None else caches[i])
        return self.ln_f(x)

    # ---- attribute decoder (teacher forced): h (N, d), target (N, A) -> list of logits per attribute (N, n_a)
    def attr_logits(self, h, target):
        N = h.shape[0]
        x0 = self.h_proj(h)[:, None, :]                                           # slot 0 sees only the context
        xs = [self.attr_in[a](target[:, a])[:, None, :] for a in self.order[:-1]]  # slot p sees the attributes decoded before p
        x = torch.cat([x0] + xs, 1) + self.slot.weight[None, : self.A, :]
        for blk in self.attr_blocks:
            x = blk(x, None)
        x = self.attr_ln(x)
        x = x[:, torch.as_tensor([self.order.index(a) for a in range(self.A)], device=x.device)]  # back to vocabulary order
        return x, [self.heads[a](x[:, a]) for a in range(self.A)]

    def pointer_logits(self, y_ptr, keys, i_idx, j_valid):
        """y_ptr (B, L, d_attr) decoder state at the ptr slot for target strokes 0..L-1; keys (B, L, d) main states of
        strokes 0..L-1.  Target i may point to j < i with i - j <= w_ptr.  Returns logits (B, L, 1 + L): [none, j...]."""
        q = self.ptr_q(y_ptr)
        s = torch.einsum("bid,bjd->bij", q, self.ptr_k(keys)) / math.sqrt(self.d)
        none = (q @ self.ptr_none)[..., None] / math.sqrt(self.d)
        s = s.masked_fill(~(j_valid if j_valid.dim() == 3 else j_valid[None]), float("-inf"))
        return torch.cat([none, s], -1)

    # ---- training forward
    def forward(self, prefix, tok_in, tok_out, bud=None, lm=None):
        """prefix (B, 6), tok_in (B, n, A) strokes fed as input, tok_out (B, n + 1, A) targets for the outputs at
        positions 5 .. 5 + n (the last one is EOS or the next stroke); bud (B, n, 2) budget bins and lm (B, n) landmark
        flags of the inputs (0.3.0, optional).  Returns dict of mean losses and counts."""
        B, n, A = tok_in.shape
        x = torch.cat([self.embed_prefix(prefix), self.embed_strokes(tok_in, bud, self.embed_global(prefix))], 1)
        L = x.shape[1]
        if self.landmarks and lm is not None:
            mask = landmark_mask(L, self.window, N_PREFIX, lm)
        elif L <= self.window:
            mask = None                                                             # 0.6: the band covers the chunk = plain causal (lets SDPA use a fused kernel)
        else:
            mask = banded_causal_mask(L, self.window, N_PREFIX, x.device)
        h = self.trunk(x, mask)
        out = {}
        out["loss_qtot"] = F.cross_entropy(self.head_qtot(h[:, 0]), prefix[:, 1])
        out["loss_ntok"] = F.cross_entropy(self.head_ntok(h[:, 1]), prefix[:, 2])
        ctx = h[:, N_PREFIX - 1: N_PREFIX + n]                                      # (B, n + 1, d) states that predict tok_out
        kind = tok_out[..., self.ai["kind"]]
        valid = kind > 0                                                            # not PAD
        is_stroke = (kind == 1) | (kind == 3)
        is_cont = kind == 3
        hc = ctx.reshape(B * (n + 1), -1); tg = tok_out.reshape(B * (n + 1), A)
        y, logits = self.attr_logits(hc, tg)
        v = valid.reshape(-1); st = is_stroke.reshape(-1); co = is_cont.reshape(-1)
        has_ptr = (tg[:, self.ai["ptr"]] > 0) & st
        # photon weight of every target stroke (0.3.0, q_weight_pow > 0): bright strokes cost more
        w = None
        if self.q_weight_pow > 0:
            lq = tg[:, self.ai["lq"]].float()
            q = torch.pow(10.0, (lq + 0.5) * 7.0 / self.sizes[self.ai["lq"]])      # matches Vocab.dec_log with q_log_max 7
            w = torch.pow(q, self.q_weight_pow)
            w = w / w[st].mean().clamp_min(1e-9) if st.any() else w
        # 0.5 rules: the same masks as the sampler, on the teacher-forced inputs (rows that break one are counted)
        R = self.rules if (self.rules is not None and self.rules.any) else None
        viol = {}
        if R is not None:
            from .rules import constrain, train_context
            R.to(x.device)
            rctx = train_context(self, tok_in)
        # CONT time rule: the t of a CONT target at position i follows the input token at position i - 1
        t_rule = None
        if self.cont and self.t_window > 0 and R is None:                                        # 0.3 / 0.4 only: with model.rules the CONT window is rules.cont_window, in training and sampling alike
            prev = torch.cat([tok_in[:, :1], tok_in[:, :-1]], 1) if n > 0 else tok_in         # position 0 has no previous token (never CONT)
            t_rule = torch.cat([self.rule_t(prev), self.rule_t(tok_in[:, -1:])], 1).reshape(-1) if n > 0 else torch.zeros(B, dtype=torch.long, device=x.device)
        losses = {}
        for a, name in enumerate(self.names):
            if name == "ptr":
                continue
            if name == "kind":
                m = v
            elif name == "s":
                m = has_ptr & ~co
            elif name in ("dx", "dy", "dz"):
                m = has_ptr
            elif name in ("ix", "iy", "iz"):
                m = st & ~has_ptr
            else:
                m = st
            if not m.any():
                continue
            lg = logits[a].float(); tga = tg[:, a]
            if R is not None:
                lg = constrain(R, self, name, lg, tg, rctx, target=tga, count=viol, active=m)
            if name == "t" and t_rule is not None and co.any():
                lg = torch.where(co[:, None], self.cont_t_mask(lg, t_rule), lg)
                tga = torch.where(co, self.cont_t_target(tga, t_rule), tga)
            if w is not None and name in ("kind", "th", "ph", "len", "lq", "lc", "coh", "lt"):
                ce = F.cross_entropy(lg[m], tga[m], reduction="none")
                losses[name] = (ce * w[m]).sum() / w[m].sum().clamp_min(1e-9)
            else:
                losses[name] = F.cross_entropy(lg[m], tga[m])
        # pointer: keys = main states of input strokes (positions 6 .. 5 + n) = strokes 0 .. n-1; target i (0..n) may point to j < i
        keys = h[:, N_PREFIX: N_PREFIX + n]                                        # (B, n, d)
        yp = y[:, self.ai["ptr"]].reshape(B, n + 1, -1)
        i_idx = torch.arange(n + 1, device=x.device)[:, None]; j_idx = torch.arange(n, device=x.device)[None, :]
        j_valid = (j_idx < i_idx) & (i_idx - j_idx <= self.w_ptr)
        if self.causal_ptr:                                                         # t of the parent <= t of the child
            it = self.ai["t"]
            j_valid = j_valid[None] & (tok_in[:, None, :, it] <= tok_out[:, :, None, it])
        pl = self.pointer_logits(yp, keys, i_idx, j_valid).float()                 # (B, n + 1, 1 + n)
        ptr = tok_out[..., self.ai["ptr"]]
        tgt = torch.where(ptr > 0, i_idx[None, :, 0] - ptr + 1, torch.zeros_like(ptr))
        m = is_stroke & ~is_cont                                                   # a CONT's parent is the previous token by rule
        if m.any():
            losses["ptr"] = F.cross_entropy(pl[m], tgt[m])
        out.update({f"loss_{k}": val for k, val in losses.items()})
        out["loss"] = out["loss_qtot"] + out["loss_ntok"] + sum(losses.values())
        out["n_tokens"] = int(v.sum())
        out.update({f"viol_{k}": c for k, c in viol.items()})
        return out

    # ---- sampling helpers
    @torch.no_grad()
    def decode_attributes(self, h, fixed, temperature=1.0, keys=None, ptr_mask=None, rng=None):
        """Sample the attributes of one stroke for a batch.  h (B, d) context; fixed: dict attr_index -> (B,) tensor of
        values that are set instead of sampled (filled as the loop reaches them; a callable gets the partial token).
        keys (B, M, d): main states of the previous strokes (for the pointer, most recent last); ptr_mask (B, M) bool valid.
        Returns tok (B, A) long."""
        B = h.shape[0]
        tok = torch.zeros(B, self.A, dtype=torch.long, device=h.device)
        x0 = self.h_proj(h)[:, None, :]
        for a in range(self.A):
            xs = [x0] + [self.attr_in[b](tok[:, b])[:, None, :] for b in range(a)]
            x = torch.cat(xs, 1) + self.slot.weight[None, : a + 1, :]
            for blk in self.attr_blocks:
                x = blk(x, None)
            y = self.attr_ln(x[:, a])
            name = self.names[a]
            if name in fixed and not callable(fixed[name]):
                tok[:, a] = fixed[name]; continue
            if name in fixed:
                val = fixed[name](tok)
                if val is not None:
                    tok[:, a] = val; continue
            if name == "ptr":
                if keys is None or keys.shape[1] == 0:
                    tok[:, a] = 0; continue
                q = self.ptr_q(y)
                s = torch.einsum("bd,bjd->bj", q, self.ptr_k(keys)) / math.sqrt(self.d)
                s = s.masked_fill(~ptr_mask, float("-inf"))
                none = (q @ self.ptr_none)[:, None] / math.sqrt(self.d)
                logits = torch.cat([none, s], 1).float()
                c = sample_logits(logits, temperature, rng)                        # 0 = none, else j + 1 with j most recent last
                M = keys.shape[1]
                tok[:, a] = torch.where(c > 0, M - (c - 1), torch.zeros_like(c))  # distance back: most recent = 1
            else:
                logits = self.heads[a](y).float()
                tok[:, a] = sample_logits(logits, temperature, rng)
        return tok


def sample_logits(logits, temperature=1.0, rng=None, top_p=None):
    if temperature <= 0:
        return logits.argmax(-1)
    p = F.softmax(logits / temperature, -1)
    if top_p is not None and top_p < 1.0:
        sp, si = torch.sort(p, descending=True, dim=-1)
        cum = torch.cumsum(sp, -1)
        keep = cum - sp < top_p
        sp = sp * keep
        p = torch.zeros_like(p).scatter_(-1, si, sp)
        p = p / p.sum(-1, keepdim=True)
    return torch.multinomial(p, 1, generator=rng).squeeze(-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
