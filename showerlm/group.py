"""ShowerLM 0.6 group detail model: the detail strokes of a full shower, grouped by the skeleton piece they belong to.
All groups are generated together, one stroke per group per step.
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .detail import TYPE_PREFIX, DetailLM, decode_detail
from .data import TokenStore, budget_bins
from .tokens import KIND_EOS, KIND_PAD, KIND_STROKE

GROUP_MODES = ("join",)
GROUP_ORDERS = ("time", "along")                  # stroke order inside a group (along)

# group plan (model.group_plan, 0.6.0.dev2): strokes and photons of a group, predicted at its GS position
GN_EXACT, GN_BINS, GN_MAX = 16, 64, 8192         # strokes: bins 0..15 exact, then 48 log bins from 16 to 8192 (x1.14 each)
GQ_BINS, GQ_LOG_MAX = 128, 8.0                   # photons: bin 0 = none, then 127 log10 bins over (1, 1e8] (x1.16 each)


def enc_gn(n):
    n = np.asarray(n, dtype=np.float64)
    lg = GN_EXACT + np.floor(np.log(np.maximum(n, GN_EXACT) / GN_EXACT) / np.log(GN_MAX / GN_EXACT) * (GN_BINS - GN_EXACT))
    return np.clip(np.where(n < GN_EXACT, n, lg), 0, GN_BINS - 1).astype(np.int64)


def dec_gn(b):
    """bin -> strokes (exact below 16; geometric centre of the bin above)"""
    b = np.asarray(b, dtype=np.float64)
    w = np.log(GN_MAX / GN_EXACT) / (GN_BINS - GN_EXACT)
    return np.where(b < GN_EXACT, b, GN_EXACT * np.exp((b - GN_EXACT + 0.5) * w))


def enc_gq(q):
    q = np.asarray(q, dtype=np.float64)
    lb = 1 + np.floor(np.log10(np.maximum(q, 1.0)) / GQ_LOG_MAX * (GQ_BINS - 1))
    return np.clip(np.where(q > 0, lb, 0), 0, GQ_BINS - 1).astype(np.int64)


def dec_gq(b):
    b = np.asarray(b, dtype=np.float64)
    return np.where(b > 0, 10.0 ** ((b - 0.5) * GQ_LOG_MAX / (GQ_BINS - 1)), 0.0)


def stroke_photons(vocab, tok):
    """photons of each stroke read from its lq bin (what the plan and the budgets count)"""
    tok = np.asarray(tok).reshape(-1, len(vocab.names))
    return vocab.dec_log(tok[:, vocab.index["lq"]], vocab.n_q, vocab.q_log_max) if len(tok) else np.zeros(0)


# ---------------------------------------------------------------- data
class GroupStore(TokenStore):
    """tokens_det_* folder written by the 0.6 tools/extract_skeleton.py (every file has `grp`).  Events without a
    skeleton are left out (a group needs a piece)."""

    def __init__(self, root, max_events=0, exclude_eids=None):
        super().__init__(root, max_events, exclude_eids)
        index = json.load(open(self.root / "index.json"))
        if all("n_skel" in index[f.name] for f in self.files):                      # 0.6 index (extract_skeleton finish)
            n_skel = np.array([index[f.name]["n_skel"] for f in self.files])
        else:
            n_skel = np.array([np.load(f)["skel"].shape[0] for f in self.files])
        keep = n_skel > 0
        self.no_skeleton = [int(e) for e in self.eids[~keep]]
        self.files = [f for f, k in zip(self.files, keep) if k]
        self.n_tok, self.q_total, self.eids, self.n_skel = self.n_tok[keep], self.q_total[keep], self.eids[keep], n_skel[keep]
        if not self.files:
            raise SystemExit(f"{root}: no event with a skeleton")
        with np.load(self.files[0]) as d:
            if "grp" not in d.files:
                raise SystemExit(f"{self.files[0]} has no 'grp': split the tokens with the 0.6 tools/extract_skeleton.py")

    def load(self, i):
        with np.load(self.files[i]) as d:
            return d["tok"].astype(np.int64), d["skel"].astype(np.int64), d["grp"].astype(np.int64)


def along_key(vocab, skel, tok, grp):
    """group_order along: position of every stroke's head point along its group's piece (projection on the
    segment, clipped to [0, L], voxel units; the arithmetic of tools/extract_skeleton.group_of)."""
    ix = vocab.index
    if len(tok) == 0:
        return np.zeros(0)
    k = np.clip(np.asarray(grp, dtype=np.int64), 0, max(len(skel) - 1, 0))
    P = vocab.head_point(skel[k])
    u = vocab.dec_dir(skel[k, ix["th"]], skel[k, ix["ph"]])
    L = vocab.dec_len(skel[k, ix["len"]]).astype(np.float64)
    return np.clip(np.einsum("nc,nc->n", vocab.head_point(tok) - P, u), 0.0, L)


def event_groups(tok, skel, grp, vocab, mode="join", order="time"):
    """Groups of one event: list of (piece k, 0, stroke indices in the group's order), every piece included (empty
    groups too), in piece order.  tok (N, A) detail strokes (time order), skel (K, A), grp (N,).
    order "time" (0.6): time order inside a group (the order of tok); "along" (0.6.2): by the position of the head along
    the group's piece (along_key), ties in time order."""
    if mode not in GROUP_MODES:
        raise ValueError(f"group mode {mode} not in {GROUP_MODES}")
    if order not in GROUP_ORDERS:
        raise ValueError(f"group order {order} not in {GROUP_ORDERS}")
    K = len(skel)
    if order == "along":
        o = np.lexsort((np.arange(len(grp)), along_key(vocab, skel, tok, grp), grp))  # piece, position along it, time order
    else:
        o = np.argsort(grp, kind="stable")                                         # stable: time order inside a group
    bounds = np.searchsorted(grp[o], np.arange(K + 1))
    return [(k, 0, o[bounds[k]:bounds[k + 1]]) for k in range(K)]


def group_ranks(groups, n):
    """rank (n,) of every detail stroke inside its group = the sampler step that decodes it."""
    r = np.zeros(n, np.int64)
    for _, _, idx in groups:
        r[idx] = np.arange(len(idx))
    return r


def pack_item(tok, skel, groups, sel, P, A, q_tok=None):
    """One training item from the groups `sel` (indices into groups) of an event: positions [GS, s1 .. sm] per group, at
    most P positions (a group longer than P - 1 strokes is cut: its first P - 1 strokes, no EOS target)."""
    if q_tok is None:
        q_tok = np.zeros(len(tok))
    plan_n = np.full(P, -1, np.int64); plan_q = np.full(P, -1, np.int64); bud = np.zeros((P, 2), np.int64)
    tok_in = np.zeros((P, A), np.int64)
    tok_out = np.full((P, A), KIND_PAD, np.int64)
    seg = np.full(P, -1, np.int64); piece = np.zeros(P, np.int64); far = np.zeros(P, np.int64); gs = np.zeros(P, bool)
    oj = np.zeros(P, np.int64)                                                     # rank of the target in its group
    p = 0
    for j, g in enumerate(sel):
        k, f, idx = groups[g]
        m = len(idx)
        n = min(m + 1, P - p)                                                      # positions of this group
        if n <= 0:
            break
        s = tok[idx[: n]]                                                          # targets s1 .. s_n (or fewer + EOS)
        gs[p] = True
        tok_in[p + 1: p + n] = tok[idx[: n - 1]]
        nt = min(n, m)
        tok_out[p: p + nt] = s[:nt]
        if n == m + 1:
            tok_out[p + m, 0] = KIND_EOS
        seg[p: p + n] = j; piece[p: p + n] = k; far[p: p + n] = f
        oj[p: p + n] = np.arange(n)
        bn, bq = int(enc_gn(m)), int(enc_gq(q_tok[idx].sum()))
        plan_n[p], plan_q[p] = bn, bq
        if n > 1:                                                                  # inputs s1 .. s_{n-1}: budget after them
            i = np.arange(1, n)
            bud[p + 1: p + n] = budget_bins(dec_gn(bn) - i, dec_gq(bq) - np.cumsum(q_tok[idx[: n - 1]]))
        p += n
    return dict(skel=torch.from_numpy(skel), tok_in=torch.from_numpy(tok_in), tok_out=torch.from_numpy(tok_out), seg=torch.from_numpy(seg),
                piece=torch.from_numpy(piece), far=torch.from_numpy(far), gs=torch.from_numpy(gs), bud=torch.from_numpy(bud),
                plan_n=torch.from_numpy(plan_n), plan_q=torch.from_numpy(plan_q), oj=torch.from_numpy(oj))


ITEM_KEYS = ("skel", "tok_in", "tok_out", "seg", "piece", "far", "gs", "bud", "plan_n", "plan_q")
OCC_KEYS = ("oj", "okeys", "ostep")                # occupancy (model.occ_feat): per position rank, then the event's occupancy
N_POS_KEYS = 11                                    # batch entries 1 .. 10 are per position (trim_group cuts them)


def pack_groups(groups, P, order):
    """Split the groups (in `order`) into consecutive packs of at most P positions (m + 1 per group; a group longer
    than P - 1 strokes forms a pack of its own and is cut)."""
    packs, cur, used = [], [], 0
    for g in order:
        n = min(len(groups[g][2]) + 1, P)
        if cur and used + n > P:
            packs.append(cur); cur, used = [], 0
        cur.append(g); used += n
    if cur:
        packs.append(cur)
    return packs


class GroupPackDataset(Dataset):
    """item = one pack of groups of one random event (pack_item); the packs of an event shuffle the group order.
    batch size 1 (collate_group): every item has its own skeleton.  item(e, j) (tools/audit_rules.py) = pack j of event
    e with the groups in piece order."""

    def __init__(self, store: GroupStore, P, n_items, seed=0, fixed=False, mode="join", k_max=4096, order="time", occ=None):
        if mode not in GROUP_MODES:
            raise ValueError(f"group mode {mode} not in {GROUP_MODES}")
        self.store, self.P, self.n_items, self.seed, self.fixed, self.mode, self.k_max = store, int(P), int(n_items), int(seed), fixed, mode, int(k_max)
        self.A = len(store.vocab.names)
        self.order, self.occ = order, occ                                           # in-group order; occupancy config (None = off)

    def __len__(self):
        return self.n_items

    def groups(self, e):
        tok, skel, grp = self.store.load(e)
        if len(skel) == 0:
            raise ValueError(f"{self.store.files[e]}: no skeleton")
        if len(skel) > self.k_max:
            raise ValueError(f"{self.store.files[e]}: skeleton of {len(skel)} tokens > k_max {self.k_max}")
        return tok, skel, event_groups(tok, skel, grp, self.store.vocab, self.mode, self.order)

    def with_occ(self, item, tok, skel, groups):
        """add the event's occupancy timeline (all groups, ranks in the model's order) to a pack item."""
        if self.occ is None:
            return item
        from .occ import event_occupancy
        keys, steps = event_occupancy(self.store.vocab, skel, tok, group_ranks(groups, len(tok)), self.occ["ranges"], self.occ["w_occ"])
        item["okeys"], item["ostep"] = torch.from_numpy(keys), torch.from_numpy(steps)
        return item

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i] if self.fixed else None)
        e = int(rng.integers(len(self.store)))
        tok, skel, groups = self.groups(e)
        packs = pack_groups(groups, self.P, rng.permutation(len(groups)))
        item = pack_item(tok, skel, groups, packs[int(rng.integers(len(packs)))], self.P, self.A, stroke_photons(self.store.vocab, tok))
        return self.with_occ(item, tok, skel, groups)

    def packs_of(self, e):
        """Every pack of event e, groups in piece order (audit: each group exactly once)."""
        tok, skel, groups = self.groups(e)
        q = stroke_photons(self.store.vocab, tok)
        return [self.with_occ(pack_item(tok, skel, groups, pk, self.P, self.A, q), tok, skel, groups) for pk in pack_groups(groups, self.P, range(len(groups)))]


def collate_group(items):
    if len(items) != 1:
        raise ValueError("group detail model: batch size must be 1 (use train.accum for larger batches)")
    it = items[0]
    return [it[k][None] for k in ITEM_KEYS + (OCC_KEYS if "okeys" in it else ())]


def trim_group(batch):
    """Cut the all-padding tail of the positions (seg < 0); the occupancy entries (0.6.2) are not per position."""
    seg = batch[3]
    n = int((seg[0] >= 0).sum())
    n = max(n, 1)
    return [batch[0]] + [x[:, :n] for x in batch[1:N_POS_KEYS]] + list(batch[N_POS_KEYS:])


# ---------------------------------------------------------------- model
def group_mask(K, seg):
    """(1, 1, P, K + P) bool: every position sees the whole skeleton; a group position sees its own group's positions
    up to itself; a padding position (seg < 0) sees only itself."""
    P = seg.shape[1]
    s = seg[0]
    i = torch.arange(P, device=seg.device)
    own = (s[:, None] == s[None, :]) & (i[None, :] <= i[:, None]) & (s[:, None] >= 0)
    own |= (s[:, None] < 0) & (i[:, None] == i[None, :])
    sk = torch.ones(P, K, dtype=torch.bool, device=seg.device)
    return torch.cat([sk, own], 1)[None, None]


def plan_kind_allowed(n_kind, gs, plan_n):
    """(N, n_kind) bool, group plan: the first target of a group (at its GS position) is EOS if the plan says 0 strokes
    and a stroke otherwise; later targets are not restricted here."""
    ok = torch.ones(gs.shape[0], n_kind, dtype=torch.bool, device=gs.device)
    empty = plan_n == 0
    ok[:, KIND_EOS] = ~gs | empty
    ok[:, KIND_STROKE] = ~gs | ~empty
    return ok


def ptr_allowed(K, piece, far, mode):
    """(N, K + 1) bool: the pointer bins a group's stroke may take: 'none' and its own piece."""
    N = piece.shape[0]
    ok = torch.zeros(N, K + 1, dtype=torch.bool, device=piece.device)
    ok[:, 0] = True
    ok[torch.arange(N, device=piece.device), piece.clamp(0, K - 1) + 1] = True
    return ok


class GroupDetailLM(DetailLM):
    def __init__(self, *args, group_mode="join", group_plan=False, occ_feat=None, pos_first=False, group_order="time", **kw):
        super().__init__(*args, **kw)
        if group_mode not in GROUP_MODES:
            raise ValueError(f"model.group_mode {group_mode} not in {GROUP_MODES}")
        if group_order not in GROUP_ORDERS:
            raise ValueError(f"model.group_order {group_order} not in {GROUP_ORDERS}")
        self.group_mode = group_mode
        self.group_plan = bool(group_plan)
        # occupancy / geometry features of the head attributes (showerlm/occ.py; None = off), head position decoded
        # before t (pos_first: t floor = the light cone at the head voxel), stroke order inside a group (time | along)
        self.pos_first, self.group_order = bool(pos_first), group_order
        self.occ_cfg, self.occ = None, None
        if occ_feat:
            from .occ import OccHeads, default_cfg
            self.occ_cfg = {**default_cfg(), **(occ_feat if isinstance(occ_feat, dict) else {})}
            self.occ = OccHeads(self.heads[0].in_features, int(self.occ_cfg["rank"]))
        if self.pos_first:
            o = [self.names[a] for a in self.order]
            if not (o.index("t") > max(o.index(c) for c in ("ptr", "s", "dx", "dy", "dz", "ix", "iy", "iz"))):
                raise ValueError(f"model.pos_first: attr_order must decode t after ptr, s, dx, dy, dz, ix, iy, iz (got {o})")
        if self.group_plan and not self.budget:
            raise ValueError("model.group_plan needs model.budget: true (the strokes see the budget left in their group)")
        self.gs_proj = nn.Linear(self.d, self.d)                                    # input embedding of the group's piece -> GS
        self.gs_far = nn.Embedding(2, self.d)
        mods = [self.gs_proj, self.gs_far]
        if self.group_plan:
            # plan of a group, read at its GS position: P(strokes) P(photons | strokes)
            self.head_gn = nn.Linear(self.d, GN_BINS)
            self.gq_cond = nn.Embedding(GN_BINS, self.d)
            self.head_gq = nn.Linear(self.d, GQ_BINS)
            mods += [self.head_gn, self.gq_cond, self.head_gq]
        for mod in mods:
            mod.apply(self._init)

    def plan_logits(self, h, plan_n):
        """h (N, d) outputs at GS positions, plan_n (N,) strokes bins -> logits of the strokes bin and of the photons bin"""
        return self.head_gn(h).float(), self.head_gq(h + self.gq_cond(plan_n.clamp(0, GN_BINS - 1))).float()

    def embed_gs(self, e_skel, piece, far):
        """GS embeddings: e_skel (1, K, d), piece / far (...) -> (..., d)"""
        K = e_skel.shape[1]
        par = e_skel[0, piece.clamp(0, K - 1)]
        return self.gs_proj(par) + self.gs_far(far) + self.type_emb.weight[TYPE_PREFIX]

    def skeleton_pass(self, skel):
        """skel (1, K, A) -> e_skel, h_skel (1, K, d) and the per-layer keys / values of the skeleton block."""
        e_skel = self.embed_skel(skel)
        caches = [dict() for _ in self.blocks]
        h_skel = self.trunk(e_skel, None, caches)                                  # causal (Block: no mask + cache = is_causal)
        return e_skel, h_skel, caches

    def occ_train_context(self, skel, piece, oj, okeys, ostep):
        """feature context of a pack (showerlm/occ.py): the event's occupancy, rank and group piece per position."""
        from .occ import occ_context
        oc = occ_context(self._vocab_occ, skel[0].cpu().numpy(), np.zeros(0, np.int64), np.zeros((0, 3), np.int64), oj[0], piece[0], skel.device)
        oc["okeys"], oc["ostep"] = okeys[0], ostep[0].to(torch.int32)
        return oc

    def _terms(self, skel, tok_in, tok_out, seg, piece, far, gs, bud=None, plan_n=None, plan_q=None, oj=None, okeys=None, ostep=None):
        if skel.shape[0] != 1:
            raise ValueError("GroupDetailLM: batch size 1")
        K = skel.shape[1]
        P, A = tok_in.shape[1], tok_in.shape[2]
        e_skel, h_skel, caches = self.skeleton_pass(skel)
        x = torch.where(gs[..., None], self.embed_gs(e_skel, piece, far), self.embed_detail(tok_in, e_skel, bud if self.group_plan else None))
        cc = [dict(k=c["k"], v=c["v"]) for c in caches]
        h = self.trunk(x, group_mask(K, seg), cc)                                   # (1, P, d)
        tg = tok_out[0]
        kind = tg[:, self.ai["kind"]]
        v = kind > 0
        s_ = kind == KIND_STROKE
        ptr = tg[:, self.ai["ptr"]]
        pf = self.ptr_feature(h_skel, ptr[None])[0]
        y, logits = self.attr_logits(h[0], tg, pf)
        has_ptr = (ptr > 0) & s_
        R = self.rules if (self.rules is not None and self.rules.any) else None
        self.last_viol = {}
        rctx = None
        if R is not None:
            from .rules import constrain, constrain_ptr, hung_context
            R.to(tok_in.device)
            rctx = group_train_context(self, tok_in[0], gs[0])
            rctx.update(hung_context(R, skel, torch.ones(1, K, dtype=torch.bool, device=skel.device), P))
        oc = None
        if self.occ is not None:
            if okeys is None:
                raise ValueError("model.occ_feat: the batch has no occupancy (GroupPackDataset occ=...)")
            oc = self.occ_train_context(skel, piece, oj, okeys, ostep)
        terms = {}
        for a, name in enumerate(self.names):
            if name in ("ptr", "lt"):
                continue
            if name == "kind":
                m = v
            elif name in ("s", "dx", "dy", "dz"):
                m = has_ptr
            elif name in ("ix", "iy", "iz"):
                m = s_ & ~has_ptr
            else:
                m = s_
            lg = logits[a].float()
            if oc is not None:
                from .occ import apply_occ
                lg = apply_occ(self, name, lg, y[:, a], tg, oc)
            if R is not None:
                lg = constrain(R, self, name, lg, tg, rctx, target=tg[:, a], count=self.last_viol, active=m)
            if name == "kind" and self.group_plan:
                from .rules import apply_mask
                lg = apply_mask(lg, plan_kind_allowed(lg.shape[1], gs[0], plan_n[0]), target=tg[:, a], count=self.last_viol, key="plan_kind", active=m)
            terms[name] = (F.cross_entropy(lg, tg[:, a], reduction="none"), m)
        pl = self.skel_pointer_logits(y[:, self.ai["ptr"]][None], h_skel, torch.ones(1, K, dtype=torch.bool, device=skel.device))[0].float()
        from .rules import apply_mask
        pl = apply_mask(pl, ptr_allowed(K, piece[0], far[0], self.group_mode), target=ptr.clamp(0, K), count=self.last_viol, key="ptr_group", active=s_)
        if R is not None:
            pl = constrain_ptr(R, self, pl, tg, rctx, target=ptr.clamp(0, K), count=self.last_viol, active=s_)
        terms["ptr"] = (F.cross_entropy(pl, ptr.clamp(0, K), reduction="none"), s_)
        if self.group_plan:
            pn, pq = plan_n[0], plan_q[0]
            mp = gs[0] & (pn >= 0)
            ln, lq = self.plan_logits(h[0], pn)
            terms["plan_n"] = (F.cross_entropy(ln, pn.clamp(0, GN_BINS - 1), reduction="none"), mp)
            terms["plan_q"] = (F.cross_entropy(lq, pq.clamp(0, GQ_BINS - 1), reduction="none"), mp)
        return terms, v

    def forward(self, skel, tok_in, tok_out, seg, piece, far, gs, bud=None, plan_n=None, plan_q=None, oj=None, okeys=None, ostep=None):
        """One pack (batch size 1, collate_group).  Returns dict of mean losses (per active target)."""
        terms, v = self._terms(skel, tok_in, tok_out, seg, piece, far, gs, bud, plan_n, plan_q, oj, okeys, ostep)
        out, losses = {}, {}
        for name, (ce, m) in terms.items():
            if m.any():
                losses[name] = ce[m].mean()
        out.update({f"loss_{k}": val for k, val in losses.items()})
        out["loss"] = sum(losses.values())
        out["n_tokens"] = int(v.sum())
        out.update({f"viol_{k}": c for k, c in self.last_viol.items()})
        return out

    @torch.no_grad()
    def position_logprobs(self, skel, tok_in, tok_out, seg, piece, far, gs, bud=None, plan_n=None, plan_q=None, oj=None, okeys=None, ostep=None):
        """Teacher-forced log-probability of every target (P,), summed over its active attributes, the group plan included
        at the GS positions (tests)."""
        terms, _ = self._terms(skel, tok_in, tok_out, seg, piece, far, gs, bud, plan_n, plan_q, oj, okeys, ostep)
        tot = torch.zeros_like(terms["kind"][0])
        for ce, m in terms.values():
            tot = tot - torch.where(m, ce, torch.zeros_like(ce))
        return tot


def group_train_context(model, tok_in, gs):
    """Rule context of every target of a pack: the previous token of target p is the input at p (none at a GS);
    time-order floor = t of that input, 0 at a GS (the first stroke of a group is free).  tok_in (P, A), gs (P,)."""
    ai = model.ai
    P, A = tok_in.shape
    prev = torch.where(gs[:, None], torch.zeros_like(tok_in), tok_in)
    has_prev = ~gs & (tok_in[:, ai["kind"]] > 0)
    z = torch.zeros(P, dtype=tok_in.dtype, device=tok_in.device)
    t_order = torch.where(has_prev, prev[:, ai["t"]], z)
    t_rule = model.rule_t(prev) if getattr(model, "cont", False) else z
    return dict(has_prev=has_prev, prev_lt=z, prev_t=prev[:, ai["t"]], prev_len=prev[:, ai["len"]], t_order=t_order, t_rule=t_rule, prev_tok=prev)


# ---------------------------------------------------------------- sampling
def block_step(blk, x, sk, sv, cache):
    """One new position per row through Block blk, with the shared skeleton keys / values (1, h, K, dh) and the row's
    own cache (dict k, v: (G, h, t, dh), updated): the same arithmetic as Block.forward (softmax over skeleton + own
    keys), without copying the skeleton keys G times."""
    G, L, d = x.shape
    q, k, v = blk.qkv(blk.ln1(x)).split(d, dim=2)
    q = q.view(G, L, blk.h, -1).transpose(1, 2); k = k.view(G, L, blk.h, -1).transpose(1, 2); v = v.view(G, L, blk.h, -1).transpose(1, 2)
    if "k" in cache:
        k = torch.cat([cache["k"], k], 2); v = torch.cat([cache["v"], v], 2)
    cache["k"], cache["v"] = k, v
    scale = 1.0 / math.sqrt(q.shape[-1])
    K = sk.shape[2]
    dh = q.shape[-1]
    qf = q.transpose(0, 1).reshape(1, blk.h, G * L, dh)                            # all rows as one query block: the
    s1 = torch.matmul(qf, sk.transpose(-1, -2)).view(blk.h, G, L, K).transpose(0, 1) * scale   # skeleton keys are not copied G times
    s2 = torch.matmul(q, k.transpose(-1, -2)) * scale                               # (G, h, L, t)
    if L > 1:
        t = k.shape[2]
        cm = torch.ones(L, t, dtype=torch.bool, device=x.device).tril(t - L)
        s2 = s2.masked_fill(~cm, float("-inf"))
    p = torch.softmax(torch.cat([s1, s2], -1), -1)
    ps = p[..., :K].transpose(0, 1).reshape(1, blk.h, G * L, K)
    y = torch.matmul(ps, sv).view(blk.h, G, L, dh).transpose(0, 1) + torch.matmul(p[..., K:], v)
    y = y.transpose(1, 2).reshape(G, L, d)
    x = x + blk.proj(y)
    return x + blk.mlp(blk.ln2(x))


def trunk_step(model, x, skel_caches, caches):
    for i, blk in enumerate(model.blocks):
        x = block_step(blk, x, skel_caches[i]["k"], skel_caches[i]["v"], caches[i])
    return model.ln_f(x)


def anchors_one(vocab, skel, piece, s_bin):
    """Vocab.anchor of skeleton pieces skel[piece] (N,) at s bins (N,), vectorised (same float64 arithmetic)."""
    ix = vocab.index
    par = skel[piece]
    head = vocab.head_point(par)
    u = vocab.dec_dir(par[:, ix["th"]], par[:, ix["ph"]])
    L = vocab.dec_len(par[:, ix["len"]]).astype(np.float64)
    s = np.where(L > 0, vocab.dec_s(s_bin), 0.0)
    return np.clip(np.floor(head + (s * L)[:, None] * u).astype(np.int64), 0, vocab.g - 1)


def hung_positions(vocab, skel, tok):
    """Head voxels (N, 3) of hung strokes: clip(anchor(piece ptr - 1, s) + d); rows without a pointer get 0."""
    ix = vocab.index
    out = np.zeros((len(tok), 3), np.int64)
    p = tok[:, ix["ptr"]]
    h = np.flatnonzero(p > 0)
    if len(h):
        a = anchors_one(vocab, skel, p[h] - 1, tok[h, ix["s"]])
        d = np.stack([vocab.dec_d(tok[h, ix[c]]) for c in ("dx", "dy", "dz")], 1)
        out[h] = np.clip(a + d, 0, vocab.g - 1)
    return out


@torch.no_grad()
def generate_group_detail(model, vocab, skel, device, temperature=1.0, max_steps=4096, seed=0, rng=None, forced_groups=None, attr_temp=None,
                          q_total_range=None, plan_tries=64, stop_at_plan=False, q_target=None, target_scale="q", target_clip=(0.5, 2.0),
                          placer=None):
    """Detail strokes of ONE shower given its skeleton (K, A): every group of the shower is a row of one batch (rows share
    the skeleton keys / values); a row leaves the batch at its EOS.  forced_groups (tests, timing): list of (piece,
    far, (m, A) tokens), scored instead of sampled; info['logp'] = summed log-probability per group (equals
    GroupDetailLM.position_logprobs).  Returns (tokens (T, A) sorted by t, info)."""
    A = len(vocab.names)
    skel = np.asarray(skel, dtype=np.int64).reshape(-1, A)
    K = len(skel)
    if K == 0:
        return np.zeros((0, A), np.int64), dict(groups=0, steps=0, max_group=0, **({"place": dict(placer.stats)} if placer is not None else {}))
    if rng is None:
        rng = torch.Generator(device=device).manual_seed(seed)
    plan = bool(getattr(model, "group_plan", False))
    if q_target is not None:
        if not plan or forced_groups is not None or q_total_range is not None:
            raise ValueError("q_target needs a group-plan model, sampled groups and no q_total_range")
        if not np.isfinite(float(q_target)):
            raise ValueError(f"q_target {q_target} is not finite")
        if target_scale not in ("q", "nq") or not (0.0 < target_clip[0] <= 1.0 <= target_clip[1]):
            raise ValueError(f"target_scale {target_scale!r} (q | nq) / target_clip {target_clip} (lo <= 1 <= hi, lo > 0)")
    if placer is not None and placer.active and forced_groups is not None:
        raise ValueError("placement rules apply to sampled groups only (not with forced_groups)")
    order_names = [model.names[a] for a in model.order]
    if placer is not None and placer.active and order_names[0] != "kind":
        raise ValueError(f"placement redraws keep the kind: model.attr_order must start with kind (got {order_names[:3]})")
    if placer is not None and placer.active and placer.keep_t > 0 and order_names[:2] != ["kind", "t"]:
        raise ValueError(f"placement keep_t > 0 keeps kind and t: model.attr_order must start with kind, t (got {order_names[:3]}; use keep_t 0)")
    n_keep_t = 2                                                                    # placement redraws at the stroke's own t keep kind and t
    sk_t = torch.from_numpy(skel).to(device)[None]
    e_skel, h_skel, skel_caches = model.skeleton_pass(sk_t)
    forced_all = forced_plan = None
    if forced_groups is not None:
        piece = torch.as_tensor([g[0] for g in forced_groups], dtype=torch.long, device=device)
        far = torch.as_tensor([g[1] for g in forced_groups], dtype=torch.long, device=device)
        mlen = max([len(g[2]) for g in forced_groups] + [0])
        fa = np.zeros((len(forced_groups), mlen + 1, A), np.int64)
        fa[:, :, model.ai["kind"]] = KIND_EOS
        for j, g in enumerate(forced_groups):
            if len(g[2]):
                fa[j, : len(g[2])] = g[2]
        forced_all = torch.from_numpy(fa).to(device)
        max_steps = min(max_steps, mlen + 1)
        forced_plan = (np.array([enc_gn(len(g[2])) for g in forced_groups], np.int64),
                       np.array([enc_gq(stroke_photons(vocab, g[2]).sum()) for g in forced_groups], np.int64))
    else:
        piece = torch.arange(K, device=device); far = torch.zeros(K, dtype=torch.long, device=device)
    G = len(piece)
    use_rules = getattr(model, "rules", None) is not None and model.rules.any
    hctx = {}
    if use_rules:
        from .rules import hung_context, sample_context
        model.rules.to(device)
        hctx = hung_context(model.rules, sk_t, torch.ones(1, K, dtype=torch.bool, device=device), G)
    rows = torch.arange(G, device=device)                                          # active rows -> group index
    ost = oc0 = None
    if getattr(model, "occ", None) is not None:                                    # occupancy as the training data builds it
        if not use_rules:
            raise ValueError("model.occ_feat needs the trained rules (the features ride in the rule context)")
        from .occ import OccState, occ_context, update_occ
        ost = OccState(vocab, skel, model.occ_cfg["ranges"], model.occ_cfg["w_occ"])
        oc0 = occ_context(vocab, skel, ost.keys, ost.steps, np.zeros(0, np.int64), np.zeros(0, np.int64), device)
    caches = [dict() for _ in model.blocks]
    x = model.embed_gs(e_skel, piece, far)[:, None, :]
    t_prev = torch.zeros(G, dtype=torch.long, device=device)
    logp_g = torch.zeros(G, device=device)
    allow_all = ptr_allowed(K, piece, far, model.group_mode)
    kept_rows, kept_tok, kept_rec, kept_step = [], [], [], []
    sv1 = torch.ones(1, K, dtype=torch.bool, device=device)                        # h_skel (1, K, d) is shared by every row
    prev = None
    fixed_fn = lambda t: torch.from_numpy(hung_positions(vocab, skel, t.cpu().numpy())).to(device)  # noqa: E731
    n_kind = vocab.sizes[vocab.index["kind"]]
    info = dict(groups=G)
    step = 0
    for step in range(max_steps):
        h = trunk_step(model, x, skel_caches, caches)[:, -1]
        n = len(rows)
        kind_allow = None
        if plan and step == 0:                                                     # every group's plan, all at once
            bn, bq, lpp, tries, qt = draw_plans(model, h, temperature, rng, forced_plan, q_total_range, plan_tries,
                                                float(stroke_photons(vocab, skel).sum()))
            logp_g += lpp
            n_plan, q_plan = dec_gn(bn.cpu().numpy()), dec_gq(bq.cpu().numpy())
            cnt = np.zeros(G); qsum = np.zeros(G)
            q_skel = float(stroke_photons(vocab, skel).sum())
            info.update(q_skel=q_skel, plan_n_total_raw=float(n_plan.sum()))
            if q_target is not None:                                               # one factor for all groups
                nb = bn.cpu().numpy()
                empty = nb == 0                                                    # planned 0 strokes: EOS first (kind_allow), photons never drawn
                info.update(plan_q_det_empty=float(q_plan[empty].sum()))
                q_plan = np.where(empty, 0.0, q_plan)
                info.update(plan_q_det_raw=float(q_plan.sum()))
                f_raw = (float(q_target) - q_skel) / max(float(q_plan.sum()), 1.0)
                f = float(np.clip(f_raw, target_clip[0], target_clip[1]))
                q_plan = q_plan * f
                if target_scale == "nq":                                           # exact-count bins stay integers (as trained):
                    x = n_plan * f                                                 # stochastic rounding, mean n f, >= 1 if planned
                    lo = np.floor(x)
                    up = np.random.default_rng([abs(int(seed)), int(int(seed) < 0), 7919]).random(len(x)) < (x - lo)
                    n_int = np.where(nb > 0, np.maximum(lo + up, 1.0), 0.0)
                    n_plan = np.where(nb < GN_EXACT, n_int, x)
                qt = q_skel + float(q_plan.sum())
                info.update(q_target=float(q_target), target_f=f, target_f_raw=f_raw, target_clipped=bool(f != f_raw))
            else:
                info.update(plan_q_det_raw=float(q_plan.sum()))
            info.update(plan_tries=tries, plan_q_total=qt, plan_n_total=float(n_plan.sum()), plan_q_det=float(q_plan.sum()),
                        plan_in_range=(None if q_total_range is None else bool(q_total_range[0] <= qt <= q_total_range[1])),
                        plan_bn=bn.cpu().numpy().copy(), plan_bq=bq.cpu().numpy().copy(), plan_n=np.array(n_plan, dtype=np.float64), plan_q=np.array(q_plan, dtype=np.float64))
            kind_allow = plan_kind_allowed(n_kind, torch.ones(G, dtype=torch.bool, device=device), bn)
        elif plan and stop_at_plan:
            done = torch.from_numpy(cnt[rows.cpu().numpy()] >= n_plan[rows.cpu().numpy()]).to(device)
            kind_allow = torch.ones(n, n_kind, dtype=torch.bool, device=device)
            kind_allow[done] = False
            kind_allow[done, KIND_EOS] = True
        rctx = None
        if use_rules:
            rctx = dict(sample_context(model, prev, t_prev, n, device), **{k: (v[:n] if k == "row_b" else v) for k, v in hctx.items()})
        if ost is not None:                                                        # rank of this step = the sampler step
            ok_, os_ = ost.tensors(device)
            rctx.update(update_occ(oc0, ok_, os_, torch.full((n,), step, dtype=torch.long, device=device), piece[rows]))
        forced = forced_all[rows, step] if forced_all is not None else None
        tok, lp = decode_detail(model, h, h_skel, sv1, None, vocab, temperature, rng,
                                attr_temp=attr_temp, forced=forced, rctx=rctx, ptr_allow=allow_all[rows], fixed_fn=fixed_fn, kind_allow=kind_allow)
        if placer is not None and placer.active:
            from .place import sub_context
            pa = allow_all[rows]
            tok, lp = placer.resolve(tok, lp, rows.cpu().numpy(), lambda sel, kt, keep_t: decode_detail(
                model, h[sel], h_skel, sv1, None, vocab, temperature, rng, attr_temp=attr_temp, rctx=sub_context(rctx, sel), ptr_allow=pa[sel],
                fixed_fn=fixed_fn, kind_allow=None if kind_allow is None else kind_allow[sel], keep=kt, n_keep=n_keep_t if keep_t else 1))
        logp_g[rows] += lp
        alive = tok[:, model.ai["kind"]] == KIND_STROKE
        keep = torch.nonzero(alive)[:, 0]
        if len(keep) == 0:
            break
        rows, tok = rows[keep], tok[keep]
        kept_rows.append(rows); kept_tok.append(tok); kept_step.append(step)
        if placer is not None and placer.active:
            kn = keep.cpu().numpy()
            kept_rec.append({k: v[kn] for k, v in placer.rec.items()})
            placer.commit(tok.cpu().numpy())
        if ost is not None:
            ost.add(tok.cpu().numpy(), step)
        t_prev = tok[:, model.ai["t"]]
        prev = tok
        for c in caches:
            c["k"], c["v"] = c["k"][keep], c["v"][keep]
        bud = None
        if plan:                                                                   # budget left in each group after this stroke
            r_np = rows.cpu().numpy()
            cnt[r_np] += 1
            qsum[r_np] += stroke_photons(vocab, tok.cpu().numpy())
            bud = torch.from_numpy(budget_bins(n_plan[r_np] - cnt[r_np], q_plan[r_np] - qsum[r_np])).to(device)[:, None, :]
        x = model.embed_detail(tok[:, None, :], e_skel.expand(len(rows), -1, -1), bud)
    if kept_tok:
        gr = torch.cat(kept_rows).cpu().numpy(); allt = torch.cat(kept_tok).cpu().numpy()
    else:
        gr = np.zeros(0, np.int64); allt = np.zeros((0, A), np.int64)
    sizes = np.bincount(gr, minlength=G)
    o = np.lexsort((np.arange(len(allt)), allt[:, vocab.index["t"]])) if len(allt) else np.zeros(0, np.int64)
    st = np.concatenate([np.full(len(r), s, np.int64) for r, s in zip(kept_rows, kept_step)]) if kept_tok else np.zeros(0, np.int64)
    info.update(steps=step + 1, max_group=int(sizes.max()) if G else 0, logp=logp_g.cpu().numpy(), group_of=gr[o], step_of=st[o])   # sampler step per stroke
    if placer is not None:
        info["place"] = dict(placer.stats)
        if kept_rec:                                                               # per detail stroke, aligned with the output
            info["place_rec"] = {k: np.concatenate([r[k] for r in kept_rec])[o] for k in kept_rec[0]}
    return allt[o], info


def draw_plans(model, h, temperature, rng, forced_plan, q_total_range, tries, q_skel):
    """Plans of all groups from their GS outputs h (G, d).  Returns strokes bins, photons bins (G,), their summed
    log-probability per group (G,), the number of draws, and the planned photons of the shower."""
    from .model import sample_logits
    dev = h.device
    ln = model.head_gn(h).float()
    lsn = torch.log_softmax(ln, -1)
    best = None
    n_draw = 1 if (forced_plan is not None or q_total_range is None) else max(1, int(tries))
    for k in range(n_draw):
        bn = torch.from_numpy(forced_plan[0]).to(dev) if forced_plan is not None else sample_logits(ln, temperature, rng)
        lq = model.plan_logits(h, bn)[1]
        bq = torch.from_numpy(forced_plan[1]).to(dev) if forced_plan is not None else sample_logits(lq, temperature, rng)
        qt = q_skel + float(dec_gq(bq.cpu().numpy()).sum())
        miss = 0.0 if q_total_range is None else max(q_total_range[0] - qt, qt - q_total_range[1], 0.0)
        if best is None or miss < best[0]:
            lp = lsn.gather(1, bn[:, None])[:, 0] + torch.log_softmax(lq, -1).gather(1, bq[:, None])[:, 0]
            best = (miss, bn, bq, lp, k + 1, qt)
        if miss == 0.0:
            break
    _, bn, bq, lp, _, qt = best
    return bn, bq, lp, k + 1, qt
