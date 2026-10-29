"""Detail sampler that decodes the groups of several showers together (lockstep).  With one shower it gives the same
tokens as group.generate_group_detail.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import torch

from .data import budget_bins
from .detail import TYPE_DETAIL, decode_detail
from .group import GN_EXACT, dec_gn, dec_gq, draw_plans, enc_gn, enc_gq, plan_kind_allowed, ptr_allowed, stroke_photons
from .place import STAT_KEYS, STROKE_KINDS, Placer, sub_context
from .prof import region

OCC_CHECK = os.environ.get("SHOWERLM_OCC_CHECK", "0") == "1"                     # tests: also keep the 0.6.3 per-shower tables and compare
from .tables import budget_bins_t, tables_for
from .tokens import KIND_EOS, KIND_STROKE


# ---------------------------------------------------------------- skeletons of a batch
class MultiSkel:
    """The skeletons of S showers: per-shower skeleton pass (embeddings, outputs, per-layer keys / values), the padded
    tensors the rules read, and the concatenated piece geometry (showerlm.occ.piece_geometry) with the anchor table."""

    def __init__(self, model, vocab, skels, device):
        from .occ import piece_geometry
        self.model, self.v, self.device = model, vocab, device
        A = len(vocab.names)
        self.skels = [np.asarray(s, dtype=np.int64).reshape(-1, A) for s in skels]
        self.S = len(self.skels)
        self.K = [len(s) for s in self.skels]
        self.Kmax = max(self.K + [0])
        self.off = np.concatenate([[0], np.cumsum(self.K)[:-1]]).astype(np.int64) if self.S else np.zeros(0, np.int64)
        self.Kall = int(sum(self.K))
        self.e, self.h, self.caches = [], [], []
        d = model.d
        for sk in self.skels:
            if len(sk) == 0:                                                       # a shower without pieces has no rows (never used)
                self.e.append(torch.zeros(1, 0, d, device=device)); self.h.append(torch.zeros(1, 0, d, device=device)); self.caches.append(None)
                continue
            sk_t = torch.from_numpy(sk).to(device)[None]
            e_skel, h_skel, caches = model.skeleton_pass(sk_t)
            self.e.append(e_skel); self.h.append(h_skel); self.caches.append(caches)
        self.e_all = torch.cat([e[0] for e in self.e], 0) if self.S else torch.zeros(0, d, device=device)       # (Kall, d)
        self.h_all = torch.cat([h[0] for h in self.h], 0) if self.S else torch.zeros(0, d, device=device)
        self.pad = torch.zeros(self.S, self.Kmax, A, dtype=torch.long, device=device)
        self.valid = torch.zeros(self.S, self.Kmax, dtype=torch.bool, device=device)
        for s, sk in enumerate(self.skels):
            if len(sk):
                self.pad[s, : len(sk)] = torch.from_numpy(sk).to(device)
                self.valid[s, : len(sk)] = True
        self.skel_all = np.concatenate(self.skels, 0) if self.S else np.zeros((0, A), np.int64)
        self.P, self.u, self.L, self.anc = piece_geometry(vocab, self.skel_all)
        self.anc_t = torch.as_tensor(self.anc, dtype=torch.long, device=device)                                   # (Kall, n_s + 1, 3)
        self.q_skel = np.array([float(stroke_photons(vocab, sk).sum()) for sk in self.skels])
        self.key_off = np.arange(self.S, dtype=np.int64) * int(vocab.g) ** 3                                      # occupancy key offset per shower

    def bounds(self, row_b):
        """Contiguous row block [a, b) of every shower in row_b (sorted shower index per row): list of (a, b)."""
        n = int(row_b.shape[0])
        if n == 0:
            return [(0, 0)] * self.S
        if self.S == 1:
            return [(0, n)]
        c = torch.bincount(row_b, minlength=self.S).tolist()
        out, a = [], 0
        for s in range(self.S):
            out.append((a, a + c[s])); a += c[s]
        return out

    def pointer_logits(self, y, row_b):
        """y (n, d_attr) decoder states at the ptr slot -> (n, 1 + Kmax) logits [none, pieces of the row's shower]."""
        m = self.model
        if self.S == 1:
            return m.skel_pointer_logits(y[:, None, :], self.h[0], self.valid[:1])[:, 0]
        n = y.shape[0]
        out = torch.full((n, 1 + self.Kmax), float("-inf"), device=y.device, dtype=y.dtype)
        for s, (a, b) in enumerate(self.bounds(row_b)):
            if b > a:
                lg = m.skel_pointer_logits(y[a:b, None, :], self.h[s], self.valid[s: s + 1, : self.K[s]])[:, 0]
                out[a:b, : lg.shape[1]] = lg
        return out

    def ptr_feature(self, ptr, row_b):
        """ptr (n,) local pointers -> (n, 1, d_attr) decoder inputs of the ptr slot (DetailLM.ptr_feature per shower)."""
        m = self.model
        if self.S == 1:
            return m.ptr_feature(self.h[0], ptr[:, None])
        parts = []
        for s, (a, b) in enumerate(self.bounds(row_b)):
            if b > a:
                parts.append(m.ptr_feature(self.h[s], ptr[a:b, None]))
        return torch.cat(parts, 0) if parts else m.ptr_feature(self.h[0], ptr[:, None])

    def hung_heads(self, tok, opoff):
        """Head voxels (n, 3) of the hung rows of tok from the anchor table (showerlm.group.hung_positions arithmetic);
        opoff (n,) = index of the row's shower's first piece."""
        T = tables_for(self.v, tok.device)
        p = tok[:, self.v.index["ptr"]]
        gp = torch.where(p > 0, p + opoff, torch.zeros_like(p))
        return T.hung_heads(tok, self.anc_t, gp)

    def embed_detail(self, tok, gidx, bud):
        """DetailLM.embed_detail with the parent piece read from the concatenated embeddings: tok (n, 1, A), gidx (n,) global
        piece index of the pointer (any value where ptr = 0), bud (n, 1, 2) or None."""
        m = self.model
        ip = m.ai["ptr"]
        ptr = tok[..., ip]
        x = sum(m.emb[a]((ptr > 0).long() if a == ip else tok[..., a]) for a in range(m.A))
        if self.Kall > 0:
            par = self.e_all[gidx.clamp(0, self.Kall - 1)][:, None, :]
            x = x + m.par_proj(par) * (ptr > 0)[..., None].to(par.dtype)
        if m.budget and bud is not None:
            x = x + m.bud_n(bud[..., 0]) + m.bud_q(bud[..., 1])
        return x + m.type_emb.weight[TYPE_DETAIL]


def block_step_multi(blk, x, ms, layer, cache, bounds):
    """group.block_step for rows of several showers: the linear layers over all rows, the attention core per shower block
    (its own skeleton keys / values; the row's own cache).  One shower: the arithmetic of block_step."""
    G, L, d = x.shape
    q, k, v = blk.qkv(blk.ln1(x)).split(d, dim=2)
    q = q.view(G, L, blk.h, -1).transpose(1, 2); k = k.view(G, L, blk.h, -1).transpose(1, 2); v = v.view(G, L, blk.h, -1).transpose(1, 2)
    if "k" in cache:
        k = torch.cat([cache["k"], k], 2); v = torch.cat([cache["v"], v], 2)
    cache["k"], cache["v"] = k, v
    scale = 1.0 / math.sqrt(q.shape[-1])
    dh = q.shape[-1]
    s2 = torch.matmul(q, k.transpose(-1, -2)) * scale                               # (G, h, L, t)
    if L > 1:
        t = k.shape[2]
        cm = torch.ones(L, t, dtype=torch.bool, device=x.device).tril(t - L)
        s2 = s2.masked_fill(~cm, float("-inf"))
    ys = []
    for s, (a, b) in enumerate(bounds):
        if b <= a:
            continue
        sk, sv = ms.caches[s][layer]["k"], ms.caches[s][layer]["v"]                 # (1, h, K_s, dh)
        K = sk.shape[2]
        n = b - a
        qf = q[a:b].transpose(0, 1).reshape(1, blk.h, n * L, dh)                   # all rows of the shower as one query block
        s1 = torch.matmul(qf, sk.transpose(-1, -2)).view(blk.h, n, L, K).transpose(0, 1) * scale
        p = torch.softmax(torch.cat([s1, s2[a:b]], -1), -1)
        ps = p[..., :K].transpose(0, 1).reshape(1, blk.h, n * L, K)
        ys.append(torch.matmul(ps, sv).view(blk.h, n, L, dh).transpose(0, 1) + torch.matmul(p[..., K:], v[a:b]))
    y = torch.cat(ys, 0) if len(ys) > 1 else ys[0]
    y = y.transpose(1, 2).reshape(G, L, d)
    x = x + blk.proj(y)
    return x + blk.mlp(blk.ln2(x))


def trunk_step_multi(model, x, ms, caches, bounds):
    for i, blk in enumerate(model.blocks):
        x = block_step_multi(blk, x, ms, i, caches[i], bounds)
    return model.ln_f(x)


# ---------------------------------------------------------------- placement rules of a batch
class MultiPlacer:
    """One showerlm.place.Placer per shower; the pending rows of all showers are redrawn together (one decode per attempt).
    One shower: the loop of Placer.resolve."""

    def __init__(self, placers):
        self.p = list(placers)
        self.active = any(p.active for p in self.p)
        self.x = any(p.x for p in self.p)
        self.tries = self.p[0].tries if self.p else 0
        self.keep_t = self.p[0].keep_t if self.p else 0
        if self.p and any((p.tries, p.keep_t, p.mode, p.occ_kind) != (self.tries, self.keep_t, self.p[0].mode, self.p[0].occ_kind) for p in self.p):
            raise ValueError("MultiPlacer: every shower needs the same placement settings")
        self.rec = None

    def resolve(self, tok, lp, groups, sh, redo):
        """tok (n, A) torch, lp (n,), groups (n,) local group of every row, sh (n,) shower of every row (numpy);
        redo(sel, keep, keep_t) -> (tok, lp) of the rows sel.  Returns tok, lp; self.rec per row as Placer.resolve."""
        if not self.active:
            return tok, lp
        t0 = time.time()
        v = self.p[0].v
        ka, ta = v.index["kind"], v.index["t"]
        tn = tok.cpu().numpy()
        t_first = tn[:, ta].copy()
        pending = np.flatnonzero(tn[:, ka] == KIND_STROKE)
        for s in range(len(self.p)):
            self.p[s].stats["place_strokes"] += int((sh[pending] == s).sum())
        taken = [set() for _ in self.p]
        redrawn = np.zeros(len(tn), bool)
        n_redraw = np.zeros(len(tn), np.int16)
        fb = np.zeros(len(tn), np.uint8)
        for attempt in range(self.tries + 1):
            if len(pending) == 0:
                break
            last = attempt == self.tries
            bx = np.zeros(len(pending), bool); bu = np.zeros(len(pending), bool)
            ps = sh[pending]
            with region("det.place_check"):
                for s in np.unique(ps):
                    m = ps == s
                    bx[m], bu[m] = self.p[s].check(tn[pending[m]], groups[pending[m]], taken[s], last)
            bad = bx | bu
            if last:                                                               # fallback: keep the last draw
                for s in np.unique(ps):
                    m = ps == s
                    self.p[s].stats["place_fallback_x"] += int(bx[m].sum()); self.p[s].stats["place_fallback_u"] += int(bu[m].sum())
                fb[pending[bx]] |= 1
                fb[pending[bu]] |= 2
                ok = pending
            else:
                for s in np.unique(ps):
                    m = ps == s
                    self.p[s].stats["place_rej_x"] += int(bx[m].sum()); self.p[s].stats["place_rej_u"] += int(bu[m].sum())
                ok = pending[~bad]
            if self.x and len(ok):
                so = sh[ok]
                for s in np.unique(so):
                    taken[s].update(self.p[s].heads(tn[ok[so == s]]).tolist())
            if last:
                break
            pending = pending[bad]
            if len(pending):
                ps = sh[pending]
                for s in np.unique(ps):
                    m = int((ps == s).sum())
                    self.p[s].stats["place_redraws"] += m
                    self.p[s].stats["place_redraws_t"] += 0 if attempt < self.keep_t else m
                redrawn[pending] = True
                n_redraw[pending] += 1
                with region("det.place_redo"):
                    sel = torch.from_numpy(pending).to(tok.device)
                    t2, l2 = redo(sel, tok[sel], attempt < self.keep_t)
                    tok = tok.clone(); lp = lp.clone()
                    tok[sel] = t2; lp[sel] = l2
                    tn[pending] = t2.cpu().numpy()
        for s in range(len(self.p)):
            m = redrawn & (sh == s)
            self.p[s].stats["place_rows_redrawn"] += int(m.sum())
            self.p[s].stats["place_t_shift"] += int((tn[m, ta] - t_first[m]).sum())
        self.rec = dict(redraws=n_redraw, dt=(tn[:, ta] - t_first).astype(np.int16), fb=fb)
        dt = (time.time() - t0) / max(len(self.p), 1)
        for p in self.p:
            p.stats["place_s"] += dt
        return tok, lp

    def commit(self, tok_np, sh):
        """0.6.4a: the accepted strokes of all showers drawn in one renderer call (per stroke, so the same keys as one call
        per shower), then each shower's keys go to its own Placer."""
        if not self.x or len(tok_np) == 0:
            return
        t0 = time.time()
        p0 = self.p[0]
        if p0.occ_kind == "path":
            from .occ import drawn_entries
            rows, keys = drawn_entries(p0.v, tok_np, p0.ranges, p0.w_occ)
        else:
            rows = np.flatnonzero(np.isin(tok_np[:, p0.v.index["kind"]], STROKE_KINDS))
            keys = p0.heads(tok_np[rows])
        ks = sh[rows]
        for s in np.unique(sh):
            self.p[s].occ.update(keys[ks == s].tolist())
            self.p[s].stats["place_occ_end"] = len(self.p[s].occ)
        dt = (time.time() - t0) / len(self.p)
        for p in self.p:
            p.stats["place_s"] += dt


# ---------------------------------------------------------------- the sampler
@torch.no_grad()
def generate_lockstep(model, vocab, skels, device, temperature=1.0, max_steps=4096, seeds=None, forced_groups=None, attr_temp=None,
                      q_targets=None, target_scale="q", target_clip=(0.5, 2.0), placers=None, ring=None, ring_clip=(0.5, 2.0), rng=None):
    """Detail strokes of S showers (skels: list of (K_s, A)) in lockstep.  seeds: per shower (the batch generator takes the
    first; the nq stochastic rounding of every shower its own).  forced_groups: per shower, as generate_group_detail (tests).
    q_targets: per shower (None = no target); target_scale / target_clip as generate_group_detail.  placers: per shower
    showerlm.place.Placer or None.  ring: showerlm.ringtab.RingTable (nq_r).  Returns [(tokens, info)] per shower."""
    A = len(vocab.names)
    S = len(skels)
    if S == 0:
        return []
    seeds = list(seeds) if seeds is not None else [0] * S
    if len(seeds) != S:
        raise ValueError("generate_lockstep: one seed per shower")
    if q_targets is None:
        q_targets = [None] * S
    plan = bool(getattr(model, "group_plan", False))
    if any(q is not None for q in q_targets):
        if not plan or forced_groups is not None:
            raise ValueError("q_target needs a group-plan model and sampled groups")
        if any(q is not None and not np.isfinite(float(q)) for q in q_targets):
            raise ValueError("q_target is not finite")
        if target_scale not in ("q", "nq") or not (0.0 < target_clip[0] <= 1.0 <= target_clip[1]):
            raise ValueError(f"target_scale {target_scale!r} (q | nq) / target_clip {target_clip} (lo <= 1 <= hi, lo > 0)")
    if ring is not None and (not plan or target_scale != "nq" or forced_groups is not None):
        raise ValueError("the ring table (nq_r) needs a group-plan model, --target-scale nq and sampled groups")
    order_names = [model.names[a] for a in model.order]
    mp = MultiPlacer([p for p in placers if p is not None]) if placers is not None and any(p is not None for p in placers) else None
    if mp is not None and len(mp.p) != S:
        raise ValueError("generate_lockstep: a Placer for every shower or for none")
    if mp is not None and mp.active:
        if forced_groups is not None:
            raise ValueError("placement rules apply to sampled groups only (not with forced_groups)")
        if order_names[0] != "kind":
            raise ValueError(f"placement redraws keep the kind: model.attr_order must start with kind (got {order_names[:3]})")
        if mp.keep_t > 0 and order_names[:2] != ["kind", "t"]:
            raise ValueError(f"placement keep_t > 0 keeps kind and t: model.attr_order must start with kind, t (got {order_names[:3]}; use keep_t 0)")
    n_keep_t = 2
    with region("det.setup"):
        ms = MultiSkel(model, vocab, skels, device)
    T = tables_for(vocab, device)
    if rng is None:
        rng = torch.Generator(device=device).manual_seed(int(seeds[0]))
    # ---- rows: every group of every shower; showers without pieces have no rows (their output is empty)
    g_shower = np.concatenate([np.full(K, s, np.int64) for s, K in enumerate(ms.K)]) if ms.Kall else np.zeros(0, np.int64)
    g_local = np.concatenate([np.arange(K, dtype=np.int64) for K in ms.K]) if ms.Kall else np.zeros(0, np.int64)
    g_global = g_local + ms.off[g_shower] if ms.Kall else np.zeros(0, np.int64)
    forced_all = forced_plan = None
    if forced_groups is not None:
        if len(forced_groups) != S:
            raise ValueError("forced_groups: one list per shower")
        fg = [(s, g) for s, lst in enumerate(forced_groups) for g in lst]
        g_shower = np.array([s for s, _ in fg], np.int64)
        g_local = np.array([int(g[0]) for _, g in fg], np.int64)
        g_global = g_local + ms.off[g_shower] if len(fg) else np.zeros(0, np.int64)
        far_np = np.array([int(g[1]) for _, g in fg], np.int64)
        mlen = max([len(g[2]) for _, g in fg] + [0])
        fa = np.zeros((len(fg), mlen + 1, A), np.int64)
        fa[:, :, model.ai["kind"]] = KIND_EOS
        for j, (_, g) in enumerate(fg):
            if len(g[2]):
                fa[j, : len(g[2])] = g[2]
        forced_all = torch.from_numpy(fa).to(device)
        max_steps = min(max_steps, mlen + 1)
        forced_plan = (np.array([enc_gn(len(g[2])) for _, g in fg], np.int64), np.array([enc_gq(stroke_photons(vocab, g[2]).sum()) for _, g in fg], np.int64))
    else:
        far_np = np.zeros(len(g_global), np.int64)
    G = len(g_global)
    infos = [dict(groups=int((g_shower == s).sum())) for s in range(S)]
    if G == 0:
        out = []
        for s in range(S):
            inf = dict(groups=0, steps=0, max_group=0)
            if mp is not None:
                inf["place"] = dict(mp.p[s].stats)
            out.append((np.zeros((0, A), np.int64), inf))
        return out
    piece_t = torch.from_numpy(g_local).to(device); far_t = torch.from_numpy(far_np).to(device)
    gshow_t = torch.from_numpy(g_shower).to(device); gglob_t = torch.from_numpy(g_global).to(device)
    koff_t = torch.from_numpy(ms.key_off[g_shower]).to(device); poff_t = torch.from_numpy(ms.off[g_shower]).to(device)
    use_rules = getattr(model, "rules", None) is not None and model.rules.any
    if not use_rules:
        raise ValueError("generate_lockstep needs a model trained with the 0.5 rules (model.rules); older models: the legacy sampler")
    from .rules import hung_context, sample_context
    model.rules.to(device)
    hctx = hung_context(model.rules, ms.pad, ms.valid, 1)
    hctx.pop("row_b", None)
    g_in = np.concatenate([np.arange(int((g_shower == s).sum()), dtype=np.int64) for s in range(S)]) if G else np.zeros(0, np.int64)   # index inside its shower
    rows = torch.arange(G, device=device)
    osts = oc0 = otab = None
    if getattr(model, "occ", None) is not None:
        if not use_rules:
            raise ValueError("model.occ_feat needs the trained rules (the features ride in the rule context)")
        from .occ import OccState, OccTable, occ_context, update_occ
        with region("det.occ_add"):
            otab = OccTable(vocab, ms.skels, ms.key_off, model.occ_cfg["ranges"], model.occ_cfg["w_occ"], device)
        osts = [OccState(vocab, sk, model.occ_cfg["ranges"], model.occ_cfg["w_occ"]) for sk in ms.skels] if OCC_CHECK else []
        oc0 = occ_context(vocab, ms.skel_all, np.zeros(0, np.int64), np.zeros((0, 3), np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64), device)
    caches = [dict() for _ in model.blocks]
    x = model.embed_gs(ms.e_all[None], gglob_t, far_t)[:, None, :] if ms.Kall else None
    t_prev = torch.zeros(G, dtype=torch.long, device=device)
    logp_g = torch.zeros(G, device=device)
    allow_all = ptr_allowed(ms.Kmax, piece_t, far_t, model.group_mode)
    kept_rows, kept_tok, kept_rec, kept_step = [], [], [], []
    prev = None
    n_kind = vocab.sizes[vocab.index["kind"]]
    fixed_fn = lambda t, rc: ms.hung_heads(t, rc["opoff"])  # noqa: E731
    n_plan_t = q_plan_t = cnt = qsum = None
    step = 0
    for step in range(max_steps):
        row_b = gshow_t[rows]
        bounds = ms.bounds(row_b)
        with region("det.trunk"):
            h = trunk_step_multi(model, x, ms, caches, bounds)[:, -1]
        n = len(rows)
        kind_allow = None
        if plan and step == 0:                                                     # every group's plan of every shower, all at once
            bn, bq, lpp, tries, _ = draw_plans(model, h, temperature, rng, forced_plan, None, 1, 0.0)
            logp_g += lpp
            nb = bn.cpu().numpy(); qb = bq.cpu().numpy()
            n_plan, q_plan = dec_gn(nb), dec_gq(qb)
            for s in range(S):
                m = g_shower == s
                inf = infos[s]
                q_skel = float(ms.q_skel[s])
                inf.update(q_skel=q_skel, plan_n_total_raw=float(n_plan[m].sum()))
                qt_s = q_targets[s]
                if qt_s is not None:                                               # 0.6.1t: one factor per shower
                    empty = nb[m] == 0
                    inf.update(plan_q_det_empty=float(q_plan[m][empty].sum()))
                    qm = np.where(empty, 0.0, q_plan[m])
                    inf.update(plan_q_det_raw=float(qm.sum()))
                    f_raw = (float(qt_s) - q_skel) / max(float(qm.sum()), 1.0)
                    f = float(np.clip(f_raw, target_clip[0], target_clip[1]))
                    qm = qm * f
                    nm = n_plan[m]
                    if target_scale == "nq":
                        xx = nm * f
                        lo = np.floor(xx)
                        up = np.random.default_rng([abs(int(seeds[s])), int(int(seeds[s]) < 0), 7919]).random(len(xx)) < (xx - lo)
                        n_int = np.where(nb[m] > 0, np.maximum(lo + up, 1.0), 0.0)
                        nm = np.where(nb[m] < GN_EXACT, n_int, xx)
                    if ring is not None:                                           # 0.6.3 nq_r: per ring x depth cell of the piece
                        nm, rinfo = ring.apply(vocab, ms.skels[s], nm, nb[m], ring_clip, seed=int(seeds[s]))
                        inf.update(**rinfo)
                    n_plan[m] = nm; q_plan[m] = qm
                    inf.update(q_target=float(qt_s), target_f=f, target_f_raw=f_raw, target_clipped=bool(f != f_raw), plan_q_total=q_skel + float(qm.sum()))
                else:
                    inf.update(plan_q_det_raw=float(q_plan[m].sum()), plan_q_total=q_skel + float(q_plan[m].sum()))
                inf.update(plan_tries=tries, plan_n_total=float(n_plan[m].sum()), plan_q_det=float(q_plan[m].sum()), plan_in_range=None,
                           plan_bn=nb[m].copy(), plan_bq=qb[m].copy(), plan_n=np.array(n_plan[m], dtype=np.float64), plan_q=np.array(q_plan[m], dtype=np.float64))
            n_plan_t = torch.as_tensor(np.asarray(n_plan, dtype=np.float64), device=device)
            q_plan_t = torch.as_tensor(np.asarray(q_plan, dtype=np.float64), device=device)
            cnt = torch.zeros(G, dtype=torch.float64, device=device); qsum = torch.zeros(G, dtype=torch.float64, device=device)
            kind_allow = plan_kind_allowed(n_kind, torch.ones(G, dtype=torch.bool, device=device), bn)
        rctx = dict(sample_context(model, prev, t_prev, n, device), **hctx, row_b=row_b, okoff=koff_t[rows], opoff=poff_t[rows])
        if otab is not None:                                                      # 0.6.4a: the table stays on the device
            if OCC_CHECK:                                                          # the 0.6.3 per-shower tables, concatenated: must be equal
                ok_ = np.concatenate([osts[s].keys + ms.key_off[s] for s in range(S)])
                os_ = np.concatenate([o.steps for o in osts], 0)
                if not (np.array_equal(ok_, otab.keys.cpu().numpy()) and np.array_equal(os_, otab.steps.cpu().numpy())):
                    raise AssertionError(f"OccTable differs from the per-shower OccState tables at step {step}")
            rctx.update(update_occ(oc0, otab.keys, otab.steps, torch.full((n,), step, dtype=torch.long, device=device), gglob_t[rows], koff_t[rows], poff_t[rows]))
        forced = forced_all[rows, step] if forced_all is not None else None
        pa = allow_all[rows]
        with region("det.decode"):
            tok, lp = decode_detail(model, h, ms.pad, ms.valid, None, vocab, temperature, rng, attr_temp=attr_temp, forced=forced, rctx=rctx,
                                    ptr_allow=pa, fixed_fn=fixed_fn, kind_allow=kind_allow, ms=ms)
        if mp is not None and mp.active:
            rows_np = rows.cpu().numpy()
            tok, lp = mp.resolve(tok, lp, g_local[rows_np], g_shower[rows_np], lambda sel, kt, keep_t: decode_detail(
                model, h[sel], ms.pad, ms.valid, None, vocab, temperature, rng, attr_temp=attr_temp, rctx=sub_context(rctx, sel), ptr_allow=pa[sel],
                fixed_fn=fixed_fn, kind_allow=None if kind_allow is None else kind_allow[sel], keep=kt, n_keep=n_keep_t if keep_t else 1, ms=ms))
        logp_g[rows] += lp
        alive = tok[:, model.ai["kind"]] == KIND_STROKE
        keep = torch.nonzero(alive)[:, 0]
        if len(keep) == 0:
            break
        rows, tok = rows[keep], tok[keep]
        kept_rows.append(rows); kept_tok.append(tok); kept_step.append(step)
        need_np = (mp is not None and mp.active) or otab is not None
        if need_np:
            tok_np = tok.cpu().numpy(); kn = keep.cpu().numpy(); sh_np = g_shower[rows.cpu().numpy()]
        if mp is not None and mp.active:
            kept_rec.append({k: v[kn] for k, v in mp.rec.items()})
            with region("det.place_commit"):
                mp.commit(tok_np, sh_np)
        if otab is not None:
            with region("det.occ_add"):
                otab.add(tok_np, sh_np, step)
            for s in (np.unique(sh_np) if OCC_CHECK else ()):
                osts[s].add(tok_np[sh_np == s], step)
        t_prev = tok[:, model.ai["t"]]
        prev = tok
        for c in caches:
            c["k"], c["v"] = c["k"][keep], c["v"][keep]
        bud = None
        if plan:                                                                   # budget left in each group after this stroke
            cnt[rows] += 1.0
            qsum[rows] += T.photons(tok)
            bud = budget_bins_t(n_plan_t[rows] - cnt[rows], q_plan_t[rows] - qsum[rows])[:, None, :]
        with region("det.embed"):
            x = ms.embed_detail(tok[:, None, :], (tok[:, model.ai["ptr"]] - 1 + poff_t[rows]), bud)
    # ---- per shower outputs
    if kept_tok:
        gr = torch.cat(kept_rows).cpu().numpy(); allt = torch.cat(kept_tok).cpu().numpy()
        st = np.concatenate([np.full(len(r), s_, np.int64) for r, s_ in zip(kept_rows, kept_step)])
    else:
        gr = np.zeros(0, np.int64); allt = np.zeros((0, A), np.int64); st = np.zeros(0, np.int64)
    rec_all = {k: np.concatenate([r[k] for r in kept_rec]) for k in kept_rec[0]} if kept_rec else None
    logp_np = logp_g.cpu().numpy()
    out = []
    for s in range(S):
        m = g_shower[gr] == s
        gl = g_in[gr[m]]; ts = allt[m]; ss = st[m]                                 # group index inside the shower (= piece index for sampled groups)
        sizes = np.bincount(gl, minlength=int((g_shower == s).sum()))
        o = np.lexsort((np.arange(len(ts)), ts[:, vocab.index["t"]])) if len(ts) else np.zeros(0, np.int64)
        last = int(ss.max()) if len(ss) else -1
        inf = infos[s]
        gi = np.flatnonzero(g_shower == s)
        inf.update(steps=min(max_steps, last + 2), max_group=int(sizes.max()) if len(sizes) else 0, logp=logp_np[gi], group_of=gl[o], step_of=ss[o])
        if mp is not None:
            inf["place"] = dict(mp.p[s].stats)
            if rec_all is not None:
                inf["place_rec"] = {k: v[m][o] for k, v in rec_all.items()}
        out.append((ts[o], inf))
    return out
