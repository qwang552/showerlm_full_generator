"""Strokes -> tokens and back.  One token = one stroke with discrete attributes: parent pointer and offset or an
absolute head voxel, offset inside the voxel, time, direction, length, photons and a few others (see Vocab.attrs).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from . import chains as CH
from .common import cache_meta, load_event, log, pct, split_ids, write_json
from .raster import render, stroke_voxels

KIND_PAD, KIND_STROKE, KIND_EOS, KIND_CONT = 0, 1, 2, 3     # CONT (0.3.0 vocabulary only): the next piece of the previous token's chain
D_REP = np.array([-30, -20, -14, -11, -9] + list(range(-8, 9)) + [9, 11, 14, 20, 30])
COH_EDGES = np.array([0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0001])


class Vocab:
    """Bin definitions of every attribute; encode / decode between physical values and integer bins.

    v3 (0.3.0): kind has 4 values (PAD, STROKE, EOS, CONT) and a 20th attribute `lt` = the length of the chain that is
    still to come after this piece (voxel units, log bins; 0 = this piece ends the chain).  So the first piece of a
    track says how long the track will be, every later piece is a CONT whose parent is the previous token (ptr 1,
    s = end), and "does the chain go on" is read from lt instead of being guessed again at every piece.  Old token
    files and checkpoints (no `v3` in their vocab dict) keep the 0.1 / 0.2 vocabulary."""

    def __init__(self, g=192, t_max=13.0, n_t=512, len_max=32.0, n_len=65, w_ptr=2048, n_th=180, n_ph=360, n_q=96, q_log_max=7.0,
                 n_c=32, c_log_max=4.0, n_s=16, n_off=8, v3=False, n_lt=24, lt_max=400.0, t_cut=12.0):
        self.t_cut = float(t_cut)                                     # raw-time cut of the cache (ns): the t_cap rule
        self.g, self.t_max, self.n_t, self.len_max, self.n_len, self.w_ptr = int(g), float(t_max), int(n_t), float(len_max), int(n_len), int(w_ptr)
        self.n_th, self.n_ph, self.n_q, self.q_log_max, self.n_c, self.c_log_max = int(n_th), int(n_ph), int(n_q), float(q_log_max), int(n_c), float(c_log_max)
        self.n_s, self.n_off = int(n_s), int(n_off)
        self.v3, self.n_lt, self.lt_max = bool(v3), int(n_lt), float(lt_max)
        self.attrs = [("kind", 4 if self.v3 else 3), ("ptr", self.w_ptr + 1), ("s", self.n_s + 1), ("dx", len(D_REP) + 1), ("dy", len(D_REP) + 1), ("dz", len(D_REP) + 1),
                      ("ix", self.g), ("iy", self.g), ("iz", self.g), ("ox", self.n_off), ("oy", self.n_off), ("oz", self.n_off),
                      ("t", self.n_t), ("th", self.n_th), ("ph", self.n_ph), ("len", self.n_len), ("lq", self.n_q), ("lc", self.n_c), ("coh", len(COH_EDGES) - 1)]
        if self.v3:
            self.attrs.append(("lt", self.n_lt))
        self.names = [a for a, _ in self.attrs]
        self.sizes = [s for _, s in self.attrs]
        self.index = {a: i for i, a in enumerate(self.names)}

    def to_dict(self):
        d = dict(g=self.g, t_max=self.t_max, n_t=self.n_t, len_max=self.len_max, n_len=self.n_len, w_ptr=self.w_ptr, n_th=self.n_th, n_ph=self.n_ph,
                 n_q=self.n_q, q_log_max=self.q_log_max, n_c=self.n_c, c_log_max=self.c_log_max, n_s=self.n_s, n_off=self.n_off)
        if self.v3:
            d.update(v3=True, n_lt=self.n_lt, lt_max=self.lt_max)
        if self.t_cut != 12.0:                                        # 0.6 full showers; 12 ns vocabularies stay byte-identical
            d["t_cut"] = self.t_cut
        return d

    @property
    def t_cap(self):
        """Last t bin below the cache's raw-time cut (the t_cap rule): enc_t(12 ns) for the 12 ns cache."""
        return int(self.enc_t(self.t_cut - 1e-9))

    @property
    def s_end(self):
        """s bin of 'the end of the parent' (s = 1): what every CONT piece carries."""
        return self.n_s

    def enc_lt(self, L):
        """Remaining chain length in voxel units -> bin (0 = nothing follows; 1.. = log scale up to lt_max)."""
        L = np.asarray(L, dtype=np.float64)
        b = 1 + np.floor(np.log(np.maximum(L, 1e-9)) / np.log(self.lt_max) * (self.n_lt - 2))
        return np.where(L <= 0, 0, np.clip(b, 1, self.n_lt - 1)).astype(np.int64)

    def dec_lt(self, b):
        b = np.asarray(b, dtype=np.float64)
        return np.where(b <= 0, 0.0, np.exp((np.maximum(b, 1) - 1 + 0.5) / (self.n_lt - 2) * np.log(self.lt_max)))

    def t_bins_of_len(self, len_bin):
        """Light-travel time of a piece of `len_bin` (+ 1 voxel to the next head) in t bins: the CONT time rule."""
        from .common import C_CM_PER_NS, VOX_CM
        L = (self.dec_len(len_bin) + 1.0) * VOX_CM
        return np.rint(L / C_CM_PER_NS / (self.t_max / self.n_t)).astype(np.int64)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    # ---- encoders / decoders (vectorised)
    def enc_t(self, t):
        return np.clip(np.floor(np.asarray(t, dtype=np.float64) / self.t_max * self.n_t), 0, self.n_t - 1).astype(np.int64)

    def dec_t(self, b):
        return (np.asarray(b, dtype=np.float64) + 0.5) * self.t_max / self.n_t

    def enc_dir(self, u):
        u = np.asarray(u, dtype=np.float64).reshape(-1, 3)
        th = np.degrees(np.arccos(np.clip(u[:, 2], -1, 1)))
        ph = np.degrees(np.arctan2(u[:, 1], u[:, 0]))
        bth = np.clip(np.floor(th / 180.0 * self.n_th), 0, self.n_th - 1).astype(np.int64)
        bph = np.clip(np.floor((ph + 180.0) / 360.0 * self.n_ph), 0, self.n_ph - 1).astype(np.int64)
        return bth, bph

    def dec_dir(self, bth, bph):
        th = np.radians((np.atleast_1d(bth).astype(np.float64) + 0.5) * 180.0 / self.n_th)
        ph = np.radians((np.atleast_1d(bph).astype(np.float64) + 0.5) * 360.0 / self.n_ph - 180.0)
        return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)

    def enc_len(self, length):
        return np.clip(np.rint(np.asarray(length, dtype=np.float64) / self.len_max * (self.n_len - 1)), 0, self.n_len - 1).astype(np.int64)

    def dec_len(self, b):
        return np.asarray(b, dtype=np.float64) * self.len_max / (self.n_len - 1)

    def enc_log(self, x, n, lmax):
        return np.clip(np.floor(np.log10(np.maximum(np.asarray(x, dtype=np.float64), 1.0)) / lmax * n), 0, n - 1).astype(np.int64)

    def dec_log(self, b, n, lmax):
        return 10.0 ** ((np.asarray(b, dtype=np.float64) + 0.5) * lmax / n)

    def enc_coh(self, coh):
        return np.clip(np.searchsorted(COH_EDGES, np.asarray(coh, dtype=np.float64), side="right") - 1, 0, len(COH_EDGES) - 2)

    def dec_coh(self, b):
        b = np.asarray(b, dtype=np.int64)
        return 0.5 * (COH_EDGES[b] + np.minimum(COH_EDGES[b + 1], 1.0))

    def enc_off(self, off):
        return np.clip(np.floor((np.asarray(off, dtype=np.float64) + 0.5) * self.n_off), 0, self.n_off - 1).astype(np.int64)

    def dec_off(self, b):
        return (np.asarray(b, dtype=np.float64) + 0.5) / self.n_off - 0.5

    def enc_d(self, d):
        return np.argmin(np.abs(np.asarray(d, dtype=np.float64).reshape(-1)[:, None] - D_REP[None, :]), axis=1) + 1      # 0 = n/a

    def dec_d(self, b):
        return D_REP[np.maximum(np.asarray(b, dtype=np.int64) - 1, 0)]

    def enc_s(self, s):
        return np.clip(np.floor(np.asarray(s, dtype=np.float64) * self.n_s), 0, self.n_s - 1).astype(np.int64) + 1          # 0 = n/a

    def dec_s(self, b):
        return (np.maximum(np.asarray(b, dtype=np.float64) - 1, 0) + 0.5) / self.n_s

    # ---- whole-token helpers
    def head_point(self, tok):
        """Head point in voxel units from the absolute attributes of tokens (T, A)."""
        tok = np.asarray(tok).reshape(-1, len(self.names))
        ijk = tok[:, [self.index["ix"], self.index["iy"], self.index["iz"]]].astype(np.float64)
        off = np.stack([self.dec_off(tok[:, self.index[a]]) for a in ("ox", "oy", "oz")], 1)
        return ijk + 0.5 + off

    def anchor(self, tok_parent, s_bin):
        """Anchor voxel: the point at fraction s (quantised) along the parent stroke, from the parent's own attributes."""
        head = self.head_point(tok_parent)[0]
        u = self.dec_dir(tok_parent[self.index["th"]], tok_parent[self.index["ph"]])[0]
        length = float(self.dec_len(tok_parent[self.index["len"]]))
        s = float(self.dec_s(s_bin)) if length > 0 else 0.0
        return np.clip(np.floor(head + s * length * u).astype(np.int64), 0, self.g - 1)

    def decode(self, tok, use_pointer=True):
        """Tokens (T, A) -> stroke arrays for the renderer.  With use_pointer, the head voxel of a token with a parent is
        anchor + (dx, dy, dz) (the absolute attributes are overwritten); without it the absolute (ix, iy, iz) is used."""
        tok = np.asarray(tok, dtype=np.int64).copy()
        tok = tok[is_stroke(tok[:, self.index["kind"]])]
        ix = self.index
        if use_pointer:
            for i in range(len(tok)):
                p = int(tok[i, ix["ptr"]])
                if p > 0 and i - p >= 0:
                    a = self.anchor(tok[i - p], tok[i, ix["s"]])
                    d = np.array([self.dec_d(tok[i, ix[k]]) for k in ("dx", "dy", "dz")]).reshape(3)
                    tok[i, [ix["ix"], ix["iy"], ix["iz"]]] = np.clip(a + d, 0, self.g - 1)
        ijk = tok[:, [ix["ix"], ix["iy"], ix["iz"]]]
        off = np.stack([self.dec_off(tok[:, ix[a]]) for a in ("ox", "oy", "oz")], 1)
        u = self.dec_dir(tok[:, ix["th"]], tok[:, ix["ph"]])
        return dict(ijk=ijk, off=off, u=u, len=self.dec_len(tok[:, ix["len"]]), t=self.dec_t(tok[:, ix["t"]]),
                    Q=self.dec_log(tok[:, ix["lq"]], self.n_q, self.q_log_max), C=self.dec_log(tok[:, ix["lc"]], self.n_c, self.c_log_max),
                    coh=self.dec_coh(tok[:, ix["coh"]]), tok=tok)


def is_stroke(kind):
    """True for the tokens that draw light (STROKE and CONT); False for PAD and EOS."""
    kind = np.asarray(kind)
    return (kind == KIND_STROKE) | (kind == KIND_CONT)


def pieces_of_event(ev, ex, ct, pa, len_max_vox):
    """Cut chains into straight pieces of <= len_max_vox voxel units (greedy along the chain).  Returns per-piece arrays."""
    cid, cpos = ex["chain_id"], ex["chain_pos"]
    order = np.lexsort((cpos, cid))
    n_ch = ex["n_chains"]
    starts = np.searchsorted(cid[order], np.arange(n_ch))
    nvox = ct["n_vox"]
    q = ev["q"].astype(np.float64); cnt = ex["cnt"].astype(np.float64); coh = ex["coh"]
    p_vox = ex["p"] / ex["vox"]; t_raw = ex["t"]; u_vox = ex["u"]
    head, length, photons, records, cohw, dirn, chain_of, k_of, t0 = [], [], [], [], [], [], [], [], []
    piece_of_voxel = np.zeros(len(q), np.int64)
    for c in range(n_ch):
        vs = order[starts[c]:starts[c] + nvox[c]]
        k, i0 = 0, 0
        while i0 < len(vs):
            i1 = i0
            while i1 + 1 < len(vs) and np.linalg.norm(p_vox[vs[i1 + 1]] - p_vox[vs[i0]]) <= len_max_vox:
                i1 += 1
            v = vs[i0:i1 + 1]
            j = len(head)
            piece_of_voxel[v] = j
            head.append(v[0]); photons.append(q[v].sum()); records.append(cnt[v].sum())
            cohw.append((q[v] * coh[v]).sum() / max(q[v].sum(), 1e-9)); chain_of.append(c); k_of.append(k); t0.append(t_raw[v[0]])
            if len(v) >= 2:
                d = p_vox[v[-1]] - p_vox[v[0]]; nd = np.linalg.norm(d)
                length.append(nd); dirn.append(d / nd if nd > 1e-9 else u_vox[v[0]])
            else:
                length.append(0.0); dirn.append(u_vox[v[0]] if coh[v[0]] > 1e-6 else np.array([0.0, 0.0, -1.0]))
            k += 1; i0 = i1 + 1
    P = len(head)
    head = np.array(head, np.int64); length = np.array(length); dirn = np.array(dirn).reshape(P, 3)
    parent = np.full(P, -1, np.int64); s = np.zeros(P)
    chain_of = np.array(chain_of, np.int64); k_of = np.array(k_of, np.int64)
    # lt: length of the chain still to come after this piece (sum of the later pieces' lengths; pieces of a chain are consecutive)
    lt = np.zeros(P)
    for j in range(P - 2, -1, -1):
        if chain_of[j + 1] == chain_of[j]:
            lt[j] = lt[j + 1] + length[j + 1]
    for j in range(P):
        if k_of[j] > 0:
            parent[j] = j - 1; s[j] = 1.0
        else:
            c = chain_of[j]
            pv = pa["voxel"][c]
            if pv >= 0:
                jp = piece_of_voxel[pv]
                parent[j] = jp
                if length[jp] > 0:
                    s[j] = float(np.clip((p_vox[pv] - p_vox[head[jp]]) @ dirn[jp] / length[jp], 0.0, 1.0))
    return dict(head=head, len=length, photons=np.array(photons), records=np.array(records), coh=np.array(cohw), dir=dirn,
                parent=parent, s=s, chain=chain_of, k=k_of, t=np.array(t0), lt=lt)


def tokenize_event(ev, g, ranges, vocab, coh_min=0.9):
    """Event arrays -> (tok (T, A) int64, info dict).  Chains in stroke mode, pieces, causal order, closed-loop anchors."""
    ex, ct, pa = CH.process_event(ev, g, ranges, "stroke", coh_min)
    pc = pieces_of_event(ev, ex, ct, pa, vocab.len_max)
    P = len(pc["head"])
    ijk = ev["ijk"].astype(np.int64)[pc["head"]]
    # chains ordered by head time (ties: z, y, x of the head voxel); pieces of a chain stay together in order
    c0 = pc["k"] == 0
    chain_rank = np.zeros(ex["n_chains"], np.int64)
    co = np.lexsort((ijk[c0, 0], ijk[c0, 1], ijk[c0, 2], pc["t"][c0]))
    chain_rank[pc["chain"][c0][co]] = np.arange(c0.sum())
    order = np.lexsort((pc["k"], chain_rank[pc["chain"]]))
    rank = np.empty(P, np.int64); rank[order] = np.arange(P)
    ix = vocab.index
    tok = np.zeros((P, len(vocab.names)), np.int64)
    tok[:, ix["kind"]] = KIND_STROKE
    if vocab.v3:
        tok[pc["k"][order] > 0, ix["kind"]] = KIND_CONT
        tok[:, ix["lt"]] = vocab.enc_lt(pc["lt"][order])
    tok[:, ix["ix"]] = ijk[order, 0]; tok[:, ix["iy"]] = ijk[order, 1]; tok[:, ix["iz"]] = ijk[order, 2]
    off = ev["off"].astype(np.float64)[pc["head"]][order]
    tok[:, ix["ox"]] = vocab.enc_off(off[:, 0]); tok[:, ix["oy"]] = vocab.enc_off(off[:, 1]); tok[:, ix["oz"]] = vocab.enc_off(off[:, 2])
    tok[:, ix["t"]] = vocab.enc_t(pc["t"][order])
    bth, bph = vocab.enc_dir(pc["dir"][order]); tok[:, ix["th"]] = bth; tok[:, ix["ph"]] = bph
    tok[:, ix["len"]] = vocab.enc_len(pc["len"][order])
    tok[:, ix["lq"]] = vocab.enc_log(pc["photons"][order], vocab.n_q, vocab.q_log_max)
    tok[:, ix["lc"]] = vocab.enc_log(pc["records"][order], vocab.n_c, vocab.c_log_max)
    tok[:, ix["coh"]] = vocab.enc_coh(pc["coh"][order])
    parent = pc["parent"][order]
    pr = np.where(parent >= 0, rank[np.maximum(parent, 0)], -1)
    back = np.arange(P) - pr
    has = (parent >= 0) & (back >= 1) & (back <= vocab.w_ptr)
    tok[has, ix["ptr"]] = back[has]
    tok[has, ix["s"]] = vocab.enc_s(pc["s"][order][has])
    hijk = ijk[order]
    for i in np.flatnonzero(has):
        d = hijk[i] - vocab.anchor(tok[i - back[i]], tok[i, ix["s"]])
        tok[i, ix["dx"]] = vocab.enc_d(d[0])[0]; tok[i, ix["dy"]] = vocab.enc_d(d[1])[0]; tok[i, ix["dz"]] = vocab.enc_d(d[2])[0]
    qp = pc["photons"][order]
    info = dict(n_tok=int(P), q_total=float(qp.sum()), n_vox=int(len(ev["q"])), frac_with_ptr=float(has.mean()),
                frac_parent_lost=float(((parent >= 0) & ~has).mean()), n_pieces_k1=int((pc["k"] > 0).sum()),
                q_cum=np.concatenate([[0.0], np.cumsum(qp)[:-1]]) / max(qp.sum(), 1e-9))
    if vocab.v3:
        info["cont_t_residual"] = cont_t_residual(tok, vocab)
    return tok, info


def cont_t_residual(tok, vocab):
    """For every CONT token: its t bin minus the rule (previous token's t bin + light travel of the previous piece).
    The model decodes the t of a CONT inside a window around this rule, so the residual must be small (`--check` reports it)."""
    ix = vocab.index
    c = np.flatnonzero(tok[1:, ix["kind"]] == KIND_CONT) + 1
    if len(c) == 0:
        return np.zeros(0, np.int64)
    return tok[c, ix["t"]] - (tok[c - 1, ix["t"]] + vocab.t_bins_of_len(tok[c - 1, ix["len"]]))


def check_event(ev, tok, vocab, g, ranges):
    """Render the truth tokens and compare with the truth voxels (representation ceiling)."""
    out = {}
    rend = None
    for name, use_ptr in (("pointer", True), ("absolute", False)):
        r = render(vocab.decode(tok, use_pointer=use_ptr), g, ranges)
        if use_ptr:
            rend = r
        kt = (ev["ijk"][:, 2].astype(np.int64) * g + ev["ijk"][:, 1]) * g + ev["ijk"][:, 0]
        kr = (r["ijk"][:, 2].astype(np.int64) * g + r["ijk"][:, 1]) * g + r["ijk"][:, 0]
        common, it, ir = np.intersect1d(kt, kr, return_indices=True)
        qt = ev["q"].astype(np.float64); qr = r["q"].astype(np.float64)
        union = len(kt) + len(kr) - len(common)
        qmin = np.minimum(qt[it], qr[ir]).sum()
        qmax = qt.sum() + qr.sum() - qmin
        dlog = np.abs(np.log10(qt[it] / np.maximum(qr[ir], 1e-9)))
        mt = ev["m"][it].astype(np.float64); mr = r["m"][ir].astype(np.float64)
        cosm = np.einsum("nc,nc->n", mt, mr) / np.maximum(np.linalg.norm(mt, axis=1) * np.linalg.norm(mr, axis=1), 1e-9)
        out[name] = dict(iou=float(len(common) / max(union, 1)), q_overlap=float(qmin / max(qmax, 1e-9)),
                         n_vox_truth=int(len(kt)), n_vox_render=int(len(kr)), q_rel_err=float(qr.sum() / qt.sum() - 1),
                         dlogq_pct_matched=pct(dlog, [50, 90]), dt_ns_pct_matched=pct(np.abs(ev["t"][it] - r["t"][ir]), [50, 90]),
                         cos_m_pct_matched=pct(cosm, [10, 50]))

    def chain_stats(e):
        ex, ct, pa = CH.process_event(e, g, ranges, "stroke")
        L = ct["length"]
        return dict(n_ge8=int((L >= 8).sum()), n_ge20=int((L >= 20).sum()), n_ge60=int((L >= 60).sum()), q_frac_ge8=float(ct["photons"][L >= 8].sum() / e["q"].sum()))
    out["chains_truth"] = chain_stats(ev); out["chains_render"] = chain_stats(rend)
    return out, rend


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="/data/user/qwang/corsika_sparse_v1/data/processed_192")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=0, help="events (0 = all of the split)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/data/user/qwang/showerlm/out")
    ap.add_argument("--name", default=None, help="output folder name under --out (shards share it)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--coh-min", type=float, default=0.9)
    ap.add_argument("--t-max", type=float, default=13.0, help="raw-time range of the t attribute (ns); the cache keeps raw t < 12 ns")
    ap.add_argument("--n-t", type=int, default=512, help="t bins (0.6 full showers: 3150 over 80 ns, the same 25.4 ps bins)")
    ap.add_argument("--t-cut", type=float, default=None, help="raw-time cut of the cache (ns; default: metadata time_max, else 12)")
    ap.add_argument("--n-lt", type=int, default=24, help="lt bins (v3)")
    ap.add_argument("--lt-max", type=float, default=400.0, help="largest remaining chain length of lt (voxels; full showers: 1400 = 29 m)")
    ap.add_argument("--ids-file", default=None, help="text file of event ids (one per line) instead of --split / --n")
    ap.add_argument("--w-ptr", type=int, default=2048, help="largest pointer distance (tokens)")
    ap.add_argument("--v3", action="store_true", help="0.3.0 vocabulary: kind CONT for the later pieces of a chain + attribute lt (remaining chain length)")
    ap.add_argument("--check", action="store_true", help="also render the tokens back and compare with the truth voxels")
    ap.add_argument("--save-render", action="store_true", help="with --check: save the rendered events too")
    a = ap.parse_args()
    meta, g, ranges = cache_meta(a.cache)
    if a.ids_file:
        ids = [int(x) for x in open(a.ids_file).read().split()][a.shard::a.nshards]
    else:
        ids = split_ids(meta, a.split, a.n, a.seed)[a.shard::a.nshards]
    t_cut = a.t_cut if a.t_cut is not None else float(meta.get("time_max") or 12.0)
    if t_cut > a.t_max:
        raise SystemExit(f"--t-max {a.t_max} must be >= the cache's time cut {t_cut}")
    out = Path(a.out) / (a.name or f"tokens_{a.split}")
    (out / "events").mkdir(parents=True, exist_ok=True)
    vocab = Vocab(g=g, t_max=a.t_max, n_t=a.n_t, w_ptr=a.w_ptr, v3=a.v3, n_lt=a.n_lt, lt_max=a.lt_max, t_cut=t_cut)
    vpath = out / "vocab.json"
    if vpath.exists():
        if json.load(open(vpath)) != vocab.to_dict():
            raise SystemExit(f"{vpath} holds a different vocabulary; use another --name")
    else:                                                                       # atomic: parallel shards read it
        tmp = out / f".vocab.json.{a.shard}.tmp"
        write_json(tmp, vocab.to_dict())
        os.replace(tmp, vpath)
    log(f"tokens: {len(ids)} {a.split} events (shard {a.shard}/{a.nshards}) -> {out}")
    checks = []; infos = []; resid = []
    for i, eid in enumerate(ids):
        f = out / "events" / f"tokens_{eid:05d}.npz"
        if f.exists():
            log(f"skip {f.name} (exists)"); continue
        ev = load_event(a.cache, eid)
        if "m" not in ev or not np.any(ev["m"]) or "cnt" not in ev:
            raise SystemExit("this cache has no direction (m) or record count (cnt): tokens need both")
        t0 = time.time()
        tok, info = tokenize_event(ev, g, ranges, vocab, a.coh_min)
        info["seconds"] = round(time.time() - t0, 2)
        np.savez_compressed(f, tok=tok.astype(np.int16), q_cum=info["q_cum"].astype(np.float32), eid=np.int64(eid),
                            q_total=np.float64(info["q_total"]), n_vox=np.int64(info["n_vox"]))
        infos.append(dict(eid=eid, n_tok=info["n_tok"], q_total=info["q_total"], n_vox=info["n_vox"], frac_with_ptr=info["frac_with_ptr"],
                          frac_parent_lost=info["frac_parent_lost"], n_cont=int((tok[:, vocab.index["kind"]] == KIND_CONT).sum()),
                          seconds=info["seconds"], t_at_cap=int((tok[:, vocab.index["t"]] >= vocab.t_cap).sum())))
        if "cont_t_residual" in info:
            resid.append(info["cont_t_residual"])
        if a.check:
            ck, rend = check_event(ev, tok, vocab, g, ranges); ck["eid"] = eid; checks.append(ck)
            if a.save_render:
                np.savez_compressed(out / "events" / f"render_{eid:05d}.npz", **rend)
            log(f"check {eid}: iou {ck['pointer']['iou']:.3f} q_overlap {ck['pointer']['q_overlap']:.3f} chains>=8 truth {ck['chains_truth']['n_ge8']} render {ck['chains_render']['n_ge8']}")
        if (i + 1) % 16 == 0 or i == len(ids) - 1:
            log(f"{i + 1}/{len(ids)}: eid {eid} tokens {info['n_tok']} voxels {info['n_vox']} with pointer {info['frac_with_ptr']:.3f}")
    if resid and sum(len(r) for r in resid):
        r = np.concatenate(resid)
        log(f"CONT time rule: {len(r)} continuation pieces, t residual (bins) 5/50/95% = {np.percentile(r, [5, 50, 95]).tolist()}, |res| <= 6: {float((np.abs(r) <= 6).mean()):.4f}")
    ipath = out / f"info_shard{a.shard}.json"
    if ipath.exists():                                                          # a resumed shard keeps the events of the earlier run
        new = {e["eid"] for e in infos}
        infos = [e for e in json.load(open(ipath)).get("events", []) if e["eid"] not in new] + infos
    write_json(ipath, dict(args=vars(a), events=infos))
    if a.check and checks:
        tag = "" if a.nshards == 1 else f"_shard{a.shard}"                    # shards share the folder: one file each
        write_json(out / f"check{tag}.json", checks)
        (out / f"check{tag}.md").write_text(check_markdown(checks))
        log("written", out / f"check{tag}.md")
    log("done")


def check_markdown(checks):
    def col(path):
        v = []
        for c in checks:
            x = c
            for k in path:
                x = x[k]
            v.append(x)
        return np.array(v, dtype=np.float64)
    L = [f"# Token representation check ({len(checks)} events): truth tokens rendered back vs truth voxels", "",
         "Everything the model can learn passes through the tokens, so this is the ceiling of what generation can reproduce.", "",
         "| quantity | pointer decode (mean) | absolute decode (mean) |", "|---|---|---|"]
    for key, desc in [(("iou",), "voxel IoU"), (("q_overlap",), "photon overlap sum min / sum max"), (("q_rel_err",), "total photons, relative error"),
                      (("dlogq_pct_matched", 0), "|dlog10 q| per matched voxel, median"), (("dt_ns_pct_matched", 0), "|dt| per matched voxel, median (ns)"),
                      (("cos_m_pct_matched", 1), "cos(direction) per matched voxel, median")]:
        L.append(f"| {desc} | {col(('pointer',) + key).mean():.3f} | {col(('absolute',) + key).mean():.3f} |")
    L += ["", "| chain statistic (stroke mode, per event mean) | truth | rendered |", "|---|---|---|"]
    for k, desc in [("n_ge8", "chains >= 8 cm"), ("n_ge20", "chains >= 20 cm"), ("n_ge60", "chains >= 60 cm"), ("q_frac_ge8", "photon fraction in chains >= 8 cm")]:
        L.append(f"| {desc} | {col(('chains_truth', k)).mean():.3g} | {col(('chains_render', k)).mean():.3g} |")
    L += ["", f"voxels per event: truth {col(('pointer', 'n_vox_truth')).mean():.0f}, rendered {col(('pointer', 'n_vox_render')).mean():.0f}"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
