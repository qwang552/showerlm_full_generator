"""Sampling-time placement rules for the group detail model: a detail stroke whose head lands on a voxel already crossed
by a track (x) or that points back along it (u) is redrawn.  Off unless asked for.
"""
from __future__ import annotations

import time

import numpy as np

from .raster2 import Prepared
from .tokens import KIND_CONT, KIND_STROKE

STROKE_KINDS = (KIND_STROKE, KIND_CONT)                                           # tokens that draw light (tokens.is_stroke)

PLACE_MODES = ("off", "x", "u", "xu")
OCC_KINDS = ("heads", "skel", "path")
STAT_KEYS = ("place_strokes", "place_rej_x", "place_rej_u", "place_redraws", "place_redraws_t", "place_rows_redrawn", "place_t_shift",
             "place_fallback_x", "place_fallback_u", "place_occ_skel", "place_occ_end", "place_s")


def voxel_keys(ijk, g):
    """(N, 3) voxel indices -> int64 keys (the key of raster2._merge)."""
    ijk = np.asarray(ijk, dtype=np.int64).reshape(-1, 3)
    return (ijk[:, 2] * g + ijk[:, 1]) * g + ijk[:, 0]


def drawn_keys(vocab, tok, ranges, w_occ):
    """Keys of the voxels the path renderer draws for stroke tokens (N, A) with absolute head voxels: crossed for >= w_occ
    voxel, plus the first and last voxel of every stroke (raster2.Prepared.entries_path)."""
    tok = np.asarray(tok, dtype=np.int64).reshape(-1, len(vocab.names))
    if len(tok) == 0:
        return np.zeros(0, np.int64)
    st = vocab.decode(tok, use_pointer=False)
    if len(st["len"]) == 0:
        return np.zeros(0, np.int64)
    ijk = Prepared(st, vocab.g, ranges).entries_path(w_occ)[1]
    return voxel_keys(ijk, vocab.g)


def nearest_piece(vocab, skel, tok, block=4e6):
    """Nearest skeleton piece (K pieces) to the head point of every token (N, A): tools/extract_skeleton.group_of for
    unhung strokes (point-to-segment distance in voxels, argmin = the earliest piece on ties), same arithmetic."""
    ix = vocab.index
    tok = np.asarray(tok, dtype=np.int64).reshape(-1, len(vocab.names))
    K = len(skel)
    out = np.zeros(len(tok), np.int64)
    if len(tok) == 0 or K == 0:
        return out
    P = vocab.head_point(skel)
    u = vocab.dec_dir(skel[:, ix["th"]], skel[:, ix["ph"]])
    Lk = vocab.dec_len(skel[:, ix["len"]]).astype(np.float64)
    hp = vocab.head_point(tok)
    step = max(1, int(block // K))
    for b0 in range(0, len(tok), step):
        h = hp[b0:b0 + step]
        w = h[:, None, :] - P[None]
        s = np.clip(np.einsum("nkc,kc->nk", w, u), 0.0, Lk[None])
        d2 = ((w - s[..., None] * u[None]) ** 2).sum(2)
        out[b0:b0 + step] = d2.argmin(1)
    return out


class Placer:
    """Placement rules of ONE shower (generate_group_detail(placer=...)).  mode: off | x | u | xu; ranges (3, 2) metres
    (the render box of the run); occ: heads | path (x); w_occ: drawn-voxel threshold of path occupancy; tries: redraws per
    stroke and step, the first keep_t of them at the stroke's own t (0 = exact rejection sampling)."""

    def __init__(self, vocab, skel, ranges, mode="off", w_occ=0.5, tries=16, occ="heads", keep_t=0):
        if mode not in PLACE_MODES or occ not in OCC_KINDS:
            raise ValueError(f"place mode {mode!r} not in {PLACE_MODES} / occ {occ!r} not in {OCC_KINDS}")
        if not (0.0 <= float(w_occ) < 1.0) or int(tries) < 0 or int(keep_t) < 0:
            raise ValueError(f"place w_occ {w_occ} (0 <= w < 1) / tries {tries} / keep_t {keep_t} (>= 0)")
        self.v, self.g = vocab, int(vocab.g)
        self.skel = np.asarray(skel, dtype=np.int64).reshape(-1, len(vocab.names))
        self.ranges = np.asarray(ranges, dtype=np.float64).reshape(3, 2)
        self.mode, self.x, self.u = mode, "x" in mode, "u" in mode
        self.w_occ, self.tries, self.occ_kind, self.keep_t = float(w_occ), int(tries), occ, int(keep_t)
        self.occ = set()
        t0 = time.time()
        if self.x and len(self.skel):
            self.occ.update(self.occupied(self.skel, skeleton=True).tolist())
        self.stats = {k: 0 for k in STAT_KEYS}
        self.stats["place_occ_skel"] = self.stats["place_occ_end"] = len(self.occ)
        self.stats["place_s"] = time.time() - t0

    @property
    def active(self):
        return self.x or self.u

    def heads(self, tok):
        ix = self.v.index
        return voxel_keys(tok[:, [ix["ix"], ix["iy"], ix["iz"]]], self.g)

    def occupied(self, tok, skeleton=False):
        """keys of the voxels stroke tokens (absolute heads) occupy: their drawn voxels (occ path; occ skel for the skeleton)
        or their head voxels"""
        if self.occ_kind == "path" or (skeleton and self.occ_kind == "skel"):
            return drawn_keys(self.v, tok, self.ranges, self.w_occ)
        return self.heads(tok[np.isin(tok[:, self.v.index["kind"]], STROKE_KINDS)])

    def check(self, tok, groups, taken, last=False):
        """Rows (n, A) of this step, their groups (n,), keys of the heads accepted earlier in this step (set).  Returns
        bad_x, bad_u (n,) bool; the head of a row passes x if it is not occupied, not taken, and no earlier row that claims
        its head has the same head (sequential in row order).  A row claims its head if it passes u, or if this is the
        last attempt (every row is kept then)."""
        n = len(tok)
        bad_u = np.zeros(n, bool)
        if self.u and len(self.skel):
            unh = np.flatnonzero(tok[:, self.v.index["ptr"]] == 0)
            if len(unh):
                bad_u[unh] = nearest_piece(self.v, self.skel, tok[unh]) != groups[unh]
        bad_x = np.zeros(n, bool)
        if self.x:
            k = self.heads(tok)
            occ = np.fromiter((int(c) in self.occ or int(c) in taken for c in k), bool, n)
            claim = ~bad_u | last
            cand = np.flatnonzero(~occ & claim)
            first = np.zeros(n, bool)
            if len(cand):
                _, i0 = np.unique(k[cand], return_index=True)                       # lowest row of every head among candidates
                first[cand[i0]] = True
            bad_x = occ | (claim & ~first)
        return bad_x, bad_u

    def resolve(self, tok, lp, groups, redo):
        """tok (n, A) torch long and lp (n,) of one step, groups (n,) numpy group of every row; redo(sel, keep, keep_t) ->
        (tok, lp) of the rows sel (torch long) decoded again after their kind (keep_t: after their kind and t).  Returns
        tok, lp (rows replaced).  self.rec: per row of the step, redraws (int16), dt = final - first t bin (int16), fb =
        fallback bits (1 x, 2 u; with tries 0 the strokes the rules would refuse)."""
        import torch
        if not self.active:
            return tok, lp
        t0 = time.time()
        ka, ta = self.v.index["kind"], self.v.index["t"]
        tn = tok.cpu().numpy()
        t_first = tn[:, ta].copy()
        pending = np.flatnonzero(tn[:, ka] == KIND_STROKE)
        self.stats["place_strokes"] += len(pending)
        taken = set()
        redrawn = np.zeros(len(tn), bool)
        n_redraw = np.zeros(len(tn), np.int16)
        fb = np.zeros(len(tn), np.uint8)
        for attempt in range(self.tries + 1):
            if len(pending) == 0:
                break
            last = attempt == self.tries
            bx, bu = self.check(tn[pending], groups[pending], taken, last)
            bad = bx | bu
            if last:                                                               # fallback: keep the last draw
                self.stats["place_fallback_x"] += int(bx.sum()); self.stats["place_fallback_u"] += int(bu.sum())
                fb[pending[bx]] |= 1
                fb[pending[bu]] |= 2
                ok = pending
            else:
                self.stats["place_rej_x"] += int(bx.sum()); self.stats["place_rej_u"] += int(bu.sum())
                ok = pending[~bad]
            if self.x and len(ok):
                taken.update(self.heads(tn[ok]).tolist())
            if last:
                break
            pending = pending[bad]
            if len(pending):
                self.stats["place_redraws"] += len(pending)
                self.stats["place_redraws_t"] += 0 if attempt < self.keep_t else len(pending)
                redrawn[pending] = True
                n_redraw[pending] += 1
                sel = torch.from_numpy(pending).to(tok.device)
                t2, l2 = redo(sel, tok[sel], attempt < self.keep_t)
                tok = tok.clone(); lp = lp.clone()
                tok[sel] = t2; lp[sel] = l2
                tn[pending] = t2.cpu().numpy()
        self.stats["place_rows_redrawn"] += int(redrawn.sum())
        self.stats["place_t_shift"] += int((tn[redrawn, ta] - t_first[redrawn]).sum())
        self.rec = dict(redraws=n_redraw, dt=(tn[:, ta] - t_first).astype(np.int16), fb=fb)
        self.stats["place_s"] += time.time() - t0
        return tok, lp

    def commit(self, tok):
        """Accepted strokes (m, A) of a step (absolute heads): their drawn voxels become occupied."""
        if not self.x or len(tok) == 0:
            return
        t0 = time.time()
        self.occ.update(self.occupied(tok).tolist())
        self.stats["place_occ_end"] = len(self.occ)
        self.stats["place_s"] += time.time() - t0


def sub_context(rctx, sel):
    """Rule context (showerlm.rules.sample_context + hung_context, as generate_group_detail builds it) of the rows sel:
    per-row entries indexed, per-skeleton entries kept, cached entries (keys starting with '_') dropped."""
    if rctx is None:
        return None
    per_row = ("has_prev", "prev_lt", "prev_t", "prev_len", "t_order", "t_rule", "prev_tok", "row_b", "oj", "opiece", "okoff", "opoff")   # oj, opiece: 0.6.2 occupancy; okoff, opoff: 0.6.3 lockstep
    return {k: (v[sel] if k in per_row else v) for k, v in rctx.items() if not k.startswith("_")}
