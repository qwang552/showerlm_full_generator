"""Renderer variants (no learned parameters).  The evaluation uses mode "path" with w_min 0.15 and causal=True: as
raster.render with a smaller w_min, and no light earlier than the light cone from the injection point.
"""
from __future__ import annotations

import numpy as np

from .common import C_CM_PER_NS

INJECTION_M = (0.0, 0.0, 50.0)       # injection point of every shower (m); clock zero assumed there, as the cache light-front correction does


def traverse(head, u, length, g):
    """All strokes at once.  head (T,3) voxel units, u (T,3) unit, length (T,) voxel units.  Returns per crossed voxel
    (in stroke order, then along the stroke): stroke index, ijk (n,3), s_in, s_out.  Same steps and stops as
    raster.stroke_voxels with w_min = 0 (a stroke of zero length or direction: its head voxel, s_in = s_out = 0)."""
    T = len(length)
    cell = np.floor(head).astype(np.int64)
    zero = (length <= 1e-6) | (np.linalg.norm(u, axis=1) < 1e-6)
    step = np.where(u > 0, 1, -1)
    big = np.abs(u) > 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        t_delta = np.where(big, 1.0 / np.abs(u), np.inf)
        t_max = np.where(big, (cell + (u > 0) - head) / np.where(big, u, 1.0), np.inf)
    t_entry = np.zeros(T)
    limit = (4 * length).astype(np.int64) + 4
    active = ~zero
    S, I, A, B, K = [], [], [], [], []
    zi = np.flatnonzero(zero)
    if len(zi):
        S.append(zi); I.append(cell[zi].copy()); A.append(np.zeros(len(zi))); B.append(np.zeros(len(zi))); K.append(np.zeros(len(zi), np.int64))
    it = 0
    while active.any():
        idx = np.flatnonzero(active)
        c = cell[idx]
        inside = np.all((c >= 0) & (c < g), axis=1) & (it < limit[idx])
        active[idx[~inside]] = False
        idx = idx[inside]
        if not len(idx):
            break
        tm = t_max[idx]
        a = np.argmin(tm, axis=1)
        t_exit = np.minimum(tm[np.arange(len(idx)), a], length[idx])
        S.append(idx); I.append(cell[idx].copy()); A.append(t_entry[idx].copy()); B.append(t_exit); K.append(np.full(len(idx), it))
        done = t_exit >= length[idx]
        active[idx[done]] = False
        go = ~done
        gi, ga = idx[go], a[go]
        cell[gi, ga] += step[gi, ga]
        t_max[gi, ga] += t_delta[gi, ga]
        t_entry[gi] = t_exit[go]
        it += 1
    if not S:
        return np.zeros(0, np.int64), np.zeros((0, 3), np.int64), np.zeros(0), np.zeros(0)
    s, ijk, a_, b_, k_ = (np.concatenate(x) for x in (S, I, A, B, K))
    o = np.lexsort((k_, s))
    return s[o], ijk[o], a_[o], b_[o]


def _merge(ijk, q, off, m, t, cnt, g, vox_cm, z_top_cm, lightfront):
    key = (ijk[:, 2] * g + ijk[:, 1]) * g + ijk[:, 0]
    uniq, inv = np.unique(key, return_inverse=True)
    nv = len(uniq)
    qs = np.bincount(inv, weights=q, minlength=nv)
    qw = np.maximum(qs, 1e-9)
    offs = np.stack([np.bincount(inv, weights=q * off[:, a], minlength=nv) for a in range(3)], 1) / qw[:, None]
    ms = np.stack([np.bincount(inv, weights=q * m[:, a], minlength=nv) for a in range(3)], 1) / qw[:, None]
    ts = np.bincount(inv, weights=q * t, minlength=nv) / qw
    cs = np.bincount(inv, weights=cnt, minlength=nv)
    out_ijk = np.stack([uniq % g, (uniq // g) % g, uniq // (g * g)], 1)
    if lightfront:
        ts = ts - (z_top_cm - (out_ijk[:, 2] + 0.5) * vox_cm) / C_CM_PER_NS
    return dict(ijk=out_ijk.astype(np.int32), q=qs.astype(np.float32), off=offs.astype(np.float32), m=ms.astype(np.float32),
                t=ts.astype(np.float32), cnt=cs.astype(np.int32))


class Prepared:
    """The strokes of one shower decoded once; render(variant) draws them with one variant."""

    def __init__(self, strokes, g, ranges):
        ranges = np.asarray(ranges, dtype=np.float64)
        self.g = int(g)
        self.vox_cm = float((ranges[0, 1] - ranges[0, 0]) / g * 100.0)
        self.z_top_cm = float((ranges[2, 1] - ranges[2, 0]) * 100.0)
        self.inj_vox = (np.asarray(INJECTION_M) - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0]) * self.g
        self.off0 = np.asarray(strokes["off"], dtype=np.float64)
        self.head = np.asarray(strokes["ijk"], dtype=np.float64) + 0.5 + np.clip(self.off0, -0.499, 0.499)
        self.u = np.asarray(strokes["u"], dtype=np.float64)
        self.len = np.asarray(strokes["len"], dtype=np.float64)
        self.t0 = np.asarray(strokes["t"], dtype=np.float64)
        self.Q = np.asarray(strokes["Q"], dtype=np.float64)
        self.C = np.asarray(strokes["C"], dtype=np.float64)
        self.coh = np.asarray(strokes["coh"], dtype=np.float64)
        self._trav = None

    def trav(self):
        if self._trav is None:
            self._trav = traverse(self.head, self.u, self.len, self.g)
        return self._trav

    def entries_path(self, w_min):
        s, ijk, a, b = self.trav()
        T = len(self.len)
        w = b - a
        n = np.bincount(s, minlength=T)
        start = np.concatenate([[0], np.cumsum(n)[:-1]])
        pos = np.arange(len(s)) - start[s]
        first, last = pos == 0, pos == n[s] - 1
        keep = (w >= w_min) | first | last
        s, ijk, a, b, w, first = s[keep], ijk[keep], a[keep], b[keep], w[keep], first[keep]
        nk = np.bincount(s, minlength=T)
        sum_rest = np.bincount(s, weights=np.where(first, 0.0, w), minlength=T)
        multi = nk[s] > 1
        w = np.where(first & multi, sum_rest[s] / np.maximum(nk[s] - 1, 1), w)
        w = np.where(first & ~multi & (self.len[s] <= 1e-6), 1.0, w)       # zero-length stroke: weight 1, as raster
        wn = w / np.maximum(np.bincount(s, weights=w, minlength=T)[s], 1e-9)
        q = self.Q[s] * wn
        cnt = np.maximum(1, np.rint(self.C[s] * wn)).astype(np.int64)
        s_mid = 0.5 * (a + b)
        pts = self.head[s] + s_mid[:, None] * self.u[s]
        off = np.clip(pts - (ijk + 0.5), -0.5, 0.5)
        off[first] = self.off0[s[first]]
        return s, ijk, q, off, s_mid, cnt

    def entries_records(self, phase, j0=0.0, j1=0.0, seed=0):
        T = len(self.len)
        n = np.maximum(1, np.rint(self.C)).astype(np.int64)
        s = np.repeat(np.arange(T), n)
        start = np.concatenate([[0], np.cumsum(n)[:-1]])
        k = np.arange(len(s)) - start[s]
        sp = (k + phase) / n[s] * self.len[s]
        pts = self.head[s] + sp[:, None] * self.u[s]
        sig = j0 + j1 * (1.0 - self.coh[s])
        if np.any(sig > 0):                                                     # transverse jitter of the record points
            uu = self.u[s]
            ref = np.eye(3)[np.argmin(np.abs(uu), axis=1)]
            e1 = np.cross(uu, ref); e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
            e2 = np.cross(uu, e1)
            z = np.random.default_rng(seed).standard_normal((len(s), 2)) * sig[:, None]
            pts = pts + z[:, :1] * e1 + z[:, 1:] * e2
        ijk = np.floor(pts).astype(np.int64)
        ok = np.all((ijk >= 0) & (ijk < self.g), axis=1)
        s, ijk, sp, pts = s[ok], ijk[ok], sp[ok], pts[ok]
        n_in = np.bincount(s, minlength=T)
        q = self.Q[s] / np.maximum(n_in[s], 1)
        off = pts - (ijk + 0.5)
        return s, ijk, q, off, sp, np.ones(len(s), np.int64)

    def render(self, mode="path", w_min=0.35, phase=0.5, j0=0.0, j1=0.0, f0=0.0, f1=0.0, spread="six", seed=0, causal=False, beta=1.0, head_t=False, lightfront=True):
        if mode == "path":
            s, ijk, q, off, sp, cnt = self.entries_path(w_min)
        elif mode == "records":
            s, ijk, q, off, sp, cnt = self.entries_records(phase, j0, j1, seed)
        else:
            raise ValueError(mode)
        st = sp
        if head_t and mode == "path":                                           # the head voxel keeps the token time (its mean record time)
            n = np.bincount(s, minlength=len(self.len))
            st = np.where(np.arange(len(s)) == np.concatenate([[0], np.cumsum(n)[:-1]])[s], 0.0, sp)
        t = self.t0[s] + st * self.vox_cm / (beta * C_CM_PER_NS)
        if causal:                                                              # no light before the light cone of the injection point
            p = ijk + 0.5 + off if mode == "records" else self.head[s] + sp[:, None] * self.u[s]
            t = np.maximum(t, np.linalg.norm(p - self.inj_vox, axis=1) * self.vox_cm / C_CM_PER_NS)
        m = self.coh[s, None] * self.u[s]
        if f0 > 0 or f1 > 0:
            f = np.clip(f0 + f1 * (1.0 - self.coh[s]), 0.0, 0.9)
            nb = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], np.int64)
            if spread == "perp":
                ax = np.argmax(np.abs(self.u[s]), axis=1)                       # the 4 neighbours across the main axis
                use = (np.abs(nb)[None, :, :] * (np.arange(3)[None, None, :] == ax[:, None, None])).sum(2) == 0
            else:
                use = np.ones((len(s), 6), bool)
            nij = ijk[:, None, :] + nb[None, :, :]
            use &= np.all((nij >= 0) & (nij < self.g), axis=2)
            nuse = use.sum(1)
            f = np.where(nuse > 0, f, 0.0)
            share = q * f / np.maximum(nuse, 1)
            r, c = np.nonzero(use)
            ijk = np.concatenate([ijk, nij[r, c]])
            q = np.concatenate([q * (1 - f), share[r]])
            off = np.concatenate([off, np.zeros((len(r), 3))])
            m = np.concatenate([m, m[r]])
            t = np.concatenate([t, t[r]])
            cnt = np.concatenate([cnt, np.zeros(len(r), np.int64)])
        if len(q) == 0:
            z = np.zeros((0, 3), np.float32)
            return dict(ijk=np.zeros((0, 3), np.int32), q=np.zeros(0, np.float32), off=z, m=z, t=np.zeros(0, np.float32), cnt=np.zeros(0, np.int32))
        return _merge(ijk, q, off, m, t, cnt, self.g, self.vox_cm, self.z_top_cm, lightfront)


def render_variant(strokes, g, ranges, **variant):
    return Prepared(strokes, g, ranges).render(**variant)
