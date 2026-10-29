"""The vocabulary's decoders as GPU lookup tables, so the samplers can keep their bookkeeping on the device.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .data import BUD_N_MAX, BUD_Q_MAX, N_BUD

_CACHE = {}


class Tables:
    def __init__(self, vocab, device):
        self.v = vocab
        dev = torch.device(device)
        self.device = dev
        ai = vocab.index
        f = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=torch.float64, device=dev)  # noqa: E731
        self.off = f(vocab.dec_off(np.arange(vocab.n_off)))                                 # (n_off,)
        self.len = f(vocab.dec_len(np.arange(vocab.n_len)))                                 # (n_len,) voxel units
        self.s = f(vocab.dec_s(np.arange(vocab.sizes[ai["s"]])))                            # (n_s + 1,)
        self.d = torch.as_tensor(vocab.dec_d(np.arange(vocab.sizes[ai["dx"]])), dtype=torch.long, device=dev)   # (n_d,) voxels
        th, ph = np.meshgrid(np.arange(vocab.n_th), np.arange(vocab.n_ph), indexing="ij")
        self.dir = f(vocab.dec_dir(th.ravel(), ph.ravel()).reshape(vocab.n_th, vocab.n_ph, 3))   # (n_th, n_ph, 3)
        self.q = f(vocab.dec_log(np.arange(vocab.n_q), vocab.n_q, vocab.q_log_max))         # (n_q,) photons of an lq bin
        self.t = f(vocab.dec_t(np.arange(vocab.n_t)))                                       # (n_t,) ns
        self.ai = ai
        self.g = int(vocab.g)

    # ---- token geometry (float64, the numpy order of operations)
    def head_point(self, tok):
        """(N, 3) float64 head point in voxel units (Vocab.head_point)."""
        ai = self.ai
        ijk = tok[:, [ai["ix"], ai["iy"], ai["iz"]]].to(torch.float64)
        off = torch.stack([self.off[tok[:, ai[a]].clamp(0, len(self.off) - 1)] for a in ("ox", "oy", "oz")], 1)
        return ijk + 0.5 + off

    def direction(self, tok):
        return self.dir[tok[:, self.ai["th"]].clamp(0, self.dir.shape[0] - 1), tok[:, self.ai["ph"]].clamp(0, self.dir.shape[1] - 1)]

    def length(self, tok):
        return self.len[tok[:, self.ai["len"]].clamp(0, len(self.len) - 1)]

    def photons(self, tok):
        """(N,) photons of every token from its lq bin (showerlm.group.stroke_photons)."""
        return self.q[tok[:, self.ai["lq"]].clamp(0, len(self.q) - 1)]

    def anchors(self, parent, s_bin):
        """Anchor voxels (N, 3) long of parent tokens (N, A) at s bins (N,): Vocab.anchor / Rules.anchors, same float64
        arithmetic (head + (s * L) * u, floor, clip)."""
        head = self.head_point(parent)
        u = self.direction(parent)
        L = self.length(parent)
        sv = torch.where(L > 0, self.s[s_bin.clamp(0, len(self.s) - 1)], torch.zeros_like(L))
        return torch.floor(head + (sv * L)[:, None] * u).to(torch.long).clamp(0, self.g - 1)

    def d_offsets(self, tok):
        """(N, 3) long: dec_d of the dx, dy, dz bins."""
        ai = self.ai
        return torch.stack([self.d[tok[:, ai[k]].clamp(0, len(self.d) - 1)] for k in ("dx", "dy", "dz")], 1)

    def hung_heads(self, tok, anc_table, gptr):
        """Head voxels (N, 3) long of hung strokes from an anchors table (M, n_s + 1, 3) (showerlm.occ.piece_geometry):
        clip(anc[piece, s] + d); rows without a pointer get 0.  gptr (N,) = pointer with the global piece offset."""
        ai = self.ai
        N = tok.shape[0]
        out = torch.zeros(N, 3, dtype=torch.long, device=tok.device)
        if anc_table.shape[0] == 0 or N == 0:
            return out
        hung = gptr > 0
        k = (gptr - 1).clamp(0, anc_table.shape[0] - 1)
        a = anc_table[k, tok[:, ai["s"]].clamp(0, anc_table.shape[1] - 1)]
        h = (a + self.d_offsets(tok)).clamp(0, self.g - 1)
        return torch.where(hung[:, None], h, out)

    def slice_light(self, tok, edges_z, edges_t, ranges):
        """Light of every token per depth slice (N, Kz) and per lateness slice (N, Kt), float64, as showerlm.zplan.
        token_slice_light / showerlm.tplan.token_tslice_light (3 points at 1/6, 1/2, 5/6 of the length, Q / 3 each).
        edges_z / edges_t: float64 tensors or None (then that output is None).  Rows that are not strokes must be masked
        by the caller."""
        from .common import C_CM_PER_NS, C_M_PER_NS
        ai = self.ai
        r = np.asarray(ranges, dtype=np.float64).reshape(3, 2)
        vm = float((r[0, 1] - r[0, 0]) / self.g)
        N = tok.shape[0]
        hz = tok[:, ai["iz"]].to(torch.float64) + 0.5 + self.off[tok[:, ai["oz"]].clamp(0, len(self.off) - 1)].clamp(-0.499, 0.499)
        L = self.length(tok)
        uz = self.direction(tok)[:, 2]
        frac = torch.as_tensor(np.array([1.0, 3.0, 5.0]) / 6.0, dtype=torch.float64, device=tok.device)
        z = float(r[2, 0]) + (hz[:, None] + (frac[None, :] * L[:, None]) * uz[:, None]) * vm            # (N, 3) metres
        depth = float(r[2, 1]) - z
        Q3 = (self.photons(tok) / 3.0)[:, None].expand(N, 3)
        out = []
        for edges, tau in ((edges_z, depth), (edges_t, None)):
            if edges is None:
                out.append(None); continue
            if tau is None:                                                                      # lateness tau = t - depth / c
                tt = self.t[tok[:, ai["t"]].clamp(0, len(self.t) - 1)][:, None] + (frac[None, :] * L[:, None]) * vm * 100.0 / C_CM_PER_NS
                tau = tt - depth / C_M_PER_NS
            K = int(edges.shape[0])
            k = (torch.searchsorted(edges, tau.contiguous(), right=True) - 1).clamp(0, K - 1)
            o = torch.zeros(N, K, dtype=torch.float64, device=tok.device)
            o.scatter_add_(1, k, Q3)
            out.append(o)
        return out[0], out[1]


def tables_for(vocab, device):
    key = (repr(sorted(vocab.to_dict().items())), str(torch.device(device)))
    if key not in _CACHE:
        _CACHE[key] = Tables(vocab, device)
    return _CACHE[key]


def enc_bud_t(x, xmax):
    """showerlm.data.enc_bud on tensors (float64)."""
    x = x.to(torch.float64).clamp_min(0.0)
    return torch.floor(torch.log1p(x) / math.log1p(xmax) * (N_BUD - 1) + 1e-9).clamp(0, N_BUD - 1).to(torch.long)


def budget_bins_t(rem_n, rem_q):
    """showerlm.data.budget_bins on tensors: (..., 2) long."""
    return torch.stack([enc_bud_t(rem_n, BUD_N_MAX), enc_bud_t(rem_q, BUD_Q_MAX)], -1)


def enc_z_t(x, n_bins, q_max):
    """showerlm.zplan.enc_z on tensors."""
    x = x.to(torch.float64).clamp_min(0.0)
    return torch.floor(torch.log1p(x) / math.log1p(q_max) * (n_bins - 1) + 1e-9).clamp(0, n_bins - 1).to(torch.long)
