"""Occupancy and geometry features of the head position of a detail stroke (which voxels are already lit, distance to
the skeleton), used by the group detail model.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .raster2 import Prepared
from .tokens import KIND_CONT, KIND_STROKE

OCC_ATTRS = ("dx", "dy", "dz", "ix", "iy", "iz")
N_OCC, N_FEAT = 3, 5
BIG = 2 ** 30
D_HALF = 8                                                                          # |d| support of hung strokes (rules.D_MAX)
CHUNK = 256                                                                         # rows per occupancy lookup of the 17^3 cube (GPU memory)
STROKE_KINDS = (KIND_STROKE, KIND_CONT)


def default_cfg():
    # w_occ 0.15 = the w_min of the rvC renderer that defines the evaluated voxels (tools/macro_eval_rv.py): the drawn
    # channels then mark exactly the voxels a stroke lights in the metric.  lr_mult: showerlm/train.py (optimizer groups).
    return dict(w_occ=0.15, rank=8, lr_mult=10.0, ranges=[-10.0, 10.0, -10.0, 10.0, 30.0, 50.0])


def voxel_keys(ijk, g):
    ijk = np.asarray(ijk, dtype=np.int64).reshape(-1, 3)
    return (ijk[:, 2] * g + ijk[:, 1]) * g + ijk[:, 0]


def drawn_entries(vocab, tok, ranges, w_occ):
    """(stroke row, voxel key) of every voxel the path renderer draws for the stroke tokens tok (N, A) with absolute head
    voxels (crossed >= w_occ, plus first and last voxel; showerlm/place.drawn_keys).  Rows that are not strokes draw nothing."""
    tok = np.asarray(tok, dtype=np.int64).reshape(-1, len(vocab.names))
    rows = np.flatnonzero(np.isin(tok[:, vocab.index["kind"]], STROKE_KINDS))
    if len(rows) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    st = vocab.decode(tok[rows], use_pointer=False)
    s, ijk = Prepared(st, vocab.g, np.asarray(ranges, dtype=np.float64).reshape(3, 2)).entries_path(w_occ)[:2]
    return rows[s], voxel_keys(ijk, vocab.g)


def head_entries(vocab, tok):
    tok = np.asarray(tok, dtype=np.int64).reshape(-1, len(vocab.names))
    rows = np.flatnonzero(np.isin(tok[:, vocab.index["kind"]], STROKE_KINDS))
    ix = vocab.index
    return rows, voxel_keys(tok[rows][:, [ix["ix"], ix["iy"], ix["iz"]]], vocab.g)


def merge(parts):
    """parts: list of (keys (n,), channel, steps (n,)) -> sorted unique keys (M,) int64 and first step per channel (M, 3)
    int64 (BIG = never)."""
    parts = [p for p in parts if len(p[0])]
    if not parts:
        return np.zeros(0, np.int64), np.zeros((0, N_OCC), np.int64)
    keys = np.concatenate([p[0] for p in parts])
    u, inv = np.unique(keys, return_inverse=True)
    st = np.full((len(u), N_OCC), BIG, np.int64)
    o = 0
    for k, c, s in parts:
        np.minimum.at(st[:, c], inv[o: o + len(k)], np.asarray(s, dtype=np.int64))
        o += len(k)
    return u, st


def event_occupancy(vocab, skel, det, rank, ranges, w_occ):
    """Occupancy timeline of one event (training): skeleton tokens (K, A), detail tokens (N, A) with true head voxels,
    rank (N,) = position of every detail stroke in its group (the sampler step that decodes it).  Returns (keys, steps)."""
    rank = np.asarray(rank, dtype=np.int64)
    sr, sk = drawn_entries(vocab, skel, ranges, w_occ)
    hr, hk = head_entries(vocab, skel)
    dr, dk = drawn_entries(vocab, det, ranges, w_occ)
    er, ek = head_entries(vocab, det)
    return merge([(sk, 0, np.full(len(sk), -1)), (hk, 2, np.full(len(hk), -1)), (dk, 1, rank[dr]), (ek, 2, rank[er])])


class OccState:
    """Occupancy of one shower while sampling: the skeleton at creation, then add(tokens, step) after every step."""

    def __init__(self, vocab, skel, ranges, w_occ):
        self.v, self.ranges, self.w = vocab, np.asarray(ranges, dtype=np.float64).reshape(3, 2), float(w_occ)
        skel = np.asarray(skel, dtype=np.int64).reshape(-1, len(vocab.names))
        sr, sk = drawn_entries(vocab, skel, self.ranges, self.w)
        hr, hk = head_entries(vocab, skel)
        self.keys, self.steps = merge([(sk, 0, np.full(len(sk), -1)), (hk, 2, np.full(len(hk), -1))])
        self._t = None

    def add(self, tok, step):
        """Accepted detail strokes (m, A) of step `step` (absolute head voxels).  0.6.3: incremental (the new keys are
        merged into the sorted table; same keys / steps as rebuilding the table with merge(), which sorted every key again
        at every step)."""
        dr, dk = drawn_entries(self.v, tok, self.ranges, self.w)
        er, ek = head_entries(self.v, tok)
        new_k, new_s = merge([(dk, 1, np.full(len(dk), step)), (ek, 2, np.full(len(ek), step))])
        if len(new_k):
            pos = np.searchsorted(self.keys, new_k)
            found = (pos < len(self.keys)) & (self.keys[np.minimum(pos, max(len(self.keys) - 1, 0))] == new_k) if len(self.keys) else np.zeros(len(new_k), bool)
            if found.any():                                                        # keys already in the table: earliest step per channel
                i = pos[found]
                self.steps[i] = np.minimum(self.steps[i], new_s[found])
            if (~found).any():                                                     # new keys: inserted in order
                ins = pos[~found]
                self.keys = np.insert(self.keys, ins, new_k[~found])
                self.steps = np.insert(self.steps, ins, new_s[~found], axis=0)
        self._t = None

    def tensors(self, device):
        if self._t is None or self._t[0].device != torch.device(device):
            self._t = (torch.from_numpy(self.keys).to(device), torch.from_numpy(self.steps).to(device))
        return self._t


class OccTable:
    """0.6.4a: the occupancy of ALL showers of a lockstep batch in one table kept on the device: keys = every shower's keys
    plus its offset s * g^3, steps (M, 3).  Equal, at every step, to concatenating the per-shower OccState tables (what the
    0.6.3 sampler uploaded whole at every step): the offsets keep the showers' keys apart, so merging the new entries of all
    showers at once gives the union of the per-shower merges.  add() draws the accepted strokes of all showers in ONE
    renderer call (drawn_entries is per stroke) instead of one call per shower, and merges on the device instead of
    inserting into numpy tables and uploading them again."""

    def __init__(self, vocab, skels, key_off, ranges, w_occ, device):
        self.v, self.ranges, self.w, self.dev = vocab, np.asarray(ranges, dtype=np.float64).reshape(3, 2), float(w_occ), device
        self.key_off = np.asarray(key_off, dtype=np.int64)
        sts = [OccState(vocab, sk, self.ranges, self.w) for sk in skels]
        k = np.concatenate([o.keys + self.key_off[s] for s, o in enumerate(sts)]) if sts else np.zeros(0, np.int64)
        st = np.concatenate([o.steps for o in sts], 0) if sts else np.zeros((0, N_OCC), np.int64)
        self.keys, self.steps = torch.from_numpy(k).to(device), torch.from_numpy(st).to(device)

    def add(self, tok, sh, step):
        """Accepted detail strokes tok (m, A) numpy of step `step`, sh (m,) their showers."""
        tok = np.asarray(tok, dtype=np.int64).reshape(-1, len(self.v.names))
        dr, dk = drawn_entries(self.v, tok, self.ranges, self.w)
        er, ek = head_entries(self.v, tok)
        new_k, new_s = merge([(dk + self.key_off[sh[dr]], 1, np.full(len(dk), step)), (ek + self.key_off[sh[er]], 2, np.full(len(ek), step))])
        if len(new_k) == 0:
            return
        allk = torch.cat([self.keys, torch.from_numpy(new_k).to(self.dev)])
        alls = torch.cat([self.steps, torch.from_numpy(new_s).to(self.dev)], 0)
        u, inv = torch.unique(allk, sorted=True, return_inverse=True)
        st = torch.full((len(u), N_OCC), BIG, dtype=alls.dtype, device=self.dev)
        st.scatter_reduce_(0, inv[:, None].expand(-1, N_OCC), alls, reduce="amin", include_self=True)
        self.keys, self.steps = u, st


# ---------------------------------------------------------------- context
def piece_geometry(vocab, skel):
    """Per skeleton piece: head point P (K, 3), unit direction u (K, 3), length L (K,) in voxel units (float64), and the
    anchor voxel of every s bin (K, n_s + 1, 3) (showerlm.group.anchors_one, the arithmetic of the hung head voxels)."""
    from .group import anchors_one
    ix = vocab.index
    skel = np.asarray(skel, dtype=np.int64).reshape(-1, len(vocab.names))
    K, S = len(skel), vocab.n_s + 1
    P = vocab.head_point(skel) if K else np.zeros((0, 3))
    u = vocab.dec_dir(skel[:, ix["th"]], skel[:, ix["ph"]]) if K else np.zeros((0, 3))
    L = vocab.dec_len(skel[:, ix["len"]]).astype(np.float64) if K else np.zeros(0)
    anc = anchors_one(vocab, skel, np.repeat(np.arange(K), S), np.tile(np.arange(S), K)).reshape(K, S, 3) if K else np.zeros((0, S, 3), np.int64)
    return P, u, L, anc


def occ_context(vocab, skel, keys, steps, oj, opiece, device, okoff=None, opoff=None):
    """Context of the feature function: occupancy (keys (M,), steps (M, 3)), per row the rank oj (N,) of the stroke being
    decoded and the piece opiece (N,) of its group, and the skeleton's piece geometry.
    0.6.3 (several showers decoded in lockstep, showerlm/lockstep.py): skel = the pieces of ALL showers one after the
    other, keys = every shower's keys plus its offset s * g^3, and per row okoff (N,) = that offset and opoff (N,) = the
    index of its shower's first piece (added to the row's pointer); zeros (the default) = one shower, as 0.6.2."""
    P, u, L, anc = piece_geometry(vocab, skel)
    dm = np.full(vocab.sizes[vocab.index["dx"]], -1, np.int64)
    b = np.arange(1, len(dm))
    dv = vocab.dec_d(b)
    ok = np.abs(dv) <= D_HALF
    dm[b[ok]] = dv[ok] + D_HALF
    t = lambda x, dt=torch.long: torch.as_tensor(x, dtype=dt, device=device)  # noqa: E731
    n = len(np.asarray(oj))
    return dict(okeys=t(keys), ostep=t(steps, torch.int32), oj=t(oj), opiece=t(opiece), oP=t(P, torch.float64), ou=t(u, torch.float64), oL=t(L, torch.float64),
                oanc=t(anc), odmap=t(dm), og=int(vocab.g),
                okoff=t(np.zeros(n, np.int64)) if okoff is None else t(okoff), opoff=t(np.zeros(n, np.int64)) if opoff is None else t(opoff))


def update_occ(ctx, keys, steps, oj, opiece, okoff=None, opoff=None):
    """Same skeleton, new occupancy / rows (sampler steps).  okoff / opoff (0.6.3): per-row offsets, zeros when not given."""
    ctx = dict(ctx)
    n = oj.shape[0]
    ctx.update(okeys=keys, ostep=steps.to(torch.int32), oj=oj, opiece=opiece,
               okoff=torch.zeros(n, dtype=torch.long, device=oj.device) if okoff is None else okoff,
               opoff=torch.zeros(n, dtype=torch.long, device=oj.device) if opoff is None else opoff)
    return {k: v for k, v in ctx.items() if not k.startswith("_")}


def _occupied(oc, keys, j):
    """keys (n, ...) int64, j (n,) rank -> (n, ..., 3) float: voxel occupied (first step < j) per channel."""
    M = oc["okeys"].shape[0]
    if M == 0:
        return torch.zeros(*keys.shape, N_OCC, device=keys.device)
    flat = keys.reshape(-1)
    idx = torch.searchsorted(oc["okeys"], flat).clamp(max=M - 1)
    found = oc["okeys"][idx] == flat
    st = oc["ostep"][idx]
    st = torch.where(found[:, None], st, torch.full_like(st, BIG))
    st = st.view(*keys.shape, N_OCC)
    jj = j.view(-1, *([1] * (keys.dim() - 1)), 1).to(st.dtype)
    return (st < jj).float()


def _keys(x, y, z, g):
    return (z * g + y) * g + x


def _rad(x, y, g):
    """log-scaled radius of voxel centres (x, y indices) from the axis (between voxels g/2 - 1 and g/2)"""
    r = torch.hypot(x.double() + 0.5 - g / 2, y.double() + 0.5 - g / 2)
    return (torch.log1p(r) / math.log1p(g / 2 * math.sqrt(2))).float()


def _seg_dist(p, P, u, L):
    """point (..., 3) to segment P + s u, s in [0, L] (P, u (n, 3), L (n,) broadcast over the middle dims): distance."""
    sh = (-1,) + (1,) * (p.dim() - 2) + (3,)
    w = p - P.view(sh)
    s = (w * u.view(sh)).sum(-1).clamp(min=0.0)
    s = torch.minimum(s, L.view(sh[:-1]))
    return torch.linalg.norm(w - s[..., None] * u.view(sh), dim=-1)


def _ldist(d, g):
    return (torch.log1p(d) / math.log1p(g * math.sqrt(3))).float()


def features(name, tok, oc, rows, ai):
    """Features (n, n_bins, 5) of attribute `name` for the rows `rows` (index tensor into the per-row context and tok):
    hung rows for dx / dy / dz, unhung rows for ix / iy / iz (callers select them).  tok (N, A): the stroke with the
    attributes decoded before `name` filled in."""
    g = oc["og"]
    dev = tok.device
    t = tok[rows]
    j = oc["oj"][rows]
    n = len(rows)
    zr = torch.zeros(len(oc["oj"]), dtype=torch.long, device=dev)
    koff = oc.get("okoff", zr)[rows]                                                            # 0.6.3: key / piece offsets of the row's shower
    poff = oc.get("opoff", zr)[rows]
    if name in ("dx", "dy", "dz"):
        nb = oc["odmap"].shape[0]
        k = (t[:, ai["ptr"]] - 1 + poff).clamp(0, max(oc["oanc"].shape[0] - 1, 0))
        a = oc["oanc"][k, t[:, ai["s"]].clamp(0, oc["oanc"].shape[1] - 1)]                  # (n, 3)
        dv = torch.arange(-D_HALF, D_HALF + 1, device=dev)
        cube = oc.get("_cube")
        if cube is None or oc.get("_cube_rows") is None or not torch.equal(oc["_cube_rows"], rows):
            X = (a[:, 0, None] + dv).clamp(0, g - 1); Y = (a[:, 1, None] + dv).clamp(0, g - 1); Z = (a[:, 2, None] + dv).clamp(0, g - 1)
            cube = torch.cat([_occupied(oc, _keys(X[c:c + CHUNK, :, None, None], Y[c:c + CHUNK, None, :, None], Z[c:c + CHUNK, None, None, :], g)
                                        + koff[c:c + CHUNK, None, None, None], j[c:c + CHUNK])
                              for c in range(0, n, CHUNK)]) if n else torch.zeros(0, 17, 17, 17, N_OCC, device=dev)   # (n, 17, 17, 17, 3)
            oc["_cube"], oc["_cube_rows"] = cube, rows
        dmap = oc["odmap"]
        if name == "dx":
            occ = cube.mean((2, 3))                                                              # (n, 17, 3)
            off = torch.stack(torch.broadcast_tensors(dv[None, :], torch.zeros_like(dv)[None, :], torch.zeros_like(dv)[None, :]), -1).expand(n, -1, -1)
        else:
            xi = dmap[t[:, ai["dx"]].clamp(0, nb - 1)]
            okx = xi >= 0
            xs = xi.clamp(min=0)
            r_ = torch.arange(n, device=dev)
            dxv = (xs - D_HALF)
            if name == "dy":
                occ = cube[r_, xs].mean(2)                                                       # (n, 17, 3)
                off = torch.stack([dxv[:, None].expand(n, len(dv)), dv[None, :].expand(n, -1), torch.zeros(n, len(dv), dtype=torch.long, device=dev)], -1)
                bad = ~okx
            else:
                yi = dmap[t[:, ai["dy"]].clamp(0, nb - 1)]
                ys = yi.clamp(min=0)
                occ = cube[r_, xs, ys]                                                           # (n, 17, 3)
                off = torch.stack([dxv[:, None].expand(n, len(dv)), (ys - D_HALF)[:, None].expand(n, len(dv)), dv[None, :].expand(n, -1)], -1)
                bad = ~okx | (yi < 0)
            occ = torch.where(bad[:, None, None], torch.zeros_like(occ), occ)
        c = (a[:, None, :] + off).clamp(0, g - 1)                                                 # (n, 17, 3) candidate voxels
        rad = _rad(c[..., 0], c[..., 1], g)
        dist = _ldist(_seg_dist(c.double() + 0.5, oc["oP"][k], oc["ou"][k], oc["oL"][k]), g)
        f17 = torch.cat([occ, rad[..., None], dist[..., None]], -1)                              # (n, 17, 5)
        if name != "dx":
            f17 = torch.where(bad[:, None, None], torch.zeros_like(f17), f17)
        F = torch.zeros(n, nb, N_FEAT, device=dev)
        valid = dmap >= 0
        F[:, valid] = f17[:, dmap[valid]]
        return F
    # unhung rows: absolute head voxel
    k = oc["opiece"][rows].clamp(0, max(oc["oP"].shape[0] - 1, 0))
    P, u, L = oc["oP"][k], oc["ou"][k], oc["oL"][k]
    b = torch.arange(g, device=dev)
    F = torch.zeros(n, g, N_FEAT, device=dev)
    if name == "ix":
        x = b.double() + 0.5
        F[..., 3] = (torch.log1p((x - g / 2).abs()) / math.log1p(g / 2 * math.sqrt(2))).float()[None, :].expand(n, -1)
        e = P + L[:, None] * u
        lo, hi = torch.minimum(P[:, 0], e[:, 0]), torch.maximum(P[:, 0], e[:, 0])
        F[..., 4] = _ldist(torch.maximum(torch.maximum(lo[:, None] - x[None, :], x[None, :] - hi[:, None]), torch.zeros(1, dtype=torch.float64, device=dev)), g)
        return F
    xi = t[:, ai["ix"]]
    if name == "iy":
        F[..., 3] = _rad(xi[:, None].expand(n, g), b[None, :].expand(n, -1), g)
        p = torch.stack([xi[:, None].double().expand(n, g) + 0.5, b[None, :].double().expand(n, g) + 0.5], -1)   # (n, g, 2)
        uu = u[:, :2]
        nrm = torch.linalg.norm(uu, dim=1)
        u2 = uu / nrm.clamp(min=1e-12)[:, None]
        L2 = L * nrm
        d2 = _seg_dist(torch.cat([p, torch.zeros_like(p[..., :1])], -1), torch.cat([P[:, :2], torch.zeros_like(P[:, :1])], 1),
                       torch.cat([u2, torch.zeros_like(u2[:, :1])], 1), torch.where(nrm > 1e-12, L2, torch.zeros_like(L2)))
        F[..., 4] = _ldist(d2, g)
        return F
    yi = t[:, ai["iy"]]
    kk = _keys(xi[:, None], yi[:, None], b[None, :], g) + koff[:, None]                           # (n, g)
    F[..., :N_OCC] = _occupied(oc, kk, j)
    F[..., 3] = _rad(xi, yi, g)[:, None].expand(n, g)
    p = torch.stack([xi[:, None].double().expand(n, g) + 0.5, yi[:, None].double().expand(n, g) + 0.5, b[None, :].double().expand(n, g) + 0.5], -1)
    F[..., 4] = _ldist(_seg_dist(p, P, u, L), g)
    return F


# ---------------------------------------------------------------- learned bias
class OccHeads(nn.Module):
    """Per head attribute: logit bias from the features, b = F w + sum_r (F U)_r (V y)_r; w, U zero-initialised."""

    def __init__(self, d_attr, rank=8):
        super().__init__()
        self.w = nn.ParameterDict({a: nn.Parameter(torch.zeros(N_FEAT)) for a in OCC_ATTRS})
        self.U = nn.ParameterDict({a: nn.Parameter(torch.zeros(N_FEAT, rank)) for a in OCC_ATTRS})
        self.V = nn.ModuleDict({a: nn.Linear(d_attr, rank) for a in OCC_ATTRS})
        for m in self.V.values():
            nn.init.normal_(m.weight, std=0.02); nn.init.zeros_(m.bias)

    def bias(self, name, F, y):
        """F (n, n_bins, 5), y (n, d_attr) -> (n, n_bins)"""
        F = F.to(self.w[name].dtype)
        return F @ self.w[name] + ((F @ self.U[name]) * self.V[name](y.to(self.w[name].dtype))[:, None, :]).sum(-1)


def occ_rows(name, tok, ai):
    """Rows whose `name` gets features: hung strokes for dx / dy / dz, unhung for ix / iy / iz."""
    st = (tok[:, ai["kind"]] == KIND_STROKE) | (tok[:, ai["kind"]] == KIND_CONT)
    hung = tok[:, ai["ptr"]] > 0
    return torch.nonzero(st & (hung if name in ("dx", "dy", "dz") else ~hung))[:, 0]


def apply_occ(model, name, lg, y, tok, oc):
    """Logits (N, n_bins) of attribute `name` plus the occupancy bias on the rows that have features (others unchanged)."""
    if name not in OCC_ATTRS or oc is None or "okeys" not in oc:
        return lg
    rows = occ_rows(name, tok, model.ai)
    if len(rows) == 0:
        return lg
    F = features(name, tok, oc, rows, model.ai)
    out = lg.clone()
    out[rows] = lg[rows] + model.occ.bias(name, F, y[rows]).to(lg.dtype)
    return out
