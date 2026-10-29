"""Renderer: stroke tokens -> cache-format voxels (ijk, q, off, m, t, cnt).  No learned parameters.

A stroke is a straight segment from a point in its head voxel along its direction; the voxels it crosses share its
photons in proportion to the path inside them (voxels crossed for less than W_MIN are skipped).
"""
from __future__ import annotations

import numpy as np

from .common import C_CM_PER_NS

W_MIN = 0.35          # minimum path (voxel units) for a crossed voxel to be lit (the head and tail voxels always are)


def stroke_voxels(head, u, length, g, w_min=W_MIN):
    """head: point in voxel units (3 floats); u: unit direction; length: segment length in voxel units.
    Returns ijk (k, 3) int64, s_mid (k,) distance along the line at the middle of each voxel's path (voxel units, 0 at
    the head point), w (k,) path length inside each voxel (voxel units; the head voxel gets the mean of the others)."""
    cell = np.floor(head).astype(np.int64)
    if length <= 1e-6 or np.linalg.norm(u) < 1e-6:
        return cell[None, :].copy(), np.zeros(1), np.ones(1)
    cells, s_in, s_out = [], [], []
    step = np.where(u > 0, 1, -1)
    with np.errstate(divide="ignore"):
        t_delta = np.where(np.abs(u) > 1e-12, 1.0 / np.abs(u), np.inf)
        boundary = cell + (u > 0)
        t_max = np.where(np.abs(u) > 1e-12, (boundary - head) / u, np.inf)
    t_entry = 0.0
    for _ in range(int(4 * length) + 4):
        if np.any(cell < 0) or np.any(cell >= g):
            break
        a = int(np.argmin(t_max))
        t_exit = min(float(t_max[a]), float(length))
        cells.append(cell.copy()); s_in.append(t_entry); s_out.append(t_exit)
        if t_exit >= length:
            break
        cell[a] += step[a]; t_max[a] += t_delta[a]; t_entry = t_exit
    ijk = np.array(cells, dtype=np.int64).reshape(-1, 3)
    s_in = np.array(s_in); s_out = np.array(s_out)
    w = s_out - s_in
    keep = (w >= w_min)
    keep[0] = True
    if len(keep) > 1:
        keep[-1] = True
    ijk, s_in, s_out, w = ijk[keep], s_in[keep], s_out[keep], w[keep]
    if len(w) > 1:
        w[0] = w[1:].mean()
    return ijk, 0.5 * (s_in + s_out), w


def render(strokes, g, ranges, lightfront=True):
    """strokes: dict of arrays over T strokes: ijk (T,3) head voxel, off (T,3) head offset in [-0.5,0.5], u (T,3) unit
    direction, len (T,) length in voxel units, t (T,) raw time of the head (ns), Q (T,) photons, C (T,) records, coh (T,) |m|.
    Returns cache-format arrays (ijk, q, off, m, t, cnt) sorted by voxel key."""
    ranges = np.asarray(ranges, dtype=np.float64)
    vox_cm = float((ranges[0, 1] - ranges[0, 0]) / g * 100.0)
    z_top_cm = float((ranges[2, 1] - ranges[2, 0]) * 100.0)
    IJK, Q, OFF, M, T, CNT = [], [], [], [], [], []
    for i in range(len(strokes["len"])):
        head = strokes["ijk"][i].astype(np.float64) + 0.5 + np.clip(strokes["off"][i], -0.499, 0.499)
        u = strokes["u"][i].astype(np.float64)
        ijk, s_mid, w = stroke_voxels(head, u, float(strokes["len"][i]), g)
        k = len(ijk)
        if k == 0:
            continue
        wn = w / max(w.sum(), 1e-9)
        q = float(strokes["Q"][i]) * wn
        cnt = np.maximum(1, np.rint(float(strokes["C"][i]) * wn)).astype(np.int64)
        pts = head[None, :] + s_mid[:, None] * u[None, :]
        off = np.clip(pts - (ijk + 0.5), -0.5, 0.5)
        off[0] = strokes["off"][i]
        t = float(strokes["t"][i]) + s_mid * vox_cm / C_CM_PER_NS
        IJK.append(ijk); Q.append(q); OFF.append(off); CNT.append(cnt); T.append(t)
        M.append(np.repeat((float(strokes["coh"][i]) * u)[None, :], k, 0))
    if not IJK:
        z = np.zeros((0, 3), np.float32)
        return dict(ijk=np.zeros((0, 3), np.int32), q=np.zeros(0, np.float32), off=z, m=z, t=np.zeros(0, np.float32), cnt=np.zeros(0, np.int32))
    ijk = np.concatenate(IJK); q = np.concatenate(Q); off = np.concatenate(OFF); m = np.concatenate(M); t = np.concatenate(T); cnt = np.concatenate(CNT)
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
        z_cm = (out_ijk[:, 2] + 0.5) * vox_cm                                  # light front at the voxel centre, as the cache preprocess
        ts = ts - (z_top_cm - z_cm) / C_CM_PER_NS
    return dict(ijk=out_ijk.astype(np.int32), q=qs.astype(np.float32), off=offs.astype(np.float32), m=ms.astype(np.float32),
                t=ts.astype(np.float32), cnt=cs.astype(np.int32))
