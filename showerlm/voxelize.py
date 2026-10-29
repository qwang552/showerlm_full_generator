"""Raw CORSIKA 8 Cherenkov parquet -> one cache-format event (ijk, q, off, m, t, cnt), the same rules as
corsika_sparse_v2 `preprocess` (sum NPhotons per voxel, photon-weighted centroid / direction / time, t relative
to the light front).  Used for tests on a downloaded event and for re-preprocessing with another box or time window.

    python -m showerlm.voxelize --parquet .../shower_group_1772/cherenkov/light.parquet --out event_01772.npz \
        [--time-max 12] [--ranges -2 2 -2 2 46 50] [--grid 192] [--raw-time]
"""
from __future__ import annotations

import argparse

import numpy as np

from .common import C_M_PER_NS, log


def voxelize(cols, ranges, grid, time_max=12.0, lightfront=True):
    """cols: dict of 1-D float64 arrays NPhotons, time, posX/Y/Z and optionally dirX/Y/Z.  Returns the cache arrays."""
    ranges = np.asarray(ranges, dtype=np.float64)
    w = cols["NPhotons"]; t = cols["time"]
    xyz = np.stack([cols["posX"], cols["posY"], cols["posZ"]], 1)
    keep = np.isfinite(w) & (w > 0) & np.isfinite(t) & np.all(np.isfinite(xyz), 1)
    if time_max is not None:
        keep &= t < time_max
    keep &= np.all((xyz >= ranges[:, 0]) & (xyz <= ranges[:, 1]), 1)
    w, t, xyz = w[keep], t[keep], xyz[keep]
    has_dir = all(k in cols for k in ("dirX", "dirY", "dirZ"))
    d = np.stack([cols["dirX"], cols["dirY"], cols["dirZ"]], 1)[keep] if has_dir else None
    span = ranges[:, 1] - ranges[:, 0]
    idx = np.clip(np.floor((xyz - ranges[:, 0]) / span * grid).astype(np.int64), 0, grid - 1)
    key = (idx[:, 2] * grid + idx[:, 1]) * grid + idx[:, 0]
    uniq, inv = np.unique(key, return_inverse=True)
    n = len(uniq)
    q = np.bincount(inv, weights=w, minlength=n)
    cnt = np.bincount(inv, minlength=n)
    wx = np.stack([np.bincount(inv, weights=w * xyz[:, a], minlength=n) for a in range(3)], 1)
    wt = np.bincount(inv, weights=w * t, minlength=n)
    ijk = np.stack([uniq % grid, (uniq // grid) % grid, uniq // (grid * grid)], 1)
    centers = ranges[:, 0] + (ijk + 0.5) * span / grid
    off = (wx / q[:, None] - centers) / (span / grid)
    tt = wt / q
    if lightfront:
        tt = tt - (ranges[2, 1] - centers[:, 2]) / C_M_PER_NS
    m = np.stack([np.bincount(inv, weights=w * d[:, a], minlength=n) for a in range(3)], 1) / q[:, None] if has_dir else np.zeros((n, 3))
    return dict(ijk=ijk.astype(np.int32), q=q.astype(np.float32), off=off.astype(np.float32), m=m.astype(np.float32),
                t=tt.astype(np.float32), cnt=cnt.astype(np.int32)), dict(kept_rows=int(keep.sum()), raw_rows=int(len(keep)), q_kept=float(w.sum()))


def read_parquet(path, columns=("NPhotons", "time", "posX", "posY", "posZ", "dirX", "dirY", "dirZ")):
    import pyarrow.parquet as pq
    f = pq.ParquetFile(path)
    cols = [c for c in columns if c in f.schema.names]
    tab = f.read(columns=cols)
    return {c: tab.column(c).to_numpy(zero_copy_only=False).astype(np.float64) for c in cols}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-max", type=float, default=12.0, help="raw time cut in ns (<= 0: no cut)")
    ap.add_argument("--ranges", type=float, nargs=6, default=[-2, 2, -2, 2, 46, 50])
    ap.add_argument("--grid", type=int, default=192)
    ap.add_argument("--raw-time", action="store_true", help="store raw time instead of light-front time")
    a = ap.parse_args()
    cols = read_parquet(a.parquet)
    ranges = np.asarray(a.ranges, dtype=np.float64).reshape(3, 2)
    arrays, info = voxelize(cols, ranges, a.grid, None if a.time_max <= 0 else a.time_max, lightfront=not a.raw_time)
    np.savez(a.out, **arrays)
    log(f"{a.out}: {len(arrays['q'])} voxels, {info}")


if __name__ == "__main__":
    main()
