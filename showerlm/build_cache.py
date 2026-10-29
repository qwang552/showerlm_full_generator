"""Build the full-shower voxel cache from raw CORSIKA 8 Cherenkov parquet files.

Box x, y in [-10, 10] m, z in [30, 50] m, 960^3 voxels (2.083 cm), raw time < 80 ns.  Per voxel: summed photons,
photon-weighted position, direction and time relative to the light front.  One shard of a condor job array per call.

    python -m showerlm.build_cache --ids-file ids.txt --out <cache dir> --shard 0 --nshards 10
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .common import log
from .voxelize import read_parquet, voxelize

RAW_ROOT = "/data/user/qwang/z0918_cherenkov"
OLD_CACHE = "/data/user/qwang/corsika_sparse_v1/data/processed_192"


def raw_path(raw_root, eid):
    return Path(raw_root) / f"shower_group_{int(eid)}" / "cherenkov" / "light.parquet"


def split_lists(old_cache, ids):
    """{train/val/test: [ids]} for the given ids, from the 12 ns cache's metadata (ids it does not list -> 'none')."""
    tag = {}
    meta = Path(old_cache) / "metadata.json"
    if meta.exists():
        for k, v in json.load(open(meta)).get("split", {}).items():
            for e in v:
                tag[int(e)] = k
    out = {}
    for e in ids:
        out.setdefault(tag.get(int(e), "none"), []).append(int(e))
    return out


def write_atomic(path, write):
    """write(tmp_path) then rename: a killed job never leaves a half-written file under the final name."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    write(tmp)
    os.replace(tmp, path)


def check_box(ranges, grid):
    """The geometric conventions the rest of the code relies on (see the module docstring)."""
    span = ranges[:, 1] - ranges[:, 0]
    if not np.allclose(span, span[0]):
        raise SystemExit(f"box must be a cube (one grid size for x, y, z): spans {span.tolist()} m")
    if not (np.isclose(ranges[0, 0], -ranges[0, 1]) and np.isclose(ranges[1, 0], -ranges[1, 1])):
        raise SystemExit("x and y ranges must be symmetric about the shower axis (x = y = 0)")
    if not np.isclose(ranges[2, 1], 50.0):
        raise SystemExit("the top of the box must be the injection depth z = 50 m (rules.cone / chains assume it)")
    return float(span[0] / grid * 100.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", default=RAW_ROOT)
    ap.add_argument("--old-cache", default=OLD_CACHE, help="12 ns cache: only metadata.json is read, for the train / val / test lists")
    ap.add_argument("--ids-file", required=True, help="text file of shower ids (whitespace separated)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ranges", type=float, nargs=6, default=[-10, 10, -10, 10, 30, 50], help="x0 x1 y0 y1 z0 z1 (m)")
    ap.add_argument("--grid", type=int, default=960)
    ap.add_argument("--time-max", type=float, default=80.0, help="raw time cut (ns)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    ranges = np.asarray(a.ranges, dtype=np.float64).reshape(3, 2)
    vox_cm = check_box(ranges, a.grid)
    ids_all = [int(x) for x in open(a.ids_file).read().split()]
    ids = ids_all[a.shard::a.nshards]
    out = Path(a.out)
    (out / "events").mkdir(parents=True, exist_ok=True)
    (out / "info").mkdir(parents=True, exist_ok=True)
    meta = dict(builder="showerlm.build_cache 0.6.0", grid=a.grid, ranges=ranges.tolist(), time_max=a.time_max, voxel_cm=vox_cm,
                raw_root=a.raw_root, ids_file=str(a.ids_file), n_ids=len(ids_all), split=split_lists(a.old_cache, ids_all))
    mpath = out / "metadata.json"
    if mpath.exists():
        old = json.load(open(mpath))
        diff = [k for k in ("grid", "ranges", "time_max", "raw_root", "split") if old.get(k) != meta.get(k)]
        if diff:
            raise SystemExit(f"{mpath} was written with other settings ({diff}); use another --out")
    else:
        write_atomic(mpath, lambda p: Path(p).write_text(json.dumps(meta, indent=1)))
    log(f"build_cache: {len(ids)} showers (shard {a.shard}/{a.nshards}) -> {out}; box {ranges.tolist()} m, grid {a.grid} "
        f"({vox_cm:.4f} cm voxels), t < {a.time_max} ns; host {os.uname().nodename}")
    for i, eid in enumerate(ids):
        f = out / "events" / f"event_{eid:05d}.npz"
        if f.exists():
            log(f"skip {f.name} (exists)")
            continue
        src = raw_path(a.raw_root, eid)
        if not src.exists():
            log(f"WARNING: no raw file for shower {eid}: {src}")
            continue
        t0 = time.time()
        cols = read_parquet(src)
        t_read = time.time() - t0
        arrays, info = voxelize(cols, ranges, a.grid, a.time_max, lightfront=True)
        if int(arrays["ijk"].max(initial=0)) >= 32767:
            raise SystemExit("grid too large for the int16 token format")
        w = cols["NPhotons"]
        info.update(eid=eid, q_raw=float(np.nansum(w)), n_vox=int(len(arrays["q"])), nph_max=float(np.nanmax(w)) if len(w) else 0.0,
                    q_kept_frac=float(info["q_kept"] / max(np.nansum(w), 1e-9)), seconds_read=round(t_read, 2),
                    seconds=round(time.time() - t0, 2))
        write_atomic(f, lambda p: np.savez_compressed(open(p, "wb"), **arrays))
        (out / "info" / f"event_{eid:05d}.json").write_text(json.dumps(info))
        log(f"{i + 1}/{len(ids)}: shower {eid}: {info['n_vox']} voxels, {info['q_kept_frac']:.4f} of the photons kept, {info['seconds']} s")
    (out / f"shard_{a.shard}.done").write_text(time.strftime("%Y-%m-%dT%H:%M:%S\n"))
    log("done")


if __name__ == "__main__":
    main()
