"""Small helpers shared by every module (no torch here)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

C_CM_PER_NS = 29.9792458
C_M_PER_NS = 0.299792458
VOX_CM = 400.0 / 192.0          # 2.083 cm voxels of the 192^3 cache
YIELD_CM = 343.0                # photons per cm at beta = 1 (max NPhotons per 1 cm record; Frank-Tamm 300-650 nm, n = 1.31)
N_ICE = 1.31


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def read_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def fresh_dir(base, name):
    """<base>/<name>, or <name>_2, _3 ... if it exists: outputs are never overwritten."""
    base = Path(base)
    out = base / name
    k = 2
    while out.exists():
        out = base / f"{name}_{k}"
        k += 1
    out.mkdir(parents=True)
    return out


def cache_meta(cache):
    meta = read_json(Path(cache) / "metadata.json")
    g = int(meta.get("grid", meta.get("data", {}).get("grid", 192)))
    ranges = np.asarray(meta.get("ranges", meta.get("data", {}).get("ranges", [[-2, 2], [-2, 2], [46, 50]])), dtype=np.float64)
    return meta, g, ranges


def split_ids(meta, split, n=0, seed=0):
    ids = [int(e) for e in meta["split"][split]]
    if n and n < len(ids):
        rng = np.random.default_rng(seed)
        ids = sorted(int(x) for x in rng.choice(ids, size=n, replace=False))
    return ids


def event_path(cache, eid):
    return Path(cache) / "events" / f"event_{int(eid):05d}.npz"


def load_event(cache, eid, keys=("ijk", "q", "off", "m", "t", "cnt")):
    with np.load(event_path(cache, eid)) as f:
        return {k: np.asarray(f[k]) for k in keys if k in f.files}


def pct(a, ps):
    a = np.asarray(a, dtype=np.float64)
    return [float(x) for x in np.percentile(a, ps)] if a.size else [float("nan")] * len(ps)
