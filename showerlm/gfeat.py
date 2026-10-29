"""Global features G of a skeleton (9 numbers: depth and radius quantiles, light times, long-track counts), the
conditioning of the gplan skeleton model, and the table of training showers they are drawn from at sampling time.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .common import C_CM_PER_NS, log, write_json
from .tokens import KIND_CONT, Vocab, is_stroke

NAMES = ["depth10", "depth50", "depth90", "r50", "r90", "t50", "t90", "n_long1m", "n_long2m"]
UNITS = ["m", "m", "m", "m", "m", "ns", "ns", "", ""]
CUBE = [-10, 10, -10, 10, 30, 50]          # 0.6 cache box (m); the injection point is (0, 0, top of the box)
N_BINS = 32


def wquantile(x, w, qs):
    """Weighted quantiles (linear in the cumulative weight)."""
    x = np.asarray(x, dtype=np.float64); w = np.asarray(w, dtype=np.float64)
    if len(x) == 0 or w.sum() <= 0:
        return np.zeros(len(qs))
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w) - 0.5 * w
    return np.interp(np.asarray(qs) * w.sum(), c, x)


def chain_lengths(tok, vocab):
    """Length (m) of every chain of a skeleton: a STROKE and the CONT pieces that follow it."""
    ix = vocab.index
    tok = np.asarray(tok).reshape(-1, len(vocab.names))
    tok = tok[is_stroke(tok[:, ix["kind"]])]
    if len(tok) == 0:
        return np.zeros(0)
    new = tok[:, ix["kind"]] != KIND_CONT
    new[0] = True
    cid = np.cumsum(new) - 1
    seg_m = vocab.dec_len(tok[:, ix["len"]]) * vox_m(vocab)
    return np.bincount(cid, weights=seg_m)


def vox_m(vocab, ranges=CUBE):
    return float((ranges[1] - ranges[0]) / vocab.g)


def features(tok, vocab, ranges=CUBE):
    """The 9 raw global variables (NAMES) of one skeleton (T, A) int array."""
    r = np.asarray(ranges, dtype=np.float64).reshape(3, 2)
    s = vocab.decode(np.asarray(tok, dtype=np.int64).reshape(-1, len(vocab.names)), use_pointer=False)
    if len(s["len"]) == 0:
        return np.zeros(len(NAMES))
    vm = vox_m(vocab, ranges)
    head = s["ijk"].astype(np.float64) + 0.5 + np.clip(s["off"], -0.499, 0.499)            # voxel units
    frac = np.array([1.0, 3.0, 5.0]) / 6.0
    p = head[:, None, :] + (frac[None, :, None] * s["len"][:, None, None]) * s["u"][:, None, :]   # (T, 3, 3) voxel units
    xyz = r[:, 0][None, None, :] + p * vm                                                  # metres
    depth = (r[2, 1] - xyz[..., 2]).ravel()
    rad = np.hypot(xyz[..., 0], xyz[..., 1]).ravel()
    t = (s["t"][:, None] + frac[None, :] * s["len"][:, None] * vm * 100.0 / C_CM_PER_NS).ravel()
    w = np.repeat(s["Q"] / 3.0, 3)
    d10, d50, d90 = wquantile(depth, w, [0.1, 0.5, 0.9])
    r50, r90 = wquantile(rad, w, [0.5, 0.9])
    t50, t90 = wquantile(t, w, [0.5, 0.9])
    cl = chain_lengths(tok, vocab)
    return np.array([d10, d50, d90, r50, r90, t50, t90, float((cl >= 1.0).sum()), float((cl >= 2.0).sum())])


def make_edges(raw, n_bins=N_BINS):
    """Equal-count bin edges per variable (inner edges only; duplicates removed, so integer counts get fewer bins)."""
    qs = np.arange(1, n_bins) / n_bins
    return [np.unique(np.quantile(raw[:, k], qs)).tolist() for k in range(raw.shape[1])]


def to_bins(raw, edges):
    raw = np.atleast_2d(raw)
    return np.stack([np.searchsorted(np.asarray(e), raw[:, k], side="right") for k, e in enumerate(edges)], 1).astype(np.int64)


def _one(args):
    f, vocab_d, ranges = args
    vocab = Vocab.from_dict(vocab_d)
    with np.load(f) as d:
        return int(d["eid"]), features(d["tok"].astype(np.int64), vocab, ranges)


def table(tok_dir, workers=8, ranges=CUBE, max_events=0):
    tok_dir = Path(tok_dir)
    vocab_d = json.load(open(tok_dir / "vocab.json"))
    files = sorted((tok_dir / "events").glob("tokens_*.npz"))
    if max_events:
        files = files[:max_events]
    jobs = [(f, vocab_d, ranges) for f in files]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_one, jobs, chunksize=16))
    else:
        res = [_one(j) for j in jobs]
    eid = np.array([r[0] for r in res], dtype=np.int64)
    raw = np.stack([r[1] for r in res]) if res else np.zeros((0, len(NAMES)))
    return eid, raw


class GTable:
    """gfeat_<split>.npz + meta.json of a G directory: bins by eid (training) and the prior rows (generation)."""

    def __init__(self, gdir, split="train"):
        self.gdir = Path(gdir)
        self.meta = json.load(open(self.gdir / "meta.json"))
        d = np.load(self.gdir / f"gfeat_{split}.npz")
        self.eid, self.raw, self.bins = d["eid"], d["raw"], d["bins"]
        self.by_eid = {int(e): i for i, e in enumerate(self.eid)}
        self.edges = self.meta["edges"]

    def bins_of(self, eid):
        return self.bins[self.by_eid[int(eid)]]

    def draw(self, n, seed=0):
        """n rows of the table (with replacement): raw (n, K), bins (n, K), row indices."""
        idx = np.random.default_rng(seed).integers(0, len(self.eid), n)
        return self.raw[idx], self.bins[idx], idx

    def bins_of_raw(self, raw):
        return to_bins(raw, self.edges)


def markdown(eid_tr, raw_tr, raw_va, edges, q_tr=None):
    L = ["# G (physical global variables of the skeleton), 0.6.1g", "",
         f"{len(eid_tr)} training / {len(raw_va)} validation showers.  Photon-weighted over 3 points per skeleton stroke.", "",
         "| variable | unit | train p5 | p50 | p95 | val p50 | bins | corr. with skeleton photons |", "|---|---|---|---|---|---|---|---|"]
    for k, n in enumerate(NAMES):
        p5, p50, p95 = np.percentile(raw_tr[:, k], [5, 50, 95]) if len(raw_tr) else (0, 0, 0)
        v50 = np.percentile(raw_va[:, k], 50) if len(raw_va) else 0
        c = float(np.corrcoef(raw_tr[:, k], q_tr)[0, 1]) if (q_tr is not None and len(raw_tr) > 2 and raw_tr[:, k].std() > 0) else float("nan")
        L.append(f"| {n} | {UNITS[k]} | {p5:.3g} | {p50:.3g} | {p95:.3g} | {v50:.3g} | {len(edges[k]) + 1} | {c:.2f} |")
    if len(raw_tr) > 2:
        cc = np.corrcoef(raw_tr.T)
        L += ["", "Correlations between the variables (train):", "", "| | " + " | ".join(NAMES) + " |", "|---|" + "---|" * len(NAMES)]
        for k, n in enumerate(NAMES):
            L.append(f"| {n} | " + " | ".join(f"{cc[k, j]:.2f}" for j in range(len(NAMES))) + " |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="skeleton token folder of the training split (tokens_skel40_train)")
    ap.add_argument("--val", required=True, help="skeleton token folder of the validation split")
    ap.add_argument("--out", required=True, help="new folder for gfeat_train.npz, gfeat_val.npz, meta.json, gfeat.md")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-bins", type=int, default=N_BINS)
    ap.add_argument("--max-events", type=int, default=0, help="tests only")
    ap.add_argument("--ranges", type=float, nargs=6, default=CUBE)
    a = ap.parse_args()
    out = Path(a.out)
    if (out / "meta.json").exists():
        raise SystemExit(f"{out}/meta.json exists: G tables are never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    eid_tr, raw_tr = table(a.train, a.workers, a.ranges, a.max_events)
    log(f"train: {len(eid_tr)} skeletons")
    eid_va, raw_va = table(a.val, a.workers, a.ranges, a.max_events)
    log(f"val: {len(eid_va)} skeletons")
    edges = make_edges(raw_tr, a.n_bins)
    np.savez(out / "gfeat_train.npz", eid=eid_tr, raw=raw_tr, bins=to_bins(raw_tr, edges))
    np.savez(out / "gfeat_val.npz", eid=eid_va, raw=raw_va, bins=to_bins(raw_va, edges))
    write_json(out / "meta.json", dict(names=NAMES, units=UNITS, edges=edges, n_bins=int(a.n_bins), ranges=list(a.ranges), train=str(a.train), val=str(a.val),
                                       n_train=int(len(eid_tr)), n_val=int(len(eid_va))))
    q_tr = None
    try:
        idx = json.load(open(Path(a.train) / "index.json"))
        qe = {int(v["eid"]): float(v["q_total"]) for v in idx.values()}
        q_tr = np.array([qe.get(int(e), np.nan) for e in eid_tr])
        q_tr = None if np.isnan(q_tr).any() else q_tr
    except (OSError, ValueError, KeyError):
        pass
    (out / "gfeat.md").write_text(markdown(eid_tr, raw_tr, raw_va, edges, q_tr))
    log(f"G tables -> {out} (gfeat.md)")


if __name__ == "__main__":
    main()
