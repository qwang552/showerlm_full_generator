"""Depth plan of a skeleton ("zplan"): its light in 12 depth slices, the conditioning of the zplan skeleton model, and
the table of training showers it is drawn from at sampling time.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .common import log, write_json
from .tokens import Vocab

EDGES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0]   # slice lower bounds (m); 12 slices, the last open
CUBE = [-10, 10, -10, 10, 30, 50]                                        # 0.6 cache box (m); injection at (0, 0, top)
N_BINS = 48
Q_MAX = 1.0e8


def default_cfg():
    return dict(edges=list(EDGES), ranges=list(CUBE), n_bins=N_BINS, q_max=Q_MAX, jitter=0.0)


def token_slice_light(tok, vocab, edges=EDGES, ranges=CUBE):
    """(T, K) photons of each STROKE / CONT token in each depth slice (tok (T, A) int, stroke tokens only)."""
    tok = np.asarray(tok, dtype=np.int64).reshape(-1, len(vocab.names))
    K = len(edges)
    if len(tok) == 0:
        return np.zeros((0, K))
    s = vocab.decode(tok, use_pointer=False)
    if len(s["len"]) != len(tok):
        raise ValueError("token_slice_light: give stroke tokens only (PAD / EOS rows would shift the rows)")
    r = np.asarray(ranges, dtype=np.float64).reshape(3, 2)
    vm = float((r[0, 1] - r[0, 0]) / vocab.g)
    head_z = s["ijk"][:, 2].astype(np.float64) + 0.5 + np.clip(s["off"][:, 2], -0.499, 0.499)     # voxel units
    frac = np.array([1.0, 3.0, 5.0]) / 6.0
    z = r[2, 0] + (head_z[:, None] + frac[None, :] * s["len"][:, None] * s["u"][:, 2][:, None]) * vm   # (T, 3) metres
    depth = r[2, 1] - z
    k = np.clip(np.searchsorted(np.asarray(edges, dtype=np.float64), depth, side="right") - 1, 0, K - 1)
    out = np.zeros((len(tok), K))
    rows = np.repeat(np.arange(len(tok)), 3)
    np.add.at(out, (rows, k.ravel()), np.repeat(s["Q"] / 3.0, 3))
    return out


def plan_of(tok, vocab, edges=EDGES, ranges=CUBE):
    """(K,) planned photons per slice of a whole skeleton = the sum of token_slice_light."""
    return token_slice_light(tok, vocab, edges, ranges).sum(0)


def enc_z(x, n_bins=N_BINS, q_max=Q_MAX):
    """Photons (planned or still to come) -> log1p bin (vectorised); 0 = nothing (left)."""
    x = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    return np.clip(np.floor(np.log1p(x) / np.log1p(q_max) * (n_bins - 1) + 1e-9), 0, n_bins - 1).astype(np.int64)


def remaining_bins(tok, vocab, cfg, scale=None):
    """Per position k = 0 .. T: bins of the light still to come in every slice after the first k tokens, (T + 1, K),
    and the plan (K,) photons.  Row 0 is the plan itself (prefix), row k + 1 the budget input of token k.
    scale (K,): the plan is the tokens' slice light times scale (training jitter); None = exact."""
    sl = token_slice_light(tok, vocab, cfg["edges"], cfg["ranges"])
    plan = sl.sum(0) if scale is None else sl.sum(0) * np.asarray(scale, dtype=np.float64)
    rem = plan[None, :] - np.concatenate([np.zeros((1, sl.shape[1])), np.cumsum(sl, 0)], 0)
    rem[np.abs(rem) < 0.5] = 0.0                                                      # float noise of the cumulative sum
    return enc_z(rem, cfg["n_bins"], cfg["q_max"]), plan


def _one(args):
    f, vocab_d, edges, ranges = args
    vocab = Vocab.from_dict(vocab_d)
    with np.load(f) as d:
        return int(d["eid"]), plan_of(d["tok"].astype(np.int64), vocab, edges, ranges)


def table(tok_dir, workers=8, edges=EDGES, ranges=CUBE, max_events=0):
    tok_dir = Path(tok_dir)
    vocab_d = json.load(open(tok_dir / "vocab.json"))
    files = sorted((tok_dir / "events").glob("tokens_*.npz"))
    if max_events:
        files = files[:max_events]
    jobs = [(f, vocab_d, list(edges), list(ranges)) for f in files]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_one, jobs, chunksize=16))
    else:
        res = [_one(j) for j in jobs]
    eid = np.array([r[0] for r in res], dtype=np.int64)
    plan = np.stack([r[1] for r in res]) if res else np.zeros((0, len(edges)))
    return eid, plan


class ZTable:
    """zplan_<split>.npz + meta.json of a zplan directory: plans by eid (with gplan the row of the G draw's eid)."""

    def __init__(self, zdir, split="train"):
        self.zdir = Path(zdir)
        self.meta = json.load(open(self.zdir / "meta.json"))
        d = np.load(self.zdir / f"zplan_{split}.npz")
        self.eid, self.plan = d["eid"], d["plan"]
        self.by_eid = {int(e): i for i, e in enumerate(self.eid)}

    def plan_of_eids(self, eids):
        missing = [int(e) for e in eids if int(e) not in self.by_eid]
        if missing:
            raise SystemExit(f"zplan table {self.zdir} has no row for eids {missing[:5]}")
        return self.plan[[self.by_eid[int(e)] for e in eids]]

    def draw(self, n, seed=0):
        idx = np.random.default_rng(seed).integers(0, len(self.eid), n)
        return self.plan[idx], idx

    def check(self, cfg):
        for k in ("edges", "ranges"):
            if [float(x) for x in self.meta[k]] != [float(x) for x in cfg[k]]:
                raise SystemExit(f"zplan table {self.zdir}: {k} {self.meta[k]} != the model's {cfg[k]}")


def slice_names(edges):
    e = list(edges)
    return [f"{e[k]:g}-{e[k + 1]:g}" if k + 1 < len(e) else f">{e[k]:g}" for k in range(len(e))]


def markdown(eid_tr, plan_tr, plan_va, edges):
    names = slice_names(edges)
    L = ["# zplan (skeleton light per depth slice), 0.6.1z", "",
         f"{len(eid_tr)} training / {len(plan_va)} validation skeletons.  3 points per stroke, as showerlm.gfeat.", "",
         "| depth slice (m) | mean photons | mean fraction | fraction p5 | p50 | p95 | std / mean of fraction | val mean fraction |",
         "|---|---|---|---|---|---|---|---|"]
    tot = plan_tr.sum(1, keepdims=True).clip(1.0, None)
    fr = plan_tr / tot
    frv = plan_va / plan_va.sum(1, keepdims=True).clip(1.0, None) if len(plan_va) else np.zeros((0, len(edges)))
    for k, n in enumerate(names):
        p5, p50, p95 = np.percentile(fr[:, k], [5, 50, 95]) if len(fr) else (0, 0, 0)
        m = fr[:, k].mean() if len(fr) else 0.0
        cv = fr[:, k].std() / m if m > 0 else float("nan")
        vm = frv[:, k].mean() if len(frv) else float("nan")
        L.append(f"| {n} | {plan_tr[:, k].mean():.3g} | {m:.4f} | {p5:.4f} | {p50:.4f} | {p95:.4f} | {cv:.2f} | {vm:.4f} |")
    L += ["", f"Skeleton photons per shower: mean {tot.mean():.4g}, std {tot.std():.3g}."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="skeleton token folder of the training split (tokens_skel40_train)")
    ap.add_argument("--val", required=True, help="skeleton token folder of the validation split")
    ap.add_argument("--out", required=True, help="new folder for zplan_train.npz, zplan_val.npz, meta.json, zplan.md")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-events", type=int, default=0, help="tests only")
    ap.add_argument("--edges", type=float, nargs="+", default=EDGES)
    ap.add_argument("--ranges", type=float, nargs=6, default=CUBE)
    a = ap.parse_args()
    out = Path(a.out)
    if (out / "meta.json").exists():
        raise SystemExit(f"{out}/meta.json exists: zplan tables are never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    eid_tr, plan_tr = table(a.train, a.workers, a.edges, a.ranges, a.max_events)
    log(f"train: {len(eid_tr)} skeletons")
    eid_va, plan_va = table(a.val, a.workers, a.edges, a.ranges, a.max_events)
    log(f"val: {len(eid_va)} skeletons")
    np.savez(out / "zplan_train.npz", eid=eid_tr, plan=plan_tr)
    np.savez(out / "zplan_val.npz", eid=eid_va, plan=plan_va)
    write_json(out / "meta.json", dict(edges=list(a.edges), ranges=list(a.ranges), names=slice_names(a.edges), train=str(a.train), val=str(a.val),
                                       n_train=int(len(eid_tr)), n_val=int(len(eid_va))))
    (out / "zplan.md").write_text(markdown(eid_tr, plan_tr, plan_va, a.edges))
    log(f"zplan tables -> {out} (zplan.md)")


if __name__ == "__main__":
    main()
