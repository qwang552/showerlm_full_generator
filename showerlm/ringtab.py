"""Optional per ring x depth rescaling of the planned detail strokes (sampling time, --target-ring); not used by the
default settings.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .common import log, write_json
from .group import GN_EXACT, GroupStore
from .tokens import Vocab
from .zplan import CUBE

R_EDGES = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]                                     # ring lower bounds (m), the last open
D_EDGES = [0.0, 2.0, 5.0, 10.0]                                                     # depth slice lower bounds (m), the last open


def cell_names(r_edges=R_EDGES, d_edges=D_EDGES):
    rn = [f"r{r_edges[k]:g}-{r_edges[k + 1]:g}" if k + 1 < len(r_edges) else f"r>{r_edges[k]:g}" for k in range(len(r_edges))]
    dn = [f"z{d_edges[k]:g}-{d_edges[k + 1]:g}" if k + 1 < len(d_edges) else f"z>{d_edges[k]:g}" for k in range(len(d_edges))]
    return [f"{r}_{d}" for d in dn for r in rn]


def piece_cells(skel, vocab, r_edges=R_EDGES, d_edges=D_EDGES, ranges=CUBE):
    """(K,) cell index of every skeleton piece: ring of its midpoint's radius x depth slice of its midpoint."""
    skel = np.asarray(skel, dtype=np.int64).reshape(-1, len(vocab.names))
    K = len(skel)
    if K == 0:
        return np.zeros(0, np.int64)
    ix = vocab.index
    r = np.asarray(ranges, dtype=np.float64).reshape(3, 2)
    vm = float((r[0, 1] - r[0, 0]) / vocab.g)
    head = vocab.head_point(skel)
    u = vocab.dec_dir(skel[:, ix["th"]], skel[:, ix["ph"]])
    L = vocab.dec_len(skel[:, ix["len"]]).astype(np.float64)
    mid = (head + 0.5 * L[:, None] * u) * vm + r[:, 0]                              # metres
    rad = np.hypot(mid[:, 0], mid[:, 1])
    depth = r[2, 1] - mid[:, 2]
    kr = np.clip(np.searchsorted(np.asarray(r_edges, dtype=np.float64), rad, side="right") - 1, 0, len(r_edges) - 1)
    kd = np.clip(np.searchsorted(np.asarray(d_edges, dtype=np.float64), depth, side="right") - 1, 0, len(d_edges) - 1)
    return kd * len(r_edges) + kr


def event_counts(tok, skel, grp, vocab, r_edges=R_EDGES, d_edges=D_EDGES, ranges=CUBE):
    """(n_c, p_c) of one event: detail strokes per cell (through the piece of their group) and pieces per cell."""
    C = len(r_edges) * len(d_edges)
    cells = piece_cells(skel, vocab, r_edges, d_edges, ranges)
    p = np.bincount(cells, minlength=C).astype(np.float64)
    g = np.clip(np.asarray(grp, dtype=np.int64), 0, max(len(skel) - 1, 0))
    n = np.bincount(cells[g], minlength=C).astype(np.float64) if len(g) and len(skel) else np.zeros(C)
    return n, p


def _one(args):
    f, vocab_d, r_edges, d_edges, ranges = args
    vocab = Vocab.from_dict(vocab_d)
    with np.load(f) as d:
        n, p = event_counts(d["tok"].astype(np.int64), d["skel"].astype(np.int64), d["grp"].astype(np.int64), vocab, r_edges, d_edges, ranges)
        return int(d["eid"]), n, p


def table(tok_dir, workers=8, r_edges=R_EDGES, d_edges=D_EDGES, ranges=CUBE, max_events=0):
    store = GroupStore(tok_dir, max_events)
    vocab_d = store.vocab.to_dict()
    jobs = [(f, vocab_d, list(r_edges), list(d_edges), list(ranges)) for f in store.files]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_one, jobs, chunksize=16))
    else:
        res = [_one(j) for j in jobs]
    C = len(r_edges) * len(d_edges)
    eid = np.array([r[0] for r in res], dtype=np.int64)
    n = np.stack([r[1] for r in res]) if res else np.zeros((0, C))
    p = np.stack([r[2] for r in res]) if res else np.zeros((0, C))
    return eid, n, p


class RingTable:
    """<dir>/meta.json (edges, ranges, m_c) + ringtab_<split>.npz."""

    def __init__(self, rdir, split="train"):
        self.rdir = Path(rdir)
        self.meta = json.load(open(self.rdir / "meta.json"))
        self.r_edges, self.d_edges, self.ranges = [float(x) for x in self.meta["r_edges"]], [float(x) for x in self.meta["d_edges"]], [float(x) for x in self.meta["ranges"]]
        self.m = np.asarray(self.meta["m"], dtype=np.float64)                        # strokes per piece per cell (pooled)
        d = np.load(self.rdir / f"ringtab_{split}.npz")
        self.eid, self.n, self.p = d["eid"], d["n"], d["p"]
        self.names = cell_names(self.r_edges, self.d_edges)

    def apply(self, vocab, skel, n_plan, nb, clip=(0.5, 2.0), seed=0):
        """n_plan (G,) planned strokes per group after nq (float; group g = piece g), nb (G,) drawn strokes bins (0 = planned
        empty).  Returns the rescaled plans (G,) and an info dict."""
        n_plan = np.asarray(n_plan, dtype=np.float64).copy()
        nb = np.asarray(nb)
        G = len(n_plan)
        C = len(self.m)
        cells = piece_cells(skel, vocab, self.r_edges, self.d_edges, self.ranges)
        if len(cells) != G:
            raise ValueError(f"ring apply: {G} groups but {len(cells)} pieces")
        live = nb > 0
        planned = np.bincount(cells[live], weights=n_plan[live], minlength=C)
        pieces = np.bincount(cells, minlength=C).astype(np.float64)
        target = self.m * pieces
        tot0 = float(n_plan[live].sum())
        tt = float(target[planned > 0].sum())                                       # the prior's shape at the planned total
        t_norm = tot0 / tt if tt > 0 else 1.0
        target = target * t_norm
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(planned > 0, target / np.maximum(planned, 1e-12), 1.0)
        gc = np.clip(g, clip[0], clip[1])
        gc = np.where(planned > 0, gc, 1.0)
        scaled = np.where(live, n_plan * gc[cells], 0.0)
        tot1 = float(scaled.sum())
        f_tot = tot0 / tot1 if tot1 > 0 else 1.0
        scaled = scaled * f_tot
        lo = np.floor(scaled)
        up = np.random.default_rng([abs(int(seed)), int(int(seed) < 0), 7927]).random(G) < (scaled - lo)
        n_int = np.where(live, np.maximum(lo + up, 1.0), 0.0)
        out = np.where(nb < GN_EXACT, n_int, scaled)
        out = np.where(live, out, 0.0)
        after = np.bincount(cells[live], weights=out[live], minlength=C)
        moved = 0.5 * float(np.abs(after - planned).sum()) / max(tot0, 1.0)
        used = planned > 0
        info = dict(ring_cells=int((pieces > 0).sum()), ring_g_min=float(gc[used].min()) if used.any() else 1.0, ring_g_max=float(gc[used].max()) if used.any() else 1.0,
                    ring_g_mean=float(gc[used].mean()) if used.any() else 1.0, ring_moved=moved, ring_clipped=int(((gc != g) & used).sum()), ring_f_tot=f_tot, ring_t_norm=t_norm,
                    ring_planned=planned, ring_after=after, ring_target=target, ring_pieces=pieces)
        return out, info

    def check(self, ranges):
        if [float(x) for x in ranges] != self.ranges:
            raise SystemExit(f"ring table {self.rdir}: ranges {self.ranges} != the run's {list(ranges)}")


def markdown(eid, n, p, n_va, p_va, r_edges, d_edges):
    names = cell_names(r_edges, d_edges)
    tn, tp = n.sum(0), p.sum(0)
    m = np.where(tp > 0, tn / np.maximum(tp, 1.0), 0.0)
    mv = np.where(p_va.sum(0) > 0, n_va.sum(0) / np.maximum(p_va.sum(0), 1.0), 0.0) if len(p_va) else np.zeros_like(m)
    L = ["# ringtab (detail strokes per skeleton piece, per ring x depth cell of the piece's midpoint), 0.6.3", "",
         f"{len(eid)} training / {len(n_va)} validation showers; rings {r_edges} m (last open), depth slices {d_edges} m (last open).", "",
         "| cell | pieces (train, total) | strokes (train, total) | strokes per piece m_c | val m_c | share of strokes |", "|---|---|---|---|---|---|"]
    for c, nm in enumerate(names):
        L.append(f"| {nm} | {tp[c]:.0f} | {tn[c]:.0f} | {m[c]:.2f} | {mv[c]:.2f} | {tn[c] / max(tn.sum(), 1.0):.4f} |")
    L += ["", f"Mean strokes per shower {n.sum(1).mean():.1f}, pieces per shower {p.sum(1).mean():.1f}."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="tokens_det40_train folder (0.6 tools/extract_skeleton.py: tok, skel, grp)")
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True, help="new folder for ringtab_train.npz, ringtab_val.npz, meta.json, ringtab.md")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-events", type=int, default=0, help="tests only")
    ap.add_argument("--r-edges", type=float, nargs="+", default=R_EDGES)
    ap.add_argument("--d-edges", type=float, nargs="+", default=D_EDGES)
    ap.add_argument("--ranges", type=float, nargs=6, default=CUBE)
    a = ap.parse_args()
    out = Path(a.out)
    if (out / "meta.json").exists():
        raise SystemExit(f"{out}/meta.json exists: ring tables are never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    eid, n, p = table(a.train, a.workers, a.r_edges, a.d_edges, a.ranges, a.max_events)
    log(f"train: {len(eid)} showers")
    eid_v, n_v, p_v = table(a.val, a.workers, a.r_edges, a.d_edges, a.ranges, a.max_events)
    log(f"val: {len(eid_v)} showers")
    np.savez(out / "ringtab_train.npz", eid=eid, n=n, p=p)
    np.savez(out / "ringtab_val.npz", eid=eid_v, n=n_v, p=p_v)
    tn, tp = n.sum(0), p.sum(0)
    m = np.where(tp > 0, tn / np.maximum(tp, 1.0), 0.0)
    write_json(out / "meta.json", dict(r_edges=list(a.r_edges), d_edges=list(a.d_edges), ranges=list(a.ranges), names=cell_names(a.r_edges, a.d_edges),
                                       m=m.tolist(), train=str(a.train), val=str(a.val), n_train=int(len(eid)), n_val=int(len(eid_v))))
    (out / "ringtab.md").write_text(markdown(eid, n, p, n_v, p_v, a.r_edges, a.d_edges))
    log(f"ring tables -> {out} (ringtab.md)")


if __name__ == "__main__":
    main()
