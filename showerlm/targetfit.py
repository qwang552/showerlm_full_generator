"""Skeleton light S, detail light D and detail strokes n of the training showers, and the fitted slope of D on S used
to set the detail light of a generated shower (sample06 --target-total prior).

    python -m showerlm.targetfit --tokens <tokens_det40_train> --out <target_fit.json> [--n 1500] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import log, write_json
from .data import TokenStore
from .group import stroke_photons


class TargetTotals:
    """Token photons (lq bins) of the showers of a tokens_det40_* folder, read when an eid is first asked for."""

    def __init__(self, root, vocab=None):
        self.store = TokenStore(root)                                              # every event, with or without skeleton
        self.vocab = self.store.vocab
        if vocab is not None and self.vocab.to_dict() != vocab.to_dict():
            raise SystemExit(f"target tokens {root}: vocabulary differs from the detail run's")
        self.file_of = {int(e): f for e, f in zip(self.store.eids, self.store.files)}
        self.cache = {}

    def check(self, eids):
        missing = sorted({int(e) for e in eids} - set(self.file_of))
        if missing:
            raise SystemExit(f"target tokens {self.store.root}: no event for {len(missing)} prior eids, e.g. {missing[:5]}")

    def of_eid(self, e):
        """(skeleton photons, detail photons, detail strokes) of one shower."""
        e = int(e)
        if e not in self.cache:
            with np.load(self.file_of[e]) as d:
                tok = d["tok"].astype(np.int64)
                skel = d["skel"].astype(np.int64) if "skel" in d.files else np.zeros((0, len(self.vocab.names)), np.int64)
            self.cache[e] = (float(stroke_photons(self.vocab, skel).sum()), float(stroke_photons(self.vocab, tok).sum()), int(len(tok)))
        return self.cache[e]


def slope(y, x):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return float(np.cov(y, x)[0, 1] / np.var(x, ddof=1)) if len(x) > 2 and np.var(x) > 0 else float("nan")


def corr(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1]) if (ok.sum() > 2 and x[ok].std() > 0 and y[ok].std() > 0) else float("nan")


def fit(totals, n=1500, seed=0):
    """b = slope of D on S over n random training showers, plus the reference statistics the reports compare with."""
    eids = np.asarray(sorted(totals.file_of), dtype=np.int64)
    pick = np.random.default_rng(seed).choice(eids, size=min(n, len(eids)), replace=False)
    v = np.array([totals.of_eid(e) for e in pick], dtype=np.float64)
    v = v[v[:, 0] > 0]                                                           # showers with a skeleton, as the generated statistics
    S, D, N = v[:, 0], v[:, 1], v[:, 2]
    T = S + D
    pps = D / np.maximum(N, 1.0)
    b = slope(D, S)
    resid = D - (D.mean() + b * (S - S.mean()))
    return dict(tokens=str(totals.store.root), n=int(len(S)), n_picked=int(len(pick)), seed=int(seed), b=b, b_se=float(resid.std(ddof=2) / (S.std(ddof=0) * np.sqrt(len(S)))) if len(S) > 2 else float("nan"),
                q_skel_mean=float(S.mean()), q_skel_std=float(S.std()), q_det_mean=float(D.mean()), q_det_std=float(D.std()),
                q_total_mean=float(T.mean()), q_total_std=float(T.std()), corr_skel_det=corr(S, D), corr_total_skel=corr(T, S),
                slope_det_on_skel=b, slope_ndet_on_skel=slope(N, S), n_det_mean=float(N.mean()),
                det_photons_per_stroke_mean=float(pps.mean()), slope_det_pps_on_skel=slope(pps, S), det_resid_std=float(resid.std()))


def load_fit(path):
    d = json.load(open(path))
    for k in ("b", "tokens", "n"):
        if k not in d:
            raise SystemExit(f"{path}: not a target fit (no {k})")
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", required=True, help="tokens_det40_* folder of the TRAINING split")
    ap.add_argument("--out", required=True, help="new json file")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"{a.out} exists: never overwritten")
    r = fit(TargetTotals(a.tokens), a.n, a.seed)
    write_json(a.out, r)
    log(f"target fit on {r['n']} training showers: b = {r['b']:.4f} +- {r['b_se']:.4f}; corr(S, D) {r['corr_skel_det']:.3f}, corr(T, S) {r['corr_total_skel']:.3f}, "
        f"total {r['q_total_mean']:.4g} +- {r['q_total_std']:.3g} -> {a.out}")


if __name__ == "__main__":
    main()
