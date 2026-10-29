"""ShowerLM 0.6 generation of full showers: the skeleton model first (chains >= 40 cm), then the group detail model.

    python -m showerlm.sample06 --skel-run <skeleton run> --det-run <detail run> --n 20 --out <dir> --name gen06

The skeleton is conditioned on a plan (G, zplan, tplan rows of a random training shower); the detail light is set
from that shower's detail light (--target-total prior).  See --help for the sampling guards.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .common import fresh_dir, log, write_json
from .detail import ATTR_CHUNK
from .group import GroupStore, generate_group_detail, stroke_photons
from .lockstep import generate_lockstep
from .place import OCC_KINDS, PLACE_MODES, STAT_KEYS, Placer
from .targetfit import TargetTotals, corr, fit, load_fit, slope
from .sample import generate, load_run
from .sample04 import save04
from .train import device_for, is_group, seq_len_at

CUBE = [-10, 10, -10, 10, 30, 50]


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def photon_summary(rows, ref=None):
    """0.6.1t: total photons, the detail-skeleton relation and how the realised detail follows the plan; NaN where it does
    not apply; ref = the truth's statistics (targetfit.fit) put next to them as truth_ref."""
    col = lambda k: np.array([float(r[k]) for r in rows])  # noqa: E731
    qs, qd, qt, nd = col("q_skel_tok"), col("q_det_tok"), col("q_total"), col("n_det")
    pd, pr, f = col("plan_q_det"), col("plan_q_det_raw"), col("target_f")
    nan = float("nan")
    lit = qs > 0                                                                 # showers with a skeleton
    pps = qd / np.maximum(nd, 1.0)
    s = dict(n_with_skeleton=int(lit.sum()), q_total_mean=float(qt.mean()), q_total_std=float(qt.std()), q_skel_mean=float(qs.mean()), q_skel_std=float(qs.std()),
             q_det_mean=float(qd.mean()), q_det_std=float(qd.std()), corr_skel_det=corr(qs[lit], qd[lit]), corr_total_skel=corr(qt[lit], qs[lit]),
             slope_det_on_skel=slope(qd[lit], qs[lit]), slope_ndet_on_skel=slope(nd[lit], qs[lit]), n_det_mean=float(nd.mean()),
             det_photons_per_stroke_mean=float(pps[lit].mean()) if lit.any() else nan, slope_det_pps_on_skel=slope(pps[lit], qs[lit]))
    ok = lit & np.isfinite(pd) & (pd > 0)
    if ok.any():
        rr = qd[ok] / pd[ok]
        p5, p50, p95 = np.percentile(rr, [5, 50, 95])
        s.update(corr_plan_det=corr(pd[ok], qd[ok]), det_over_plan_p5=float(p5), det_over_plan_p50=float(p50), det_over_plan_p95=float(p95),
                 ndet_over_plan_n=float(np.median(nd[ok] / np.maximum(col("plan_n_total")[ok], 1.0))),
                 ndet_over_plan_n_raw=float(np.median(nd[ok] / np.maximum(col("plan_n_total_raw")[ok], 1.0))))
    t = ok & np.isfinite(f)
    if t.any():
        fr, ff = col("target_f_raw")[t], f[t]
        y, x = np.log(np.maximum(qd[t], 1.0) / pr[t]), np.log(ff)
        control = bool((ff == 1.0).all())                                         # --target-clip 1 1: every shower "clipped" to 1
        s.update(control=control, target_f_mean=float(ff.mean()), target_f_std=float(ff.std()), target_f_min=float(ff.min()), target_f_max=float(ff.max()),
                 target_f_raw_mean=float(fr.mean()), target_f_raw_std=float(fr.std()),
                 n_target_clipped=None if control else int(sum(str(r["target_clipped"]) == "True" for r, k in zip(rows, t) if k)), q_target_std=float(col("q_target")[t].std()),
                 corr_total_target=corr(qt[t], col("q_target")[t]), gamma=slope(y, x), gamma_f_below_1=slope(y[ff < 1], x[ff < 1]),
                 gamma_f_above_1=slope(y[ff > 1], x[ff > 1]), n_f_below_1=int((ff < 1).sum()), n_f_above_1=int((ff > 1).sum()),
                 link_corr_skel_row=corr(qs[t], col("target_q_skel")[t]), skel_minus_row_mean=float((qs[t] - col("target_q_skel")[t]).mean()),
                 skel_minus_row_std=float((qs[t] - col("target_q_skel")[t]).std()))
    if ref is not None:
        s["truth_ref"] = {k: ref[k] for k in ("n", "q_total_mean", "q_total_std", "q_skel_mean", "q_skel_std", "q_det_mean", "q_det_std", "corr_skel_det",
                                              "corr_total_skel", "slope_det_on_skel", "slope_ndet_on_skel", "n_det_mean", "det_photons_per_stroke_mean",
                                              "slope_det_pps_on_skel") if k in ref}
    return s


class SkeletonsFrom:
    """0.6.1x --skeleton-from: the skeletons of an earlier sample06 output folder (tokens_XXXXX.npz tok_skel, in index order)
    with their stop reason and target row (timing.csv skel_stop / target_eid)."""

    def __init__(self, root, vocab):
        import csv
        self.root = Path(root)
        rj = json.loads((self.root / "run.json").read_text())
        if rj.get("vocab") != vocab.to_dict():
            raise SystemExit(f"--skeleton-from {root}: its vocabulary differs from the detail run's")
        self.rows = {int(r["index"]): r for r in csv.DictReader(open(self.root / "timing.csv"))}
        self.n = len(self.rows)
        if sorted(self.rows) != list(range(self.n)) or not (self.root / f"tokens_{self.n - 1:05d}.npz").exists():
            raise SystemExit(f"--skeleton-from {root}: timing.csv indices are not 0 .. n - 1 with a tokens file each")
        self.run = rj
        self.A = len(vocab.names)

    def load(self, i):
        with np.load(self.root / f"tokens_{i:05d}.npz") as d:
            return d["tok_skel"].astype(np.int64).reshape(-1, self.A), int(d["eid"])

    def stop(self, i):
        return self.rows[i]["skel_stop"]

    def target_eid(self, i):
        return int(self.rows[i].get("target_eid", -1))


def _norm(k, v):
    if k in ("det_run", "truth_skeleton") and v:
        return str(Path(v).resolve())
    return json.loads(json.dumps(v))


def setting_diffs(run, mine):
    """The settings of this run (dict) that differ from those an earlier run.json recorded (keys it lacks are skipped)."""
    return [f"{k} {_norm(k, v)!r} here, {_norm(k, run[k])!r} there" for k, v in mine.items() if k in run and _norm(k, v) != _norm(k, run[k])]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skel-run", default=None, help="training run of the skeleton model (ShowerLM on tokens_skel40_*)")
    ap.add_argument("--det-run", required=True, help="training run of the group detail model (model.type group)")
    ap.add_argument("--checkpoint", default="best", help="best | last (both runs, unless --skel-checkpoint says otherwise)")
    ap.add_argument("--skel-checkpoint", default=None)
    ap.add_argument("--truth-skeleton", default=None, help="tokens_det40_* folder: its TRUE skeletons instead of the skeleton model (diagnostic)")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--skel-batch", type=int, default=1, help="skeletons generated together (the detail part is one shower at a time)")
    ap.add_argument("--out", default="/data/user/qwang/showerlm_full/out")
    ap.add_argument("--name", default="gen06")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--skel-max-tokens", type=int, default=4096, help="skeleton length cap (also capped at the detail model's k_max)")
    ap.add_argument("--max-group", type=int, default=4096, help="detail: at most this many serial steps (strokes of the longest group)")
    ap.add_argument("--q-total-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="group-plan models: redraw the plans of all groups until the shower's planned photons are in [LO, HI] "
                         "(e.g. the 1st and 99th percentile of the training showers, tools/macro_eval.py prints them); off by default")
    ap.add_argument("--plan-tries", type=int, default=64, help="with --q-total-range: at most this many draws of the plans")
    ap.add_argument("--stop-at-plan", action="store_true", help="group-plan models: end a group at its planned number of strokes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ranges", type=float, nargs=6, default=CUBE, help="render box in metres (default: the 0.6 cube)")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--g-prior", default=None, help="gplan skeleton models: G folder whose TRAINING table is the prior (default: data.gfeat of the skeleton run)")
    ap.add_argument("--z-prior", default=None, help="zplan skeleton models: zplan folder whose TRAINING table is the prior (default: data.zplan_table of the skeleton run)")
    ap.add_argument("--target-total", choices=["off", "prior"], default="off",
                    help="0.6.1t, group-plan models: scale the drawn plans so that the planned detail photons = D_row + b (S_gen - S_row) (row = the G row's / "
                         "zplan row's / a random training shower; with --truth-skeleton the true shower itself); off = as 0.6")
    ap.add_argument("--target-scale", choices=["q", "nq"], default="q", help="with --target-total: scale the planned photons (q) or strokes and photons (nq)")
    ap.add_argument("--target-clip", type=float, nargs=2, default=[0.5, 2.0], metavar=("LO", "HI"), help="with --target-total: the scale factor is clipped to [LO, HI]")
    ap.add_argument("--target-tokens", default=None, help="with --target-total: tokens_det40_* folder of the TRAINING split (default: data.tokens_train of the detail run)")
    ap.add_argument("--target-fit", default=None, help="with --target-total: json of python -m showerlm.targetfit (slope b, truth statistics); default: fit here")
    ap.add_argument("--target-slope", type=float, default=None, help="with --target-total: b of D_row + b (S_gen - S_row) instead of the fitted one (-1 = full cancellation)")
    ap.add_argument("--place", choices=PLACE_MODES, default="off", help="0.6.1x placement rules of the detail strokes: x (head not on a drawn voxel), "
                    "u (unhung head nearest to its group's piece), xu (both); off = as 0.6.1t")
    ap.add_argument("--place-occ", choices=OCC_KINDS, default="heads", help="with --place x: what occupies a voxel: heads (head voxels of skeleton "
                    "and accepted detail strokes; exact for the truth), skel (the voxels the renderer draws for the skeleton + detail head voxels) "
                    "or path (the voxels the renderer draws for skeleton and detail)")
    ap.add_argument("--place-w-occ", type=float, default=0.5, help="with --place-occ skel / path: a voxel crossed for >= this many voxels is occupied")
    ap.add_argument("--place-tries", type=int, default=16, help="with --place: redraws per stroke and step; 0 = count only (tokens as off)")
    ap.add_argument("--place-keep-t", type=int, default=0, help="with --place: the first this many redraws keep the stroke's t, later ones redraw t too "
                    "(0 = every redraw redraws t: exact rejection sampling; > 0 is an approximation, see showerlm/place.py)")
    ap.add_argument("--skeleton-from", default=None, help="0.6.1x: a sample06 output folder whose skeletons (and target rows) are used instead of the skeleton model")
    ap.add_argument("--same-as", default=None, help="0.6.1x: a sample06 output folder; refuse unless this run's detail run, true-skeleton folder, n and detail settings equal its")
    ap.add_argument("--same-as-allow", nargs="*", default=[], choices=["det_run", "sampler", "ring"],
                    help="0.6.2: settings --same-as does not compare (det_run: a new detail model on the baseline's skeletons and settings; "
                         "0.6.3: sampler = --sampler / --det-batch, ring = --target-ring / --ring-clip)")
    ap.add_argument("--det-batch", type=int, default=1, help="0.6.3: showers whose detail is decoded in lockstep (one GPU); 1 = one at a time")
    ap.add_argument("--sampler", choices=["lockstep", "legacy"], default="lockstep", help="0.6.3: the lockstep detail sampler (default) or the 0.6.2 per-shower one")
    ap.add_argument("--t-prior", default=None, help="0.6.3 tplan skeleton models: tplan folder whose TRAINING table is the prior (default: data.tplan_table of the skeleton run)")
    ap.add_argument("--target-ring", default=None, help="0.6.3 nq_r: ring table folder (python -m showerlm.ringtab); needs --target-total prior --target-scale nq")
    ap.add_argument("--ring-clip", type=float, nargs=2, default=[0.5, 2.0], metavar=("LO", "HI"), help="with --target-ring: the per-cell factor is clipped to [LO, HI]")
    ap.add_argument("--skel-min-tokens", type=int, default=0, help="0.6.3.2: no skeleton EOS before this many tokens (sample.generate min_tokens; 0 = off). "
                    "The shortest training skeleton has 572 tokens; 0.6.3t / tz drew a few far shorter ones (33, 618) that the detail cannot make up")
    ap.add_argument("--skel-min-plan", type=float, default=0.0, help="0.6.3.2: no skeleton EOS while its light is below this fraction of its planned "
                    "skeleton light (zplan / tplan row; 0 = off).  0.7: normal showers reach >= 0.8 (v063z_fulla40 min 0.80); the early-EOS outliers 0.09 / 0.5")
    ap.add_argument("--skel-late-guard", type=float, default=0.0, help="0.6.4a/b: redraw new skeleton chain heads at lateness > 8 ns once the shower's light there (zplan models: plus the planned light above depth c (t - 8 ns), which the time order forces late) would "
                    "exceed max(--skel-late-k x its tplan light there, this fraction x its planned skeleton light) (sample.generate late_guard; 0 = off).  "
                    "0.05: training showers have <= 0.4%% of their light after 20 ns, the bad generated ones ~30%%")
    ap.add_argument("--skel-late-k", type=float, default=3.0, help="0.6.4a, with --skel-late-guard: multiple of the planned late light (tplan models)")
    ap.add_argument("--skel-late-tries", type=int, default=16, help="0.6.4a, with --skel-late-guard / --skel-late-jump: redraws per step before the last draw is kept")
    ap.add_argument("--skel-late-jump", type=float, default=0.0, help="0.6.4b (zplan models): also redraw a new chain head whose time forces more than this fraction "
                    "of the planned skeleton light late (the plan still to draw above depth c (t - 8 ns); sample.generate late_jump; 0 = off)")
    ap.add_argument("--skel-cap-stop", type=float, default=0.0, metavar="T_NS", help="0.6.4c (rules skeleton models): a new skeleton chain head drawn at t >= T_NS "
                    "(its t bin or later; >= the box end = the cap bin) ends the skeleton instead (stop 'cap'; sample.generate cap_stop_t; 0 = off).  "
                    "80: v064b drew up to 290 tokens per shower at the 80 ns cap, the truth 1 in 256 showers")
    ap.add_argument("--skel-max-plan", type=float, default=0.0, help="0.6.4c (zplan / tplan models): the skeleton stops at the first chain end once its light reaches "
                    "this multiple of its planned skeleton light (stop 'plan'; sample.generate max_plan_frac; 0 = off; >= --skel-min-plan).  v064b skeletons ended at 1.03-1.05")
    ap.add_argument("--skel-check-every", type=int, default=8, help="0.6.3: the skeleton sampler tests 'any shower alive' every this many steps (one host sync)")
    a = ap.parse_args()
    if a.skel_min_tokens < 0 or not (0.0 <= a.skel_min_plan < 1.0) or ((a.skel_min_tokens or a.skel_min_plan) and (a.truth_skeleton or a.skeleton_from)):
        raise SystemExit("--skel-min-tokens >= 0 / --skel-min-plan in [0, 1), and only with --skel-run (generated skeletons)")
    if (not (0.0 <= a.skel_late_guard < 1.0) or not (0.0 <= a.skel_late_jump < 1.0) or a.skel_late_k < 0 or a.skel_late_tries < 0
            or ((a.skel_late_guard or a.skel_late_jump) and (a.truth_skeleton or a.skeleton_from))):
        raise SystemExit("--skel-late-guard / --skel-late-jump in [0, 1), --skel-late-k / --skel-late-tries >= 0, and only with --skel-run (generated skeletons)")
    if (a.skel_cap_stop < 0 or not (a.skel_max_plan == 0 or max(0.5, a.skel_min_plan) <= a.skel_max_plan <= 2.0)
            or ((a.skel_cap_stop or a.skel_max_plan) and (a.truth_skeleton or a.skeleton_from))):
        raise SystemExit("--skel-cap-stop >= 0, --skel-max-plan 0 or in [max(0.5, --skel-min-plan), 2], and only with --skel-run (generated skeletons)")
    if a.det_batch < 1:
        raise SystemExit("--det-batch must be >= 1")
    if a.det_batch > 1 and (a.q_total_range or a.stop_at_plan):
        raise SystemExit("--det-batch > 1 does not support --q-total-range / --stop-at-plan (use --det-batch 1)")
    if a.target_ring and (a.target_total == "off" or a.target_scale != "nq"):
        raise SystemExit("--target-ring needs --target-total prior --target-scale nq")
    if a.target_ring and a.sampler != "lockstep":
        raise SystemExit("--target-ring needs --sampler lockstep")
    if a.target_ring and not (0.0 < a.ring_clip[0] <= 1.0 <= a.ring_clip[1]):
        raise SystemExit(f"--ring-clip {a.ring_clip}: need 0 < LO <= 1 <= HI")
    if not a.truth_skeleton and not a.skel_run and not a.skeleton_from:
        raise SystemExit("give --skel-run (or --truth-skeleton for the diagnostic, or --skeleton-from)")
    if a.truth_skeleton and a.skeleton_from:
        raise SystemExit("--truth-skeleton and --skeleton-from exclude each other")
    if not (0.0 <= a.place_w_occ < 1.0) or a.place_tries < 0 or a.place_keep_t < 0:
        raise SystemExit(f"--place-w-occ {a.place_w_occ} (need 0 <= w < 1) / --place-tries {a.place_tries} / --place-keep-t {a.place_keep_t} (need >= 0)")
    device = device_for(a.allow_cpu)
    det_model, vocab, det_cfg, det_ck = load_run(a.det_run, device, a.checkpoint)
    if not is_group(det_cfg):
        raise SystemExit(f"{a.det_run} is not a group detail run (model.type group)")
    has_plan = bool(getattr(det_model, "group_plan", False))
    if (a.q_total_range or a.stop_at_plan) and not has_plan:
        raise SystemExit("--q-total-range / --stop-at-plan need a group-plan detail model (model.group_plan)")
    target = a.target_total != "off"
    if target and (not has_plan or a.q_total_range):
        raise SystemExit("--target-total needs a group-plan detail model and no --q-total-range")
    if target and not (0.0 < a.target_clip[0] <= 1.0 <= a.target_clip[1]):
        raise SystemExit(f"--target-clip {a.target_clip}: need 0 < LO <= 1 <= HI")
    k_max = int(det_cfg["model"].get("k_max", 4096))
    skel_cap = min(a.skel_max_tokens, k_max)
    skel_model = skel_cfg = skel_ck = None
    if a.skel_run and (a.truth_skeleton or a.skeleton_from):
        log(f"--truth-skeleton / --skeleton-from given: --skel-run {a.skel_run} is not used")
        a.skel_run = None
    sfrom = SkeletonsFrom(a.skeleton_from, vocab) if a.skeleton_from else None
    if a.skel_run:
        skel_model, skel_vocab, skel_cfg, skel_ck = load_run(a.skel_run, device, a.skel_checkpoint or a.checkpoint)
        if skel_vocab.to_dict() != vocab.to_dict():
            raise SystemExit("the skeleton and detail runs use different vocabularies")
        skel_chunk = seq_len_at(skel_cfg["train"]["seq_len_schedule"], 10 ** 9)
        if skel_cap > skel_chunk:
            raise SystemExit(f"skeleton cap {skel_cap} > the skeleton model's training chunk {skel_chunk}: give --skel-max-tokens <= {skel_chunk}")
    gprior = None
    if skel_model is not None and getattr(skel_model, "n_gfeat", 0):
        from .gfeat import GTable
        gprior = GTable(a.g_prior or skel_cfg["data"]["gfeat"], "train")
        log(f"gplan skeleton: G drawn from the training table {gprior.gdir} ({len(gprior.eid)} rows)")
    zprior = None
    if skel_model is not None and getattr(skel_model, "n_zslice", 0):
        from .zplan import ZTable
        zprior = ZTable(a.z_prior or skel_cfg["data"]["zplan_table"], "train")
        zprior.check(skel_model.zcfg)
        log(f"zplan skeleton: plans drawn from the training table {zprior.zdir} ({len(zprior.eid)} rows)" + (", same shower as the G row" if gprior is not None else ""))
    tprior = None
    if skel_model is not None and getattr(skel_model, "n_tslice", 0):
        from .tplan import TTable
        tprior = TTable(a.t_prior or skel_cfg["data"]["tplan_table"], "train")
        tprior.check(skel_model.tcfg)
        if gprior is None and zprior is None:
            raise SystemExit("a tplan skeleton model needs a G or zplan prior to pick the training shower of its plan")
        log(f"tplan skeleton: plans drawn from the training table {tprior.tdir} ({len(tprior.eid)} rows), same shower as the G / zplan row")
    ring = None
    if a.target_ring:
        from .ringtab import RingTable
        ring = RingTable(a.target_ring, "train")
        ring.check(a.ranges)
        log(f"nq_r: planned strokes rescaled per ring x depth cell from {ring.rdir} ({len(ring.eid)} training showers), clip {a.ring_clip}")
    totals = tfit = None
    b_slope = nan = float("nan")
    if target:
        if a.target_fit:
            tfit = load_fit(a.target_fit)
        if not a.truth_skeleton:
            totals = TargetTotals(a.target_tokens or det_cfg["data"]["tokens_train"], vocab)
            if gprior is not None:
                totals.check(gprior.eid)
            elif zprior is not None:
                totals.check(zprior.eid)
            if tfit is None:
                tfit = fit(totals, 1500, 0)
            elif str(Path(tfit["tokens"]).resolve()) != str(totals.store.root.resolve()):
                log(f"WARNING: target fit {a.target_fit} was made on {tfit['tokens']}, the targets come from {totals.store.root}")
        b_slope = a.target_slope if a.target_slope is not None else (float(tfit["b"]) if tfit is not None else -1.0)
        if not np.isfinite(b_slope):
            raise SystemExit(f"target slope b = {b_slope} is not finite (fit {a.target_fit or 'made here'})")
        log(f"target total: rows = " + ("the true showers themselves" if a.truth_skeleton else f"training showers of {totals.store.root} ("
            + ("those of " + a.skeleton_from if sfrom is not None else "the G row's" if gprior is not None else "the zplan row's" if zprior is not None else "random") + ")")
            + f"; planned detail = D_row + b (S_gen - S_row), b = {b_slope:.4f}" + (" (given)" if a.target_slope is not None else " (fit)" if tfit is not None else "")
            + f"; scale {a.target_scale}, clip {a.target_clip}")
    elif a.target_fit or a.target_slope is not None or a.target_tokens:
        log("--target-fit / --target-slope / --target-tokens are ignored without --target-total")
    grows, zrows, trows, rrows = [], [], [], []
    truth = GroupStore(a.truth_skeleton) if a.truth_skeleton else None
    n_total = min(a.n, len(truth)) if truth else (min(a.n, sfrom.n) if sfrom else a.n)
    if sfrom is not None and target:
        if sfrom.run.get("truth_skeleton"):
            raise SystemExit(f"--skeleton-from {a.skeleton_from} has true skeletons: with --target-total use --truth-skeleton {sfrom.run['truth_skeleton']} "
                             "(the same skeletons; target row = the own true shower)")
        bad = [i for i in range(n_total) if sfrom.target_eid(i) < 0]
        if bad:
            raise SystemExit(f"--skeleton-from {a.skeleton_from}: no target row (target_eid) for showers {bad[:5]} ...; was it made with --target-total?")
        totals.check([sfrom.target_eid(i) for i in range(n_total)])
    mine = dict(seed=a.seed, temperature=a.temperature, max_group=a.max_group, stop_at_plan=a.stop_at_plan,
                ranges=np.asarray(a.ranges, dtype=np.float64).reshape(3, 2).tolist(), q_total_range=a.q_total_range, plan_tries=a.plan_tries,
                target_total=a.target_total, target_scale=a.target_scale if target else None, target_clip=a.target_clip if target else None,
                target_slope=b_slope if target else None)
    # 0.6.3: the sampler / detail batch (the random stream of the detail) and the ring table change the detail too; a run made
    # with --det-batch 1 by either sampler has the 0.6.2 detail stream, so those compare as equal
    s_mine = dict(sampler=("legacy" if a.det_batch == 1 else a.sampler), det_batch=a.det_batch)
    r_mine = dict(target_ring=str(Path(a.target_ring).resolve()) if a.target_ring else None, ring_clip=a.ring_clip if a.target_ring else None)
    same_run = json.loads((Path(a.same_as) / "run.json").read_text()) if a.same_as else None

    def cmp_settings(rj, allow):
        m = dict(mine)
        if "sampler" not in allow:
            m.update({k: x for k, x in s_mine.items()})
        if "ring" not in allow:
            m.update(r_mine)
        return m
    for flag, rj, extra, allow in (("--skeleton-from " + str(a.skeleton_from), sfrom.run if sfrom else None, {}, ["sampler", "ring"]),
                                   ("--same-as " + str(a.same_as), same_run, {k: x for k, x in dict(det_run=a.det_run, truth_skeleton=a.truth_skeleton, n=n_total).items()
                                                                              if k not in a.same_as_allow}, a.same_as_allow)):
        # a run.json of 0.6.2 or older lacks the 0.6.3 keys: it was made one shower at a time without a ring table
        rj_full = {**dict(sampler="legacy", det_batch=1, target_ring=None, ring_clip=None), **rj} if rj is not None else None
        diff = setting_diffs(rj_full, {**extra, **cmp_settings(rj_full, allow)}) if rj is not None else []
        if diff:
            raise SystemExit(f"{flag}: differs from that run in what changes the detail: {'; '.join(diff)}")
    out = fresh_dir(a.out, a.name)
    if tfit is not None:
        write_json(out / "target_fit.json", tfit)
    ranges = np.asarray(a.ranges, dtype=np.float64).reshape(3, 2)
    sk_src = "TRUE skeletons from " + a.truth_skeleton if truth else ("skeletons of " + a.skeleton_from if sfrom else str(skel_ck))
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    log(f"skeleton: {sk_src}; detail: {det_ck}; on {device} ({gpu_name}); {n_total} showers -> {out}; sampler {a.sampler}, detail batch {a.det_batch}, skeleton batch {a.skel_batch}"
        + (f"; placement rules {a.place} (occ {a.place_occ}, w_occ {a.place_w_occ}, tries {a.place_tries}, keep t {a.place_keep_t})" if a.place != "off" else ""))
    write_json(out / "run.json", dict(mode="generate06", skel_run=a.skel_run, det_run=a.det_run, skel_checkpoint=str(skel_ck), det_checkpoint=str(det_ck),
                                      truth_skeleton=a.truth_skeleton, n=n_total, skel_batch=a.skel_batch, temperature=a.temperature, skel_max_tokens=skel_cap,
                                      max_group=a.max_group, seed=a.seed, group_plan=has_plan, q_total_range=a.q_total_range, plan_tries=a.plan_tries,
                                      stop_at_plan=a.stop_at_plan, vocab=vocab.to_dict(), ranges=ranges.tolist(), skel_config=skel_cfg, det_config=det_cfg,
                                      device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else "",
                                      g_prior=str(gprior.gdir) if gprior is not None else None, z_prior=str(zprior.zdir) if zprior is not None else None,
                                      target_total=a.target_total, target_scale=a.target_scale if target else None, target_clip=a.target_clip if target else None,
                                      target_tokens=(str(totals.store.root) if totals is not None else ("own true shower" if target else None)),
                                      target_slope=b_slope if target else None, target_fit=(a.target_fit or "fitted here") if tfit is not None else None,
                                      place=a.place, place_occ=a.place_occ, place_w_occ=a.place_w_occ, place_tries=a.place_tries, place_keep_t=a.place_keep_t, skeleton_from=a.skeleton_from, same_as=a.same_as, same_as_allow=a.same_as_allow,
                                      sampler=s_mine["sampler"], det_batch=a.det_batch, sampler_flag=a.sampler, skel_check_every=a.skel_check_every, skel_min_tokens=a.skel_min_tokens, skel_min_plan=a.skel_min_plan,
                                      skel_late_guard=a.skel_late_guard, skel_late_k=a.skel_late_k, skel_late_tries=a.skel_late_tries, skel_late_jump=a.skel_late_jump, skel_cap_stop=a.skel_cap_stop, skel_max_plan=a.skel_max_plan,
                                      t_prior=str(tprior.tdir) if tprior is not None else None, target_ring=r_mine["target_ring"], ring_clip=r_mine["ring_clip"],
                                      skeleton_from_run=(dict(skel_run=sfrom.run.get("skel_run"), skel_checkpoint=sfrom.run.get("skel_checkpoint"), seed=sfrom.run.get("seed"),
                                                              target_fit=sfrom.run.get("target_fit")) if sfrom else None)))
    rows = []
    oom_splits = []
    nan = float("nan")
    i = 0
    while i < n_total:
        B = min(a.skel_batch if not (truth or sfrom) else max(a.skel_batch, a.det_batch), n_total - i)   # 0.6.3.1: given skeletons are read det-batch at a time
        sync(device)
        t0 = time.time()
        if sfrom is not None:
            loaded = [sfrom.load(i + b) for b in range(B)]
            skels = [x[0] for x in loaded]
            eids = [x[1] for x in loaded]
            sk_stop = [sfrom.stop(i + b) for b in range(B)]
            sk_late = (np.zeros(B, np.int64), np.zeros(B, np.int64), np.zeros(B, np.int64), np.zeros(B, np.int64))
            n_det_truth = [-1] * B
            t_eid = [sfrom.target_eid(i + b) for b in range(B)] if target else [-1] * B
        elif truth:
            loaded = [truth.load(i + b) for b in range(B)]
            skels = [x[1] for x in loaded]
            eids = [int(truth.eids[i + b]) for b in range(B)]
            sk_stop = ["truth"] * B
            sk_late = (np.zeros(B, np.int64), np.zeros(B, np.int64), np.zeros(B, np.int64), np.zeros(B, np.int64))
            n_det_truth = [len(x[0]) for x in loaded]
            t_eid = list(eids) if target else [-1] * B
        else:
            g_raw = g_bins = g_idx = None
            if gprior is not None:
                g_raw, g_bins, g_idx = gprior.draw(B, seed=a.seed * 100003 + i + 13)
            z_plan = z_eid = None
            if zprior is not None:
                if gprior is not None:
                    z_eid = gprior.eid[g_idx]
                    z_plan = zprior.plan_of_eids(z_eid)
                else:
                    z_plan, z_idx = zprior.draw(B, seed=a.seed * 100003 + i + 17)
                    z_eid = zprior.eid[z_idx]
            t_plan = None
            if tprior is not None:
                t_plan = tprior.plan_of_eids(z_eid if z_eid is not None else gprior.eid[g_idx])
            skels, sk_plan = generate(skel_model, vocab, B, device, a.temperature, skel_cap, 10 ** 9, seed=a.seed * 100003 + i, verbose=False,
                                      birth_ptr=bool(skel_cfg["data"].get("birth_ptr", True)), constraints=True, gbins=g_bins, zplan=z_plan, tplan=t_plan,
                                      check_every=a.skel_check_every, min_tokens=a.skel_min_tokens, min_plan_frac=a.skel_min_plan,
                                      late_guard=a.skel_late_guard, late_k=a.skel_late_k, late_tries=a.skel_late_tries, late_jump=a.skel_late_jump,
                                      cap_stop_t=int(vocab.enc_t(min(a.skel_cap_stop, vocab.t_cut - 1e-9))) if a.skel_cap_stop > 0 else None,
                                      max_plan_frac=a.skel_max_plan)
            if tprior is not None:
                from .tplan import plan_of as tplan_of
                tc = skel_model.tcfg
                te = z_eid if z_eid is not None else gprior.eid[g_idx]
                for b in range(B):
                    got = tplan_of(skels[b], vocab, tc["edges"], tc["ranges"])
                    trows.append(dict(index=i + b, prior_eid=int(te[b]), **{f"want_t{k}": round(float(t_plan[b, k]), 1) for k in range(len(got))},
                                      **{f"got_t{k}": round(float(got[k]), 1) for k in range(len(got))}))
            if zprior is not None:
                from .zplan import plan_of
                zc = skel_model.zcfg
                for b in range(B):
                    got = plan_of(skels[b], vocab, zc["edges"], zc["ranges"])
                    zrows.append(dict(index=i + b, prior_eid=int(z_eid[b]), **{f"want_z{k}": round(float(z_plan[b, k]), 1) for k in range(len(got))},
                                      **{f"got_z{k}": round(float(got[k]), 1) for k in range(len(got))}))
            if gprior is not None:
                from .gfeat import NAMES, features
                for b in range(B):
                    got = features(skels[b], vocab, a.ranges)
                    grows.append(dict(index=i + b, prior_row=int(g_idx[b]), prior_eid=int(gprior.eid[g_idx[b]]),
                                      **{f"want_{n}": round(float(g_raw[b, k]), 4) for k, n in enumerate(NAMES)},
                                      **{f"got_{n}": round(float(got[k]), 4) for k, n in enumerate(NAMES)}))
            eids = [-1] * B
            sk_stop = sk_plan["stop"]
            sk_late = (sk_plan["late_redraws"], sk_plan["late_fallbacks"], sk_plan["late_jumps"], sk_plan["cap_stops"])
            n_det_truth = [-1] * B
            t_eid = [-1] * B
            if totals is not None:
                if gprior is not None:
                    t_eid = [int(e) for e in gprior.eid[g_idx]]
                elif zprior is not None:
                    t_eid = [int(e) for e in z_eid]
                else:
                    t_eid = [int(e) for e in np.random.default_rng(a.seed * 100003 + i + 19).choice(totals.store.eids, B)]
        sync(device)
        skel_s = (time.time() - t0) / B
        q_tgt, t_row = [None] * B, [(nan, nan)] * B                              # targets: outside the skeleton timing (file reads)
        if target:
            for b in range(B):
                if truth and sfrom is None:
                    tok_b, sk_b = loaded[b][0], loaded[b][1]
                    t_row[b] = (float(stroke_photons(vocab, sk_b).sum()), float(stroke_photons(vocab, tok_b).sum()))
                else:
                    t_row[b] = totals.of_eid(t_eid[b])[:2]
                s_gen = float(stroke_photons(vocab, skels[b]).sum())
                q_tgt[b] = s_gen + t_row[b][1] + b_slope * (s_gen - t_row[b][0])
        S = max(1, a.det_batch)
        results = {}
        todo = [list(range(b0, min(B, b0 + S))) for b0 in range(0, B, S)]
        while todo:
            bs = todo.pop(0)
            sync(device)
            t1 = time.time()
            placers = [Placer(vocab, skels[b], ranges, a.place, a.place_w_occ, a.place_tries, a.place_occ, a.place_keep_t) if a.place != "off" else None for b in bs]
            if a.sampler == "lockstep":
                try:
                    res = generate_lockstep(det_model, vocab, [skels[b] for b in bs], device, a.temperature, max_steps=a.max_group,
                                            seeds=[a.seed * 100003 + i + b + 7 for b in bs], q_targets=[q_tgt[b] for b in bs], target_scale=a.target_scale,
                                            target_clip=tuple(a.target_clip), placers=placers, ring=ring, ring_clip=tuple(a.ring_clip))
                except torch.cuda.OutOfMemoryError:                                # 0.6.3.1: this group again in two halves (fresh placers)
                    if len(bs) == 1:
                        raise
                    torch.cuda.empty_cache()
                    h = len(bs) // 2
                    todo[:0] = [bs[:h], bs[h:]]
                    oom_splits.append(dict(index=i + bs[0], n=len(bs)))
                    log(f"detail batch of {len(bs)} showers (from {i + bs[0]}) ran out of GPU memory: again as {h} + {len(bs) - h}")
                    continue
            else:
                res = [generate_group_detail(det_model, vocab, skels[b], device, a.temperature, max_steps=a.max_group, seed=a.seed * 100003 + i + b + 7,
                                             q_total_range=a.q_total_range, plan_tries=a.plan_tries, stop_at_plan=a.stop_at_plan,
                                             q_target=q_tgt[b], target_scale=a.target_scale, target_clip=tuple(a.target_clip), placer=pl) for b, pl in zip(bs, placers)]
            sync(device)
            det_batch_s = (time.time() - t1) / len(bs)
            for b, (det, info), pl in zip(bs, res, placers):
                results[b] = (det, info, pl, det_batch_s)
        for b in range(B):
            det, info, placer, det_s = results[b]
            r = save04(out, i + b, skels[b], det, vocab, ranges, skel_s + det_s, eids[b])
            if "ring_cells" in info:
                rrows.append(dict(index=i + b, **{k: (round(float(info[k]), 4) if isinstance(info[k], float) else int(info[k])) for k in
                                                  ("ring_cells", "ring_g_min", "ring_g_max", "ring_g_mean", "ring_moved", "ring_clipped", "ring_f_tot", "ring_t_norm")},
                                  **{f"planned_{n}": round(float(x), 1) for n, x in zip(ring.names, info["ring_planned"])},
                                  **{f"after_{n}": round(float(x), 1) for n, x in zip(ring.names, info["ring_after"])},
                                  **{f"target_{n}": round(float(x), 1) for n, x in zip(ring.names, info["ring_target"])}))
            if "plan_bn" in info:                                                      # 0.6.1t: per-group plans, for per-group checks
                np.savez_compressed(out / f"plans_{i + b:05d}.npz", bn=info["plan_bn"], bq=info["plan_bq"], n=info["plan_n"], q=info["plan_q"],
                                    group_of=np.asarray(info["group_of"], np.int32), step=np.asarray(info.get("step_of", np.zeros(0)), np.int32))
            elif not has_plan:                                                         # 0.6.1x: the generating row, no-plan models too
                np.savez_compressed(out / f"groups_{i + b:05d}.npz", group_of=np.asarray(info.get("group_of", np.zeros(0)), np.int32),
                                    step=np.asarray(info.get("step_of", np.zeros(0)), np.int32))             # 0.6.2: sampler step of every stroke
            if placer is not None:                                                     # 0.6.1x: per detail stroke of tok_det
                rec = info.get("place_rec") or dict(redraws=np.zeros(0, np.int16), dt=np.zeros(0, np.int16), fb=np.zeros(0, np.uint8))
                np.savez_compressed(out / f"place_{i + b:05d}.npz", **rec)
            pst = info.get("place", {})
            rows.append(dict(index=i + b, eid=eids[b], n_skel=int(len(skels[b])), skel_stop=sk_stop[b], skel_late_redraws=int(sk_late[0][b]), skel_late_fallbacks=int(sk_late[1][b]), skel_late_jumps=int(sk_late[2][b]), skel_cap_stops=int(sk_late[3][b]), n_det=int(len(det)), n_det_truth=n_det_truth[b],
                             groups=info["groups"], steps=info["steps"], max_group=info["max_group"], hit_max_group=int(info["steps"] >= a.max_group),
                             plan_q_total=round(info.get("plan_q_total", float("nan")), 1), plan_n_total=info.get("plan_n_total", float("nan")),
                             plan_tries=info.get("plan_tries", 0), plan_in_range=info.get("plan_in_range"),
                             n_vox=int(len(r["q"])), q_total=float(r["q"].sum()), skel_s=round(skel_s, 3), det_s=round(det_s, 3), total_s=round(skel_s + det_s, 3),
                             q_skel_tok=round(float(stroke_photons(vocab, skels[b]).sum()), 1), q_det_tok=round(float(stroke_photons(vocab, det).sum()), 1),
                             plan_q_det_raw=round(info.get("plan_q_det_raw", nan), 1), plan_q_det=round(info.get("plan_q_det", nan), 1),
                             plan_n_total_raw=info.get("plan_n_total_raw", nan), plan_q_det_empty=round(info.get("plan_q_det_empty", nan), 1),
                             target_eid=t_eid[b], target_q_skel=round(t_row[b][0], 1), target_q_det=round(t_row[b][1], 1),
                             q_target=round(q_tgt[b], 1) if q_tgt[b] is not None else nan, target_f=round(info.get("target_f", nan), 6),
                             target_f_raw=round(info.get("target_f_raw", nan), 6), target_clipped=info.get("target_clipped"),
                             **{k: (round(float(pst[k]), 3) if k == "place_s" else int(pst[k])) if k in pst else -1 for k in STAT_KEYS}))
            log(f"  {i + b + 1}/{n_total}: {rows[-1]}")
        i += B
    cols = list(rows[0])
    with open(out / "timing.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in cols) + "\n")
    if grows:
        gc = list(grows[0])
        with open(out / "gcheck.csv", "w") as f:
            f.write(",".join(gc) + "\n")
            for r in grows:
                f.write(",".join(str(r[k]) for k in gc) + "\n")
        from .gfeat import NAMES
        want = np.array([[r[f"want_{n}"] for n in NAMES] for r in grows]); got = np.array([[r[f"got_{n}"] for n in NAMES] for r in grows])
        cc = [float(np.corrcoef(want[:, k], got[:, k])[0, 1]) if (len(grows) > 2 and want[:, k].std() > 0 and got[:, k].std() > 0) else float("nan") for k in range(len(NAMES))]
        write_json(out / "gcheck_summary.json", {n: dict(want_mean=float(want[:, k].mean()), got_mean=float(got[:, k].mean()), corr=cc[k]) for k, n in enumerate(NAMES)})
        log("G followed? " + " ".join(f"{n} corr {c:.2f}" for n, c in zip(NAMES, cc)))
    if zrows:
        zc = list(zrows[0])
        with open(out / "zcheck.csv", "w") as f:
            f.write(",".join(zc) + "\n")
            for r in zrows:
                f.write(",".join(str(r[k]) for k in zc) + "\n")
        from .zplan import slice_names
        names = slice_names(skel_model.zcfg["edges"])
        K = len(names)
        want = np.array([[r[f"want_z{k}"] for k in range(K)] for r in zrows]); got = np.array([[r[f"got_z{k}"] for k in range(K)] for r in zrows])
        fw = want / want.sum(1, keepdims=True).clip(1.0, None); fg = got / got.sum(1, keepdims=True).clip(1.0, None)

        def corr(x, y):
            return float(np.corrcoef(x, y)[0, 1]) if (len(x) > 2 and x.std() > 0 and y.std() > 0) else float("nan")

        def depth_q(fr, q):                                                      # depth (m) by which q of the light is emitted, from the slices
            e = np.asarray(skel_model.zcfg["edges"], dtype=np.float64); hi = np.append(e[1:], e[-1] + 6.0)
            c = np.cumsum(fr, 1); res = np.full(len(fr), np.nan)
            for n in range(len(fr)):
                if c[n, -1] < 0.5:                                               # a skeleton without light: no depth
                    continue
                k = int(np.searchsorted(c[n], q)); k = min(k, K - 1)
                c0 = c[n, k - 1] if k > 0 else 0.0
                res[n] = e[k] + (hi[k] - e[k]) * (q - c0) / max(fr[n, k], 1e-12)
            return res
        summ = {names[k]: dict(want_mean_frac=float(fw[:, k].mean()), got_mean_frac=float(fg[:, k].mean()), want_std_frac=float(fw[:, k].std()),
                               got_std_frac=float(fg[:, k].std()), corr_frac=corr(fw[:, k], fg[:, k]), corr_photons=corr(want[:, k], got[:, k])) for k in range(K)}
        for q in (0.1, 0.5, 0.9):
            dw, dg = depth_q(fw, q), depth_q(fg, q)
            ok = np.isfinite(dw) & np.isfinite(dg)
            nan = float("nan")
            summ[f"depth{int(q * 100)}_from_slices"] = dict(want_mean=float(dw[ok].mean()) if ok.any() else nan, got_mean=float(dg[ok].mean()) if ok.any() else nan,
                                                            want_std=float(dw[ok].std()) if ok.any() else nan, got_std=float(dg[ok].std()) if ok.any() else nan,
                                                            corr=corr(dw[ok], dg[ok]), n_without_light=int((~ok).sum()))
        write_json(out / "zcheck_summary.json", summ)
        log("zplan followed? " + " ".join(f"{names[k]} corr {summ[names[k]]['corr_frac']:.2f}" for k in range(K))
            + " | " + " ".join(f"depth{q} corr {summ[f'depth{q}_from_slices']['corr']:.2f} std {summ[f'depth{q}_from_slices']['got_std']:.2f}/{summ[f'depth{q}_from_slices']['want_std']:.2f}" for q in (10, 50, 90)))
    if trows:
        tcl = list(trows[0])
        with open(out / "tcheck.csv", "w") as f:
            f.write(",".join(tcl) + "\n")
            for r in trows:
                f.write(",".join(str(r[k]) for k in tcl) + "\n")
        from .tplan import slice_names as t_slice_names
        tnames = t_slice_names(skel_model.tcfg["edges"])
        Kt = len(tnames)
        want = np.array([[r[f"want_t{k}"] for k in range(Kt)] for r in trows]); got = np.array([[r[f"got_t{k}"] for k in range(Kt)] for r in trows])
        fw = want / want.sum(1, keepdims=True).clip(1.0, None); fg = got / got.sum(1, keepdims=True).clip(1.0, None)

        def tcorr(x, y):
            return float(np.corrcoef(x, y)[0, 1]) if (len(x) > 2 and x.std() > 0 and y.std() > 0) else float("nan")
        tsumm = {tnames[k]: dict(want_mean_frac=float(fw[:, k].mean()), got_mean_frac=float(fg[:, k].mean()), want_std_frac=float(fw[:, k].std()),
                                 got_std_frac=float(fg[:, k].std()), corr_frac=tcorr(fw[:, k], fg[:, k]), corr_photons=tcorr(want[:, k], got[:, k]),
                                 got_over_want_photons=float(got[:, k].sum() / max(want[:, k].sum(), 1.0))) for k in range(Kt)}
        write_json(out / "tcheck_summary.json", tsumm)
        log("tplan followed? " + " ".join(f"{tnames[k]} corr {tsumm[tnames[k]]['corr_frac']:.2f} got/want {tsumm[tnames[k]]['got_over_want_photons']:.3f}" for k in range(Kt)))
    if rrows:
        rcl = list(rrows[0])
        with open(out / "ringcheck.csv", "w") as f:
            f.write(",".join(rcl) + "\n")
            for r in rrows:
                f.write(",".join(str(r[k]) for k in rcl) + "\n")
        log("nq_r: " + " ".join(f"{k} {np.mean([r[k] for r in rrows]):.3f}" for k in ("ring_g_min", "ring_g_max", "ring_g_mean", "ring_moved", "ring_clipped")))
    from .rules import sample_fallbacks
    SAMPLE_FALLBACKS = sample_fallbacks()
    med = lambda k: float(np.median([r[k] for r in rows]))  # noqa: E731
    photons = photon_summary(rows, tfit)
    log("photons: " + " ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in photons.items() if k != "truth_ref"))
    place = None
    if a.place != "off":
        tot = {k: int(sum(r[k] for r in rows)) for k in STAT_KEYS if k not in ("place_s", "place_occ_skel", "place_occ_end")}
        ns = max(tot["place_strokes"], 1)
        place = dict(mode=a.place, occ=a.place_occ, w_occ=a.place_w_occ, tries=a.place_tries, keep_t=a.place_keep_t, **tot, rej_x_per_stroke=tot["place_rej_x"] / ns, rej_u_per_stroke=tot["place_rej_u"] / ns,
                     fallback_x_per_stroke=tot["place_fallback_x"] / ns, fallback_u_per_stroke=tot["place_fallback_u"] / ns,
                     t_shift_per_redrawn_stroke=tot["place_t_shift"] / max(tot["place_rows_redrawn"], 1), median_place_s=med("place_s"))
        log("placement: " + " ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in place.items()))
    timing = dict(gpu=gpu_name, device=str(device), sampler=s_mine["sampler"], det_batch=a.det_batch, skel_batch=a.skel_batch, n=len(rows),
                  skel_s_median=med("skel_s"), det_s_median=med("det_s"), place_s_median=med("place_s") if a.place != "off" else 0.0, total_s_median=med("total_s"),
                  skel_s_mean=float(np.mean([r["skel_s"] for r in rows])), det_s_mean=float(np.mean([r["det_s"] for r in rows])), total_s_mean=float(np.mean([r["total_s"] for r in rows])),
                  skeleton_source=("model" if not (truth or sfrom) else "truth" if truth else "skeleton_from"),
                  attr_chunk=ATTR_CHUNK, det_oom_splits=oom_splits)
    from .prof import ON as PROF_ON, snapshot
    if PROF_ON:                                                                    # 0.6.4a: SHOWERLM_PROF=1
        write_json(out / "prof.json", dict(n=len(rows), skel_batch=a.skel_batch, det_batch=a.det_batch, gpu=gpu_name, regions=snapshot(len(rows))))
    write_json(out / "summary.json", dict(n=len(rows), rule_fallbacks=dict(SAMPLE_FALLBACKS), photons=photons, place=place, median_n_skel=med("n_skel"), median_n_det=med("n_det"),
                                          median_max_group=med("max_group"), n_hit_max_group=int(sum(r["hit_max_group"] for r in rows)),
                                          median_skel_s=med("skel_s"), median_det_s=med("det_s"), median_total_s=med("total_s"), median_q_total=med("q_total"),
                                          skel_stop={k: sum(r["skel_stop"] == k for r in rows) for k in sorted({r["skel_stop"] for r in rows})}, timing=timing))
    log(f"done: median {med('total_s'):.1f} s per shower (skeleton {med('skel_s'):.1f} s, detail {med('det_s'):.1f} s; {gpu_name}, detail batch {a.det_batch}, skeleton batch {a.skel_batch}); rule fallbacks {dict(SAMPLE_FALLBACKS)}")


if __name__ == "__main__":
    main()
