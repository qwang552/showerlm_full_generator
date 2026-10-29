"""Voxels -> particle tracks ("chains") from what the cache stores: position, photon-weighted direction, time,
photons and record count.  A voxel's successor lies ahead along its direction, slightly later.
"""
from __future__ import annotations

import argparse

import numpy as np

from .common import C_CM_PER_NS, cache_meta, fresh_dir, load_event, log, pct, split_ids, write_json

OFFS = np.array([(dx, dy, dz) for dx in range(-2, 3) for dy in range(-2, 3) for dz in range(-2, 3) if (dx, dy, dz) != (0, 0, 0)], dtype=np.int64)
MODES = ("stroke", "pure")


def positions(ev, g, ranges):
    """Voxel centroids in cm (box origin at ranges[:,0]), voxel size in cm, raw time in ns (light-front undone)."""
    ranges = np.asarray(ranges, dtype=np.float64)
    vox = float((ranges[:, 1] - ranges[:, 0])[0] / g * 100.0)
    p = (ev["ijk"].astype(np.float64) + 0.5 + ev["off"].astype(np.float64)) * vox
    z_top_cm = (ranges[2, 1] - ranges[2, 0]) * 100.0
    z_centre = (ev["ijk"][:, 2].astype(np.float64) + 0.5) * vox          # the cache subtracts the light front at the voxel centre
    t_raw = ev["t"].astype(np.float64) + (z_top_cm - z_centre) / C_CM_PER_NS
    return p, vox, t_raw


def neighbour_table(ijk, g, offs=OFFS):
    keys = (ijk[:, 2].astype(np.int64) * g + ijk[:, 1]) * g + ijk[:, 0]
    order = np.argsort(keys)
    sk = keys[order]
    n = len(ijk)
    idx = np.full((n, len(offs)), -1, np.int64)
    for k, o in enumerate(offs):
        nb = ijk.astype(np.int64) + o
        inside = np.all((nb >= 0) & (nb < g), 1)
        kk = (np.clip(nb[:, 2], 0, g - 1) * g + np.clip(nb[:, 1], 0, g - 1)) * g + np.clip(nb[:, 0], 0, g - 1)
        pos = np.clip(np.searchsorted(sk, kk), 0, n - 1)
        hit = inside & (sk[pos] == kk)
        idx[hit, k] = order[pos[hit]]
    return idx


def coherent_mask(ev, mode, coh_min=0.9):
    coh = np.linalg.norm(ev["m"].astype(np.float64), axis=1)
    if mode == "pure":
        cnt = ev["cnt"] if "cnt" in ev else np.ones(len(coh))
        return (coh >= 0.999) | ((coh >= 0.99) & (cnt <= 3))
    return coh >= coh_min


def extract(ev, g, ranges, mode="stroke", coh_min=0.9, s_min=0.6, s_max=3.6, r_perp_max=1.6, cos_min=0.75, dt_tol=0.2):
    """Link each coherent voxel to its best successor.  Returns dict(succ, pred, chain_id, chain_pos, coherent, ...).
    Every voxel gets a chain id (unlinked voxels are 1-voxel chains), so chains partition the event."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    p, vox, t = positions(ev, g, ranges)
    m = ev["m"].astype(np.float64)
    coh = np.linalg.norm(m, axis=1)
    u = m / np.maximum(coh, 1e-9)[:, None]
    n = len(p)
    coherent = coherent_mask(ev, mode, coh_min)
    nb = neighbour_table(ev["ijk"], g)
    valid = nb >= 0
    j = np.where(valid, nb, 0)
    d = p[j] - p[:, None, :]                                   # (n, K, 3) displacement to each neighbour
    s = np.einsum("nkc,nc->nk", d, u)                          # along the direction
    perp = np.linalg.norm(d - s[..., None] * u[:, None, :], axis=2)
    cosd = np.einsum("nkc,nc->nk", u[j], u)
    dt = t[j] - t[:, None]
    ok = valid & coherent[:, None] & coherent[j] & (s > s_min) & (s <= s_max) & (perp < r_perp_max) & (cosd > cos_min) \
        & (np.abs(dt - s / C_CM_PER_NS) < dt_tol)
    score = perp / r_perp_max + (1 - cosd) / (1 - cos_min) + np.abs(dt - s / C_CM_PER_NS) / dt_tol + 0.25 * s / s_max
    score = np.where(ok, score, np.inf)
    best = np.argmin(score, axis=1)
    bs = score[np.arange(n), best]
    has = np.isfinite(bs)
    succ = np.where(has, j[np.arange(n), best], -1)
    order = np.argsort(bs)                                     # one predecessor per voxel: best claimant wins
    pred = np.full(n, -1, np.int64)
    taken = np.zeros(n, bool)
    for i in order:
        if succ[i] < 0:
            continue
        if taken[succ[i]]:
            succ[i] = -1
        else:
            taken[succ[i]] = True
            pred[succ[i]] = i
    chain_id = np.full(n, -1, np.int64)
    chain_pos = np.zeros(n, np.int64)
    heads = np.flatnonzero((pred < 0) & (succ >= 0))
    cid = 0
    for h in heads:
        k, pos = h, 0
        while k >= 0 and chain_id[k] < 0:
            chain_id[k] = cid; chain_pos[k] = pos; pos += 1; k = succ[k]
        cid += 1
    single = np.flatnonzero(chain_id < 0)                      # every remaining voxel is its own chain
    chain_id[single] = cid + np.arange(len(single))
    n_chains = cid + len(single)
    cnt = ev["cnt"].astype(np.int64) if "cnt" in ev else np.ones(n, np.int64)
    return dict(succ=succ, pred=pred, chain_id=chain_id, chain_pos=chain_pos, coherent=coherent, n_chains=int(n_chains),
                p=p, t=t, u=u, coh=coh, vox=vox, cnt=cnt, mode=mode)


def chain_table(ev, ex):
    """Per-chain arrays: head/tail voxel, n voxels, length (cm, sum of links), photons, start/end positions, directions, times."""
    cid = ex["chain_id"]; n_ch = ex["n_chains"]
    q = ev["q"].astype(np.float64)
    p, u, t, succ = ex["p"], ex["u"], ex["t"], ex["succ"]
    link_len = np.zeros(len(q))
    has = succ >= 0
    link_len[has] = np.linalg.norm(p[succ[has]] - p[has], axis=1)
    n_vox = np.bincount(cid, minlength=n_ch)
    length = np.bincount(cid, weights=link_len, minlength=n_ch)
    photons = np.bincount(cid, weights=q, minlength=n_ch)
    records = np.bincount(cid, weights=ex["cnt"].astype(np.float64), minlength=n_ch)
    pos = ex["chain_pos"]
    head = np.full(n_ch, -1, np.int64); tail = np.full(n_ch, -1, np.int64)
    head[cid[pos == 0]] = np.flatnonzero(pos == 0)
    last = np.zeros(n_ch, np.int64)
    np.maximum.at(last, cid, pos)
    tsel = pos == last[cid]
    tail[cid[tsel]] = np.flatnonzero(tsel)
    # direction of the chain as a whole: tail - head for >= 2 voxels, the voxel direction for singletons
    dirn = p[tail] - p[head]
    nrm = np.linalg.norm(dirn, axis=1)
    multi = nrm > 1e-6
    dirn[multi] /= nrm[multi, None]
    dirn[~multi] = u[head[~multi]]
    return dict(n_vox=n_vox, length=length, photons=photons, records=records, head=head, tail=tail, dir=dirn,
                p0=p[head], p1=p[tail], u0=u[head], u1=u[tail], t0=t[head], t1=t[tail])


def parents(ev, ex, ct, r_max=60.0, dt_tol=0.15, dt_min=0.02, dt_max=2.5, g=None):
    """For every chain head: the nearest earlier voxel (of another chain) on its past light cone within r_max cm.
    Two stages: a 7^3 voxel stencil (<= ~6 cm, finds ~76 % of the parents at once), then a KD-tree search to r_max.
    Returns dict(chain (parent chain id or -1), voxel (parent voxel index or -1), dist cm, cos (chain dir vs parent dir),
    s (parent voxel position fraction along its chain))."""
    p, t, u = ex["p"], ex["t"], ex["u"]
    cid, cpos = ex["chain_id"], ex["chain_pos"]
    n_ch = ex["n_chains"]
    head = ct["head"]
    th = t[head]
    par = np.full(n_ch, -1, np.int64); pv = np.full(n_ch, -1, np.int64)
    dist = np.full(n_ch, np.inf)
    # stage 1: stencil of radius 3 voxels around the head voxel
    ijk = ev["ijk"].astype(np.int64)
    gg = int(ijk.max()) + 1 if g is None else g
    offs = np.array([(a, b, c) for a in range(-3, 4) for b in range(-3, 4) for c in range(-3, 4) if (a, b, c) != (0, 0, 0)], dtype=np.int64)
    keys = (ijk[:, 2] * gg + ijk[:, 1]) * gg + ijk[:, 0]
    order = np.argsort(keys); sk = keys[order]
    hijk = ijk[head]
    for o in offs:
        nb = hijk + o
        inside = np.all((nb >= 0) & (nb < gg), 1)
        kk = (np.clip(nb[:, 2], 0, gg - 1) * gg + np.clip(nb[:, 1], 0, gg - 1)) * gg + np.clip(nb[:, 0], 0, gg - 1)
        pos = np.clip(np.searchsorted(sk, kk), 0, len(sk) - 1)
        hit = inside & (sk[pos] == kk)
        if not hit.any():
            continue
        c = np.flatnonzero(hit); k = order[pos[hit]]
        dd = np.linalg.norm(p[k] - p[head[c]], axis=1)
        dtt = th[c] - t[k]
        ok = (cid[k] != c) & (dtt >= dt_min) & (dtt <= dt_max) & (np.abs(dtt - dd / C_CM_PER_NS) < dt_tol) & (dd < dist[c])
        c, k, dd = c[ok], k[ok], dd[ok]
        # several offsets never give the same (c, k); within one offset c is unique
        par[c] = cid[k]; pv[c] = k; dist[c] = dd
    # a stencil hit within 3 voxels is at most 3 * sqrt(3) * vox ~ 10.8 cm away, but a nearer voxel outside the
    # stencil cannot exist closer than 3 voxels: hits closer than 3 voxels are final, the others are re-searched
    final = dist < 3.0 * ex["vox"]
    todo = np.flatnonzero(~final)
    if todo.size:
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(p)
            for c in todo:
                cand = np.asarray(tree.query_ball_point(p[head[c]], r_max), dtype=np.int64)
                _best(c, cand, p, t, cid, head, th, dt_min, dt_max, dt_tol, r_max, par, pv, dist)
        except ImportError:
            order_t = np.argsort(t); ts = t[order_t]
            lo = np.searchsorted(ts, th - dt_max); hi = np.searchsorted(ts, th - dt_min)
            for c in todo:
                _best(c, order_t[lo[c]:hi[c]], p, t, cid, head, th, dt_min, dt_max, dt_tol, r_max, par, pv, dist)
    ok = par >= 0
    cosang = np.full(n_ch, np.nan); sfrac = np.full(n_ch, np.nan)
    cosang[ok] = np.clip(np.einsum("nc,nc->n", ct["dir"][ok], ct["dir"][par[ok]]), -1, 1)
    sfrac[ok] = cpos[pv[ok]] / np.maximum(ct["n_vox"][par[ok]] - 1, 1)
    dist[~ok] = np.nan
    return dict(chain=par, voxel=pv, dist=dist, cos=cosang, s=sfrac)


def _best(c, cand, p, t, cid, head, th, dt_min, dt_max, dt_tol, r_max, par, pv, dist):
    if cand.size == 0:
        return
    cand = cand[cid[cand] != c]
    if cand.size == 0:
        return
    dd = np.linalg.norm(p[cand] - p[head[c]], axis=1)
    dtt = th[c] - t[cand]
    ok = (dtt >= dt_min) & (dtt <= dt_max) & (dd < r_max) & (np.abs(dtt - dd / C_CM_PER_NS) < dt_tol)
    if not ok.any():
        return
    i = np.argmin(dd[ok])
    if dd[ok][i] < dist[c]:
        k = cand[ok][i]
        par[c] = cid[k]; pv[c] = k; dist[c] = dd[ok][i]


def null_parents(ev, ex, ct, g=None, shift_ns=1.0, seed=0):
    """Chance baseline for `parents`: every chain gets one random time offset, uniform in [-shift_ns, shift_ns]."""
    rng = np.random.default_rng(seed)
    exn = dict(ex)
    exn["t"] = ex["t"] + rng.uniform(-shift_ns, shift_ns, ex["n_chains"])[ex["chain_id"]]
    return parents(ev, exn, ct, g=g)


BIRTH_BINS = [0.0, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0, 60.0]


def birth_stats(ct, pa):
    """Chain heads (>= 2 voxels): fraction with a light-cone parent, distance histogram (fraction of all heads per bin),
    direction cosine to the parent."""
    sel = ct["n_vox"] >= 2
    d = pa["dist"][sel]; okb = np.isfinite(d); near = okb & (d < 3)
    co = pa["cos"][sel]
    return dict(n=int(sel.sum()), frac_with_parent_60cm=float(okb.mean()) if sel.any() else 0.0,
                dist_pct=pct(d[okb], [10, 50, 90]) if okb.any() else None,
                frac_lt_3cm=float(near.mean()) if sel.any() else None,
                mean_dist_3_60=float(d[okb & (d >= 3)].mean()) if (okb & (d >= 3)).any() else None,
                hist=(np.histogram(d[okb], BIRTH_BINS)[0] / max(int(sel.sum()), 1)).tolist(),
                cos_pct=pct(co[okb], [10, 50, 90]) if okb.any() else None)


def event_stats(ev, ex, ct, pa, long_cm=8.0, pa_null=None):
    q = ev["q"].astype(np.float64); tot = q.sum()
    L = ct["length"]; nv = ct["n_vox"]
    out = dict(n_vox=int(len(q)), q_total=float(tot), coherent_frac=float(ex["coherent"].mean()),
               coherent_q_frac=float(q[ex["coherent"]].sum() / tot), n_chains=int(ex["n_chains"]),
               n_chains_ge2=int((nv >= 2).sum()), n_ge8=int((L >= 8).sum()), n_ge20=int((L >= 20).sum()), n_ge60=int((L >= 60).sum()),
               len_pct=pct(L[nv >= 2], [50, 90, 99]), len_max=float(L.max()) if L.size else 0.0,
               q_frac_chains_ge8=float(ct["photons"][L >= 8].sum() / tot), q_frac_chains_ge2=float(ct["photons"][nv >= 2].sum() / tot),
               total_len_m=float(L.sum() / 100), nvox_pct=pct(nv, [50, 90, 99, 100]))
    sel = L >= long_cm
    if sel.any():
        out["ppc_long_pct"] = pct(ct["photons"][sel] / L[sel], [10, 25, 50, 75, 90])      # photons per cm of chain
        out["records_per_cm_long_pct"] = pct(ct["records"][sel] / L[sel], [10, 50, 90])
    cosang = np.einsum("nc,nc->n", ct["u0"], ct["u1"])
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    out["turn_deg_by_len"] = {}
    for lo, hi in [(4, 8), (8, 16), (16, 32), (32, 64), (64, 1000)]:
        m = (L >= lo) & (L < hi)
        if m.sum() >= 10:
            out["turn_deg_by_len"][f"{lo}-{hi}"] = [int(m.sum())] + pct(ang[m], [50, 90])
    succ = ex["succ"]; has = succ >= 0
    cl = np.einsum("nc,nc->n", ex["u"][has], ex["u"][succ[has]])
    out["turn_per_link_deg_pct"] = pct(np.degrees(np.arccos(np.clip(cl, -1, 1))), [50, 90, 99])
    # births: chains with >= 2 voxels and a light-cone parent; the same with chain times shifted at random (chance)
    out["births"] = birth_stats(ct, pa)
    if pa_null is not None:
        out["births_null"] = birth_stats(ct, pa_null)
    return out


def save_event_chains(path, ev, ex, ct, pa):
    np.savez_compressed(path, chain_id=ex["chain_id"].astype(np.int32), chain_pos=ex["chain_pos"].astype(np.int32),
                        succ=ex["succ"].astype(np.int32), coherent=ex["coherent"], mode=np.asarray(ex["mode"]),
                        ch_len=ct["length"].astype(np.float32), ch_q=ct["photons"].astype(np.float32), ch_records=ct["records"].astype(np.float32),
                        ch_nvox=ct["n_vox"].astype(np.int32), ch_head=ct["head"].astype(np.int32), ch_tail=ct["tail"].astype(np.int32),
                        ch_dir=ct["dir"].astype(np.float32), ch_t0=ct["t0"].astype(np.float32),
                        pa_chain=pa["chain"].astype(np.int32), pa_voxel=pa["voxel"].astype(np.int32), pa_dist=pa["dist"].astype(np.float32),
                        pa_cos=pa["cos"].astype(np.float32), pa_s=pa["s"].astype(np.float32))


def process_event(ev, g, ranges, mode="stroke", coh_min=0.9):
    ex = extract(ev, g, ranges, mode=mode, coh_min=coh_min)
    ct = chain_table(ev, ex)
    pa = parents(ev, ex, ct, g=g)
    return ex, ct, pa


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="/data/user/qwang/corsika_sparse_v1/data/processed_192")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=256, help="events (0 = all of the split)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/data/user/qwang/showerlm/out")
    ap.add_argument("--name", default=None)
    ap.add_argument("--mode", default="stroke", choices=MODES)
    ap.add_argument("--coh-min", type=float, default=0.9, help="stroke mode: |m| threshold for linking")
    ap.add_argument("--save-events", action="store_true", help="also write per-event chain files (tokens.py can reuse them)")
    ap.add_argument("--null-shift-ns", type=float, default=1.0, help="chance baseline for births: random time offset per chain (0 = off)")
    a = ap.parse_args()
    meta, g, ranges = cache_meta(a.cache)
    ids = split_ids(meta, a.split, a.n, a.seed)
    out = fresh_dir(a.out, a.name or f"chains_{a.mode}_{a.split}{len(ids)}")
    (out / "events").mkdir(exist_ok=True)
    log(f"chains ({a.mode}): {len(ids)} {a.split} events -> {out}")
    per = []
    for i, eid in enumerate(ids):
        ev = load_event(a.cache, eid)
        if "m" not in ev or not np.any(ev["m"]):
            raise SystemExit("this cache has no direction (m): chains need it")
        ex, ct, pa = process_event(ev, g, ranges, a.mode, a.coh_min)
        pa_null = null_parents(ev, ex, ct, g=g, shift_ns=a.null_shift_ns, seed=a.seed * 100003 + eid) if a.null_shift_ns > 0 else None
        st = event_stats(ev, ex, ct, pa, pa_null=pa_null); st["eid"] = eid
        per.append(st)
        if a.save_events:
            save_event_chains(out / "events" / f"chains_{eid:05d}.npz", ev, ex, ct, pa)
        if (i + 1) % 16 == 0 or i == len(ids) - 1:
            log(f"{i + 1}/{len(ids)}: eid {eid} voxels {st['n_vox']} chains>=2 {st['n_chains_ge2']} >=8cm {st['n_ge8']} >=60cm {st['n_ge60']}")
    summary = summarize(per)
    write_json(out / "chains.json", dict(summary=summary, events=per, args=vars(a)))
    (out / "chains.md").write_text(markdown(summary, len(ids), a))
    log("written", out / "chains.md")


def summarize(per):
    def arr(k):
        return np.array([e[k] for e in per], dtype=np.float64)
    S = {}
    for k in ("n_vox", "q_total", "coherent_frac", "coherent_q_frac", "n_chains", "n_chains_ge2", "n_ge8", "n_ge20", "n_ge60",
              "q_frac_chains_ge8", "q_frac_chains_ge2", "total_len_m"):
        a = arr(k); S[k] = dict(mean=float(a.mean()), std=float(a.std()), p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)))
    S["ppc_long_median_pct_over_events"] = pct([e["ppc_long_pct"][2] for e in per if "ppc_long_pct" in e], [10, 50, 90])
    S["turn_per_link_deg_median_mean"] = float(np.mean([e["turn_per_link_deg_pct"][0] for e in per]))
    keys = sorted({k for e in per for k in e["turn_deg_by_len"]}, key=lambda s: float(s.split("-")[0]))
    S["turn_deg_by_len_median_mean"] = {k: float(np.mean([e["turn_deg_by_len"][k][1] for e in per if k in e["turn_deg_by_len"]])) for k in keys}
    for key in ("births", "births_null"):
        B = [e[key] for e in per if key in e]
        if not B:
            continue
        S[key] = dict(frac_with_parent_60cm=float(np.mean([b["frac_with_parent_60cm"] for b in B])),
                      frac_lt_3cm=float(np.mean([b["frac_lt_3cm"] for b in B if b["frac_lt_3cm"] is not None])),
                      mean_dist_3_60=float(np.mean([b["mean_dist_3_60"] for b in B if b["mean_dist_3_60"] is not None])),
                      hist=np.mean([b["hist"] for b in B], axis=0).tolist(),
                      cos_median=float(np.mean([b["cos_pct"][1] for b in B if b["cos_pct"] is not None])))
    return S


def markdown(S, n, a):
    L = [f"# Chains from voxels: {n} {a.split} events (mode {a.mode}, coh_min {a.coh_min})", "",
         "A chain = voxels linked successor-to-successor by position, direction (m) and light-front time.", "",
         "| quantity (per event) | mean | std | 10% | 90% |", "|---|---|---|---|---|"]
    names = dict(n_vox="lit voxels", q_total="photons", coherent_frac="coherent voxel fraction", coherent_q_frac="photon fraction in coherent voxels",
                 n_chains="chains (every voxel belongs to one)", n_chains_ge2="chains with >= 2 voxels", n_ge8="chains >= 8 cm", n_ge20="chains >= 20 cm",
                 n_ge60="chains >= 60 cm", q_frac_chains_ge8="photon fraction in chains >= 8 cm", q_frac_chains_ge2="photon fraction in chains >= 2 voxels",
                 total_len_m="total chain length (m)")
    for k, v in S.items():
        if isinstance(v, dict) and "mean" in v:
            f = (lambda x: f"{x:.3f}") if v["mean"] < 10 else (lambda x: f"{x:.3g}")
            L.append(f"| {names.get(k, k)} | {f(v['mean'])} | {f(v['std'])} | {f(v['p10'])} | {f(v['p90'])} |")
    L += ["", f"- light per cm along chains >= 8 cm (median per event; 10 / 50 / 90 % over events): {' / '.join(f'{x:.0f}' for x in S['ppc_long_median_pct_over_events'])}"
          " (one particle at beta = 1 gives 343 per cm; more means several particles share the chain)",
          f"- direction change per link (median, mean over events): {S['turn_per_link_deg_median_mean']:.1f} deg",
          "- angle between first and last direction of a chain, median by chain length (cm): " + ", ".join(f"{k}: {v:.1f} deg" for k, v in S["turn_deg_by_len_median_mean"].items()),
          "  (a chain that bends sharply is cut by the extractor, so part of the trend with length is selection, not physics)"]
    L += births_markdown(S, a)
    L += ["", "What to read off: light per cm ~ constant and length-independent (Frank-Tamm); long chains straight, short chains bent",
          "(multiple scattering: angle ~ 1/E, short chains are low-E).  In pure mode the chains are single particles;",
          "in stroke mode they are painting primitives (several particles can share one)."]
    return "\n".join(L) + "\n"


def births_markdown(S, a):
    if "births" not in S:
        return []
    B, N = S["births"], S.get("births_null")
    L = ["", "## Births: where chains start, against chance", "",
         "Parent = the nearest earlier voxel of another chain on the head's light cone (|dt - d/c| < 0.15 ns, within 60 cm).",
         "Chance = the same search after shifting every chain by its own random time offset"
         + (f" (uniform +-{a.null_shift_ns:g} ns)." if N else " (not run: --null-shift-ns 0)."), "",
         "| fraction of chain heads (>= 2 voxels) | data | chance | data - chance |", "|---|---|---|---|"]

    def row(name, x, y):
        L.append(f"| {name} | {x:.3f} | " + (f"{y:.3f} | {x - y:+.3f} |" if y is not None else "- | - |"))
    row("with a parent within 60 cm", B["frac_with_parent_60cm"], N["frac_with_parent_60cm"] if N else None)
    row("parent within 3 cm", B["frac_lt_3cm"], N["frac_lt_3cm"] if N else None)
    for i, (lo, hi) in enumerate(zip(BIRTH_BINS[:-1], BIRTH_BINS[1:])):
        row(f"parent at {lo:g}-{hi:g} cm", B["hist"][i], N["hist"][i] if N else None)
    L += ["", f"- direction cosine head vs parent, median: data {B['cos_median']:.3f}" + (f", chance {N['cos_median']:.3f}" if N else ""),
          "- Read only the excess over chance.  The search keeps the nearest match, so a far parent (a photon that travelled unseen",
          "  before converting) is hidden whenever a chance match lies closer; a missing excess at 10-60 cm does not rule conversions out.",
          "- Not every birth is a new particle: on event 1772, where raw particle labels exist, 24 % (stroke) and 45 % (pure) of the",
          "  births within 3 cm join two pieces of one particle that the extractor cut in two."]
    return L


if __name__ == "__main__":
    main()
