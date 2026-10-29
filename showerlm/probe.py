"""Physics probe: per-stroke quantities of true and generated showers (light per unit length, direction, timing),
with textbook values beside them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import VOX_CM, YIELD_CM, fresh_dir, log, pct, write_json
from .tokens import KIND_STROKE, Vocab, is_stroke
import json


def load_tokens(path, n):
    path = Path(path)
    vocab = Vocab.from_dict(json.load(open(path / "vocab.json"))) if (path / "vocab.json").exists() else None
    files = sorted(path.glob("events/tokens_*.npz")) or sorted(path.glob("tokens_*.npz"))
    if not files:
        raise FileNotFoundError(f"no token files under {path}")
    files = files[:n] if n else files
    toks = []
    for f in files:
        with np.load(f) as d:
            toks.append(d["tok"].astype(np.int64))
    if vocab is None:
        run = path / "run.json"
        if run.exists():
            cfg = json.load(open(run))
            v = cfg.get("config", {}).get("vocab")
            vocab = Vocab.from_dict(v) if v else Vocab()
        else:
            vocab = Vocab()
    return toks, vocab


def stroke_tables(toks, vocab):
    ix = vocab.index
    rows = dict(lpl=[], birth_d=[], birth_cos=[], birth_lpl=[], turn_cos=[], turn_lpl=[], length=[], t_bin=[], lq=[], children_per_len=[], n_tok=[], q_tot=[])
    for tok in toks:
        tok = tok[is_stroke(tok[:, ix["kind"]])]
        T = len(tok)
        if T == 0:
            continue
        st = vocab.decode(tok, use_pointer=False)
        L = st["len"] * VOX_CM                                        # cm
        Q = st["Q"]
        lpl = Q / np.maximum(L, VOX_CM)                               # photons per cm (singletons: per voxel size)
        rows["lpl"].append(lpl[L >= 8]); rows["length"].append(L); rows["t_bin"].append(tok[:, ix["t"]]); rows["lq"].append(np.log10(Q))
        rows["n_tok"].append(T); rows["q_tot"].append(Q.sum())
        ptr = tok[:, ix["ptr"]]
        has = ptr > 0
        i = np.flatnonzero(has); p = i - ptr[has]
        d = np.stack([vocab.dec_d(tok[has, ix[k]]) for k in ("dx", "dy", "dz")], 1).astype(np.float64)
        s_bin = tok[has, ix["s"]]
        cont = (s_bin == vocab.n_s) & (ptr[has] == 1)                 # continuation pieces of one track (follow their parent piece directly)
        cosang = np.einsum("nc,nc->n", st["u"][i], st["u"][p])
        plpl = lpl[p]
        born = ~cont
        rows["birth_d"].append(np.linalg.norm(d[born], axis=1) * VOX_CM); rows["birth_cos"].append(cosang[born]); rows["birth_lpl"].append(plpl[born])
        same_len = cont & (L[p] >= 8)
        rows["turn_cos"].append(cosang[same_len]); rows["turn_lpl"].append(plpl[same_len])
        nchild = np.bincount(p[born], minlength=T)
        long_ = L >= 20
        rows["children_per_len"].append(nchild[long_] / L[long_] * 100)   # children per metre
    return {k: (np.concatenate(v) if k not in ("n_tok", "q_tot") else np.array(v)) for k, v in rows.items() if len(v)}


def summarize(R, vocab):
    S = {}
    S["strokes_per_shower"] = pct(R["n_tok"], [10, 50, 90]); S["photons_per_shower"] = pct(R["q_tot"], [10, 50, 90])
    S["light_per_cm_strokes_ge8cm_pct"] = pct(R["lpl"], [10, 25, 50, 75, 90])
    S["frac_light_per_cm_within_343x(0.8-1.2)"] = float(((R["lpl"] > 0.8 * YIELD_CM) & (R["lpl"] < 1.2 * YIELD_CM)).mean())
    S["length_cm_pct"] = pct(R["length"], [50, 90, 99]); S["frac_len_ge8"] = float((R["length"] >= 8).mean()); S["frac_len_ge60"] = float((R["length"] >= 60).mean())
    bd = R["birth_d"]
    S["birth_dist_cm_pct"] = pct(bd, [10, 50, 90]); S["birth_frac_lt3cm"] = float((bd < 3).mean())
    far = bd[(bd >= 3) & (bd < 60)]
    S["birth_mean_dist_3_60cm"] = float(far.mean()) if far.size else None
    S["birth_cos_pct"] = pct(R["birth_cos"], [10, 50, 90])
    # daughter angle vs parent light-per-length (energy proxy: more light per cm = several particles / higher energy)
    S["birth_angle_deg_median_by_parent_lpl"] = {}
    for lo, hi in [(0, 200), (200, 400), (400, 800), (800, 1600), (1600, 1e9)]:
        m = (R["birth_lpl"] >= lo) & (R["birth_lpl"] < hi)
        if m.sum() >= 20:
            S["birth_angle_deg_median_by_parent_lpl"][f"{lo}-{int(min(hi, 9999))}"] = [int(m.sum()), float(np.median(np.degrees(np.arccos(np.clip(R["birth_cos"][m], -1, 1)))))]
    S["turn_deg_pct_between_pieces"] = pct(np.degrees(np.arccos(np.clip(R["turn_cos"], -1, 1))), [50, 90]) if len(R["turn_cos"]) else None
    S["children_per_metre_strokes_ge20cm_pct"] = pct(R["children_per_len"], [50, 90])
    tb = R["t_bin"] * vocab.t_max / vocab.n_t
    S["strokes_vs_time_ns_hist"] = np.histogram(tb, bins=np.arange(0, vocab.t_max + 1, 2.0))[0].tolist()
    S["time_ns_pct"] = pct(tb, [10, 50, 90])
    return S


def markdown(S_truth, S_gen):
    L = ["# Physics probe: what the strokes say", "",
         "Expected (physics): one charged particle at beta = 1 emits 343 photons per cm (Frank-Tamm, 300-650 nm, n = 1.31); strokes with",
         "more light per cm hold several particles.  The parent of a stroke is the nearest earlier stroke on its light cone; in a dense",
         "shower part of these matches is chance (chains.md gives the chance baseline), so the birth rows test whether generation copies",
         "the data, not which particle made which.  Low-energy particles scatter more (Highland), so pieces of faint tracks turn more",
         "than pieces of bright ones.", ""]
    cols = [("truth", S_truth)] + ([("generated", S_gen)] if S_gen else [])
    L.append("| quantity | " + " | ".join(c for c, _ in cols) + " |"); L.append("|---|" + "---|" * len(cols))

    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.3g}"
        if isinstance(v, list):
            return " / ".join(fmt(x) for x in v)
        if isinstance(v, dict):
            return "; ".join(f"{k}: {fmt(x)}" for k, x in v.items())
        return str(v)
    names = [("strokes_per_shower", "strokes per shower (10/50/90 %)"), ("photons_per_shower", "photons per shower (10/50/90 %)"),
             ("light_per_cm_strokes_ge8cm_pct", "light per cm, strokes >= 8 cm (10/25/50/75/90 %)"), ("frac_light_per_cm_within_343x(0.8-1.2)", "fraction of strokes >= 8 cm within 0.8-1.2 x 343/cm"),
             ("length_cm_pct", "stroke length cm (50/90/99 %)"), ("frac_len_ge8", "fraction of strokes >= 8 cm"), ("frac_len_ge60", "fraction of strokes >= 60 cm"),
             ("birth_dist_cm_pct", "birth distance from parent cm (10/50/90 %)"), ("birth_frac_lt3cm", "births within 3 cm of the parent"), ("birth_mean_dist_3_60cm", "mean birth distance of the 3-60 cm births"),
             ("birth_cos_pct", "cos(daughter, parent) (10/50/90 %)"), ("birth_angle_deg_median_by_parent_lpl", "median daughter angle (deg) by parent light per cm [n, deg]"),
             ("turn_deg_pct_between_pieces", "turn between consecutive pieces of one track, deg (50/90 %)"), ("children_per_metre_strokes_ge20cm_pct", "children per metre of strokes >= 20 cm (50/90 %)"),
             ("time_ns_pct", "stroke head time ns (10/50/90 %)"), ("strokes_vs_time_ns_hist", "strokes per 2 ns bin")]
    for k, desc in names:
        L.append(f"| {desc} | " + " | ".join(fmt(S.get(k)) for _, S in cols) + " |")
    return "\n".join(L) + "\n"


def figure(path, R_truth, R_gen, vocab):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    sets = [("truth", R_truth, "k")] + ([("generated", R_gen, "C3")] if R_gen else [])
    for name, R, c in sets:
        ax[0, 0].hist(R["lpl"], bins=np.linspace(0, 2000, 101), histtype="step", color=c, label=name, density=True)
        ax[0, 1].hist(R["birth_d"], bins=np.linspace(0, 60, 61), histtype="step", color=c, label=name, density=True, log=True)
        ax[0, 2].hist(np.degrees(np.arccos(np.clip(R["birth_cos"], -1, 1))), bins=np.linspace(0, 90, 46), histtype="step", color=c, label=name, density=True)
        ax[1, 0].hist(R["length"], bins=np.linspace(0, 70, 71), histtype="step", color=c, label=name, density=True, log=True)
        if len(R["turn_cos"]) > 1:
            ax[1, 1].hist(np.degrees(np.arccos(np.clip(R["turn_cos"], -1, 1))), bins=np.linspace(0, 30, 61), histtype="step", color=c, label=name, density=True)
        ax[1, 2].hist(R["t_bin"] * vocab.t_max / vocab.n_t, bins=np.linspace(0, vocab.t_max, 53), histtype="step", color=c, label=name, density=True)
    ax[0, 0].axvline(YIELD_CM, color="C0", ls="--", label="343/cm (one particle)"); ax[0, 0].set_xlabel("light per cm of stroke (strokes >= 8 cm)")
    ax[0, 1].set_xlabel("birth distance from parent stroke (cm)"); ax[0, 2].set_xlabel("angle daughter vs parent (deg)")
    ax[1, 0].set_xlabel("stroke length (cm)"); ax[1, 1].set_xlabel("turn between consecutive pieces of one track (deg)"); ax[1, 2].set_xlabel("stroke head time (ns, raw)")
    for a in ax.ravel():
        a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", required=True, help="truth token directory")
    ap.add_argument("--samples", default=None, help="generated sample directory (tokens_*.npz)")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--out", default="/data/user/qwang/showerlm/out")
    ap.add_argument("--name", default="probe")
    a = ap.parse_args()
    out = fresh_dir(a.out, a.name)
    toks, vocab = load_tokens(a.tokens, a.n)
    log(f"truth: {len(toks)} token files")
    R_t = stroke_tables(toks, vocab); S_t = summarize(R_t, vocab)
    R_g = S_g = None
    if a.samples:
        toks_g, _ = load_tokens(a.samples, a.n)
        log(f"generated: {len(toks_g)} token files")
        R_g = stroke_tables(toks_g, vocab); S_g = summarize(R_g, vocab)
    write_json(out / "probe.json", dict(truth=S_t, generated=S_g, args=vars(a)))
    (out / "probe.md").write_text(markdown(S_t, S_g))
    if figure(out / "fig_probe.png", R_t, R_g, vocab):
        log("figure", out / "fig_probe.png")
    log("written", out / "probe.md")
    print((out / "probe.md").read_text())


if __name__ == "__main__":
    main()
