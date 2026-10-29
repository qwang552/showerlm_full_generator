#!/usr/bin/env python
"""Generate CORSIKA 8-like in-ice Cherenkov light showers (80 ns, 20 m box) with ShowerLM 0.6.4c (no training data needed).

    python generate.py --n 10 --out showers --seed 0

Writes showers/sample_XXXXX.npz (voxels: ijk, q, off, m, t, cnt) and tokens_XXXXX.npz, plus run.json, timing.csv,
summary.json and the plan checks.  Uses the GPU when one is available, otherwise the CPU (slow).  The sampling settings
are those of the 0.6.4c tz_cap evaluation (see run.json in this folder); only the batch sizes depend on the device.
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SETTINGS = json.loads((HERE / "run.json").read_text())

# device -> (skeleton batch, detail batch): the batches of the evaluation runs (A40 46 GB; GTX 1080 8 GB); cpu: small
BATCH = {"a40": (256, 32), "1080": (32, 8), "cpu": (4, 2)}


def pick_device(name):
    import torch
    if name != "auto":
        return name
    if not torch.cuda.is_available():
        return "cpu"
    return "a40" if torch.cuda.get_device_properties(0).total_memory > 20e9 else "1080"


class TargetRows:
    """priors/target_rows.npz: (skeleton photons, detail photons, detail strokes) of every training shower, i.e. what
    showerlm.targetfit.TargetTotals reads from the training tokens, precomputed so that no training data is needed."""

    def __init__(self, path, vocab=None):
        import numpy as np
        from showerlm.tokens import Vocab
        d = np.load(path)
        self.vocab = Vocab.from_dict(json.loads(str(d["vocab"])))
        if vocab is not None and self.vocab.to_dict() != vocab.to_dict():
            raise SystemExit(f"{path}: vocabulary differs from the detail model's")
        eids = [int(e) for e in d["eid"]]
        self.rows = {e: (float(s), float(q), int(n)) for e, s, q, n in zip(eids, d["q_skel"], d["q_det"], d["n_det"])}
        self.store = argparse.Namespace(root=Path(path).resolve(), eids=eids)
        self.file_of = dict.fromkeys(eids)

    def check(self, eids):
        missing = sorted({int(e) for e in eids} - set(self.rows))
        if missing:
            raise SystemExit(f"target rows {self.store.root}: no row for {len(missing)} prior eids, e.g. {missing[:5]}")

    def of_eid(self, e):
        return self.rows[int(e)]


def patch(render):
    """Point the 0.6.4c sampler at the shipped tables, and draw the voxels with the chosen renderer."""
    import numpy as np
    import showerlm.sample06 as s06
    from showerlm import targetfit
    from showerlm.sample04 import merge
    s06.TargetTotals = TargetRows
    rows = str((HERE / "priors" / "target_rows.npz").resolve())
    s06.load_fit = lambda p: dict(targetfit.load_fit(p), tokens=rows, tokens_fitted_on=targetfit.load_fit(p)["tokens"])
    if render == "v06":
        return
    from showerlm.raster2 import Prepared
    variant = SETTINGS["render_variant"]

    def save(out, i, tok_skel, tok_det, vocab, ranges, seconds, eid=-1):
        tok = merge(tok_skel, tok_det, vocab)
        r = Prepared(vocab.decode(tok, use_pointer=False), vocab.g, ranges).render(**variant)
        np.savez_compressed(out / f"sample_{i:05d}.npz", ijk=r["ijk"], q=r["q"], off=r["off"], m=r["m"], t=r["t"], cnt=r["cnt"],
                            eid=np.int64(eid), seconds=np.asarray([seconds], np.float32))
        np.savez_compressed(out / f"tokens_{i:05d}.npz", tok=tok.astype(np.int16), tok_skel=np.asarray(tok_skel).astype(np.int16),
                            tok_det=np.asarray(tok_det).astype(np.int16), eid=np.int64(eid))
        return r
    s06.save04 = save


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10, help="number of showers")
    ap.add_argument("--out", default="showers", help="output folder (must not exist)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=["auto", "a40", "1080", "cpu"], default="auto",
                    help="sets the batch sizes: auto = a40 on a GPU with > 20 GB, 1080 on a smaller GPU, cpu without one")
    ap.add_argument("--skel-batch", type=int, default=None, help="skeletons generated together (default: by --device)")
    ap.add_argument("--det-batch", type=int, default=None, help="showers whose detail is generated together (default: by --device)")
    ap.add_argument("--render", choices=["rvC", "v06"], default="rvC",
                    help="voxel renderer: rvC (raster2 path, w_min 0.15, causal: the one the metrics use) or v06 (the 0.6 renderer of sample06)")
    ap.add_argument("--no-kv", action="store_true", help="switch off the KV cache of the attribute decoder (slower, same distribution)")
    ap.add_argument("--models", default=str(HERE / "trained_model"), help="folder with skeleton/ and detail/")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out.exists():
        raise SystemExit(f"{out} exists (not overwritten)")
    os.environ.setdefault("SHOWERLM_ATTR_KV", "0" if a.no_kv else "1")
    os.environ.setdefault("SHOWERLM_ATTR_CHUNK", "2048")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.path.insert(0, str(HERE))
    dev = pick_device(a.device)
    sb, db = BATCH[dev]
    sb, db = a.skel_batch or min(sb, a.n), a.det_batch or min(db, a.n)
    patch(a.render)
    import showerlm.sample06 as s06
    m, pr, s = Path(a.models), HERE / "priors", SETTINGS["sample06"]
    argv = ["--skel-run", m / "skeleton", "--det-run", m / "detail", "--n", a.n, "--skel-batch", sb, "--det-batch", db, "--seed", a.seed,
            "--out", out.parent, "--name", out.name, "--g-prior", pr / "gfeat", "--z-prior", pr / "zplan", "--t-prior", pr / "tplan",
            "--target-tokens", pr / "target_rows.npz", "--target-fit", pr / "target_fit.json"]
    for k, v in s.items():
        argv += ["--" + k.replace("_", "-"), v]
    if dev == "cpu":
        argv.append("--allow-cpu")
    print(f"ShowerLM 0.6.4c: {a.n} showers -> {out} (device {dev}, skeleton batch {sb}, detail batch {db}, render {a.render}, "
          f"KV {os.environ['SHOWERLM_ATTR_KV']})", flush=True)
    sys.argv = ["sample06"] + [str(x) for x in argv]
    s06.main()
    rj = out / "run.json"
    if rj.exists():
        d = json.loads(rj.read_text())
        d["generator"] = dict(settings=SETTINGS["name"], device=dev, render=a.render,
                              render_variant=SETTINGS["render_variant"] if a.render == "rvC" else "raster.render (0.6)")
        rj.write_text(json.dumps(d, indent=1))


if __name__ == "__main__":
    main()
