"""ShowerLM 0.4 generation (skeleton model, then detail model); save04 writes the sample and token files.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .common import fresh_dir, log, write_json
from .raster import render
from .sample import generate, load_run
from .train import device_for, is_detail, seq_len_at


def merge(tok_skel, tok_det, vocab):
    """Skeleton tokens followed by the detail tokens; detail pointers removed (their head voxels are absolute)."""
    ix = vocab.index
    d = np.asarray(tok_det, dtype=np.int64).reshape(-1, len(vocab.names)).copy()
    for a in ("ptr", "s", "dx", "dy", "dz"):
        d[:, ix[a]] = 0
    return np.concatenate([np.asarray(tok_skel, dtype=np.int64).reshape(-1, len(vocab.names)), d])


def save04(out, i, tok_skel, tok_det, vocab, ranges, seconds, eid=-1):
    tok = merge(tok_skel, tok_det, vocab)
    r = render(vocab.decode(tok, use_pointer=False), vocab.g, ranges)
    np.savez_compressed(out / f"sample_{i:05d}.npz", ijk=r["ijk"], q=r["q"], off=r["off"], m=r["m"], t=r["t"], cnt=r["cnt"],
                        eid=np.int64(eid), seconds=np.asarray([seconds], np.float32))
    np.savez_compressed(out / f"tokens_{i:05d}.npz", tok=tok.astype(np.int16), tok_skel=np.asarray(tok_skel).astype(np.int16),
                        tok_det=np.asarray(tok_det).astype(np.int16), eid=np.int64(eid))
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skel-run", default=None, help="training run of the skeleton model (ShowerLM on tokens_skel_*)")
    ap.add_argument("--det-run", required=True, help="training run of the detail model (model.type detail)")
    ap.add_argument("--checkpoint", default="best", help="best | last (both runs, unless --skel-checkpoint / --det-checkpoint say otherwise)")
    ap.add_argument("--skel-checkpoint", default=None, help="skeleton checkpoint name (default: --checkpoint)")
    ap.add_argument("--det-checkpoint", default="auto",
                    help="detail checkpoint for the strokes; auto (default) = best_stroke if the run has it (0.5.1 training), else --checkpoint")
    ap.add_argument("--det-plan-checkpoint", default="auto",
                    help="detail checkpoint for the plan (QTOT, NTOK); auto (default) = best_plan if the run has it, else the strokes checkpoint")
    ap.add_argument("--truth-skeleton", default=None, help="tokens_det_* folder: use its TRUE skeletons instead of the skeleton model (diagnostic)")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out", default="/data/user/qwang/showerlm/out")
    ap.add_argument("--name", default="gen04")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--skel-max-tokens", type=int, default=1024, help="skeleton length cap (the detail model was trained with k_max)")
    ap.add_argument("--max-tokens", type=int, default=20000, help="detail strokes cap")
    ap.add_argument("--min-tokens", type=int, default=0, help="no EOS before this many detail strokes")
    ap.add_argument("--stop-at-plan", choices=["off", "q", "n"], default="off", help="also stop the detail part at its planned photons (q) / strokes (n)")
    ap.add_argument("--rebuild-every", type=int, default=256,
                    help="detail: once the chunk is full, rebuild it every this many new strokes; the rebuilt chunk keeps the last "
                         "(training chunk - rebuild_every) strokes (default 256: keeps 768 of 1024)")
    ap.add_argument("--no-rules", action="store_true", help="0.4 models: do not apply the exact sampling rules (t cap, light cone, lt rules); "
                    "0.5 models (model.rules) always apply the rules they were trained with")
    ap.add_argument("--time-order", action="store_true",
                    help="also forbid t earlier than the previous detail stroke / chain head (off: with a broad t distribution it pushes t to the cap)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ranges", type=float, nargs=6, default=[-2, 2, -2, 2, 46, 50])
    ap.add_argument("--allow-cpu", action="store_true")
    a = ap.parse_args()
    if not a.truth_skeleton and not a.skel_run:
        raise SystemExit("give --skel-run (or --truth-skeleton for the diagnostic)")
    from .detail import DetailStore, generate_detail
    device = device_for(a.allow_cpu)
    det_name = a.det_checkpoint
    if det_name == "auto":
        det_name = "best_stroke" if (Path(a.det_run) / "best_stroke.pt").exists() else a.checkpoint
    plan_name = a.det_plan_checkpoint
    if plan_name == "auto":
        plan_name = "best_plan" if (Path(a.det_run) / "best_plan.pt").exists() else det_name
    for nm in {det_name, plan_name}:
        if not (Path(a.det_run) / f"{nm}.pt").exists():
            raise SystemExit(f"{a.det_run}/{nm}.pt does not exist")
    det_model, vocab, det_cfg, det_ck = load_run(a.det_run, device, det_name)
    plan_model, plan_ck = None, det_ck
    if plan_name != det_name:
        plan_model, _, _, plan_ck = load_run(a.det_run, device, plan_name)
    if not is_detail(det_cfg):
        raise SystemExit(f"{a.det_run} is not a detail-model run (model.type detail)")
    if not (det_cfg["data"].get("tstart_prev") and det_cfg["data"].get("bud_plan")):
        raise SystemExit("the detail run must be trained with data.tstart_prev and data.bud_plan true (the sampler counts that way)")
    k_max = int(det_cfg["model"].get("k_max", 1024))
    det_chunk = min(seq_len_at(det_cfg["train"]["seq_len_schedule"], 10 ** 9), int(det_cfg["model"]["window"]))
    if not 0 < a.rebuild_every < det_chunk:
        raise SystemExit(f"--rebuild-every must be between 1 and the detail model's training chunk - 1 ({det_chunk - 1})")
    skel_model = skel_cfg = skel_ck = None
    if a.skel_run and a.truth_skeleton:
        log(f"--truth-skeleton given: --skel-run {a.skel_run} is not used")
        a.skel_run = None
    if a.skel_run:
        skel_model, skel_vocab, skel_cfg, skel_ck = load_run(a.skel_run, device, a.skel_checkpoint or a.checkpoint)
        if skel_vocab.to_dict() != vocab.to_dict():
            raise SystemExit("the skeleton and detail runs use different vocabularies")
        if not (skel_cfg["data"].get("bud_plan") and skel_cfg["data"].get("whole_sequence")):
            log("WARNING: the skeleton run was not trained with data.bud_plan and data.whole_sequence (0.4 skeleton config)")
        skel_chunk = seq_len_at(skel_cfg["train"]["seq_len_schedule"], 10 ** 9)
        if min(a.skel_max_tokens, k_max) > skel_chunk:
            raise SystemExit(f"--skel-max-tokens / k_max {min(a.skel_max_tokens, k_max)} > the skeleton model's training chunk {skel_chunk}")
    rules_in = {name: (m.rules is not None and m.rules.any) for name, m in (("skeleton", skel_model), ("detail", det_model)) if m is not None}
    if any(rules_in.values()):
        log(f"0.5 rules (trained with them, always applied): {rules_in}" + ("; --no-rules / --time-order ignored for these" if (a.no_rules or a.time_order) else ""))
    truth = DetailStore(a.truth_skeleton, a.n) if a.truth_skeleton else None
    out = fresh_dir(a.out, a.name)
    ranges = np.asarray(a.ranges, dtype=np.float64).reshape(3, 2)
    log(f"skeleton: {'TRUE skeletons from ' + a.truth_skeleton if truth else skel_ck}; detail strokes: {det_ck}; detail plan: {plan_ck}; on {device}; {a.n} showers -> {out}")
    write_json(out / "run.json", dict(mode="generate04", skel_run=a.skel_run, det_run=a.det_run, skel_checkpoint=str(skel_ck), det_checkpoint=str(det_ck), det_plan_checkpoint=str(plan_ck),
                                      truth_skeleton=a.truth_skeleton, n=a.n, batch=a.batch, temperature=a.temperature, skel_max_tokens=min(a.skel_max_tokens, k_max),
                                      max_tokens=a.max_tokens, min_tokens=a.min_tokens, stop_at_plan=a.stop_at_plan, det_chunk=det_chunk, rebuild_every=a.rebuild_every, rules=not a.no_rules, time_order=a.time_order, seed=a.seed,
                                      trained_rules=rules_in, vocab=vocab.to_dict(), ranges=ranges.tolist(), skel_config=skel_cfg, det_config=det_cfg))
    def qtok(t):
        """photons of a token array from its lq bins (what the plan counts)"""
        t = np.asarray(t).reshape(-1, len(vocab.names))
        return float(vocab.dec_log(t[:, vocab.index["lq"]], vocab.n_q, vocab.q_log_max).sum()) if len(t) else 0.0

    rows = []
    i = 0
    n_total = min(a.n, len(truth)) if truth else a.n
    while i < n_total:
        B = min(a.batch, n_total - i)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        if truth:
            skels = [truth.load(i + b)[2] for b in range(B)]
            eids = [int(truth.eids[i + b]) for b in range(B)]
            sk_stop = ["truth"] * B
            sk_qp = sk_np = [float("nan")] * B
        else:
            # a skeleton fits in one training chunk that starts at stroke 0 (prefix counters 0), so its prefix is never refreshed
            skels, sk_plan = generate(skel_model, vocab, B, device, a.temperature, min(a.skel_max_tokens, k_max), 10 ** 9, seed=a.seed * 100003 + i,
                                      verbose=False, birth_ptr=bool(skel_cfg["data"].get("birth_ptr", True)), constraints=not a.no_rules, time_order=a.time_order)
            eids = [-1] * B
            sk_stop = sk_plan["stop"]
            sk_qp, sk_np = sk_plan["q_plan"], sk_plan["n_plan"]
        t1 = time.time()
        dets, plan = generate_detail(det_model, vocab, skels, device, a.temperature, a.max_tokens, det_chunk, a.rebuild_every, seed=a.seed * 100003 + i + 7,
                                     verbose=True, min_tokens=a.min_tokens, stop_at_plan=a.stop_at_plan, constraints=not a.no_rules, time_order=a.time_order,
                                     plan_model=plan_model)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.time()
        sec = (t2 - t0) / B
        for b in range(B):
            r = save04(out, i + b, skels[b], dets[b], vocab, ranges, sec, eids[b])
            rows.append(dict(index=i + b, eid=eids[b], n_skel=int(len(skels[b])), n_skel_plan=float(sk_np[b]), q_skel_tok=qtok(skels[b]), q_skel_plan=float(sk_qp[b]),
                             skel_stop=sk_stop[b], n_det=int(len(dets[b])), n_det_plan=float(plan["n_plan"][b]), q_det_tok=qtok(dets[b]),
                             q_det_plan=float(plan["q_plan"][b]), det_stop=plan["stop"][b], n_vox=int(len(r["q"])), q_total=float(r["q"].sum()),
                             skel_s=round((t1 - t0) / B, 2), det_s=round((t2 - t1) / B, 2)))
        log(f"  {i + B}/{n_total}: {sec:.1f} s/shower (skeleton {(t1 - t0) / B:.1f} s, detail {(t2 - t1) / B:.1f} s); last: {rows[-1]}")
        i += B
    cols = list(rows[0])
    with open(out / "timing.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in cols) + "\n")
    from .rules import sample_fallbacks
    SAMPLE_FALLBACKS = sample_fallbacks()
    write_json(out / "summary.json", dict(n=len(rows), rule_fallbacks=dict(SAMPLE_FALLBACKS), n_skel_mean=float(np.mean([r["n_skel"] for r in rows])), n_det_mean=float(np.mean([r["n_det"] for r in rows])),
                                          q_total_mean=float(np.mean([r["q_total"] for r in rows])),
                                          q_det_over_plan_mean=float(np.mean([r["q_det_tok"] / max(r["q_det_plan"], 1.0) for r in rows])),
                                          n_det_over_plan_mean=float(np.mean([r["n_det"] / max(r["n_det_plan"], 1.0) for r in rows])), det_stop={k: sum(r["det_stop"] == k for r in rows) for k in ("eos", "plan", "max")}))
    log("done")


if __name__ == "__main__":
    main()
