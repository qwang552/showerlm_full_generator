"""Generate showers with a single trained ShowerLM, or render true tokens.

    python -m showerlm.sample --run <train dir> --n 256 --batch 16 --out <dir> --name gen_base
    python -m showerlm.sample --from-tokens <tokens dir> --n 256 --out <dir> --name truth_render
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .common import fresh_dir, log, read_json, write_json
from .data import N_CUM, TokenStore, budget_bins, dec_ntok, dec_qtot, enc_cum, enc_ntok, enc_qtot
from . import model as _model
from .model import N_PREFIX, ShowerLM, attr_kv_step, sample_logits
from .prof import region
from .tokens import KIND_CONT, KIND_EOS, KIND_STROKE, Vocab
from .raster import render
from .train import build_model, device_for, is_detail


def load_run(run, device, checkpoint="best"):
    run = Path(run)
    ck_path = run / f"{checkpoint}.pt"
    if not ck_path.exists():
        ck_path = run / "last.pt"
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    vocab = Vocab.from_dict(ck["vocab"])
    model = build_model(cfg, vocab, device)
    model.load_state_dict(ck["ema"])
    model.eval()
    return model, vocab, cfg, ck_path


def no_eos(lg, suppress, col):
    """EOS logit -> -inf: suppress is a bool (every row, the old --min-tokens) or a (B,) bool tensor (0.6.3.2: per row)."""
    if isinstance(suppress, torch.Tensor):
        lg[:, col] = lg[:, col].masked_fill(suppress, float("-inf"))
    elif suppress:
        lg[:, col] = float("-inf")


@torch.no_grad()
def generate(model, vocab, B, device, temperature=1.0, max_tokens=20000, prefix_refresh=256, seed=0, verbose=False, min_tokens=0, stop_at_plan="off", attr_temp=None, tstart_init=0, causal_ptr=False,
             birth_ptr=True, constraints=False, time_order=False, gbins=None, zplan=None, tplan=None, check_every=8, min_plan_frac=0.0,
             late_guard=0.0, late_k=3.0, late_tries=16, late_jump=0.0, cap_stop_t=None, max_plan_frac=0.0):
    """B showers in parallel.  Returns a list of (T_b, A) int arrays and a dict with the plan and why each shower stopped
    (eos = the model ended it; plan = stop_at_plan reached: photons >= planned photons (q) or strokes >= planned strokes (n);
    max = max_tokens).
    0.6.3.2 min_plan_frac > 0 (zplan / tplan models): no EOS while the skeleton light drawn so far is below min_plan_frac x
    the planned skeleton light of the shower (the zplan row total, else the tplan row total); with min_tokens both apply.
    0.6.4a late_guard > 0 (zplan / tplan models): a new chain head (kind STROKE) with light in the last lateness slice
    (tau = t - depth / c >= the last tplan edge, 8 ns; models without a tplan: 8 ns) is decoded again, at most late_tries
    times, when the shower's skeleton light in that slice would exceed late_cap = max(late_k x its planned light there (tplan
    models), late_guard x its planned skeleton light).  Why (REAL, view/t90diag_n256_b.md): 1-2 of 256 generated skeletons
    put ~70% of their light at tau > 8 ns near the 80 ns cap (t90 ~40 ns) while their plan asked for 1.2%; in 3000 training
    showers t90 <= 2.8 ns and light after 20 ns <= 0.4%.  After late_tries the last draw is kept (fallback).  A redraw
    decodes the whole batch again and keeps the new token of the refused rows only (one random stream: the other rows' later
    draws change, their distribution does not).  info: late_redraws, late_fallbacks, late_jumps (per shower; late_jumps =
    steps whose first draw was refused by late_jump).
    0.6.4b late_jump > 0 (zplan models): a chain head is also refused when its time forces more than late_jump x the planned
    skeleton light late.  Chain heads are written in time order (rule time_order: a head's t >= the previous head's t), so
    after a head at time t every later head is at t or later, and the planned light still to draw at depths < c (t - 8 ns)
    (max(0, zplan - drawn) per depth slice, linear inside a slice) can only come late.  Why (REAL, view/latediag_v064a.md,
    v064a runs 10-28): every catastrophic late shower starts with a head that jumps to t = 55-75 ns within its first 1-48
    tokens (0-16% of the plan drawn), usually deep (6-15 m), so its own lateness is < 8 ns and late_guard does not see it;
    after it 60-100% of the heads are late and late_guard redraws thousands of times and falls back (and made the A40
    skeleton batch of 256 slower, 8.9 s / shower vs 3.9 s at 64).  Normal showers draw their first late head after 50-86% of
    the plan at t 9-15 ns.  CONT pieces are never refused (a CONT follows its head's time; 0.6.4a redrew them, uselessly).
    0.6.4c cap_stop_t (a t bin; rules models): a chain head (kind STROKE) drawn at t >= cap_stop_t ends the skeleton instead
    (the step becomes EOS, also below min_plan_frac; stop "cap").  Why (REAL, view/enddiag_064b_tok.md, 256 showers each):
    skeleton tokens exactly at the 80 ns cap bin: truth 1 token in 1 shower; v064b tz 892 tokens in 21 showers (up to 290 in
    one), t 508 in 40.  Heads are time ordered, so after a head at the cap every later head can only take the cap bin; the
    shower keeps drawing there until its plan is used up (drawn / plan 1.03-1.05 at EOS).  The training data have no
    tokens after the cap (the cache cut), so the model has no learned way out but EOS.  info cap_stops (0 / 1 per shower).
    0.6.4c max_plan_frac > 0 (zplan / tplan models): the skeleton stops (stop "plan") at the first chain end (a piece with
    lt = 0) once its drawn light reaches max_plan_frac x its planned skeleton light.  Why (REAL, view/enddiag_064b.md):
    every v064b skeleton ended by EOS at 1.03-1.05 x its plan; the light past the plan is the last drawn = the latest and,
    in time order, deepest (d99 13.1 / 13.4 m vs truth 12.3 m) [inference: the tail excess is that overshoot; the DAG
    measures it].  The detail target follows the skeleton light (target slope), as for any shorter skeleton.

    0.3.0 (models trained on the v3 vocabulary): a token after a piece with lt > 0 is a CONT by rule (its pointer is the
    previous token, s = end, t inside a window around previous t + length / c); with birth_ptr False (data.birth_ptr of
    the run) every other stroke is placed by its absolute head voxel and no pointer is sampled.  Budget inputs (strokes
    and photons still to come) and landmark strokes (kept in the attention cache) follow the model's config.

    constraints (0.4, v3 models): exact rules of the truth enforced while sampling (see sampling_rules).
    time_order (with constraints): also forbid a chain head earlier than the previous head.  Off by default: true for the
    truth, but cutting the earlier half of a broad t distribution makes every step jump later, so t runs to the 12 ns cap
    and the model ends the shower early (0.4 detail model, first generation 2026-10-17).

    0.5 models (model.rules set): the rules the model was trained with are always applied, identically to the loss
    (showerlm.rules); constraints / time_order are then ignored.

    0.6.1g gplan models: gbins (B, n_gfeat) are the bins of the global variables G each shower is generated for
    (showerlm.gfeat.GTable.draw); they ride in the prefix tensor as in training.

    0.6.1z zplan models: zplan (B, K) planned skeleton photons per depth slice (showerlm.zplan.ZTable); its bins ride in the
    prefix after G, and after every drawn token the light still to come per slice (plan minus the light of the tokens drawn
    so far, counted by showerlm.zplan.token_slice_light as in training) rides in the budget input.

    0.6.3 tplan models (model.n_tslice): tplan (B, Kt) planned skeleton photons per LATENESS slice (showerlm.tplan.TTable),
    bins in the prefix after the zplan ones, the light still to come per lateness slice in the budget input after the depth
    slices (showerlm.tplan.token_tslice_light, as in training).
    0.6.3 speed: the bookkeeping (drawn strokes, photons, plan light per slice, budgets, anchors of CONT pieces) stays on
    the device (showerlm/tables.py: the same float64 arithmetic as the numpy code, so the tokens are those of 0.6.2 for the
    same batch, seed and device); the only host sync is the "any shower still alive" test every check_every steps (dead
    showers keep decoding, as before; their tokens are dropped).  info: q_plan, n_plan, stop, z_drawn, t_drawn."""
    from .tables import budget_bins_t, enc_z_t, tables_for
    use_rules = getattr(model, "rules", None) is not None and model.rules.any
    if use_rules:
        from .rules import sample_context
        model.rules.to(device)
        constraints = False
    if constraints:
        check_rule_order(model)
    rng = torch.Generator(device=device).manual_seed(seed)
    ix = vocab.index
    window = model.window
    cont = bool(getattr(model, "cont", False)) and vocab.v3
    use_bud = bool(getattr(model, "budget", False))
    use_lm = bool(getattr(model, "landmarks", False))
    lm_lq = int(getattr(model, "landmark_lq", -1))
    T = tables_for(vocab, device)
    A = len(vocab.names)
    # ---- plan: qtot ~ P(qtot | BOS), ntok ~ P(ntok | BOS, qtot)
    n_g = int(getattr(model, "n_gfeat", 0))
    if n_g and gbins is None:
        raise SystemExit("this skeleton model is conditioned on global variables (gplan): pass gbins (showerlm.gfeat.GTable.draw)")
    n_z = int(getattr(model, "n_zslice", 0))
    zc = getattr(model, "zcfg", None)
    if n_z and (zplan is None or zc is None):
        raise SystemExit("this skeleton model has a longitudinal light plan (zplan): pass zplan (showerlm.zplan.ZTable)")
    n_t = int(getattr(model, "n_tslice", 0))
    tc = getattr(model, "tcfg", None)
    if n_t and (tplan is None or tc is None):
        raise SystemExit("this skeleton model has a lateness light plan (tplan, 0.6.3): pass tplan (showerlm.tplan.TTable)")
    if n_t and n_z and list(map(float, tc["ranges"])) != list(map(float, zc["ranges"])):
        raise SystemExit(f"tplan ranges {tc['ranges']} differ from the zplan ranges {zc['ranges']}")
    prefix = torch.zeros(B, N_PREFIX + n_g + n_z + n_t, dtype=torch.long, device=device)
    prefix[:, 5] = int(tstart_init)                                              # TSTART before the first stroke
    if n_g:
        prefix[:, N_PREFIX: N_PREFIX + n_g] = torch.as_tensor(np.asarray(gbins), dtype=torch.long, device=device).reshape(B, n_g)
    f64 = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=torch.float64, device=device)  # noqa: E731
    z_plan_t = t_plan_t = edges_z = edges_t = None
    if n_z:
        from .zplan import enc_z
        zplan = np.asarray(zplan, dtype=np.float64).reshape(B, n_z)
        z_plan_t = f64(zplan)
        z_drawn = torch.zeros(B, n_z, dtype=torch.float64, device=device)      # skeleton light drawn so far per depth slice
        edges_z = f64(zc["edges"])
        prefix[:, N_PREFIX + n_g: N_PREFIX + n_g + n_z] = torch.from_numpy(enc_z(zplan, zc["n_bins"], zc["q_max"])).to(device)
    if n_t:
        from .zplan import enc_z
        tplan = np.asarray(tplan, dtype=np.float64).reshape(B, n_t)
        t_plan_t = f64(tplan)
        t_drawn = torch.zeros(B, n_t, dtype=torch.float64, device=device)      # skeleton light drawn so far per lateness slice
        edges_t = f64(tc["edges"])
        prefix[:, N_PREFIX + n_g + n_z:] = torch.from_numpy(enc_z(tplan, tc["n_bins"], tc["q_max"])).to(device)
    ranges_sl = (zc or tc or {}).get("ranges")
    g_emb = model.embed_global(prefix) if (n_g or n_z or n_t) else None
    h = model.trunk(model.embed_prefix(prefix)[:, :1])
    prefix[:, 1] = sample_logits(model.head_qtot(h[:, 0]).float(), temperature, rng)
    h = model.trunk(model.embed_prefix(prefix)[:, :2])
    prefix[:, 2] = sample_logits(model.head_ntok(h[:, 1]).float(), temperature, rng)
    q_plan = dec_qtot(prefix[:, 1].cpu().numpy()); n_plan = dec_ntok(prefix[:, 2].cpu().numpy())
    q_plan_t, n_plan_t = f64(q_plan), f64(n_plan)
    caches = [dict() for _ in model.blocks]
    h = model.trunk(model.embed_prefix(prefix), None, caches)
    ctx = h[:, -1]
    keys = torch.zeros(B, 0, model.d, device=device)
    key_t = torch.zeros(B, 0, dtype=torch.long, device=device)                   # t bin of each key stroke (causal pointer mask)
    # ---- device bookkeeping: the drawn strokes of every shower, their count, photons and last t
    buf = torch.zeros(B, max_tokens, A, dtype=torch.long, device=device)
    rows = torch.arange(B, device=device)
    n_tok = torch.zeros(B, dtype=torch.long, device=device)
    q_sum = torch.zeros(B, dtype=torch.float64, device=device)
    last_t = torch.zeros(B, dtype=torch.long, device=device)
    alive = torch.ones(B, dtype=torch.bool, device=device)
    stop_code = torch.zeros(B, dtype=torch.long, device=device)                  # 0 running / max, 1 eos, 2 plan, 3 cap (0.6.4c)
    # attention cache bookkeeping (landmarks): per cache column its global stroke index (-1 = prefix) and landmark flags per sample
    col_step = torch.full((N_PREFIX,), -1, dtype=torch.long, device=device)
    col_lm = torch.zeros(B, N_PREFIX, dtype=torch.bool, device=device)
    prev = None                                                                   # last stroke of every shower (B, A)
    bud = None
    t_head = torch.zeros(B, dtype=torch.long, device=device)                      # t of the last chain head (non-CONT stroke)
    zero_b = torch.zeros(B, dtype=torch.long, device=device)

    def fixed_pos(tok):
        """ix, iy, iz from the anchor when a pointer was decoded (the parent = the stroke ptr tokens back); s/dx/dy/dz are
        n/a (0) when it was not.  Vocab.anchor arithmetic on the device (showerlm/tables.py)."""
        p = tok[:, ix["ptr"]]
        ok = (p > 0) & (p <= n_tok)
        parent = buf[rows, (n_tok - p).clamp(0, max_tokens - 1)]
        anc = T.anchors(parent, tok[:, ix["s"]])
        vals = (anc + T.d_offsets(tok)).clamp(0, vocab.g - 1)
        return torch.where(ok[:, None], vals, torch.zeros_like(vals)), ok

    pos_cache = {}

    def fx(name, col):
        def f(tok):
            if name == "ix":
                pos_cache["v"], pos_cache["m"] = fixed_pos(tok)
            return pos_cache["v"][:, col], pos_cache["m"]
        return f

    def na_when_no_ptr(tok):
        return zero_b, tok[:, ix["ptr"]] == 0

    fixed = {"ix": fx("ix", 0), "iy": fx("iy", 1), "iz": fx("iz", 2), "s": na_when_no_ptr, "dx": na_when_no_ptr, "dy": na_when_no_ptr, "dz": na_when_no_ptr}
    check_every = max(1, int(check_every))
    if min_plan_frac > 0 and not (n_z or n_t):
        raise SystemExit("min_plan_frac needs a skeleton model with a light plan (zplan / tplan)")
    plan_light = (z_plan_t.sum(1) if n_z else t_plan_t.sum(1)) if min_plan_frac > 0 else None
    late_cap = edges_g = None
    late_red = torch.zeros(B, dtype=torch.long, device=device); late_fb = torch.zeros(B, dtype=torch.long, device=device)
    late_jmp = torch.zeros(B, dtype=torch.long, device=device)
    cap_n = torch.zeros(B, dtype=torch.long, device=device)                       # 0.6.4c cap stops
    if cap_stop_t is not None and not use_rules:
        raise SystemExit("cap_stop needs a skeleton model trained with the rules (time order, t cap)")
    if max_plan_frac > 0 and not (n_z or n_t):
        raise SystemExit("max_plan_frac needs a skeleton model with a light plan (zplan / tplan)")
    plan_max = (z_plan_t.sum(1) if n_z else t_plan_t.sum(1)) * max_plan_frac if max_plan_frac > 0 else None
    if late_jump > 0 and not n_z:
        raise SystemExit("late_jump needs a skeleton model with a depth light plan (zplan)")
    if late_guard > 0 or late_jump > 0:
        if not (n_z or n_t):
            raise SystemExit("late_guard needs a skeleton model with a light plan (zplan / tplan)")
        edges_g = edges_t if n_t else f64([0.0, 8.0])
        plan_tot = z_plan_t.sum(1) if n_z else t_plan_t.sum(1)
        if late_guard > 0:
            late_cap = torch.maximum(late_k * t_plan_t[:, -1], late_guard * plan_tot) if n_t else late_guard * plan_tot
        else:
            late_cap = torch.full_like(plan_tot, float("inf"))
        late_drawn = torch.zeros(B, dtype=torch.float64, device=device)
        jump_cap = late_jump * plan_tot
        if n_z:                                                                  # 0.6.4b: depth slices [lo, hi) of the zplan, the last one to the box floor
            from .common import C_M_PER_NS
            rz = np.asarray(ranges_sl, dtype=np.float64).reshape(3, 2)
            z_lo = edges_z
            z_hi = torch.cat([edges_z[1:], f64([max(float(rz[2, 1] - rz[2, 0]), float(edges_z[-1]) + 1e-6)])])
            tau_last = float(edges_g[-1])
    for step in range(max_tokens):
        sup = step < min_tokens
        if plan_light is not None:
            sup = (z_drawn.sum(1) if n_z else t_drawn.sum(1)) < min_plan_frac * plan_light
            if step < min_tokens:
                sup = torch.ones_like(sup)
        M = keys.shape[1]
        ptr_mask = torch.ones(B, M, dtype=torch.bool, device=device) if M else None
        force_cont = (prev[:, ix["lt"]] > 0) if (cont and prev is not None) else torch.zeros(B, dtype=torch.bool, device=device)
        t_rule = model.rule_t(prev) if (cont and prev is not None) else None
        def decode_step():
            if use_rules:
                return decode_one_rules(model, ctx, fixed, temperature, keys if M else None, ptr_mask, rng, sample_context(model, prev, t_head, B, device),
                                       suppress_eos=sup, attr_temp=attr_temp, key_t=key_t if (M and causal_ptr) else None, birth_ptr=birth_ptr,
                                       s_end=vocab.s_end)
            else:
                cons = sampling_rules(model, vocab, t_head if time_order else torch.zeros_like(t_head), prev) if constraints else None
                return decode_one(model, ctx, fixed, temperature, keys if M else None, ptr_mask, rng, suppress_eos=sup, attr_temp=attr_temp,
                                 key_t=key_t if (M and causal_ptr) else None, force_cont=force_cont if cont else None, t_rule=t_rule, birth_ptr=birth_ptr, s_end=vocab.s_end,
                                 cons=cons)
        with region("skel.decode"):
            tok = decode_step()
        if late_cap is not None:                                                  # 0.6.4a/b late guard: refuse late / jumping chain heads
            with region("skel.late_guard"):
                def late_of(t_):
                    _, tl = T.slice_light(t_, None, edges_g, ranges_sl)
                    return tl[:, -1]
                def forced_of(t_):                                                # 0.6.4b: planned light still to come above depth c (t - 8 ns)
                    t0 = T.t[t_[:, ix["t"]].clamp(0, len(T.t) - 1)].to(torch.float64)
                    D = C_M_PER_NS * (t0 - tau_last)
                    frac = ((D[:, None] - z_lo[None, :]) / (z_hi - z_lo)[None, :]).clamp(0.0, 1.0)
                    return ((z_plan_t - z_drawn).clamp_min(0.0) * frac).sum(1)
                def refuse(t_):
                    lt_ = late_of(t_)
                    bad_ = (lt_ > 0) & (late_drawn + lt_ > late_cap)
                    if late_jump > 0:
                        bad_ = bad_ | (forced_of(t_) > jump_cap)
                    return alive & (t_[:, ix["kind"]] == KIND_STROKE) & bad_
                bad = refuse(tok)
                if late_jump > 0:
                    late_jmp += (bad & (forced_of(tok) > jump_cap)).long()
                for _ in range(int(late_tries)):
                    if not bool(bad.any()):
                        break
                    late_red += bad.long()
                    t2 = decode_step()
                    tok = torch.where(bad[:, None], t2, tok)
                    bad = bad & refuse(tok)
                late_fb += bad.long()
        if cap_stop_t is not None:                                                # 0.6.4c: a chain head at the t cap ends the skeleton
            hit_cap = alive & (tok[:, ix["kind"]] == KIND_STROKE) & (tok[:, ix["t"]] >= int(cap_stop_t))
            tok[:, ix["kind"]] = torch.where(hit_cap, torch.full_like(hit_cap, KIND_EOS, dtype=torch.long), tok[:, ix["kind"]])
            cap_n += hit_cap.long()
        kind = tok[:, ix["kind"]]
        done_now = alive & (kind == KIND_EOS)
        alive = alive & ((kind == KIND_STROKE) | (kind == KIND_CONT))
        stop_code = torch.where(done_now, torch.ones_like(stop_code), stop_code)
        if cap_stop_t is not None:
            stop_code = torch.where(hit_cap, torch.full_like(stop_code, 3), stop_code)
        drawn = alive                                                             # rows that append a stroke this step (before plan stops)
        slot = n_tok.clamp(max=max_tokens - 1)
        buf[rows, slot] = torch.where(drawn[:, None], tok, buf[rows, slot])
        q_sum = q_sum + torch.where(drawn, T.photons(tok), torch.zeros_like(q_sum))
        last_t = torch.where(drawn, tok[:, ix["t"]], last_t)
        n_tok = n_tok + drawn.long()
        if stop_at_plan == "q":
            hit = drawn & (q_sum >= q_plan_t)
        elif stop_at_plan == "n":
            hit = drawn & (n_tok.to(torch.float64) >= n_plan_t)
        else:
            hit = None
        if hit is not None:
            alive = alive & ~hit
            stop_code = torch.where(hit, torch.full_like(stop_code, 2), stop_code)
        if late_cap is not None:
            late_drawn = late_drawn + torch.where(drawn, T.slice_light(tok, None, edges_g, ranges_sl)[1][:, -1], torch.zeros_like(late_drawn))
        if n_z or n_t:                                                            # every drawn stroke counts (info z_drawn / t_drawn)
            zl, tl = T.slice_light(tok, edges_z, edges_t, ranges_sl)
            if n_z:
                z_drawn = z_drawn + torch.where(drawn[:, None], zl, torch.zeros_like(zl))
            if n_t:
                t_drawn = t_drawn + torch.where(drawn[:, None], tl, torch.zeros_like(tl))
        if plan_max is not None:                                                  # 0.6.4c: stop at the first chain end past the plan
            end_ok = (tok[:, ix["lt"]] == 0) if cont else torch.ones_like(drawn)
            hit_m = drawn & alive & end_ok & ((z_drawn.sum(1) if n_z else t_drawn.sum(1)) >= plan_max)
            alive = alive & ~hit_m
            stop_code = torch.where(hit_m, torch.full_like(stop_code, 2), stop_code)
        if ((step + 1) % check_every == 0 or step + 1 == max_tokens) and not bool(alive.any()):
            break
        tok = torch.where(alive[:, None], tok, torch.zeros_like(tok)); tok[:, ix["kind"]] = torch.where(alive, torch.where(kind == KIND_CONT, kind, torch.ones_like(kind)), kind)
        prev = tok
        t_head = torch.where(alive & (tok[:, ix["kind"]] != KIND_CONT), tok[:, ix["t"]], t_head)
        if use_bud:
            bb = budget_bins_t(n_plan_t - n_tok.to(torch.float64), q_plan_t - q_sum)
            parts = [bb]
            if n_z:
                zr = z_plan_t - z_drawn
                zr = torch.where(zr.abs() < 0.5, torch.zeros_like(zr), zr)
                parts.append(enc_z_t(zr, zc["n_bins"], zc["q_max"]))
            if n_t:
                tr = t_plan_t - t_drawn
                tr = torch.where(tr.abs() < 0.5, torch.zeros_like(tr), tr)
                parts.append(enc_z_t(tr, tc["n_bins"], tc["q_max"]))
            bud = torch.cat(parts, 1)[:, None, :]
        lm_now = (tok[:, ix["lq"]] >= lm_lq) & alive if (use_lm and lm_lq >= 0) else torch.zeros(B, dtype=torch.bool, device=device)
        col_step = torch.cat([col_step, torch.full((1,), step, dtype=torch.long, device=device)]); col_lm = torch.cat([col_lm, lm_now[:, None]], 1)
        attn = None
        if use_lm:
            vis = (col_step < 0) | (col_step > step - window) | col_lm                # (B, M + 1): prefix, last window, own landmarks
            attn = vis[:, None, None, :]
        with region("skel.trunk"):
            h = model.trunk(model.embed_strokes(tok[:, None, :], bud, g_emb), attn, caches)
        ctx = h[:, -1]
        keys = torch.cat([keys, ctx[:, None, :]], 1)[:, -model.w_ptr:]
        key_t = torch.cat([key_t, tok[:, ix["t"]][:, None]], 1)[:, -model.w_ptr:]
        # rolling cache: prefix + last `window` strokes (+ landmarks of any sample); columns nobody needs are dropped
        if caches[0]["k"].shape[2] > N_PREFIX + window + 256:
            keep = (col_step < 0) | (col_step > step - window) | col_lm.any(0)
            idx = torch.nonzero(keep).squeeze(1)
            for c in caches:
                c["k"] = c["k"][:, :, idx]; c["v"] = c["v"][:, :, idx]
            col_step = col_step[idx]; col_lm = col_lm[:, idx]
        if (step + 1) % prefix_refresh == 0:                                       # refresh the running counters of the prefix
            fq = torch.clamp(q_sum / q_plan_t.clamp_min(1.0), max=0.999)
            fn = torch.clamp(n_tok.to(torch.float64) / n_plan_t.clamp_min(1.0), max=0.999)
            prefix[:, 3] = torch.floor(fq * (N_CUM - 1)).clamp(0, N_CUM - 1).long()
            prefix[:, 4] = torch.floor(fn * (N_CUM - 1)).clamp(0, N_CUM - 1).long()
            prefix[:, 5] = last_t
            fresh = [dict() for _ in model.blocks]
            model.trunk(model.embed_prefix(prefix), None, fresh)
            for c, f in zip(caches, fresh):
                c["k"][:, :, :N_PREFIX] = f["k"]; c["v"][:, :, :N_PREFIX] = f["v"]
        if verbose and (step + 1) % 1000 == 0:
            log(f"  step {step + 1}: alive {int(alive.sum())}/{B}, photons so far {float(q_sum.mean()):.3g} (plan {q_plan.mean():.3g})")
    n_np = n_tok.cpu().numpy()
    buf_np = buf[:, : int(n_np.max()) if B else 0].cpu().numpy()
    sc = stop_code.cpu().numpy()
    stop = [("eos", "plan", "cap")[c - 1] if c > 0 else "max" for c in sc]
    toks = [buf_np[b, : n_np[b]].astype(np.int64).reshape(-1, A) for b in range(B)]
    return toks, dict(q_plan=q_plan, n_plan=n_plan, stop=stop, z_drawn=z_drawn.cpu().numpy() if n_z else None, t_drawn=t_drawn.cpu().numpy() if n_t else None,
                      late_redraws=late_red.cpu().numpy(), late_fallbacks=late_fb.cpu().numpy(),
                      late_jumps=late_jmp.cpu().numpy(), cap_stops=cap_n.cpu().numpy())

@torch.no_grad()
def check_rule_order(model):
    """The sampling rules read t (and len, for lt) of the token being decoded, so those must come first in the decoding order."""
    o, ai = list(model.order), model.ai
    if o.index(ai["t"]) > o.index(ai["iz"]) or ("lt" in ai and max(o.index(ai["t"]), o.index(ai["len"])) > o.index(ai["lt"])):
        raise ValueError("the sampling rules need t decoded before iz, and t and len before lt (model.attr_order); sample with --no-rules")


def sampling_rules(model, vocab, t_head, prev):
    """0.4 sampling constraints (never broken by the truth: checked on event 1772 and the synthetic test set):
    t <= t_cap = enc_t(12 ns) (the cache keeps raw t < 12 ns); a chain head is not earlier than the previous chain head
    (chains are ordered by head time); an absolutely placed head is not deeper than light can go by the upper edge of its
    t bin (+ 1 voxel of slack); lt of a CONT is not larger than lt of the piece before; a piece whose next piece could not
    start before t_cap (t + rule - t_win_lo > t_cap) ends its chain (lt = 0)."""
    return dict(t_cap=vocab.t_cap, t_head=t_head, lt_prev=None if prev is None or "lt" not in vocab.index else prev[:, vocab.index["lt"]],
                vocab=vocab)


def cone_iz_min(vocab, t_bin):
    """Lowest iz an absolutely placed head may have at time bin t: depth below the top <= c * t_upper (+ 1 voxel)."""
    from .common import C_CM_PER_NS, VOX_CM
    t_up = (t_bin.double() + 1.0) * vocab.t_max / vocab.n_t
    return torch.floor(vocab.g - 1 - C_CM_PER_NS * t_up / VOX_CM).long() - 1


def masked_or_cap(lg, allowed, fallback):
    """lg with the bins outside `allowed` (B, n) removed; a row with nothing allowed keeps only bin `fallback` (B,)."""
    none = ~allowed.any(1)
    if none.any():
        allowed = allowed.clone()
        allowed[none] = False
        allowed[none, fallback[none].clamp(0, lg.shape[1] - 1)] = True
    return lg.masked_fill(~allowed, float("-inf"))


def decode_one(model, ctx, fixed, temperature, keys, ptr_mask, rng, suppress_eos=False, attr_temp=None, key_t=None, force_cont=None, t_rule=None, birth_ptr=True, s_end=16,
               cons=None):
    """One stroke for the batch: attributes decoded in the model's order; `fixed` callables may overwrite values for some
    samples; attr_temp {name: temperature} overrides the global temperature for single attributes.
    force_cont (B,) bool (v3 models): samples whose previous piece has lt > 0 MUST be CONT, the others must not; a CONT
    gets ptr 1, s = s_end and a t inside the window around t_rule (B,).  birth_ptr False: non-CONT strokes get ptr 0.
    cons (sampling_rules): exact constraints of the truth applied to t, iz and lt."""
    B = ctx.shape[0]
    tok = torch.zeros(B, model.A, dtype=torch.long, device=ctx.device)
    x0 = model.h_proj(ctx)[:, None, :]
    order = getattr(model, "order", list(range(model.A)))
    is_cont = torch.zeros(B, dtype=torch.bool, device=ctx.device)
    for p, a in enumerate(order):
        xs = [x0] + [model.attr_in[b](tok[:, b])[:, None, :] for b in order[:p]]
        x = torch.cat(xs, 1) + model.slot.weight[None, : p + 1, :]
        for blk in model.attr_blocks:
            x = blk(x, None)
        y = model.attr_ln(x[:, p])
        name = model.names[a]
        temperature_a = (attr_temp or {}).get(name, temperature)
        if name == "kind":
            logits = model.heads[a](y).float()
            logits[:, 0] = float("-inf")                                            # PAD is never generated
            no_eos(logits, suppress_eos, 2)
            if force_cont is not None and logits.shape[1] >= 4:
                logits[:, 3] = torch.where(force_cont, logits[:, 3], torch.full_like(logits[:, 3], float("-inf")))
                logits[:, :3] = torch.where(force_cont[:, None], torch.full_like(logits[:, :3], float("-inf")), logits[:, :3])
            elif logits.shape[1] >= 4:
                logits[:, 3] = float("-inf")
            tok[:, a] = sample_logits(logits, temperature_a, rng)
            is_cont = tok[:, a] == KIND_CONT
            continue
        if name == "ptr":
            if keys is None:
                tok[:, a] = torch.where(is_cont, torch.ones_like(tok[:, a]), tok[:, a])
                continue
            q = model.ptr_q(y)
            s = torch.einsum("bd,bjd->bj", q, model.ptr_k(keys)) / (model.d ** 0.5)
            if key_t is not None and order.index(model.ai["t"]) < p:                # causal: a parent cannot be later than the child
                ptr_mask = ptr_mask & (key_t <= tok[:, model.ai["t"]][:, None])
            s = s.masked_fill(~ptr_mask, float("-inf"))
            none = (q @ model.ptr_none)[:, None] / (model.d ** 0.5)
            c = sample_logits(torch.cat([none, s], 1).float(), temperature_a, rng)
            M = keys.shape[1]
            val = torch.where(c > 0, M - (c - 1), torch.zeros_like(c))
            if not birth_ptr:
                val = torch.zeros_like(val)
            tok[:, a] = torch.where(is_cont, torch.ones_like(val), val)
        elif name == "s" and is_cont.any():
            tok[:, a] = sample_logits(model.heads[a](y).float(), temperature_a, rng)
            tok[:, a] = torch.where(is_cont, torch.full_like(tok[:, a], int(s_end)), tok[:, a])
        elif name == "t" and ((t_rule is not None and is_cont.any() and getattr(model, "t_window", 0) > 0) or cons is not None):
            lg = model.heads[a](y).float()
            if t_rule is not None and is_cont.any() and getattr(model, "t_window", 0) > 0:
                lg = torch.where(is_cont[:, None], model.cont_t_mask(lg, t_rule), lg)
            if cons is not None:
                b = torch.arange(lg.shape[1], device=lg.device)[None, :]
                ok = (b <= cons["t_cap"]) & torch.isfinite(lg)
                ok &= is_cont[:, None] | (b >= cons["t_head"][:, None])
                lg = masked_or_cap(lg, ok, torch.full_like(cons["t_head"], cons["t_cap"]))
            tok[:, a] = sample_logits(lg, temperature_a, rng)
        elif name == "iz" and cons is not None:
            lg = model.heads[a](y).float()
            b = torch.arange(lg.shape[1], device=lg.device)[None, :]
            ok = b >= cone_iz_min(cons["vocab"], tok[:, model.ai["t"]])[:, None]
            lg = masked_or_cap(lg, ok, torch.full_like(tok[:, a], lg.shape[1] - 1))
            tok[:, a] = sample_logits(lg, temperature_a, rng)                      # heads placed by a pointer are overwritten below
        elif name == "lt" and cons is not None:
            lg = model.heads[a](y).float()
            b = torch.arange(lg.shape[1], device=lg.device)[None, :]
            ok = torch.ones_like(lg, dtype=torch.bool)
            if cons["lt_prev"] is not None:
                ok &= ~is_cont[:, None] | (b <= cons["lt_prev"][:, None])
            if model.t_window > 0:
                nxt = tok[:, model.ai["t"]] + model.t_rule[tok[:, model.ai["len"]]] - model.t_win_lo
                ok &= ~(nxt > cons["t_cap"])[:, None] | (b == 0)
            lg = masked_or_cap(lg, ok, torch.zeros_like(tok[:, a]))
            tok[:, a] = sample_logits(lg, temperature_a, rng)
        else:
            tok[:, a] = sample_logits(model.heads[a](y).float(), temperature_a, rng)
        if name in fixed:
            v, m = fixed[name](tok)
            tok[:, a] = torch.where(m, v, tok[:, a])
    return tok


def decode_one_rules(model, ctx, fixed, temperature, keys, ptr_mask, rng, rctx, suppress_eos=False, attr_temp=None, key_t=None, birth_ptr=True, s_end=16,
                     forced=None):
    """decode_one for 0.5 models: every attribute's logits pass through showerlm.rules.constrain, the function the
    training loss uses (kind: no PAD, CONT exactly when the previous piece has lt > 0; t above the time-order / CONT
    floor, as an offset when model.t_rel, below the cap and the CONT top; light cone on absolute heads; |d| <= 8 on
    CONT pieces; lt rules; photons per record).  Only suppress_eos (--min-tokens) is sampling-only.  A CONT gets
    ptr 1 and s = s_end, as in the data.  forced (B, A): take these values instead of sampling (tests: the masked
    logits must equal the training ones)."""
    from .rules import constrain
    R = model.rules
    B = ctx.shape[0]
    tok = torch.zeros(B, model.A, dtype=torch.long, device=ctx.device)
    x0 = model.h_proj(ctx)[:, None, :]
    order = model.order
    is_cont = torch.zeros(B, dtype=torch.bool, device=ctx.device)
    embs = []                                                                       # 0.6.3: decoder inputs of the attributes decoded so far
    caches = {} if _model.ATTR_KV else None                                       # 0.6.4b SHOWERLM_ATTR_KV=1
    for p, a in enumerate(order):                                                   # (each embedded once its value is final: same tensors)
        if p:
            embs.append(model.attr_in[order[p - 1]](tok[:, order[p - 1]])[:, None, :])
        if caches is not None:
            y = attr_kv_step(model, (embs[-1] if p else x0) + model.slot.weight[None, p: p + 1, :], caches)
        else:
            x = torch.cat([x0] + embs, 1) + model.slot.weight[None, : p + 1, :]
            for blk in model.attr_blocks:
                x = blk(x, None)
            y = model.attr_ln(x[:, p])
        name = model.names[a]
        ta = (attr_temp or {}).get(name, temperature)
        if name == "ptr" and forced is not None:
            tok[:, a] = forced[:, a]
            continue
        if name == "ptr":
            val = torch.zeros(B, dtype=torch.long, device=ctx.device)
            if keys is not None and birth_ptr:
                q = model.ptr_q(y)
                sc = torch.einsum("bd,bjd->bj", q, model.ptr_k(keys)) / (model.d ** 0.5)
                pm = ptr_mask
                if key_t is not None and order.index(model.ai["t"]) < p:
                    pm = pm & (key_t <= tok[:, model.ai["t"]][:, None])
                sc = sc.masked_fill(~pm, float("-inf"))
                none = (q @ model.ptr_none)[:, None] / (model.d ** 0.5)
                c = sample_logits(torch.cat([none, sc], 1).float(), ta, rng)
                val = torch.where(c > 0, keys.shape[1] - (c - 1), torch.zeros_like(c))
            tok[:, a] = torch.where(is_cont, torch.ones_like(val), val)
            continue
        lg = model.heads[a](y).float()
        if name == "kind":
            no_eos(lg, suppress_eos, KIND_EOS)
        lg = constrain(R, model, name, lg, tok, rctx)
        if forced is not None:
            tok[:, a] = forced[:, a]
            if name == "kind":
                is_cont = tok[:, a] == KIND_CONT
            continue
        val = sample_logits(lg, ta, rng)
        if name == "s":
            val = torch.where(is_cont, torch.full_like(val, int(s_end)), val)
        tok[:, a] = val
        if name == "kind":
            is_cont = val == KIND_CONT
        if name in fixed:
            v, m = fixed[name](tok)
            tok[:, a] = torch.where(m, v, tok[:, a])
    return tok


def save_shower(out, i, tok, vocab, g, ranges, seconds, eid=-1):
    st = vocab.decode(tok, use_pointer=False)
    r = render(st, g, ranges)
    np.savez_compressed(out / f"sample_{i:05d}.npz", ijk=r["ijk"], q=r["q"], off=r["off"], m=r["m"], t=r["t"], cnt=r["cnt"],
                        eid=np.int64(eid), seconds=np.asarray([seconds], np.float32))
    np.savez_compressed(out / f"tokens_{i:05d}.npz", tok=tok.astype(np.int16), eid=np.int64(eid))
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="training run directory")
    ap.add_argument("--checkpoint", default="best", help="best | last")
    ap.add_argument("--from-tokens", default=None, help="render truth token files instead of generating")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="/data/user/qwang/showerlm/out")
    ap.add_argument("--name", default="samples")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=20000)
    ap.add_argument("--min-tokens", type=int, default=0, help="no EOS before this many strokes (0 = free)")
    ap.add_argument("--stop-at-plan", choices=["off", "q", "n"], default="off",
                    help="also stop a shower when its photons reach the planned total (q) or its strokes the planned number (n)")
    ap.add_argument("--prefix-refresh", type=int, default=256)
    ap.add_argument("--tstart-init", type=int, default=0, help="TSTART prefix before the first stroke (0 for tstart_prev models; 1 matches 0.1.x training)")
    ap.add_argument("--causal-ptr", action="store_true", help="mask parents later than the child (only when t is decoded before ptr)")
    ap.add_argument("--attr-temperature", default="", help="per-attribute temperatures, e.g. 'ptr=0.7,t=0.5' (others use --temperature)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ranges", type=float, nargs=6, default=[-2, 2, -2, 2, 46, 50])
    ap.add_argument("--allow-cpu", action="store_true")
    a = ap.parse_args()
    attr_temp = {k: float(v) for k, v in (kv.split("=") for kv in a.attr_temperature.split(",") if kv)}
    if not a.from_tokens:
        device = device_for(a.allow_cpu)
        model, vocab, cfg, ck_path = load_run(a.run, device, a.checkpoint)
        if is_detail(cfg):
            raise SystemExit(f"{a.run} is a 0.4 detail-model run (model.type detail): generate with python -m showerlm.sample04")
    out = fresh_dir(a.out, a.name)
    ranges = np.asarray(a.ranges, dtype=np.float64).reshape(3, 2)
    if a.from_tokens:
        store = TokenStore(a.from_tokens, a.n)
        vocab = store.vocab
        log(f"rendering {len(store)} truth token files -> {out}")
        for i in range(len(store)):
            tok, _ = store.load(i)
            t0 = time.time(); save_shower(out, i, tok, vocab, vocab.g, ranges, 0.0, eid=int(store.eids[i]))
            if (i + 1) % 32 == 0:
                log(f"  {i + 1}/{len(store)}")
        write_json(out / "run.json", dict(mode="from_tokens", source=str(a.from_tokens), n=len(store), ranges=ranges.tolist()))
        log("done"); return
    log(f"model {ck_path} on {device}; generating {a.n} showers in batches of {a.batch} -> {out}")
    write_json(out / "run.json", dict(mode="generate", run=str(a.run), checkpoint=str(ck_path), n=a.n, batch=a.batch, temperature=a.temperature,
                                      max_tokens=a.max_tokens, min_tokens=a.min_tokens, stop_at_plan=a.stop_at_plan, attr_temperature=attr_temp, tstart_init=a.tstart_init, causal_ptr=a.causal_ptr, birth_ptr=bool(cfg["data"].get("birth_ptr", True)), seed=a.seed, ranges=ranges.tolist(), config=cfg))
    rows = []
    i = 0
    while i < a.n:
        B = min(a.batch, a.n - i)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        toks, plan = generate(model, vocab, B, device, a.temperature, a.max_tokens, a.prefix_refresh, seed=a.seed * 100003 + i, verbose=True, min_tokens=a.min_tokens, stop_at_plan=a.stop_at_plan, attr_temp=attr_temp,
                              tstart_init=a.tstart_init, causal_ptr=a.causal_ptr, birth_ptr=bool(cfg["data"].get("birth_ptr", True)))
        if device.type == "cuda":
            torch.cuda.synchronize()
        sec = (time.time() - t0) / B
        for b in range(B):
            t1 = time.time()
            r = save_shower(out, i + b, toks[b], vocab, vocab.g, ranges, sec)
            rows.append(dict(index=i + b, n_tok=int(len(toks[b])), n_vox=int(len(r["q"])), q_total=float(r["q"].sum()), q_plan=float(plan["q_plan"][b]),
                             n_plan=float(plan["n_plan"][b]), stop=plan["stop"][b], gen_s=round(sec, 3), render_s=round(time.time() - t1, 3)))
        log(f"  {i + B}/{a.n}: {sec:.2f} s/shower (batch {B}); last: {rows[-1]}")
        i += B
    with open(out / "timing.csv", "w") as f:
        f.write("index,n_tok,n_vox,q_total,q_plan,n_plan,stop,gen_s,render_s\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in ("index", "n_tok", "n_vox", "q_total", "q_plan", "n_plan", "stop", "gen_s", "render_s")) + "\n")
    write_json(out / "summary.json", dict(n=len(rows), gen_s_median=float(np.median([r["gen_s"] for r in rows])), n_tok_mean=float(np.mean([r["n_tok"] for r in rows])),
                                          n_vox_mean=float(np.mean([r["n_vox"] for r in rows])), q_total_mean=float(np.mean([r["q_total"] for r in rows])),
                                          stop={k: sum(r["stop"] == k for r in rows) for k in ("eos", "plan", "max")}))
    log("done")


if __name__ == "__main__":
    main()
