"""Train ShowerLM on token chunks (curriculum over the window length).

    python -m showerlm.train --config <config.yaml> --out <dir> --name <run>
    python -m showerlm.train --resume <run dir>
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import fresh_dir, log, read_yaml, write_json
from .data import N_CUM, N_NTOK, N_QTOT, ChunkDataset, TokenStore, trim_batch
from .model import ShowerLM, count_params


def device_for(allow_cpu):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        return torch.device("cpu")
    raise SystemExit("no GPU (use --allow-cpu for a test run)")


def amp_setting(cfg, device):
    want = cfg["train"].get("amp", "auto")
    if device.type != "cuda" or want in ("off", "none", False):
        return None, False
    if want == "auto":
        return (torch.bfloat16, False) if torch.cuda.is_bf16_supported() else (torch.float16, True)
    return (torch.bfloat16, False) if want == "bf16" else (torch.float16, True)


def is_detail(cfg):
    """A detail model (trained on tokens_det_* with the skeleton in view): 0.4-0.5 showerlm.detail.DetailLM (type detail)
    or 0.6 showerlm.group.GroupDetailLM (type group)?"""
    return cfg["model"].get("type", "showerlm") in ("detail", "group")


def is_group(cfg):
    """0.6 group detail model (showerlm.group: detail strokes generated piece by piece, all pieces at once)?"""
    return cfg["model"].get("type", "showerlm") == "group"


def build_model(cfg, vocab, device):
    m = cfg["model"]
    cls, extra = ShowerLM, {}
    if is_group(cfg):
        from .group import GroupDetailLM
        cls, extra = GroupDetailLM, dict(k_max=int(m.get("k_max", 4096)), group_mode=m.get("group_mode", "join"), group_plan=bool(m.get("group_plan", False)),
                                         occ_feat=m.get("occ_feat") or None, pos_first=bool(m.get("pos_first", False)), group_order=m.get("group_order", "time"))
        if extra["k_max"] > vocab.w_ptr:
            raise SystemExit(f"model.k_max {extra['k_max']} > w_ptr {vocab.w_ptr}: the pointer attribute holds k + 1")
    elif is_detail(cfg):
        from .detail import DetailLM
        cls, extra = DetailLM, dict(k_max=int(m.get("k_max", 1024)))
        if extra["k_max"] > vocab.w_ptr:
            raise SystemExit(f"model.k_max {extra['k_max']} > w_ptr {vocab.w_ptr}: the pointer attribute holds k + 1")
    zc = None
    if m.get("zplan"):                                                          # longitudinal light plan (showerlm.zplan)
        from .zplan import default_cfg
        if is_detail(cfg) or not m.get("budget", False):
            raise SystemExit("model.zplan is a skeleton option and needs model.budget (it rides in the budget input)")
        zc = {**default_cfg(), **m["zplan"]}
    tc = None
    if m.get("tplan"):                                                          # lateness light plan (showerlm.tplan)
        from .tplan import default_cfg as t_default
        if is_detail(cfg) or not m.get("budget", False):
            raise SystemExit("model.tplan is a skeleton option and needs model.budget (it rides in the budget input)")
        tc = {**t_default(), **m["tplan"]}
    model = cls(vocab.sizes, vocab.names, n_t=vocab.n_t, n_qtot=N_QTOT, n_ntok=N_NTOK, n_cum=N_CUM, d=m["d"], n_layers=m["n_layers"],
                     n_heads=m["n_heads"], d_attr=m["d_attr"], n_attr_layers=m["n_attr_layers"], n_attr_heads=m.get("n_attr_heads", 4),
                     window=m["window"], w_ptr=vocab.w_ptr, dropout=m.get("dropout", 0.0), attr_order=m.get("attr_order"), causal_ptr=m.get("causal_ptr", False),
                     budget=bool(m.get("budget", False)), landmarks=bool(m.get("landmarks", False)), t_window=m.get("t_window", 0),
                     t_rule_table=(vocab.t_bins_of_len(np.arange(vocab.n_len)).tolist() if vocab.v3 else None), q_weight_pow=float(m.get("q_weight_pow", 0.0)),
                     t_rel=bool(m.get("t_rel", False)), n_gfeat=int((m.get("gplan") or {}).get("n_feat", 0)),
                     n_gbins=int((m.get("gplan") or {}).get("n_bins", 32)), n_zslice=len(zc["edges"]) if zc else 0,
                     n_zbins=int(zc["n_bins"]) if zc else 48, n_tslice=len(tc["edges"]) if tc else 0, n_tbins=int(tc["n_bins"]) if tc else 48, **extra)
    model.zcfg = zc
    model.tcfg = tc
    model.cont_top_extra = int(m.get("cont_top_extra", 0))                    # extra bins on top of the cont_window top
    if m.get("rules"):                                                          # 0.5: exact rules in the loss and in sampling
        from .rules import Rules, check_order
        model.rules = Rules(vocab, m["rules"], model, detail=is_detail(cfg))
        check_order(model)
        if not is_detail(cfg) and model.rules.on["time_order"] and not cfg.get("data", {}).get("whole_sequence", False):
            raise SystemExit("rules.time_order on a ShowerLM needs data.whole_sequence (the floor is counted from the first stroke)")
    if model.t_rel and not (model.rules is not None and (model.rules.on["time_order"] or (getattr(model, "pos_first", False) and model.rules.on["cone"]))):
        raise SystemExit("model.t_rel needs rules.time_order, or (0.6.2) pos_first with rules.cone (the t head predicts the offset above that floor)")
    if getattr(model, "group_order", "time") == "along" and model.rules is not None and model.rules.on["time_order"]:
        raise SystemExit("model.group_order along: the strokes of a group are not in time order, so rules.time_order must be off")
    if getattr(model, "occ", None) is not None:
        model._vocab_occ = vocab                                                    # the feature context decodes the skeleton
    model.landmark_lq = int(cfg.get("data", {}).get("landmark_lq", -1)) if bool(m.get("landmarks", False)) else -1
    return model.to(device)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)


def save_atomic(obj, path):
    """torch.save to a temporary file, then rename: a reader (or a resume after an eviction) never sees a half-written file."""
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def seq_len_at(schedule, step):
    cur = schedule[0][1]
    for s, L in schedule:
        if step >= s:
            cur = L
    return int(cur)


def lr_at(cfg, step):
    t = cfg["train"]
    warm, total, base = int(t["warmup"]), int(t["steps"]), float(t["lr"])
    if step < warm:
        return base * (step + 1) / warm
    p = min(1.0, (step - warm) / max(1, total - warm))
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def evaluate(model, loader, device, amp_dtype, trim=False, viol=None, trim_fn=trim_batch):
    """Mean validation losses.  A batch is (prefix, tok_in, tok_out, bud, lm) for ShowerLM and (skel, skel_len, prefix,
    tok_in, tok_out, bud, lm) for the 0.4 detail model; both models take it positionally.  viol (dict, 0.5): summed
    viol_<attr> counts of the validation targets (must stay 0)."""
    model.eval()
    tot = {}; n = 0
    with torch.no_grad():
        for batch in loader:
            batch = [x.to(device) for x in (trim_fn(batch) if trim else batch)]
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(*batch)
            for k, v in out.items():
                if k.startswith("loss"):
                    tot[k] = tot.get(k, 0.0) + float(v)
                elif k.startswith("viol_") and viol is not None:
                    viol[k[5:]] = viol.get(k[5:], 0) + int(v)
            n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="/data/user/qwang/showerlm/out")
    ap.add_argument("--name", default="train")
    ap.add_argument("--resume", default=None, help="run directory to continue (uses its config.yaml and last.pt)")
    ap.add_argument("--steps", type=int, default=None, help="override train.steps")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--max-events", type=int, default=0, help="use only the first N token files (tests)")
    ap.add_argument("--val-at-start", action="store_true", help="0.6.2: score the val items with the initial (EMA = --init) weights before step 1 "
                    "(the reference val loss of the starting checkpoint on the same items), logged as step 0")
    ap.add_argument("--init", default=None, help="0.6.1g fine-tune: start from the EMA weights of this checkpoint (.pt) or run folder "
                    "(its best.pt); parameters the checkpoint lacks (the G / zplan embeddings) keep their zero init; new optimizer, step 0")
    a = ap.parse_args()
    if a.resume:
        run = Path(a.resume)
        cfg = read_yaml(run / "config.yaml")
    else:
        if not a.config:
            raise SystemExit("--config is required (or --resume)")
        cfg = read_yaml(a.config)
        run = fresh_dir(a.out, a.name)
        shutil.copy(a.config, run / "config.yaml")
    if a.steps:
        cfg["train"]["steps"] = a.steps
    device = device_for(a.allow_cpu)
    torch.manual_seed(int(cfg["train"].get("seed", 0)))
    logf = open(run / "log.txt", "a")

    def say(*s):
        msg = " ".join(str(x) for x in s)
        log(msg); logf.write(time.strftime("%H:%M:%S ") + msg + "\n"); logf.flush()

    detail, group = is_detail(cfg), is_group(cfg)
    trim_fn = trim_batch
    if group:
        from .group import GroupPackDataset, GroupStore, collate_group, trim_group
        Store, trim_fn = GroupStore, trim_group
        if int(cfg["train"]["batch"]) != 1:
            raise SystemExit("model.type group: train.batch must be 1 (one pack of groups per item; use train.accum)")
    elif detail:
        from .detail import DetailChunkDataset, DetailStore, collate_detail
        Store = DetailStore
    else:
        Store = TokenStore
    excl = cfg["data"].get("exclude_eids") or None
    store = Store(cfg["data"]["tokens_train"], a.max_events, exclude_eids=excl)
    store_val = Store(cfg["data"]["tokens_val"], a.max_events, exclude_eids=excl)
    vocab = store.vocab
    if excl:
        say(f"data.exclude_eids {list(excl)}: left out {store.excluded} (train) {store_val.excluded} (val)")
    if not (run / "vocab.json").exists():
        write_json(run / "vocab.json", vocab.to_dict())
    say(f"run {run}: {len(store)} train / {len(store_val)} val token files; device {device}")
    model = build_model(cfg, vocab, device)
    say(f"model: {count_params(model) / 1e6:.1f} M parameters; config {json.dumps(cfg['model'])}")
    if a.init and not (a.resume and (run / "last.pt").exists()):
        ip = Path(a.init)
        ip = ip / "best.pt" if ip.is_dir() else ip
        ck0 = torch.load(ip, map_location=device, weights_only=False)
        res = model.load_state_dict(ck0["ema"], strict=False)
        bad = [k for k in res.missing_keys if not k.startswith(("g_emb.", "zp_emb.", "zr_emb.", "tp_emb.", "tr_emb.", "occ."))] + list(res.unexpected_keys)
        if bad:
            raise SystemExit(f"--init {ip}: checkpoint does not match the model beyond the G / zplan / tplan embeddings / occupancy heads: {bad[:8]}")
        say(f"init from {ip} (EMA weights, step {ck0.get('step')}); new parameters kept at their init: {res.missing_keys}")
        write_json(run / "init.json", dict(init=str(ip), init_step=ck0.get("step"), new_params=res.missing_keys))
    t = cfg["train"]
    # the zero-initialised occupancy heads (occ.*) get lr x occ_feat.lr_mult (Adam moves a weight by about lr per
    # step, so a 20000-step fine-tune at 1.5e-4 could move them by ~1.6 at most; the rest of the model keeps the base lr)
    occ_p = [p for n_, p in model.named_parameters() if n_.startswith("occ.")]
    groups = [dict(params=[p for n_, p in model.named_parameters() if not n_.startswith("occ.")], mult=1.0)]
    if occ_p:
        groups.append(dict(params=occ_p, mult=float(model.occ_cfg.get("lr_mult", 1.0))))
        say(f"occupancy heads: {sum(p.numel() for p in occ_p)} parameters at lr x {groups[-1]['mult']:g}")
    opt = torch.optim.AdamW(groups, lr=float(t["lr"]), betas=(0.9, 0.95), weight_decay=float(t.get("weight_decay", 0.1)))
    ema = EMA(model, float(t.get("ema", 0.999)))
    amp_dtype, use_scaler = amp_setting(cfg, device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    step, best = 0, float("inf")
    best_part = dict(plan=float("inf"), stroke=float("inf"))                   # 0.5.1, detail runs only
    if a.resume and (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); ema.shadow.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        step, best = int(ck["step"]), float(ck.get("best", float("inf")))
        best_part.update(ck.get("best_part", {}))
        say(f"resumed at step {step}")
    schedule = [[int(s), int(L)] for s, L in t["seq_len_schedule"]]
    tsp = bool(cfg["data"].get("tstart_prev", False))
    bp, lml = bool(cfg["data"].get("birth_ptr", True)), int(cfg["data"].get("landmark_lq", -1))
    trim = bool(t.get("trim_pad", False))                                       # cut the all-PAD tail of a batch (0.4)
    k_max = int(cfg["model"].get("k_max", 1024))

    bpl = bool(cfg["data"].get("bud_plan", False))                              # 0.4: budgets / counters relative to the decoded plan
    gtab = {}
    if getattr(model, "n_gfeat", 0) and not detail:                              # G bins of every shower (showerlm.gfeat tables)
        from .gfeat import GTable
        gdir = cfg["data"].get("gfeat")
        if not gdir:
            raise SystemExit("model.gplan needs data.gfeat (the folder written by python -m showerlm.gfeat)")
        gtab = {id(store): GTable(gdir, "train"), id(store_val): GTable(gdir, "val")}
        if len(gtab[id(store)].meta["names"]) != model.n_gfeat:
            raise SystemExit(f"model.gplan.n_feat {model.n_gfeat} != {len(gtab[id(store)].meta['names'])} variables in {gdir}")
        say(f"gplan: {model.n_gfeat} global variables {gtab[id(store)].meta['names']} from {gdir}")
    if model.zcfg is not None:
        say(f"zplan: {model.n_zslice} depth slices from {model.zcfg['edges']} m, {model.n_zbins} bins up to {model.zcfg['q_max']:.3g} photons")
    if getattr(model, "tcfg", None) is not None:
        say(f"tplan: {model.n_tslice} lateness slices from {model.tcfg['edges']} ns, {model.n_tbins} bins up to {model.tcfg['q_max']:.3g} photons")

    def make_ds(st, L, n_items, seed, fixed=False):
        if group:
            return GroupPackDataset(st, L, n_items, seed=seed, fixed=fixed, mode=cfg["model"].get("group_mode", "join"), k_max=k_max,
                                    order=cfg["model"].get("group_order", "time"), occ=getattr(model, "occ_cfg", None))
        if detail:
            return DetailChunkDataset(st, L, n_items, seed=seed, fixed=fixed, tstart_prev=tsp, k_max=k_max, bud_plan=bpl)
        return ChunkDataset(st, L, n_items, seed=seed, fixed=fixed, tstart_prev=tsp, birth_ptr=bp, landmark_lq=lml, bud_plan=bpl,
                            whole_sequence=bool(cfg["data"].get("whole_sequence", False)), gtable=gtab.get(id(st)), zplan=model.zcfg,
                            tplan=getattr(model, "tcfg", None))
    collate = collate_group if group else (collate_detail if detail else None)
    if model.rules is not None and model.rules.on["time_order"] and not detail:
        n_long = max(int(store.n_tok.max()), int(store_val.n_tok.max()))
        if n_long > min(L for _, L in schedule):
            raise SystemExit(f"rules.time_order: the longest train / val sequence ({n_long} tokens) must fit in every chunk of "
                             f"seq_len_schedule (whole_sequence), or chunks would start mid-sequence with fake chain heads")
    val_ds = make_ds(store_val, seq_len_at(schedule, 10 ** 9), int(t.get("val_items", 64)), seed=1, fixed=True)
    val_loader = DataLoader(val_ds, batch_size=int(t["batch"]), num_workers=0, collate_fn=collate)
    cur_len, loader, it = None, None, None
    if a.val_at_start and step == 0:
        val_viol = {}
        val = evaluate(ema.shadow, val_loader, device, amp_dtype, trim, viol=val_viol, trim_fn=trim_fn)
        with open(run / "metrics.jsonl", "a") as f:
            f.write(json.dumps(dict(step=0, val={k: round(v, 4) for k, v in val.items()}, val_viol=val_viol, note="initial weights")) + "\n")
        say(f"step 0 VAL (initial weights) loss {val['loss']:.3f} " + " ".join(f"{k[5:]} {v:.2f}" for k, v in val.items() if k != "loss"))
    accum = int(t.get("accum", 1))
    t0 = time.time(); tok_count = 0
    viol_tot = {}                                                               # 0.5: training targets that broke a rule (must stay 0)
    model.train()
    while step < int(t["steps"]):
        L = seq_len_at(schedule, step)
        if L != cur_len:
            cur_len = L
            ds = make_ds(store, L, 10 ** 8, seed=step)
            loader = DataLoader(ds, batch_size=int(t["batch"]), num_workers=int(t.get("workers", 4)), drop_last=True, persistent_workers=int(t.get("workers", 4)) > 0,
                                collate_fn=collate)
            it = iter(loader)
            say(f"step {step}: window / chunk length {L}")
        for g in opt.param_groups:
            g["lr"] = lr_at(cfg, step) * g.get("mult", 1.0)
        opt.zero_grad(set_to_none=True)
        agg = {}
        for _ in range(accum):
            batch = next(it)
            batch = [x.to(device, non_blocking=True) for x in (trim_fn(batch) if trim else batch)]
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(*batch)
            scaler.scale(out["loss"] / accum).backward()
            for k, v in out.items():
                if k.startswith("loss"):
                    agg[k] = agg.get(k, 0.0) + float(v.detach()) / accum
                elif k.startswith("viol_"):
                    viol_tot[k] = viol_tot.get(k, 0) + int(v)
            tok_count += out["n_tokens"]
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(t.get("grad_clip", 1.0)))
        scaler.step(opt); scaler.update()
        ema.update(model)
        step += 1
        if step % int(t.get("log_every", 20)) == 0 or step == 1:
            dt = time.time() - t0
            rec = dict(step=step, lr=lr_at(cfg, step), seq_len=L, grad_norm=float(gn), tok_per_s=tok_count / max(dt, 1e-9), **{k: round(v, 4) for k, v in agg.items()}, **viol_tot)
            with open(run / "metrics.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            say(f"step {step} loss {agg['loss']:.3f} (kind {agg.get('loss_kind', 0):.2f} ptr {agg.get('loss_ptr', 0):.2f} t {agg.get('loss_t', 0):.2f} th {agg.get('loss_th', 0):.2f} "
                f"lq {agg.get('loss_lq', 0):.2f} len {agg.get('loss_len', 0):.2f}) lr {rec['lr']:.2e} {rec['tok_per_s']:.0f} tok/s"
                + (f" RULE VIOLATIONS so far {viol_tot}" if any(viol_tot.values()) else ""))
            t0 = time.time(); tok_count = 0
        if step % int(t.get("val_every", 500)) == 0 or step == int(t["steps"]):
            val_viol = {}
            val = evaluate(ema.shadow, val_loader, device, amp_dtype, trim, viol=val_viol, trim_fn=trim_fn)
            with open(run / "metrics.jsonl", "a") as f:
                f.write(json.dumps(dict(step=step, val={k: round(v, 4) for k, v in val.items()}, val_viol=val_viol)) + "\n")
            say(f"step {step} VAL loss {val['loss']:.3f} " + " ".join(f"{k[5:]} {v:.2f}" for k, v in val.items() if k != "loss")
                + (f" RULE VIOLATIONS in val {val_viol}" if any(val_viol.values()) else ""))
            ck = dict(model=model.state_dict(), ema=ema.shadow.state_dict(), opt=opt.state_dict(), step=step, best=best, config=cfg, vocab=vocab.to_dict())
            if detail and "loss_qtot" in val:
                parts = dict(plan=val["loss_qtot"] + val["loss_ntok"])
                parts["stroke"] = val["loss"] - parts["plan"]
                for part, v in parts.items():
                    if v < best_part[part]:
                        best_part[part] = v; ck["best_part"] = dict(best_part)
                        save_atomic(ck, run / f"best_{part}.pt"); say(f"  best {part} terms so far ({v:.3f}) -> best_{part}.pt")
                ck["best_part"] = dict(best_part)
            if val["loss"] < best:
                best = val["loss"]; ck["best"] = best
                save_atomic(ck, run / "best.pt"); say(f"  best so far -> best.pt")
            save_atomic(ck, run / "last.pt")
        elif step % int(t.get("ckpt_every", 1000)) == 0:
            save_atomic(dict(model=model.state_dict(), ema=ema.shadow.state_dict(), opt=opt.state_dict(), step=step, best=best, best_part=dict(best_part), config=cfg,
                             vocab=vocab.to_dict()), run / "last.pt")
    say("training finished")
    write_json(run / "run.json", dict(steps=step, best_val_loss=best, **({"best_val_part": best_part} if detail else {}), params=count_params(model)))


if __name__ == "__main__":
    main()
