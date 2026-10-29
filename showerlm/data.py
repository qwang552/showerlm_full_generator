"""Token files -> training chunks (prefix + window of strokes + targets + per-stroke budget + landmark flags)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .model import N_PREFIX
from .tokens import KIND_CONT, KIND_EOS, KIND_PAD, KIND_STROKE, Vocab

N_QTOT, N_NTOK, N_CUM = 64, 64, 33
QTOT_LOG = (6.0, 9.0)          # log10 photons per shower (12 ns box: ~7e7)
NTOK_LOG = (2.0, 5.0)          # log10 strokes per shower (~1e4)
N_BUD = 40                     # bins of the per-stroke budget inputs (0.3.0): strokes still to come, photons still to come
BUD_N_MAX, BUD_Q_MAX = 2.0e4, 1.0e9


def enc_lin(x, lo, hi, n):
    return int(np.clip(np.floor((x - lo) / (hi - lo) * n), 0, n - 1))


def dec_lin(b, lo, hi, n):
    return lo + (np.asarray(b, dtype=np.float64) + 0.5) * (hi - lo) / n


def enc_qtot(q):
    return enc_lin(np.log10(max(q, 1.0)), *QTOT_LOG, N_QTOT)


def dec_qtot(b):
    return 10.0 ** dec_lin(b, *QTOT_LOG, N_QTOT)


def enc_ntok(n):
    return enc_lin(np.log10(max(n, 1.0)), *NTOK_LOG, N_NTOK)


def dec_ntok(b):
    return 10.0 ** dec_lin(b, *NTOK_LOG, N_NTOK)


def enc_cum(f):
    return int(np.clip(np.floor(f * (N_CUM - 1)), 0, N_CUM - 1))


def enc_bud(x, xmax):
    """Remaining strokes / photons -> log1p bin (vectorised); 0 = nothing left."""
    x = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    return np.clip(np.floor(np.log1p(x) / np.log1p(xmax) * (N_BUD - 1) + 1e-9), 0, N_BUD - 1).astype(np.int64)


def budget_bins(rem_n, rem_q):
    """(..., 2) int64: [strokes still to come, photons still to come] as bins."""
    return np.stack([enc_bud(rem_n, BUD_N_MAX), enc_bud(rem_q, BUD_Q_MAX)], -1)


def event_counters(tok, q_cum, q_total, vocab, bud_plan=False):
    """Running counters of one event as the model sees them, for every position k = 0 .. T:
    ncum[k], qcum[k] = fraction of the strokes / photons before stroke k (prefix NCUM / QCUM of a chunk starting at k);
    rem_n[k], rem_q[k] = strokes / photons still to come after the first k strokes (budget input of stroke k - 1).
    bud_plan False (0.3.0): exact truth counts.  bud_plan True (0.4): the way the sampler counts them, i.e. relative to
    the PLAN it decoded from the prefix bins (dec_ntok(enc_ntok(T)), dec_qtot(enc_qtot(Q))) and with the photons of the
    strokes read from their lq bins; so the budget need not be exactly 0 at the last stroke, as in generation."""
    T = len(tok)
    k = np.arange(T + 1, dtype=np.float64)
    if not bud_plan:
        qc = np.append(np.asarray(q_cum, dtype=np.float64), 1.0)
        return k / max(T, 1), qc, T - k, q_total * (1.0 - qc)
    n_plan, q_plan = float(dec_ntok(enc_ntok(T))), float(dec_qtot(enc_qtot(q_total)))
    ql = vocab.dec_log(np.asarray(tok)[:, vocab.index["lq"]], vocab.n_q, vocab.q_log_max) if T else np.zeros(0)
    cq = np.concatenate([[0.0], np.cumsum(ql)])
    return np.minimum(k / max(n_plan, 1.0), 0.999), np.minimum(cq / max(q_plan, 1.0), 0.999), n_plan - k, q_plan - cq


class TokenStore:
    """All tokens_*.npz of a directory, with a cached index (eid, n_tok, q_total)."""

    def __init__(self, root, max_events=0, exclude_eids=None):
        """exclude_eids (0.6, config data.exclude_eids): events left out of training, e.g. the ones tools/audit_rules.py
        found breaking a rule (0.5.2 used symlink folders without them)."""
        self.root = Path(root)
        self.vocab = Vocab.from_dict(json.load(open(self.root / "vocab.json")))
        files = sorted((self.root / "events").glob("tokens_*.npz"))
        if not files:
            raise FileNotFoundError(f"no tokens_*.npz under {self.root}/events")
        idx = self.root / "index.json"
        index = json.load(open(idx)) if idx.exists() else {}
        changed = False
        for f in files:
            if f.name not in index:
                with np.load(f) as d:
                    index[f.name] = dict(n_tok=int(d["tok"].shape[0]), q_total=float(d["q_total"]), eid=int(d["eid"]))
                changed = True
        if changed:
            try:                                                                    # one step: other processes may read it
                tmp = idx.with_name(f"index.json.{os.getpid()}.tmp")
                tmp.write_text(json.dumps(index)); os.replace(tmp, idx)
            except OSError:
                pass
        self.files = files[:max_events] if max_events else files
        self.n_tok = np.array([index[f.name]["n_tok"] for f in self.files])
        self.q_total = np.array([index[f.name]["q_total"] for f in self.files])
        self.eids = np.array([index[f.name]["eid"] for f in self.files])
        self.excluded = []
        if exclude_eids:
            drop = np.isin(self.eids, np.asarray(list(exclude_eids), dtype=np.int64))
            self.excluded = [int(e) for e in self.eids[drop]]
            keep = ~drop
            self.files = [f for f, k in zip(self.files, keep) if k]
            self.n_tok, self.q_total, self.eids = self.n_tok[keep], self.q_total[keep], self.eids[keep]

    def __len__(self):
        return len(self.files)

    def load(self, i):
        with np.load(self.files[i]) as d:
            return d["tok"].astype(np.int64), d["q_cum"].astype(np.float64)


class ChunkDataset(Dataset):
    """Random chunks: item = (prefix (6,), tok_in (seq_len, A), tok_out (seq_len + 1, A), bud (seq_len, 2), lm (seq_len,));
    PAD where shorter.

    tstart_prev: prefix TSTART = t of the last stroke before the chunk (0 at the shower start), as the sampler knows it.
    birth_ptr=False (0.3.0): only CONT pieces keep their pointer (ptr 1, s = end, small offset); every other stroke is
      written with ptr 0 and its absolute head voxel, i.e. the model places new strokes by absolute position and the
      pointer is used only for continuations (whose parent is always the previous token).
    bud: per input stroke, the strokes and photons still to come AFTER it (the sampler keeps the same two counters);
      bud_plan (0.4): counted relative to the decoded plan, as the sampler counts (event_counters).
    whole_sequence (0.4 skeletons): a sequence that fits in seq_len is always taken whole from its first stroke.
    lm: landmark flag of an input stroke (lq bin >= landmark_lq): the model keeps landmarks visible outside the window.
    zplan (0.6.1z, showerlm.zplan config dict): the planned skeleton light per depth slice is appended to the prefix (after
      G) and the light still to come per slice after each input stroke to its budget (bud (seq_len, 2 + K)); random chunks
      (__getitem__) scale the plan by exp(N(0, zplan.jitter)) per slice, item(e, k) without rng uses the exact plan."""

    def __init__(self, store: TokenStore, seq_len, n_items, seed=0, fixed=False, tstart_prev=False, birth_ptr=True, landmark_lq=-1, bud_plan=False,
                 whole_sequence=False, gtable=None, zplan=None, tplan=None):
        self.store, self.seq_len, self.n_items, self.seed, self.fixed = store, int(seq_len), int(n_items), int(seed), fixed
        self.tstart_prev, self.birth_ptr, self.landmark_lq, self.bud_plan = bool(tstart_prev), bool(birth_ptr), int(landmark_lq), bool(bud_plan)
        self.whole_sequence = bool(whole_sequence)
        self.gtable = gtable                                                         # 0.6.1g: showerlm.gfeat.GTable, G bins appended to the prefix
        if gtable is not None:
            missing = [int(e) for e in store.eids if int(e) not in gtable.by_eid]
            if missing:
                raise SystemExit(f"G table {gtable.gdir} has no row for {len(missing)} events of {store.root} (first {missing[:5]})")
        self.zplan = zplan                                                           # 0.6.1z: showerlm.zplan config (edges, ranges, n_bins, q_max)
        self.tplan = tplan                                                           # 0.6.3: showerlm.tplan config, columns after the zplan ones
        self.A = len(store.vocab.names)
        self.it = store.vocab.index["t"]

    def __len__(self):
        return self.n_items

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i] if self.fixed else None)
        e = int(rng.integers(len(self.store)))
        T = int(self.store.n_tok[e])
        k = int(rng.integers(0, max(T - self.seq_len // 2, 1)))
        if self.whole_sequence and T <= self.seq_len:
            k = 0                                                                   # 0.4 skeletons: always the whole sequence, as generated
        return self.item(e, k, rng)

    def item(self, e, k, rng=None):
        """The chunk of event e that starts at stroke k (0.6: tools/audit_rules.py --chunk walks every chunk of an event)."""
        tok, q_cum = self.store.load(e)
        T = len(tok)
        q_total = float(self.store.q_total[e])
        n = min(self.seq_len, T - k)
        # TSTART = time of the last stroke BEFORE the chunk (0 at the shower start), as the sampler can know it
        t_before = int(tok[k - 1, self.it]) if (k > 0 and self.tstart_prev) else (0 if self.tstart_prev else int(tok[k, self.it]))
        ncum, qcum, rem_n, rem_q = event_counters(tok, q_cum, q_total, self.store.vocab, self.bud_plan)
        prefix = np.array([0, enc_qtot(q_total), enc_ntok(T), enc_cum(qcum[k]), enc_cum(ncum[k]), t_before], dtype=np.int64)
        if self.gtable is not None:
            prefix = np.concatenate([prefix, self.gtable.bins_of(self.store.eids[e]).astype(np.int64)])
        zrem = None
        if self.zplan is not None:
            from .zplan import remaining_bins
            jit = float(self.zplan.get("jitter", 0.0))
            scale = np.exp(rng.normal(0.0, jit, len(self.zplan["edges"]))) if (jit > 0 and rng is not None) else None
            zrem, _ = remaining_bins(tok, self.store.vocab, self.zplan, scale)       # (T + 1, K): row 0 = plan, row k + 1 = after token k
            prefix = np.concatenate([prefix, zrem[0]])
        trem = None
        if self.tplan is not None:                                                  # 0.6.3: the same for the lateness slices
            from .tplan import remaining_bins as t_remaining
            jit = float(self.tplan.get("jitter", 0.0))
            scale = np.exp(rng.normal(0.0, jit, len(self.tplan["edges"]))) if (jit > 0 and rng is not None) else None
            trem, _ = t_remaining(tok, self.store.vocab, self.tplan, scale)
            prefix = np.concatenate([prefix, trem[0]])
        tok_in = np.full((self.seq_len, self.A), KIND_PAD, np.int64)
        tok_out = np.full((self.seq_len + 1, self.A), KIND_PAD, np.int64)
        m = min(n + 1, T - k)
        chunk = tok[k:k + m].copy()
        ix = self.store.vocab.index
        ip, ina = ix["ptr"], [ix[a] for a in ("s", "dx", "dy", "dz")]
        # a parent before the chunk start is invisible: such tokens fall back to absolute coordinates (ptr = 0); a CONT
        # whose previous piece is outside the chunk becomes a plain STROKE
        lost = chunk[:, ip] > np.arange(m)
        if not self.birth_ptr:
            lost |= chunk[:, ix["kind"]] != KIND_CONT
        chunk[lost, ip] = 0
        for a in ina:
            chunk[lost, a] = 0
        chunk[lost & (chunk[:, ix["kind"]] == KIND_CONT), ix["kind"]] = KIND_STROKE
        tok_in[:n] = chunk[:n]
        tok_out[:m] = chunk
        if k + n >= T:
            tok_out[n, 0] = KIND_EOS
        # budget after each input stroke: strokes and photons still to come
        g = k + np.arange(n)
        nz = 0 if zrem is None else zrem.shape[1]
        nt = 0 if trem is None else trem.shape[1]
        bud = np.zeros((self.seq_len, 2 + nz + nt), np.int64)
        bud[:n, :2] = budget_bins(rem_n[g + 1], rem_q[g + 1])
        if zrem is not None:
            bud[:n, 2: 2 + nz] = zrem[g + 1]
        if trem is not None:
            bud[:n, 2 + nz:] = trem[g + 1]
        lm = np.zeros(self.seq_len, bool)
        if self.landmark_lq >= 0:
            lm[:n] = chunk[:n, ix["lq"]] >= self.landmark_lq
        return torch.from_numpy(prefix), torch.from_numpy(tok_in), torch.from_numpy(tok_out), torch.from_numpy(bud), torch.from_numpy(lm)


def trim_batch(batch):
    """Cut the all-PAD tail of a collated batch (PAD positions come after the strokes of every sample and carry no loss,
    so the result is the same, only faster for short sequences such as the 0.4 skeletons).  Works on ShowerLM batches
    (prefix, tok_in, tok_out, bud, lm) and on detail batches (skel, skel_len, prefix, tok_in, tok_out, bud, lm)."""
    batch = list(batch)
    o = len(batch) - 5                                                              # index of prefix
    tok_in = batch[o + 1]
    n = max(1, int((tok_in[..., 0] != KIND_PAD).sum(1).max()))
    batch[o + 1] = tok_in[:, :n]
    batch[o + 2] = batch[o + 2][:, : n + 1]
    batch[o + 3] = batch[o + 3][:, :n]
    batch[o + 4] = batch[o + 4][:, :n]
    return batch
