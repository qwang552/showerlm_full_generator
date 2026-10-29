"""Optional timing of the sampler's parts (SHOWERLM_PROF=1).  Each region is timed between two CUDA synchronisations.
"""
from __future__ import annotations

import os
import time

import torch

ON = os.environ.get("SHOWERLM_PROF", "0") == "1"
_acc: dict = {}
_cnt: dict = {}


def _sync():
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


class region:
    """with region("det.trunk"): ...  (no-op unless SHOWERLM_PROF=1)"""
    __slots__ = ("name", "t0")

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        if ON:
            _sync()
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if ON:
            _sync()
            dt = time.perf_counter() - self.t0
            _acc[self.name] = _acc.get(self.name, 0.0) + dt
            _cnt[self.name] = _cnt.get(self.name, 0) + 1
        return False


def snapshot(n_showers=None):
    out = {k: dict(s=round(v, 4), calls=_cnt[k]) for k, v in sorted(_acc.items())}
    if n_showers:
        for v in out.values():
            v["s_per_shower"] = round(v["s"] / n_showers, 4)
    return out
