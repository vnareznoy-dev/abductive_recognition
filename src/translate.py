#!/usr/bin/env python3
"""Translate per-frame features -> Prolog facts, and run the abductive theory.

This is a pure translation layer. All
reasoning lives in intent_abduce.pl (Prolog is the brain). For each frame we
retract the previous frame's facts, assert the current observation, and query
classify/2 — so the classifier is MEMORYLESS: every frame is explained on its
own (the only "memory" is the OSC_WINDOW used to compute the osc flag in
features.py, which is the minimal window the physics of oscillation requires —
disclosed honestly, see README).
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

from features import FEATURE_NAMES

_PL_PATH = os.path.join(os.path.dirname(__file__), "intent_abduce.pl")
_IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


def frame_to_facts(feat_row: np.ndarray) -> List[str]:
    """One [F] feature row -> list of Prolog fact strings."""
    v = float(feat_row[_IDX["v"]])
    psi_dot = float(feat_row[_IDX["psi_dot"]])
    m = float(feat_row[_IDX["m"]])
    r = float(feat_row[_IDX["r_goal"]])
    rdot = float(feat_row[_IDX["rdot"]])
    osc = "true" if feat_row[_IDX["osc"]] > 0.5 else "false"
    return [
        f"f_speed({v:.4f})",
        f"f_yaw_rate({psi_dot:.4f})",
        f"f_off_axis({m:.4f})",
        f"f_rgoal({r:.4f})",
        f"f_rdot({rdot:.4f})",
        f"f_osc({osc})",
    ]


class AbductiveClassifier:
    """Memoryless white-box intent classifier backed by intent_abduce.pl."""

    def __init__(self, pl_path: Optional[str] = None):
        from pyswip import Prolog
        self.prolog = Prolog()
        self.prolog.consult(pl_path or _PL_PATH)
        # SWI normalises atom comparison; intent atoms come back as bytes/str.

    def _clear(self):
        for p in ("f_speed(_)", "f_yaw_rate(_)", "f_off_axis(_)",
                  "f_rgoal(_)", "f_rdot(_)", "f_osc(_)"):
            list(self.prolog.query(f"retractall({p})"))

    def classify_frame(self, feat_row: np.ndarray) -> Tuple[str, str]:
        """Return (intent_name, justification) for one frame."""
        self._clear()
        for fact in frame_to_facts(feat_row):
            self.prolog.assertz(fact)
        res = list(self.prolog.query("classify(I, W)"))
        if not res:
            return "hold", "no_solution"
        intent = _atom(res[0]["I"])
        why = _term_str(res[0]["W"])
        return intent, why

    def classify_trajectory(self, feats: np.ndarray) -> List[str]:
        """[T, F] -> per-frame intent names (the latency curve uses this)."""
        return [self.classify_frame(feats[t])[0] for t in range(feats.shape[0])]


def _atom(x) -> str:
    if isinstance(x, bytes):
        return x.decode()
    return str(x)


def _term_str(x) -> str:
    """Best-effort readable justification from a pyswip term."""
    if isinstance(x, bytes):
        return x.decode()
    return str(x)
