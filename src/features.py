#!/usr/bin/env python3
"""Per-frame geometric features shared by BOTH classifiers (fairness).

Given raw kinematics we derive flight-geometry features. The SAME [T, F] feature array feeds the inductive
LSTM and the abductive Prolog theory, so any performance gap is about the
*method*, not the inputs.

Sensor model: a realistic tracker observes POSITIONS (noisy), not velocities.
We therefore estimate velocity from lightly-smoothed positions (moving average
+ central finite difference) — exactly what a state estimator does — instead of
trusting a per-frame velocity. This keeps yaw rate well-defined and stops
observation noise from manufacturing phantom oscillation. Both methods see the
identical smoothed features.

Features (per frame)
--------------------
  v       speed                                          (m/s)
  psi_dot yaw rate from smoothed heading                 (rad/s) [frame 0 = 0]
  m       off-axis miss: perpendicular distance of the GOAL from the velocity
          ray.  m ~ 0 => heading straight at the goal.            (m)
  r_goal  range evader -> goal                            (m)
  rdot    closing rate on the goal, d(r_goal)/dt          (m/s) [<0 = closing]
  osc     serpentine flag: a sign change among SIGNIFICANT yaw excursions
          inside a short window, while moving fast enough for heading to be
          meaningful.                                              (0.0 / 1.0)

NOTE on the bug in the original plan sketch: yaw rate is NOT
``arctan2(V[1], V) / dt`` — that passes a vector as the 2nd arg and cannot yield
a rate from one frame. A rate needs two frames: the wrapped heading difference
over dt. Fixed here, and computed from a smoothed heading.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = ["v", "psi_dot", "m", "r_goal", "rdot", "osc"]
DT = 0.1

POS_SMOOTH = 5        # moving-average window on positions (estimator)
YAW_SMOOTH = 3        # moving-average window on yaw rate (feature only)
OSC_WINDOW = 13       # frames the oscillation flag integrates (~1.3 s ~ 1 period)
TREND_WINDOW = 13     # window whose mean heading is the local "intended course"
RES_AMP = 0.18        # rad — per-sample heading residual that counts as a swing
RES_SPAN = 0.50       # rad — peak-to-peak residual in the window to call it weave
MIN_OSC_SPEED = 3.0   # m/s — below this, heading is too noisy to call serpentine


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _movavg(x, w):
    """Causal-ish centred moving average along axis 0, edge-padded."""
    if w <= 1:
        return x
    pad = w // 2
    xp = np.pad(x, [(pad, pad)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    ker = np.ones(w) / w
    if x.ndim == 1:
        return np.convolve(xp, ker, mode="valid")
    out = np.empty_like(x)
    for c in range(x.shape[1]):
        out[:, c] = np.convolve(xp[:, c], ker, mode="valid")
    return out


def trajectory_features(ev_pos, ev_vel, pu_pos, goal):
    """Return [T, F] float features for ONE trajectory.

    Velocity/heading are ESTIMATED from smoothed positions (ev_vel is accepted
    for signature parity but intentionally not trusted — a tracker sees pos).
    pu_pos is accepted for future features.
    """
    T = ev_pos.shape[0]
    pos = _movavg(ev_pos, POS_SMOOTH)

    vel = np.zeros((T, 2))
    vel[1:-1] = (pos[2:] - pos[:-2]) / (2 * DT)     # central difference
    vel[0] = (pos[1] - pos[0]) / DT
    vel[-1] = (pos[-1] - pos[-2]) / DT

    v = np.linalg.norm(vel, axis=1)
    psi = np.arctan2(vel[:, 1], vel[:, 0])

    psi_dot = np.zeros(T)
    psi_dot[1:] = _wrap(psi[1:] - psi[:-1]) / DT
    psi_dot = _movavg(psi_dot, YAW_SMOOTH)

    dvec = goal[None, :] - pos                       # goal seen from evader
    r_goal = np.linalg.norm(dvec, axis=1)
    u = vel / (v[:, None] + 1e-9)
    along = np.sum(dvec * u, axis=1)
    perp = dvec - along[:, None] * u
    m = np.linalg.norm(perp, axis=1)

    rdot = np.zeros(T)
    rdot[1:] = (r_goal[1:] - r_goal[:-1]) / DT

    # Oscillation detected on HEADING (1st derivative of position), not yaw rate
    # (2nd derivative, noise-dominated). A serpentine weave makes the heading
    # wobble periodically AROUND its intended course; a steady flank turn is
    # absorbed by the local-mean trend and leaves a small residual; a straight
    # rush leaves only noise. osc fires when the detrended heading swings with
    # real amplitude AND changes sign inside the window.
    psi_u = np.unwrap(psi)
    trend = _movavg(psi_u, TREND_WINDOW)
    resid = psi_u - trend
    osc = np.zeros(T)
    for t in range(T):
        if v[t] < MIN_OSC_SPEED:
            continue
        lo = max(0, t - OSC_WINDOW + 1)
        w = resid[lo:t + 1]
        if w.size < 3 or (w.max() - w.min()) < RES_SPAN:
            continue
        sig = np.sign(w[np.abs(w) >= RES_AMP])
        if sig.size >= 2 and np.any(sig[1:] * sig[:-1] < 0):
            osc[t] = 1.0

    return np.stack([v, psi_dot, m, r_goal, rdot, osc], axis=1)


def dataset_features(ds):
    """Compute [N, T, F] features for a whole dataset dict from generate_dataset."""
    N = ds["ev_pos"].shape[0]
    feats = [trajectory_features(ds["ev_pos"][i], ds["ev_vel"][i],
                                 ds["pu_pos"][i], ds["goal"][i]) for i in range(N)]
    return np.asarray(feats, np.float64)
