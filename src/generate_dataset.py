#!/usr/bin/env python3
"""Controlled, labelled pursuit-evasion trajectory generator.

This is the ground-truth source for the abductive-vs-inductive intent
classification experiment (see README.md). We script K evader *intents*
(behavioural archetypes) as continuous-2D unicycle policies. Because WE sample the intent, the per-trajectory label is
ground truth — exactly what an accuracy/latency comparison needs and what a
raw replay of someone else's logs cannot give us.

Determinism: every roll-out is seeded, so the dataset is exactly
reproducible (paper setting — train: seed=1, test: seed=99, switch: seed=7;
observation noise 0.3 m).

Intents
-------
  rush  : steer straight at the goal — closing, low off-axis miss, steady.
  flank : steer at a constant angular offset to the goal bearing — closing,
          large off-axis miss, one-sided curve, no oscillation.
  weave : goalward bias + sinusoidal heading dither — serpentine, oscillates.
  hold  : slow loiter, not committing — low speed, not closing on the goal.

Each frame logs raw kinematics only (positions + velocities + goal); all
geometric features are derived downstream in features.py so BOTH classifiers
see identical inputs (fairness).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

INTENTS = ["rush", "flank", "weave", "hold"]
INTENT_ID = {name: i for i, name in enumerate(INTENTS)}

DT = 0.1            # s per step
WORLD = 100.0       # square arena edge (m)
GOAL = np.array([WORLD / 2, WORLD / 2])   # the evader's destination / defended point


def _wrap(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class IntentParams:
    speed: float
    turn_gain: float
    turn_max: float       # rad/s clamp
    flank_off: float      # constant heading offset for flank (rad)
    weave_amp: float      # rad
    weave_hz: float       # Hz


PARAMS = {
    "rush":  IntentParams(8.0, 2.0, 1.5, 0.0, 0.0, 0.0),
    "flank": IntentParams(8.0, 2.0, 1.5, 0.6, 0.0, 0.0),
    "weave": IntentParams(8.0, 3.0, 2.5, 0.0, 1.0, 0.8),
    "hold":  IntentParams(2.0, 1.0, 1.0, 0.0, 0.0, 0.0),
}


def _desired_heading(intent: str, ev: np.ndarray, t_step: int) -> float:
    """Desired heading for the given intent at this frame."""
    p = PARAMS[intent]
    bearing = np.arctan2(GOAL[1] - ev[1], GOAL[0] - ev[0])
    if intent == "rush":
        return bearing
    if intent == "flank":
        return bearing + p.flank_off
    if intent == "weave":
        return bearing + p.weave_amp * np.sin(2 * np.pi * p.weave_hz * t_step * DT)
    if intent == "hold":
        # Slow loiter: heading rotates gently, ignoring the goal -> not closing.
        return _wrap(np.pi / 2 + 0.4 * t_step * DT)
    raise ValueError(intent)


def _roll_out(rng: np.random.RandomState, label_seq, steps: int, obs_noise: float):
    """Roll out one unicycle evader + a pure-pursuit pursuer.

    label_seq: list of length `steps` giving the active intent per frame
    (constant for normal trajectories, switched mid-way for the switch set).
    Returns dict of float32 arrays.
    """
    # Evader starts on an annulus around the goal; pursuer somewhere in arena.
    ang = rng.uniform(0, 2 * np.pi)
    r0 = rng.uniform(38.0, 45.0)
    ev = GOAL + r0 * np.array([np.cos(ang), np.sin(ang)])
    psi = np.arctan2(GOAL[1] - ev[1], GOAL[0] - ev[0]) + rng.uniform(-0.3, 0.3)
    pu = np.array([rng.uniform(0, WORLD), rng.uniform(0, WORLD)])

    ev_pos = np.zeros((steps, 2), np.float64)
    ev_vel = np.zeros((steps, 2), np.float64)
    pu_pos = np.zeros((steps, 2), np.float64)

    for t in range(steps):
        intent = label_seq[t]
        p = PARAMS[intent]
        psi_des = _desired_heading(intent, ev, t)
        dpsi = np.clip(p.turn_gain * _wrap(psi_des - psi), -p.turn_max, p.turn_max)
        psi = _wrap(psi + dpsi * DT)
        vel = p.speed * np.array([np.cos(psi), np.sin(psi)])
        ev_pos[t] = ev
        ev_vel[t] = vel
        ev = ev + vel * DT
        # Pure-pursuit pursuer (does not affect labels; makes geometry realistic).
        to_ev = ev - pu
        d = np.linalg.norm(to_ev) + 1e-9
        pu = pu + 9.0 * (to_ev / d) * DT
        pu_pos[t] = pu

    # Observation noise on POSITIONS only — a realistic tracker sees noisy
    # positions and estimates velocity from them (see features.py). Fairness:
    # neither method gets perfectly clean data.
    if obs_noise > 0:
        ev_pos = ev_pos + rng.normal(0, obs_noise, ev_pos.shape)

    return {"ev_pos": ev_pos, "ev_vel": ev_vel, "pu_pos": pu_pos}


def make_dataset(n_per_class: int, steps: int, seed: int, obs_noise: float,
                 switch: bool = False):
    """Build a dataset.

    switch=False: one constant intent per trajectory.
    switch=True : intent flips from A to B at steps//2 (for the switch-robustness
                  test). frame_label captures the per-frame ground truth.
    """
    rng = np.random.RandomState(seed)
    ev_pos, ev_vel, pu_pos, goal, label, frame_label = [], [], [], [], [], []

    if not switch:
        for cls, name in enumerate(INTENTS):
            for _ in range(n_per_class):
                seq = [name] * steps
                roll = _roll_out(rng, seq, steps, obs_noise)
                ev_pos.append(roll["ev_pos"]); ev_vel.append(roll["ev_vel"])
                pu_pos.append(roll["pu_pos"]); goal.append(GOAL.copy())
                label.append(cls)
                frame_label.append(np.full(steps, cls, np.int64))
    else:
        # Diverse A->B switches (skip A==B).
        pairs = [(a, b) for a in INTENTS for b in INTENTS if a != b]
        for _ in range(n_per_class):
            for a, b in pairs:
                half = steps // 2
                seq = [a] * half + [b] * (steps - half)
                roll = _roll_out(rng, seq, steps, obs_noise)
                ev_pos.append(roll["ev_pos"]); ev_vel.append(roll["ev_vel"])
                pu_pos.append(roll["pu_pos"]); goal.append(GOAL.copy())
                label.append(INTENT_ID[b])   # nominal label = post-switch intent
                fl = np.empty(steps, np.int64)
                fl[:half] = INTENT_ID[a]; fl[half:] = INTENT_ID[b]
                frame_label.append(fl)

    return {
        "ev_pos": np.asarray(ev_pos, np.float64),
        "ev_vel": np.asarray(ev_vel, np.float64),
        "pu_pos": np.asarray(pu_pos, np.float64),
        "goal": np.asarray(goal, np.float64),
        "label": np.asarray(label, np.int64),
        "frame_label": np.asarray(frame_label, np.int64),
        "intent_names": np.asarray(INTENTS),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=80)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obs-noise", type=float, default=0.3)
    ap.add_argument("--switch", action="store_true",
                    help="generate the mid-episode intent-switch set instead")
    ap.add_argument("--out", default="dataset.npz")
    args = ap.parse_args()

    ds = make_dataset(args.n_per_class, args.steps, args.seed, args.obs_noise,
                      switch=args.switch)
    np.savez_compressed(args.out, **ds)
    n = len(ds["label"])
    print(f"wrote {args.out}: {n} trajectories, {args.steps} steps, "
          f"intents={list(ds['intent_names'])}, switch={args.switch}")


if __name__ == "__main__":
    main()
