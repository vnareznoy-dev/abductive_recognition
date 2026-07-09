#!/usr/bin/env python3
"""Export the benchmark dataset to CSV (the exact files shipped in data/).

Writes 320 trajectories — 240 train (seed=1) + 80 test (seed=99), 4 balanced
intent classes, 50 steps at 0.1 s, observation noise 0.3 m — as one CSV per
trajectory plus a manifest.json index. Deterministic: re-running this script
reproduces data/ byte-for-byte.

Usage:
  python3 src/export_dataset.py            # writes into ./data next to src/
  python3 src/export_dataset.py --out DIR  # custom output directory
"""

from __future__ import annotations

import argparse
import json
import os

from generate_dataset import make_dataset, INTENTS

STEPS = 50
DT = 0.1
N_PER_CLASS_TRAIN = 60
N_PER_CLASS_TEST = 20
SEED_TRAIN = 1
SEED_TEST = 99
OBS_NOISE = 0.3

HEADER = "t,ev_x,ev_y,goal_x,goal_y,pursuer_x,pursuer_y"


def _num(x: float, nd: int) -> str:
    return str(round(float(x), nd))


def _write_split(ds, split: str, start_idx: int, out_csv: str, manifest: list) -> int:
    n = ds["label"].shape[0]
    idx = start_idx
    for i in range(n):
        intent = str(ds["intent_names"][ds["label"][i]])
        fname = f"traj_{idx:04d}_{split}_{intent}.csv"
        rows = [HEADER]
        for t in range(STEPS):
            rows.append(",".join([
                _num(t * DT, 1),
                _num(ds["ev_pos"][i, t, 0], 4), _num(ds["ev_pos"][i, t, 1], 4),
                _num(ds["goal"][i, 0], 4), _num(ds["goal"][i, 1], 4),
                _num(ds["pu_pos"][i, t, 0], 4), _num(ds["pu_pos"][i, t, 1], 4),
            ]))
        with open(os.path.join(out_csv, fname), "w") as f:
            f.write("\n".join(rows) + "\n")
        manifest.append({"file": fname, "split": split, "intent": intent})
        idx += 1
    return idx


def main():
    default_out = os.path.join(os.path.dirname(__file__), "..", "data")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=default_out)
    args = ap.parse_args()

    out_csv = os.path.join(args.out, "csv")
    os.makedirs(out_csv, exist_ok=True)

    train = make_dataset(N_PER_CLASS_TRAIN, STEPS, seed=SEED_TRAIN, obs_noise=OBS_NOISE)
    test = make_dataset(N_PER_CLASS_TEST, STEPS, seed=SEED_TEST, obs_noise=OBS_NOISE)

    manifest = []
    idx = _write_split(train, "train", 0, out_csv, manifest)
    idx = _write_split(test, "test", idx, out_csv, manifest)

    meta = {
        "n_total": idx,
        "n_train": int(train["label"].shape[0]),
        "n_test": int(test["label"].shape[0]),
        "steps": STEPS,
        "dt_s": DT,
        "classes": INTENTS,
        "seed_train": SEED_TRAIN,
        "seed_test": SEED_TEST,
        "obs_noise": OBS_NOISE,
        "trajectories": manifest,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {idx} trajectories to {os.path.abspath(out_csv)}")


if __name__ == "__main__":
    main()
