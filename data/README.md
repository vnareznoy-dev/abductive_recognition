# Intent Recognition Trajectory Dataset (320 trajectories)

Synthetic pursuit–evasion trajectory dataset for **intent recognition of
dynamic objects**, used in the comparative study of abductive logic
programming vs. recurrent neural networks (LSTM, GRU).

## Contents
- `csv/traj_NNNN_<split>_<intent>.csv` — one trajectory per file.
- `manifest.json` — index: file, split (train/test), intent label.

## Composition
- **320 trajectories** total: **240 train** + **80 test**.
- **4 intent classes** (balanced): `rush`, `flank`, `weave`, `hold`.
- Each trajectory: **50 steps**, time step **0.1 s**.

### Intent definitions
- `rush`  — evader steers straight at the goal (closing, low off-axis miss).
- `flank` — steers at a constant angular offset to the goal bearing (closing, large off-axis miss, one-sided curve).
- `weave` — goalward bias with sinusoidal heading dither (serpentine).
- `hold`  — slow loiter, not committing (low speed, not closing).

## CSV columns
`t` (s), `ev_x`, `ev_y` (evader position, m), `goal_x`, `goal_y` (goal, m),
`pursuer_x`, `pursuer_y` (pure-pursuit pursuer, m; geometry only, does not
affect labels).

## Reproduction
Deterministically generated — `python3 src/export_dataset.py` rewrites this
directory byte-for-byte:
- train: `make_dataset(n_per_class=60, steps=50, seed=1,  obs_noise=0.3)`
- test:  `make_dataset(n_per_class=20, steps=50, seed=99, obs_noise=0.3)`

Observation noise 0.3 m (Gaussian) is added to positions; velocity/heading
are estimated downstream from lightly smoothed positions (state-estimator
model, see `src/features.py`).

## License
Synthetic data (no personal data), released under the repository's MIT
license for research reproducibility.
