#!/usr/bin/env python3
"""Head-to-head: abductive Prolog (System 2) vs inductive LSTM (System 1).

Runs the full experiment on the controlled dataset and prints the comparison
table + saves results.json. Honest, multi-axis comparison — NOT a rigged
"we win everything" table. Axes:

  1. accuracy        whole-trajectory intent accuracy on a held-out test set
  2. latency         frames-to-correct-and-stable decision (how fast each method
                     commits to the right intent)
  3. data efficiency LSTM accuracy vs #training trajectories; abduction is flat
                     (consumes ZERO training data)
  4. switch robust.  on mid-episode intent switches, frames to re-classify after
                     the switch (memoryless abduction vs LSTM hidden-state lag)
  5. explainability  abduction emits a proof/justification per decision; the
                     LSTM emits only a softmax (shown qualitatively)

Run:
  python3 compare.py                 # full run
  python3 compare.py --quick         # smaller/faster
  python3 lstm_baseline.py --grad-check   # verify the baseline first
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from generate_dataset import make_dataset, INTENTS
from features import dataset_features
from lstm_baseline import LSTMClassifier, grad_check
from gru_baseline import GRUClassifier
from translate import AbductiveClassifier


def _stable_decision_frame(preds, truth, hold=3):
    """First frame index from which preds == truth for `hold` consecutive frames
    and stays correct to the end. Returns len(preds) if never stable."""
    T = len(preds)
    for t in range(T):
        if all(preds[k] == truth for k in range(t, min(T, t + hold))) and \
           all(preds[k] == truth for k in range(t, T)):
            return t
    return T


def run(args):
    has_swi = True
    try:
        abd = AbductiveClassifier()
    except Exception as exc:
        print(f"[warn] Prolog backend unavailable ({exc}); abductive side skipped.")
        has_swi = False
        abd = None

    npc = 20 if args.quick else 60
    steps = 45 if args.quick else 50

    # --- datasets ----------------------------------------------------------
    train = make_dataset(npc, steps, seed=1, obs_noise=0.3)
    test = make_dataset(max(10, npc // 3), steps, seed=99, obs_noise=0.3)
    switch = make_dataset(6 if args.quick else 12, steps, seed=7,
                          obs_noise=0.3, switch=True)

    Xtr, ytr = dataset_features(train), train["label"]
    Xte, yte = dataset_features(test), test["label"]
    Xsw, sw_frame_label = dataset_features(switch), switch["frame_label"]

    F, K = Xtr.shape[-1], len(INTENTS)

    # --- inductive LSTM ----------------------------------------------------
    print("training inductive LSTM baseline ...")
    lstm = LSTMClassifier(F, K, hidden=16, seed=0)
    lstm.fit(Xtr, ytr, epochs=40 if args.quick else 80, lr=0.01, verbose=True)
    lstm_pred = lstm.predict(Xte)
    lstm_acc = float(np.mean(lstm_pred == yte))

    # --- inductive GRU (lighter recurrent baseline) ------------------------
    print("training inductive GRU baseline ...")
    gru = GRUClassifier(F, K, hidden=16, seed=0)
    gru.fit(Xtr, ytr, epochs=40 if args.quick else 80, lr=0.01, verbose=True)
    gru_pred = gru.predict(Xte)
    gru_acc = float(np.mean(gru_pred == yte))

    # --- abductive Prolog --------------------------------------------------
    abd_acc = None
    abd_lat = lstm_lat = gru_lat = None
    if has_swi:
        print("running abductive Prolog classifier ...")
        name_to_id = {n: i for i, n in enumerate(INTENTS)}
        abd_pred = []
        for i in range(Xte.shape[0]):
            # Memoryless: each frame is explained independently. The single
            # trajectory label is the MAJORITY vote of the per-frame abductions
            # (the natural way to fuse independent evidence without a model).
            seq = [name_to_id.get(s, name_to_id["hold"])
                   for s in abd.classify_trajectory(Xte[i])]
            abd_pred.append(int(np.bincount(seq, minlength=len(INTENTS)).argmax()))
        abd_pred = np.asarray(abd_pred)
        abd_acc = float(np.mean(abd_pred == yte))

        # --- latency (frames-to-stable-correct) ----------------------------
        n_lat = min(40, Xte.shape[0])
        lstm_step = lstm.predict_per_step(Xte[:n_lat])      # [n,T]
        gru_step = gru.predict_per_step(Xte[:n_lat])        # [n,T]
        abd_lat_list, lstm_lat_list, gru_lat_list = [], [], []
        for i in range(n_lat):
            truth = int(yte[i])
            abd_seq = [name_to_id.get(s, name_to_id["hold"])
                       for s in abd.classify_trajectory(Xte[i])]
            abd_lat_list.append(_stable_decision_frame(abd_seq, truth))
            lstm_lat_list.append(_stable_decision_frame(list(lstm_step[i]), truth))
            gru_lat_list.append(_stable_decision_frame(list(gru_step[i]), truth))
        abd_lat = float(np.mean(abd_lat_list))
        lstm_lat = float(np.mean(lstm_lat_list))
        gru_lat = float(np.mean(gru_lat_list))

    # --- data efficiency (LSTM only; abduction is flat at 0 data) ----------
    print("data-efficiency sweep (LSTM) ...")
    eff = {}
    per_class = npc
    sizes = [2, 5, 10, 20, per_class] if not args.quick else [2, 5, 10, npc]
    for sz in sizes:
        sub = make_dataset(sz, steps, seed=1, obs_noise=0.3)
        net = LSTMClassifier(F, K, hidden=16, seed=0)
        net.fit(dataset_features(sub), sub["label"],
                epochs=40 if args.quick else 80, lr=0.01)
        eff[sz * K] = float(np.mean(net.predict(Xte) == yte))   # key = total traj

    # --- switch robustness -------------------------------------------------
    half = steps // 2
    sw_lstm_lag = sw_abd_lag = sw_gru_lag = None
    if has_swi:
        name_to_id = {n: i for i, n in enumerate(INTENTS)}
        lstm_sw = lstm.predict_per_step(Xsw)                    # [n,T]
        gru_sw = gru.predict_per_step(Xsw)                      # [n,T]
        abd_lags, lstm_lags, gru_lags = [], [], []
        for i in range(Xsw.shape[0]):
            post = int(sw_frame_label[i, -1])                   # intent after switch
            abd_seq = [name_to_id.get(s, name_to_id["hold"])
                       for s in abd.classify_trajectory(Xsw[i])]
            abd_lags.append(_reclassify_lag(abd_seq, post, half))
            lstm_lags.append(_reclassify_lag(list(lstm_sw[i]), post, half))
            gru_lags.append(_reclassify_lag(list(gru_sw[i]), post, half))
        sw_gru_lag = float(np.nanmean(gru_lags))
        sw_abd_lag = float(np.nanmean(abd_lags))
        sw_lstm_lag = float(np.nanmean(lstm_lags))

    # --- explainability sample --------------------------------------------
    expl = None
    if has_swi:
        intent, why = abd.classify_frame(Xte[0, -1])
        expl = {"intent": intent, "justification": why,
                "lstm_equivalent": "softmax probability vector only (black-box)"}

    results = {
        "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0]),
        "steps": steps, "intents": INTENTS,
        "accuracy": {"abductive": abd_acc, "inductive_lstm": lstm_acc,
                     "inductive_gru": gru_acc},
        "latency_frames": {"abductive": abd_lat, "inductive_lstm": lstm_lat,
                           "inductive_gru": gru_lat},
        "data_efficiency_lstm": eff,
        "abductive_training_trajectories": 0,
        "switch_reclassify_lag_frames": {"abductive": sw_abd_lag,
                                         "inductive_lstm": sw_lstm_lag,
                                         "inductive_gru": sw_gru_lag},
        "explainability_sample": expl,
    }
    _print_table(results)
    out = os.path.join(os.path.dirname(__file__), "..", "results", "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {out}")
    return results


def _reclassify_lag(seq, post_intent, half):
    """Frames after the switch (at `half`) until the prediction locks onto the
    post-switch intent and stays. NaN if it never does."""
    T = len(seq)
    for t in range(half, T):
        if all(seq[k] == post_intent for k in range(t, T)):
            return t - half
    return np.nan


def _fmt(x, p=3):
    return "n/a" if x is None else (f"{x:.{p}f}" if isinstance(x, float) else str(x))


def _print_table(r):
    print("\n" + "=" * 64)
    print("  ABDUCTIVE (Prolog)  vs  INDUCTIVE (LSTM, GRU)")
    print("=" * 64)
    a = r["accuracy"]["abductive"]
    l, g = r["accuracy"]["inductive_lstm"], r["accuracy"]["inductive_gru"]
    al = r["latency_frames"]["abductive"]
    ll, gl = r["latency_frames"]["inductive_lstm"], r["latency_frames"]["inductive_gru"]
    sa = r["switch_reclassify_lag_frames"]["abductive"]
    sl = r["switch_reclassify_lag_frames"]["inductive_lstm"]
    sg = r["switch_reclassify_lag_frames"]["inductive_gru"]
    rows = [
        ("metric", "abductive", "LSTM", "GRU"),
        ("-" * 26, "-" * 10, "-" * 10, "-" * 10),
        ("test accuracy", _fmt(a), _fmt(l), _fmt(g)),
        ("train trajectories needed", "0", str(r["n_train"]), str(r["n_train"])),
        ("latency (frames to lock)", _fmt(al, 1), _fmt(ll, 1), _fmt(gl, 1)),
        ("switch re-classify lag", _fmt(sa, 1), _fmt(sl, 1), _fmt(sg, 1)),
        ("explainability", "proof tree", "softmax", "softmax"),
    ]
    for c1, c2, c3, c4 in rows:
        print(f"  {c1:26s} {c2:>10s} {c3:>10s} {c4:>10s}")
    print("-" * 64)
    print("  LSTM accuracy vs #training trajectories (data efficiency):")
    for n, acc in sorted(r["data_efficiency_lstm"].items(), key=lambda kv: int(kv[0])):
        bar = "#" * int(acc * 30)
        print(f"    {int(n):4d} traj : {acc:.3f}  {bar}")
    print("    (abductive: 0 traj, accuracy = {} — flat, no training)"
          .format(_fmt(a)))
    if r["explainability_sample"]:
        e = r["explainability_sample"]
        print("-" * 64)
        print(f"  explainability sample — abductive decision on one frame:")
        print(f"    intent       = {e['intent']}")
        print(f"    justification= {e['justification']}")
        print(f"    LSTM gives   : {e['lstm_equivalent']}")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller/faster run")
    ap.add_argument("--grad-check", action="store_true",
                    help="verify LSTM gradients then exit")
    args = ap.parse_args()
    if args.grad_check:
        grad_check()
        return
    run(args)


if __name__ == "__main__":
    main()
