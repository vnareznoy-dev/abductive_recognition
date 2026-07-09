# Abductive Intent Recognition of Dynamic Objects from Spatial Trajectories

Companion repository (dataset + code) for the article:

> **Нарєзной В. Ю.** «Абдуктивне логічне програмування у задачах розпізнавання намірів динамічних об'єктів за їхніми просторовими траєкторіями»
> *(Volodymyr Nareznoy. "Abductive logic programming for intent recognition of dynamic objects based on their spatial trajectories")*

The study compares an **abductive logic programming (ALP)** intent classifier
(SWI-Prolog, zero training data, proof-tree explanations) against **LSTM** and
**GRU** recurrent baselines (pure-numpy implementations with verified
gradients) on a controlled pursuit–evasion trajectory benchmark.

## Contents

```
data/                 320 trajectories (240 train + 80 test), CSV + manifest
src/
  generate_dataset.py   deterministic trajectory generator (4 intents)
  export_dataset.py     regenerates data/ byte-for-byte
  features.py           shared spatio-temporal features (both classifiers)
  intent_abduce.pl      abductive theory: rules, integrity constraints, classify/2
  translate.py          features -> Prolog facts bridge (pyswip)
  lstm_baseline.py      LSTM sequence classifier, numpy, BPTT + grad-check
  gru_baseline.py       GRU sequence classifier, numpy, BPTT + grad-check
  compare.py            full experiment -> results/results.json
results/results.json  reference results (Table 4 of the article)
```

## Quick start

Requirements: Python ≥ 3.9, [SWI-Prolog](https://www.swi-prolog.org/) on PATH.

```bash
# macOS: brew install swi-prolog     Ubuntu/Debian: sudo apt install swi-prolog
pip install -r requirements.txt

python3 src/compare.py            # full experiment (~ a few minutes, CPU only)
python3 src/compare.py --quick    # smaller sanity run
python3 src/compare.py --grad-check   # verify LSTM backprop numerically
```

`compare.py` regenerates the dataset in memory from fixed seeds (train
`seed=1`, test `seed=99`, switch set `seed=7`), trains the neural baselines,
runs the abductive classifier and writes `results/results.json`.

## Dataset

320 synthetic pursuit–evasion trajectories, 4 balanced intent classes
(`rush`, `flank`, `weave`, `hold`), 50 steps at 0.1 s, Gaussian observation
noise 0.3 m on positions. See [data/README.md](data/README.md) for the data
card and column description. To regenerate the CSVs:

```bash
python3 src/export_dataset.py     # rewrites data/ deterministically
```

## Key results (reference run)

| Metric                          | Abductive | LSTM  | GRU   |
|---------------------------------|-----------|-------|-------|
| Test accuracy                   | 0.988     | 1.000 | 1.000 |
| Training trajectories consumed  | 0         | 240   | 240   |
| Decision latency, frames        | 18.6      | 20.4  | 6.3   |
| Re-classification lag, frames   | 11.3      | 10.8  | 13.2  |
| Explanation of a decision       | proof tree| softmax only | softmax only |

The point of the comparison is **not** accuracy (the task is nearly saturated
for trained networks) but *interpretability* and *zero-training operation*:
the abductive classifier reaches comparable accuracy with **zero** training
trajectories and returns a logical justification (e.g.
`aimed_at_goal(low_miss, closing)`) for every decision.

## The abductive model in one paragraph

For each frame the observed features (speed, yaw rate, off-axis miss, range,
closing rate, oscillation flag) are asserted as Prolog facts. The theory
`intent_abduce.pl` defines one rule per intent plus integrity constraints
(mutually incompatible intents) and a specificity order. The classifier
searches for the minimal hypothesis Δ such that `T ∪ Δ ⊨ O` and
`T ∪ Δ ∪ IC ⊭ ⊥` — the fired rule body *is* the explanation. See the
article (formula (7) and Fig. 3) for the formal statement.

## Reproducibility notes

- Everything is deterministic: fixed seeds, no GPU, no external ML frameworks.
- The neural baselines are hand-written in numpy with a numerical
  gradient check (`--grad-check`) so the comparison is not against a
  crippled baseline.
- Reference environment: Python 3.9+, numpy ≥ 1.24, pyswip ≥ 0.2.10,
  SWI-Prolog 9.x.

## Опис українською

Репозиторій супроводжує статтю про абдуктивне розпізнавання намірів
динамічних об'єктів за просторовими траєкторіями. Містить: набір даних із
320 траєкторій (240 навчальних + 80 тестових, чотири класи намірів),
абдуктивний класифікатор на SWI-Prolog, базові моделі LSTM і GRU (numpy) та
скрипт відтворення порівняльного експерименту (табл. 4 статті):

```bash
pip install -r requirements.txt
python3 src/compare.py
```

## License

MIT — see [LICENSE](LICENSE). The dataset is synthetic (no personal data)
and is released under the same terms for research reproducibility.

## Citation

See [CITATION.cff](CITATION.cff). Citation metadata will be updated with the
journal reference upon publication.
