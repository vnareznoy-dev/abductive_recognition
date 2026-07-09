#!/usr/bin/env python3
"""Inductive baseline #2: a compact GRU sequence classifier in pure numpy.

Companion to lstm_baseline.py — same role (System 1, learned intent->kinematics
mapping), same from-scratch philosophy (forward + BPTT + Adam, dependency-light,
deterministic), same interface (`.fit`, `.predict`, `.predict_per_step`) so the
benchmark harness swaps it in beside the LSTM without changes. A GRU has two
gates instead of the LSTM's three and no separate cell state, so it is the
lighter recurrent baseline; reporting both shows the abductive comparison is
not cherry-picked against one architecture.

Equations (batch B, input D, hidden H, classes K), per timestep:
    z = sigmoid(x_t @ Wxz + h @ Whz + bz)      update gate
    r = sigmoid(x_t @ Wxr + h @ Whr + br)      reset gate
    n = tanh(x_t @ Wxn + (r * h) @ Whn + bn)   candidate
    h = (1 - z) * n + z * h                    new hidden
    logits_t = h @ V + bv
Sequence label uses logits at the LAST timestep; loss = softmax cross-entropy.
A numerical gradient check (`--grad-check`) proves the BPTT is correct so the
baseline is not silently crippled.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class GRUClassifier:
    def __init__(self, n_features: int, n_classes: int, hidden: int = 16,
                 seed: int = 0):
        self.D, self.K, self.H = n_features, n_classes, hidden
        rng = np.random.RandomState(seed)
        s = 1.0 / np.sqrt(hidden)
        # gate params: update (z), reset (r), candidate (n)
        self.Wxz = rng.uniform(-s, s, (n_features, hidden))
        self.Whz = rng.uniform(-s, s, (hidden, hidden))
        self.bz = np.zeros(hidden)
        self.Wxr = rng.uniform(-s, s, (n_features, hidden))
        self.Whr = rng.uniform(-s, s, (hidden, hidden))
        self.br = np.zeros(hidden)
        self.Wxn = rng.uniform(-s, s, (n_features, hidden))
        self.Whn = rng.uniform(-s, s, (hidden, hidden))
        self.bn = np.zeros(hidden)
        self.V = rng.uniform(-s, s, (hidden, n_classes))
        self.bv = np.zeros(n_classes)
        self._mu = None
        self._sd = None

    _PARAMS = ("Wxz", "Whz", "bz", "Wxr", "Whr", "br",
               "Wxn", "Whn", "bn", "V", "bv")

    # --- normalisation (fit on train only) ---------------------------------
    def _normalize(self, X, fit=False):
        if fit:
            flat = X.reshape(-1, X.shape[-1])
            self._mu = flat.mean(0)
            self._sd = flat.std(0) + 1e-6
        return (X - self._mu) / self._sd

    # --- forward pass over a full batch of sequences -----------------------
    def _forward(self, X):
        """X: [B,T,D] -> (last_logits [B,K], cache, per_step_h [B,T,H])."""
        B, T, _ = X.shape
        H = self.H
        h = np.zeros((B, H))
        cache = {"X": X, "z": [], "r": [], "n": [], "h_prev": [], "rh": []}
        hs = np.zeros((B, T, H))
        for t in range(T):
            x = X[:, t, :]
            z = _sigmoid(x @ self.Wxz + h @ self.Whz + self.bz)
            r = _sigmoid(x @ self.Wxr + h @ self.Whr + self.br)
            rh = r * h
            n = np.tanh(x @ self.Wxn + rh @ self.Whn + self.bn)
            h_prev = h
            h = (1.0 - z) * n + z * h_prev
            for k, val in (("z", z), ("r", r), ("n", n),
                           ("h_prev", h_prev), ("rh", rh)):
                cache[k].append(val)
            hs[:, t, :] = h
        logits = h @ self.V + self.bv
        return logits, cache, hs

    def _loss_and_grads(self, X, y):
        B, T, _ = X.shape
        H = self.H
        logits, cache, _ = self._forward(X)
        probs = _softmax(logits)
        loss = -np.mean(np.log(probs[np.arange(B), y] + 1e-12))

        grads = {k: np.zeros_like(getattr(self, k)) for k in self._PARAMS}
        dlogits = probs.copy()
        dlogits[np.arange(B), y] -= 1.0
        dlogits /= B
        grads["V"] = cache["h_prev"][-1].T @ dlogits if False else None
        # last hidden h_T is (1-z)*n + z*h_prev at t=T-1; V multiplies h_T.
        # Recover h_T from cache: h_T = hs[-1]; but we need it for V grad.
        # Simpler: recompute h_T from the last step's cached pieces.
        zT, nT, hpT = cache["z"][-1], cache["n"][-1], cache["h_prev"][-1]
        hT = (1.0 - zT) * nT + zT * hpT
        grads["V"] = hT.T @ dlogits
        grads["bv"] = dlogits.sum(0)

        dh = dlogits @ self.V.T          # grad into last hidden
        for t in reversed(range(T)):
            x = cache["X"][:, t, :]
            z, r, n = cache["z"][t], cache["r"][t], cache["n"][t]
            h_prev, rh = cache["h_prev"][t], cache["rh"][t]
            # h = (1-z)*n + z*h_prev
            dz = dh * (h_prev - n)
            dn = dh * (1.0 - z)
            dh_prev = dh * z
            # n = tanh(a_n),  a_n = x@Wxn + (r*h_prev)@Whn + bn
            da_n = dn * (1.0 - n ** 2)
            grads["Wxn"] += x.T @ da_n
            grads["Whn"] += rh.T @ da_n
            grads["bn"] += da_n.sum(0)
            drh = da_n @ self.Whn.T
            dr = drh * h_prev
            dh_prev += drh * r
            # gates z, r
            da_z = dz * z * (1.0 - z)
            da_r = dr * r * (1.0 - r)
            grads["Wxz"] += x.T @ da_z
            grads["Whz"] += h_prev.T @ da_z
            grads["bz"] += da_z.sum(0)
            grads["Wxr"] += x.T @ da_r
            grads["Whr"] += h_prev.T @ da_r
            grads["br"] += da_r.sum(0)
            dh_prev += da_z @ self.Whz.T + da_r @ self.Whr.T
            dh = dh_prev
        return loss, grads

    # --- training ----------------------------------------------------------
    def fit(self, X, y, epochs=80, lr=0.01, batch=32, seed=0, verbose=False):
        X = self._normalize(X, fit=True)
        rng = np.random.RandomState(seed)
        params = self._PARAMS
        mt = {p: np.zeros_like(getattr(self, p)) for p in params}
        vt = {p: np.zeros_like(getattr(self, p)) for p in params}
        b1, b2, eps = 0.9, 0.999, 1e-8
        step = 0
        N = X.shape[0]
        for ep in range(epochs):
            order = rng.permutation(N)
            tot = 0.0
            for s in range(0, N, batch):
                idx = order[s:s + batch]
                loss, grads = self._loss_and_grads(X[idx], y[idx])
                tot += loss * len(idx)
                step += 1
                gn = np.sqrt(sum(float(np.sum(g ** 2)) for g in grads.values()))
                if gn > 5.0:
                    scale = 5.0 / (gn + 1e-12)
                    for p in params:
                        grads[p] *= scale
                for p in params:
                    mt[p] = b1 * mt[p] + (1 - b1) * grads[p]
                    vt[p] = b2 * vt[p] + (1 - b2) * grads[p] ** 2
                    mhat = mt[p] / (1 - b1 ** step)
                    vhat = vt[p] / (1 - b2 ** step)
                    setattr(self, p, getattr(self, p) - lr * mhat / (np.sqrt(vhat) + eps))
            if verbose and (ep % 10 == 0 or ep == epochs - 1):
                print(f"  epoch {ep:3d}  train_loss={tot / N:.4f}")
        return self

    # --- inference ---------------------------------------------------------
    def predict(self, X):
        Xn = self._normalize(X, fit=False)
        logits, _, _ = self._forward(Xn)
        return logits.argmax(1)

    def predict_per_step(self, X):
        Xn = self._normalize(X, fit=False)
        _, _, hs = self._forward(Xn)               # [B,T,H]
        logits = hs @ self.V + self.bv             # [B,T,K]
        return logits.argmax(-1)


def grad_check(seed=0):
    """Numerical vs analytic gradient on a tiny random problem."""
    rng = np.random.RandomState(seed)
    B, T, D, K, H = 4, 6, 3, 3, 5
    X = rng.randn(B, T, D)
    y = rng.randint(0, K, B)
    net = GRUClassifier(D, K, hidden=H, seed=seed)
    net._mu = np.zeros(D); net._sd = np.ones(D)     # bypass normalisation
    _, grads = net._loss_and_grads(X, y)
    eps = 1e-5
    print("GRU gradient check (max relative error per param):")
    ok = True
    for p in net._PARAMS:
        W = getattr(net, p)
        num = np.zeros_like(W)
        it = np.nditer(W, flags=["multi_index"])
        while not it.finished:
            ix = it.multi_index
            old = W[ix]
            W[ix] = old + eps; lp, _ = net._loss_and_grads(X, y)
            W[ix] = old - eps; lm, _ = net._loss_and_grads(X, y)
            W[ix] = old
            num[ix] = (lp - lm) / (2 * eps)
            it.iternext()
        denom = np.maximum(1e-8, np.abs(num) + np.abs(grads[p]))
        rel = np.max(np.abs(num - grads[p]) / denom)
        flag = "OK " if rel < 1e-5 else "BAD"
        ok = ok and rel < 1e-5
        print(f"  {p:4s}: {rel:.2e}  {flag}")
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    if "--grad-check" in sys.argv:
        grad_check()
