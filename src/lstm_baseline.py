#!/usr/bin/env python3
"""Inductive baseline: a compact LSTM sequence classifier in pure numpy.

This is the inductive baseline (System 1): it LEARNS the
intent->kinematics mapping from labelled trajectories by gradient descent, with
no hand-authored rules. We implement it from scratch (forward + BPTT + Adam) so
the whole benchmark stays dependency-light and deterministic — and we ship a
numerical gradient check (`--grad-check`) so the baseline is provably NOT
secretly crippled (a broken baseline would invalidate the comparison).

Scaling to a full PyTorch LSTM is a drop-in swap of this class; the
experiment harness only needs `.fit`, `.predict`, and `.predict_per_step`.

Equations (batch B, input D, hidden H, classes K), per timestep:
    z = x_t @ Wx + h @ Wh + b            z in R^{B x 4H}
    i,f,g,o = sigmoid(zi), sigmoid(zf), tanh(zg), sigmoid(zo)
    c = f * c_prev + i * g
    h = o * tanh(c)
    logits_t = h @ V + bv                (used for per-step / latency curves)
Sequence label uses logits at the LAST timestep; loss = softmax cross-entropy.
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


class LSTMClassifier:
    def __init__(self, n_features: int, n_classes: int, hidden: int = 16,
                 seed: int = 0):
        self.D, self.K, self.H = n_features, n_classes, hidden
        rng = np.random.RandomState(seed)
        s = 1.0 / np.sqrt(hidden)
        self.Wx = rng.uniform(-s, s, (n_features, 4 * hidden))
        self.Wh = rng.uniform(-s, s, (hidden, 4 * hidden))
        self.b = np.zeros(4 * hidden)
        self.b[hidden:2 * hidden] = 1.0          # forget-gate bias = 1 (stability)
        self.V = rng.uniform(-s, s, (hidden, n_classes))
        self.bv = np.zeros(n_classes)
        self._mu = None
        self._sd = None

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
        h = np.zeros((B, H)); c = np.zeros((B, H))
        cache = {"X": X, "i": [], "f": [], "g": [], "o": [],
                 "c": [], "c_prev": [], "h": [], "tanh_c": []}
        hs = np.zeros((B, T, H))
        for t in range(T):
            z = X[:, t, :] @ self.Wx + h @ self.Wh + self.b
            i = _sigmoid(z[:, :H]); f = _sigmoid(z[:, H:2 * H])
            g = np.tanh(z[:, 2 * H:3 * H]); o = _sigmoid(z[:, 3 * H:])
            c_prev = c
            c = f * c_prev + i * g
            tc = np.tanh(c)
            h = o * tc
            for k, val in (("i", i), ("f", f), ("g", g), ("o", o),
                           ("c", c), ("c_prev", c_prev), ("h", h), ("tanh_c", tc)):
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

        grads = {k: np.zeros_like(getattr(self, k))
                 for k in ("Wx", "Wh", "b", "V", "bv")}
        dlogits = probs.copy()
        dlogits[np.arange(B), y] -= 1.0
        dlogits /= B
        grads["V"] = cache["h"][-1].T @ dlogits
        grads["bv"] = dlogits.sum(0)

        dh = dlogits @ self.V.T          # grad into last hidden
        dc = np.zeros((B, H))
        for t in reversed(range(T)):
            i, f, g, o = cache["i"][t], cache["f"][t], cache["g"][t], cache["o"][t]
            c, c_prev, tc = cache["c"][t], cache["c_prev"][t], cache["tanh_c"][t]
            do = dh * tc
            dc = dc + dh * o * (1 - tc ** 2)
            di = dc * g
            dg = dc * i
            df = dc * c_prev
            dc_prev = dc * f
            dz_i = di * i * (1 - i)
            dz_f = df * f * (1 - f)
            dz_g = dg * (1 - g ** 2)
            dz_o = do * o * (1 - o)
            dz = np.concatenate([dz_i, dz_f, dz_g, dz_o], axis=1)   # [B,4H]
            grads["Wx"] += X[:, t, :].T @ dz
            h_prev = cache["h"][t - 1] if t > 0 else np.zeros((B, H))
            grads["Wh"] += h_prev.T @ dz
            grads["b"] += dz.sum(0)
            dh = dz @ self.Wh.T
            dc = dc_prev
        return loss, grads

    # --- training ----------------------------------------------------------
    def fit(self, X, y, epochs=80, lr=0.01, batch=32, seed=0, verbose=False):
        X = self._normalize(X, fit=True)
        rng = np.random.RandomState(seed)
        params = ("Wx", "Wh", "b", "V", "bv")
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
                # Global-norm gradient clipping (BPTT can spike early on).
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
        """Whole-sequence prediction (uses the last timestep)."""
        Xn = self._normalize(X, fit=False)
        logits, _, _ = self._forward(Xn)
        return logits.argmax(1)

    def predict_per_step(self, X):
        """[B,T,D] -> [B,T] argmax class at EACH timestep (for latency curves)."""
        Xn = self._normalize(X, fit=False)
        _, _, hs = self._forward(Xn)               # [B,T,H]
        logits = hs @ self.V + self.bv             # [B,T,K]
        return logits.argmax(-1)


def grad_check(seed=0):
    """Numerical vs analytic gradient on a tiny random problem.

    Prints max relative error per parameter; < 1e-5 means BPTT is correct and
    the baseline learns honestly (no silent crippling).
    """
    rng = np.random.RandomState(seed)
    B, T, D, K, H = 4, 6, 3, 3, 5
    X = rng.randn(B, T, D)
    y = rng.randint(0, K, B)
    net = LSTMClassifier(D, K, hidden=H, seed=seed)
    net._mu = np.zeros(D); net._sd = np.ones(D)     # bypass normalisation
    _, grads = net._loss_and_grads(X, y)
    eps = 1e-5
    print("gradient check (max relative error per param):")
    ok = True
    for p in ("Wx", "Wh", "b", "V", "bv"):
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
        print(f"  {p:3s}: {rel:.2e}  {flag}")
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    if "--grad-check" in sys.argv:
        grad_check()
