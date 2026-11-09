"""Rasch (1PL) item-difficulty calibration: centralised, local, count-based, federated and DP-federated.

Model: P(y=1 | u, i) = σ(θ_u − b_i). Objective (protocol §2): log-likelihood − λ/2(‖θ‖² + ‖b‖²), λ = 1.
Responses are given as integer arrays (u, i, y) indexed 0..n_users-1 and 0..n_items-1.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .dp import sigma_for

LAMBDA = 1.0


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


@dataclass
class Responses:
    u: np.ndarray
    i: np.ndarray
    y: np.ndarray
    n_users: int
    n_items: int

    def subset(self, mask) -> "Responses":
        return Responses(self.u[mask], self.i[mask], self.y[mask], self.n_users, self.n_items)


def theta_newton(r: Responses, b: np.ndarray, theta: np.ndarray, steps: int, lam: float = LAMBDA) -> np.ndarray:
    for _ in range(steps):
        p = sigmoid(theta[r.u] - b[r.i])
        g = np.bincount(r.u, r.y - p, r.n_users) - lam * theta
        h = np.bincount(r.u, p * (1 - p), r.n_users) + lam
        theta = theta + g / h
    return theta


def b_grad_hess(r: Responses, b: np.ndarray, theta: np.ndarray):
    """Gradient / Hessian-diagonal of the (unpenalised) log-likelihood w.r.t. b."""
    p = sigmoid(theta[r.u] - b[r.i])
    g = np.bincount(r.i, p - r.y, r.n_items)
    h = np.bincount(r.i, p * (1 - p), r.n_items)
    return g, h


def fit_central(r: Responses, iters: int = 100, tol: float = 1e-6, lam: float = LAMBDA):
    theta = np.zeros(r.n_users)
    b = np.zeros(r.n_items)
    for it in range(iters):
        theta = theta_newton(r, b, theta, 1, lam)
        g, h = b_grad_hess(r, b, theta)
        step = (g - lam * b) / (h + lam)
        b = b + step
        if np.max(np.abs(step)) < tol:
            break
    theta = theta_newton(r, b, theta, 3, lam)
    return b, theta, it + 1


def fit_counts(parts: Sequence[Responses], n_items: int) -> np.ndarray:
    c = np.zeros(n_items)
    n = np.zeros(n_items)
    for r in parts:
        c += np.bincount(r.i, r.y, n_items)
        n += np.bincount(r.i, minlength=n_items)
    p = (c + 1) / (n + 2)
    return -np.log(p / (1 - p))


@dataclass
class FedLog:
    rounds: int = 0
    bytes_up: int = 0
    bytes_down: int = 0
    seconds: float = 0.0
    b_history: List[np.ndarray] = field(default_factory=list)
    participation: List[List[bool]] = field(default_factory=list)


def fit_federated(parts: Sequence[Responses], n_items: int, rounds: int = 100, local_steps: int = 1,
                  damping: float = 1.0, tol: float = 1e-6, lam: float = LAMBDA, gate=None, keep_history=False,
                  init_b=None):
    """Federated Rasch. `gate(round, k) -> bool` is the per-round authorisation hook (policy engine);
    an institution whose gate is False contributes nothing in that round. `init_b` warm-starts the difficulties
    (the penalised objective is concave, so the fixed point does not depend on the start)."""
    b = np.zeros(n_items) if init_b is None else np.array(init_b, dtype=float)
    thetas = [np.zeros(r.n_users) for r in parts]
    log = FedLog()
    t0 = time.perf_counter()
    for t in range(rounds):
        G = np.zeros(n_items)
        H = np.zeros(n_items)
        part = []
        for k, r in enumerate(parts):
            ok = True if gate is None else bool(gate(t, k))
            part.append(ok)
            if not ok:
                continue
            thetas[k] = theta_newton(r, b, thetas[k], local_steps, lam)
            g, h = b_grad_hess(r, b, thetas[k])
            G += g
            H += h
            log.bytes_up += 2 * n_items * 8
            log.bytes_down += n_items * 8
        step = damping * (G - lam * b) / (H + lam)
        b = b + step
        log.rounds = t + 1
        log.participation.append(part)
        if keep_history:
            log.b_history.append(b.copy())
        if np.max(np.abs(step)) < tol:
            break
    thetas = [theta_newton(r, b, th, 3, lam) for r, th in zip(parts, thetas)]
    log.seconds = time.perf_counter() - t0
    return b, thetas, log


class _PairIndex:
    """Aggregates responses to (student, item) pairs so per-student contributions can be clipped."""

    def __init__(self, r: Responses):
        key = r.u.astype(np.int64) * r.n_items + r.i
        uniq, inv = np.unique(key, return_inverse=True)
        self.inv = inv
        self.pu = (uniq // r.n_items).astype(np.int64)
        self.pi = (uniq % r.n_items).astype(np.int64)
        self.n_pairs = len(uniq)
        self.n_users = r.n_users
        self.n_items = r.n_items
        self.pair_count = np.bincount(inv, minlength=self.n_pairs).astype(float)

    def clipped_item_sum(self, per_resp: np.ndarray, C: float, per_pair: bool = False) -> np.ndarray:
        v = per_resp if per_pair else np.bincount(self.inv, per_resp, self.n_pairs)
        norm = np.sqrt(np.bincount(self.pu, v * v, self.n_users))
        scale = np.minimum(1.0, C / np.maximum(norm, 1e-12))
        return np.bincount(self.pi, v * scale[self.pu], self.n_items)


def fit_dp_federated(parts: Sequence[Responses], n_items: int, eps: float, delta: float, rounds: int,
                     clip: float, clip_count: float, lr: float, seed: int, local_steps: int = 1,
                     lam: float = LAMBDA, gate=None, ledger=None, task_id: str = "task", org_ids=None,
                     pseudo_count: bool = True, damping: bool = True):
    """Student-level DP federated Rasch, v2 (protocol §4; change recorded as deviation D-1).

    Release 0 (count statistics): each student's concatenated (count, correct) vector over items is clipped
    jointly to L2 norm `clip_count`; the per-item sums get Gaussian noise. They give the warm start
    b0 = −logit((c̃+1)/(ñ+2)) (rounds = 0 is the DP-CNT estimator) and the preconditioner.
    Releases 1..T: per-item gradient sums, each student's contribution clipped to `clip`, plus Gaussian noise.
    The server step is damped by the public per-round noise scale τ = sqrt(K)·σ·clip.
    Every release has sensitivity equal to its clip norm and noise σ·clip; σ is set so that T+1 releases
    meet (ε, δ) under zCDP composition. θ never leaves an institution.
    """
    rng = np.random.default_rng(seed)
    sigma = sigma_for(eps, delta, rounds + 1)
    idx = [_PairIndex(r) for r in parts]
    org_ids = org_ids or [f"org{k}" for k in range(len(parts))]
    K = len(parts)
    if ledger is not None:
        for k in range(K):
            ledger.charge(org_ids[k], task_id, sigma, releases=rounds + 1, eps_target=eps, clip=clip,
                          clip_count=clip_count, rounds=rounds)
    n_noisy = np.zeros(n_items)
    c_noisy = np.zeros(n_items)
    for r, ix in zip(parts, idx):
        n_pair = ix.pair_count
        c_pair = np.bincount(ix.inv, r.y, ix.n_pairs)
        norm = np.sqrt(np.bincount(ix.pu, n_pair ** 2 + c_pair ** 2, ix.n_users))
        scale = np.minimum(1.0, clip_count / np.maximum(norm, 1e-12))[ix.pu]
        n_noisy += np.bincount(ix.pi, n_pair * scale, n_items) + rng.normal(0.0, sigma * clip_count, n_items)
        c_noisy += np.bincount(ix.pi, c_pair * scale, n_items) + rng.normal(0.0, sigma * clip_count, n_items)
    n_hat = np.maximum(n_noisy, 0.0)
    c_hat = np.clip(c_noisy, 0.0, n_hat)
    # Noise-aware pseudo-count: shrink the warm start toward p = 0.5 (b = 0) when count noise dominates.
    m = max(2.0, np.sqrt(K) * sigma * clip_count) if pseudo_count else 2.0
    p0 = (c_hat + m / 2) / (n_hat + m)
    b = -np.log(p0 / (1 - p0))
    tau = np.sqrt(K) * sigma * clip if damping else 0.0
    precond = 0.25 * n_hat + lam + tau
    thetas = [np.zeros(r.n_users) for r in parts]
    log = FedLog()
    t0 = time.perf_counter()
    for t in range(rounds):
        G = np.zeros(n_items)
        part = []
        for k, (r, ix) in enumerate(zip(parts, idx)):
            ok = True if gate is None else bool(gate(t, k))
            part.append(ok)
            if not ok:
                continue
            thetas[k] = theta_newton(r, b, thetas[k], local_steps, lam)
            p = sigmoid(thetas[k][r.u] - b[r.i])
            G += ix.clipped_item_sum(p - r.y, clip)
            G += rng.normal(0.0, sigma * clip, n_items)
            log.bytes_up += n_items * 8
            log.bytes_down += n_items * 8
        b = b + lr * (G - lam * b) / precond
        log.rounds = t + 1
        log.participation.append(part)
    thetas = [theta_newton(r, b, th, 5, lam) for r, th in zip(parts, thetas)]
    log.seconds = time.perf_counter() - t0
    return b, thetas, log, sigma


def predict(r: Responses, b: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return sigmoid(theta[r.u] - b[r.i])


def auc(y: np.ndarray, p: np.ndarray) -> float:
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(p)
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def brier(y, p) -> float:
    return float(np.mean((p - y) ** 2))


def logloss(y, p) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
