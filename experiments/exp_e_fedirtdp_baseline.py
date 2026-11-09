"""Experiment E (exploratory): DP-FED vs a trusted-aggregator (FedIRT-DP style) baseline.

The baseline keeps the same estimator but assumes secure aggregation: the coordinator sees
only the sum of clipped messages and adds Gaussian noise calibrated to the SUM sensitivity
(C for gradients, C_n for counts), i.e. sqrt(K) times less total noise than per-institution
releases. This quantifies the price of the "no trusted aggregator" design choice.

usage: python exp_e_fedirtdp_baseline.py {syn|junyi}
Writes one JSON row per (dataset, seed, method, eps) to results/expE/<dataset>.jsonl
"""
import json, sys, time
import numpy as np
from scipy.stats import spearmanr

from common import *
sys.path.insert(0, str(ROOT / "src"))
from lfchain.data.datasets import load_junyi, make_synthetic
from lfchain.rasch import theta_newton, sigmoid, predict, auc, _PairIndex
from lfchain.dp import sigma_for

H = json.loads((RESULTS / "hparams_frozen.json").read_text())
EPS_GRID = [0.5, 1, 2, 4, 8]
DELTA = 1e-5


def fit_dp_central_noise(parts, n_items, eps, delta, rounds, clip, clip_count, lr, seed,
                         local_steps=1, lam=1.0):
    """DP-FED variant assuming secure aggregation: noise std sigma*clip on the SUM (once per
    round), not sigma*clip per institution. Same zCDP budget for the sum release."""
    rng = np.random.default_rng(seed)
    sigma = sigma_for(eps, delta, rounds + 1)
    idx = [_PairIndex(r) for r in parts]
    K = len(parts)
    n_noisy = np.zeros(n_items)
    c_noisy = np.zeros(n_items)
    for r, ix in zip(parts, idx):
        n_pair = ix.pair_count
        c_pair = np.bincount(ix.inv, r.y, ix.n_pairs)
        norm = np.sqrt(np.bincount(ix.pu, n_pair ** 2 + c_pair ** 2, ix.n_users))
        scale = np.minimum(1.0, clip_count / np.maximum(norm, 1e-12))[ix.pu]
        n_noisy += np.bincount(ix.pi, n_pair * scale, n_items)
        c_noisy += np.bincount(ix.pi, c_pair * scale, n_items)
    n_noisy += rng.normal(0.0, sigma * clip_count, n_items)      # one noise draw on the sum
    c_noisy += rng.normal(0.0, sigma * clip_count, n_items)
    n_hat = np.maximum(n_noisy, 0.0)
    c_hat = np.clip(c_noisy, 0.0, n_hat)
    m = max(2.0, sigma * clip_count)                              # no sqrt(K) factor
    p0 = (c_hat + m / 2) / (n_hat + m)
    b = -np.log(p0 / (1 - p0))
    tau = sigma * clip
    precond = 0.25 * n_hat + lam + tau
    thetas = [np.zeros(r.n_users) for r in parts]
    t0 = time.perf_counter()
    for t in range(rounds):
        G = np.zeros(n_items)
        for k, (r, ix) in enumerate(zip(parts, idx)):
            thetas[k] = theta_newton(r, b, thetas[k], local_steps, lam)
            p = sigmoid(thetas[k][r.u] - b[r.i])
            G += ix.clipped_item_sum(p - r.y, clip)
        G += rng.normal(0.0, sigma * clip, n_items)               # one noise draw on the sum
        b = b + lr * (G - lam * b) / precond
    thetas = [theta_newton(r, b, th, 5, lam) for r, th in zip(parts, thetas)]
    return b, thetas, time.perf_counter() - t0, sigma


def evaluate(d, b, thetas, ref_b, seen):
    """Same evaluation as the frozen runs: abilities were fitted on each student's training responses."""
    ys, ps = [], []
    for k, r in enumerate(d.test):
        ys.append(r.y); ps.append(predict(r, b, thetas[k]))
    y, p = np.concatenate(ys), np.concatenate(ps)
    e = (b - b.mean()) - (ref_b - ref_b.mean())
    return {"auc": auc(y, p), "b_rmse": float(np.sqrt(np.mean(e[seen] ** 2))),
            "b_spearman": float(spearmanr(b[seen], ref_b[seen]).statistic)}


def run(d, seed, log, out):
    K = len(d.train)
    seen = np.zeros(d.n_items, bool)
    for r in d.train:
        seen[np.unique(r.i)] = True
    ref = d.true_b if d.true_b is not None else cached_central(d, seed)
    hp = H["DP-FED"]
    for eps in EPS_GRID:
        for name, fn in (("DP-FED", None), ("DP-FED-SECAGG", fit_dp_central_noise)):
            if fn is None:
                from lfchain.rasch import fit_dp_federated
                b, ths, dlog, sigma = fit_dp_federated(
                    d.train, d.n_items, eps=eps, delta=DELTA, rounds=hp["rounds"],
                    clip=hp["clip"], clip_count=hp["clip_count"], lr=hp["lr"], seed=seed + 7919)
                sec = dlog.seconds
            else:
                b, ths, sec, sigma = fn(d.train, d.n_items, eps=eps, delta=DELTA, rounds=hp["rounds"],
                                        clip=hp["clip"], clip_count=hp["clip_count"], lr=hp["lr"],
                                        seed=seed + 7919)
            row = {"dataset": d.name, "seed": seed, "K": K, "method": name, "eps": eps,
                   "sigma": sigma, "seconds": sec, **evaluate(d, b, ths, ref, seen)}
            out.write(json.dumps(row, default=float) + "\n"); out.flush()
            log(f"  {d.name} s={seed} {name:16s} eps={eps} auc={row['auc']:.4f} "
                f"b_rmse={row['b_rmse']:.4f} rho={row['b_spearman']:.3f}")


def main(which):
    outdir = RESULTS / "expE"
    if not outdir.exists():
        outdir.mkdir(parents=True)
    out = open(outdir / f"{which}.jsonl", "w")
    logf = open(outdir / f"{which}.log", "w")

    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    for seed in SEEDS:
        ds = [make_synthetic(K, iid, seed) for K in (3, 10) for iid in (True, False)] \
            if which == "syn" else [load_junyi(seed)]
        for d in ds:
            log(f"[{time.strftime('%H:%M:%S')}] {d.name} seed={seed}")
            run(d, seed, log, out)
    log("done")


if __name__ == "__main__":
    main(sys.argv[1])
