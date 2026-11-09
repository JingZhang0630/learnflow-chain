"""Experiment F (exploratory): clipping sensitivity and observed clip rates.

Part 1 (exact, cheap): fraction of students whose count vector exceeds C_n and whose
gradient contribution exceeds C, on synthetic non-IID K=10 and Junyi.
Part 2 (exploratory): DP-FED difficulty RMSE / AUC at eps=2 under a grid of C and C_n.

usage: python exp_f_clipping.py {syn|junyi}
Writes results/expF/<dataset>.jsonl (sensitivity) and results/expF/clip_rates_<dataset>.json
"""
import json, sys, time
import numpy as np
from scipy.stats import spearmanr

from common import *
sys.path.insert(0, str(ROOT / "src"))
from lfchain.data.datasets import load_junyi, make_synthetic
from lfchain.rasch import theta_newton, sigmoid, _PairIndex, predict, auc
from lfchain.dp import sigma_for

H = json.loads((RESULTS / "hparams_frozen.json").read_text())
DELTA = 1e-5
C_GRID = [0.25, 0.5, 1.0, 2.0]
CN_GRID = [2.0, 4.0, 8.0]


def clip_rates(d):
    """Exact clip fractions. Count vectors are student-level, deterministic.
    Gradient norms are evaluated at the non-private warm start b_cnt with fitted theta
    (deterministic reference point, same for all C)."""
    out = {"dataset": d.name, "n_users": d.n_users}
    cnt_clipped = {}
    for Cn in CN_GRID:
        tot = hit = 0
        for r in d.train:
            ix = _PairIndex(r)
            n_pair = ix.pair_count
            c_pair = np.bincount(ix.inv, r.y, ix.n_pairs)
            norm = np.sqrt(np.bincount(ix.pu, n_pair ** 2 + c_pair ** 2, ix.n_users))
            own = np.bincount(ix.pu, minlength=ix.n_users) > 0      # students of this institution only
            tot += int(own.sum()); hit += int((norm[own] > Cn).sum())
        cnt_clipped[str(Cn)] = hit / tot
    out["count_clip_fraction"] = cnt_clipped
    # gradient norms at b_cnt warm start
    c = np.zeros(d.n_items); n = np.zeros(d.n_items)
    for r in d.train:
        c += np.bincount(r.i, r.y, d.n_items); n += np.bincount(r.i, minlength=d.n_items)
    b0 = -np.log(((c + 1) / (n + 2)) / (1 - (c + 1) / (n + 2)))
    grad_clipped = {}
    for C in C_GRID:
        tot = hit = 0
        for r in d.train:
            ix = _PairIndex(r)
            th = theta_newton(r, b0, np.zeros(r.n_users), 10)
            p = sigmoid(th[r.u] - b0[r.i])
            v = np.bincount(ix.inv, p - r.y, ix.n_pairs)
            norm = np.sqrt(np.bincount(ix.pu, v * v, ix.n_users))
            own = np.bincount(ix.pu, minlength=ix.n_users) > 0
            tot += int(own.sum()); hit += int((norm[own] > C).sum())
        grad_clipped[str(C)] = hit / tot
    out["grad_clip_fraction"] = grad_clipped
    return out


def dp_run_track_clips(d, eps, C, Cn, seed):
    """fit_dp_federated clone that records the per-round fraction of clipped students."""
    from lfchain.rasch import LAMBDA
    rng = np.random.default_rng(seed)
    sigma = sigma_for(eps, DELTA, H["DP-FED"]["rounds"] + 1)
    idx = [_PairIndex(r) for r in d.train]
    K = len(d.train)
    lam = LAMBDA
    n_noisy = np.zeros(d.n_items); c_noisy = np.zeros(d.n_items)
    for r, ix in zip(d.train, idx):
        n_pair = ix.pair_count
        c_pair = np.bincount(ix.inv, r.y, ix.n_pairs)
        norm = np.sqrt(np.bincount(ix.pu, n_pair ** 2 + c_pair ** 2, ix.n_users))
        scale = np.minimum(1.0, Cn / np.maximum(norm, 1e-12))[ix.pu]
        n_noisy += np.bincount(ix.pi, n_pair * scale, d.n_items) + rng.normal(0.0, sigma * Cn, d.n_items)
        c_noisy += np.bincount(ix.pi, c_pair * scale, d.n_items) + rng.normal(0.0, sigma * Cn, d.n_items)
    n_hat = np.maximum(n_noisy, 0.0); c_hat = np.clip(c_noisy, 0.0, n_hat)
    m = max(2.0, np.sqrt(K) * sigma * Cn)
    p0 = (c_hat + m / 2) / (n_hat + m)
    b = -np.log(p0 / (1 - p0))
    tau = np.sqrt(K) * sigma * C
    precond = 0.25 * n_hat + lam + tau
    thetas = [np.zeros(r.n_users) for r in d.train]
    clip_hist = []
    for t in range(H["DP-FED"]["rounds"]):
        G = np.zeros(d.n_items)
        clipped = 0; total = 0
        for k, (r, ix) in enumerate(zip(d.train, idx)):
            thetas[k] = theta_newton(r, b, thetas[k], 1, lam)
            p = sigmoid(thetas[k][r.u] - b[r.i])
            v = np.bincount(ix.inv, p - r.y, ix.n_pairs)
            norm = np.sqrt(np.bincount(ix.pu, v * v, ix.n_users))
            own = np.bincount(ix.pu, minlength=ix.n_users) > 0
            clipped += int((norm[own] > C).sum()); total += int(own.sum())
            G += ix.clipped_item_sum(p - r.y, C) + rng.normal(0.0, sigma * C, d.n_items)
        clip_hist.append(clipped / total)
        b = b + H["DP-FED"]["lr"] * (G - lam * b) / precond
    thetas = [theta_newton(r, b, th, 5, lam) for r, th in zip(d.train, thetas)]
    return b, thetas, float(np.mean(clip_hist))


def sensitivity(d, seed, log, out):
    seen = np.zeros(d.n_items, bool)
    for r in d.train:
        seen[np.unique(r.i)] = True
    ref = d.true_b if d.true_b is not None else cached_central(d, seed)
    for C in C_GRID:
        for Cn in CN_GRID:
            b, thetas, frac = dp_run_track_clips(d, 2.0, C, Cn, seed + 7919)
            ys, ps = [], []
            for k, r in enumerate(d.test):
                ys.append(r.y); ps.append(predict(r, b, thetas[k]))
            y, p = np.concatenate(ys), np.concatenate(ps)
            e = (b - b.mean()) - (ref - ref.mean())
            row = {"dataset": d.name, "seed": seed, "eps": 2.0, "C": C, "Cn": Cn,
                   "grad_clip_fraction": frac,
                   "auc": auc(y, p), "b_rmse": float(np.sqrt(np.mean(e[seen] ** 2))),
                   "b_spearman": float(spearmanr(b[seen], ref[seen]).statistic)}
            out.write(json.dumps(row, default=float) + "\n"); out.flush()
            log(f"  {d.name} s={seed} C={C} Cn={Cn} clip={frac:.3f} auc={row['auc']:.4f} rmse={row['b_rmse']:.4f}")


def main(which):
    outdir = RESULTS / "expF"
    if not outdir.exists():
        outdir.mkdir(parents=True)
    out = open(outdir / f"{which}.jsonl", "w")
    logf = open(outdir / f"{which}.log", "w")

    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    seeds = SEEDS if which == "syn" else SEEDS[:3]
    for seed in seeds:
        ds = [make_synthetic(10, False, seed)] if which == "syn" else [load_junyi(seed)]
        for d in ds:
            log(f"[{time.strftime('%H:%M:%S')}] {d.name} seed={seed}")
            if seed == SEEDS[0]:
                cr = clip_rates(d)
                (outdir / f"clip_rates_{which}.json").write_text(json.dumps(cr, indent=2))
                log(f"  clip rates: {json.dumps(cr)}")
            sensitivity(d, seed, log, out)
    log("done")


if __name__ == "__main__":
    main(sys.argv[1])
