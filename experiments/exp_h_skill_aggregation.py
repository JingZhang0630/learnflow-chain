"""Experiment H (exploratory): does calibrating per skill rescue a small consortium under differential privacy?

Same ASSISTments responses, students and train/test split as the frozen item-level runs; the unit of calibration is
the skill. Frozen DP-FED hyper-parameters. Reference for difficulty agreement: skill-level CEN.

usage: python exp_h_skill_aggregation.py
"""
import json, sys
import numpy as np
from scipy.stats import spearmanr

from common import *
sys.path.insert(0, str(ROOT / "src"))
from lfchain.data.datasets import load_assist09
from lfchain.rasch import Responses, fit_central, fit_counts, fit_federated, fit_dp_federated, theta_newton, predict, auc

H = json.loads((RESULTS / "hparams_frozen.json").read_text())


def score(d, b, thetas, ref, seen):
    ys = np.concatenate([r.y for r in d.test])
    ps = np.concatenate([predict(r, b, t) for r, t in zip(d.test, thetas)])
    rmse, _ = centred_rmse_mae(b, ref, seen)
    return {"auc": auc(ys, ps), "b_rmse": rmse, "b_spearman": float(spearmanr(b[seen], ref[seen]).statistic)}


def main():
    (RESULTS / "expH").mkdir(exist_ok=True)
    out = open(RESULTS / "expH" / "assist09_skill.jsonl", "w")
    hp = H["DP-FED"]
    for seed in SEEDS:
        d = load_assist09(seed, level="skill")
        item_view = load_assist09(seed)
        assert sum(len(r.y) for r in d.test) == sum(len(r.y) for r in item_view.test), "split must match the item-level runs"
        seen = np.zeros(d.n_items, bool)
        for r in d.train:
            seen[np.unique(r.i)] = True
        pooled = Responses(np.concatenate([r.u for r in d.train]), np.concatenate([r.i for r in d.train]),
                           np.concatenate([r.y for r in d.train]), d.n_users, d.n_items)
        b_cen, th_cen, _ = fit_central(pooled, iters=500, tol=1e-6)
        base = {"seed": seed, "n_skills": d.n_items, "n_students": d.n_users, "resp_per_skill": sum(len(r.y) for r in d.train) / d.n_items}
        rows = [{**base, "method": "CEN", **score(d, b_cen, [th_cen] * len(d.test), b_cen, seen)}]
        b_cnt = fit_counts(d.train, d.n_items)
        th_cnt = [theta_newton(r, b_cnt, np.zeros(d.n_users), 10) for r in d.train]
        rows.append({**base, "method": "CNT", **score(d, b_cnt, th_cnt, b_cen, seen)})
        for eps in (0.5, 1, 2, 4, 8):
            b, th, _, _ = fit_dp_federated(d.train, d.n_items, eps=eps, delta=DELTA, rounds=hp["rounds"], clip=hp["clip"],
                                           clip_count=hp["clip_count"], lr=hp["lr"], seed=seed + 7919)
            rows.append({**base, "method": "DP-FED", "eps": eps, **score(d, b, th, b_cen, seen)})
        for r in rows:
            out.write(json.dumps(r, default=float) + "\n"); out.flush()
            print(seed, r["method"], r.get("eps", ""), f"auc={r['auc']:.4f} rmse={r['b_rmse']:.3f} rho={r['b_spearman']:.3f}", flush=True)


if __name__ == "__main__":
    main()
