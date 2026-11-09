"""Exploratory ablation of DP-FED v2 components (NOT in the frozen protocol; reported as exploratory).

Variants: full v2; no noise-aware pseudo-count (m = 2); no noise damping (tau = 0); neither.
Frozen hyper-parameters, eps in {1, 2, 8}, all seeds. usage: python exp_b_ablation.py {syn|junyi|assist09}
"""
import json, sys, time
import numpy as np
from scipy.stats import spearmanr
from common import *
sys.path.insert(0, str(ROOT / "src"))
from lfchain.data.datasets import load_assist09, load_junyi, make_synthetic
from lfchain.rasch import Responses, fit_central, fit_dp_federated, predict, auc

H = json.loads((RESULTS / "hparams_frozen.json").read_text())["DP-FED"]
VARIANTS = {"full": (True, True), "no_pseudo": (False, True), "no_damping": (True, False), "neither": (False, False)}


def main(which):
    out = RESULTS / "expB_ablation" / f"{which}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out, "w")
    for seed in SEEDS:
        d = {"syn": lambda: make_synthetic(10, False, seed), "junyi": lambda: load_junyi(seed),
             "assist09": lambda: load_assist09(seed)}[which]()
        if d.true_b is not None:
            ref = d.true_b
        else:
            pooled = Responses(np.concatenate([r.u for r in d.train]), np.concatenate([r.i for r in d.train]),
                               np.concatenate([r.y for r in d.train]), d.n_users, d.n_items)
            ref, _, _ = fit_central(pooled, iters=500, tol=1e-6)
        seen = np.zeros(d.n_items, bool)
        for r in d.train:
            seen[np.unique(r.i)] = True
        for eps in (1, 2, 8):
            for name, (pc, dmp) in VARIANTS.items():
                b, th, _, sigma = fit_dp_federated(d.train, d.n_items, eps=eps, delta=DELTA, rounds=H["rounds"], clip=H["clip"],
                                                   clip_count=H["clip_count"], lr=H["lr"], seed=seed + 7919,
                                                   pseudo_count=pc, damping=dmp)
                ys = np.concatenate([r.y for r in d.test])
                ps = np.concatenate([predict(r, b, t) for r, t in zip(d.test, th)])
                rmse, _ = centred_rmse_mae(b, ref, seen)
                row = {"dataset": d.name, "seed": seed, "eps": eps, "variant": name, "auc": auc(ys, ps),
                       "b_rmse": rmse, "b_spearman": float(spearmanr(b[seen], ref[seen]).statistic),
                       "finite": bool(np.all(np.isfinite(b)))}
                fh.write(json.dumps(row) + "\n"); fh.flush()
                print(f"[{time.strftime('%H:%M:%S')}] {d.name} s={seed} eps={eps} {name:10s} auc={row['auc']:.4f} rmse={rmse:.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
