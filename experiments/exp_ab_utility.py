"""Experiments A (utility) and B (privacy–utility), protocol §2 and §4.

usage: python exp_ab_utility.py {syn|assist09|junyi}
Writes one JSON row per (dataset, seed, method[, eps]) to results/expAB/<dataset>.jsonl
"""
import json, sys, time
import numpy as np
from scipy.stats import spearmanr

from common import *
sys.path.insert(0, str(ROOT / "src"))
from lfchain.data.datasets import load_assist09, load_junyi, make_synthetic
from lfchain.rasch import (Responses, fit_central, fit_counts, fit_federated, fit_dp_federated, theta_newton,
                           predict, auc, brier, logloss)

H = json.loads((RESULTS / "hparams_frozen.json").read_text())
EPS_GRID = [0.5, 1, 2, 4, 8]


def pooled(parts, n_users, n_items):
    return Responses(np.concatenate([r.u for r in parts]), np.concatenate([r.i for r in parts]),
                     np.concatenate([r.y for r in parts]), n_users, n_items)


def evaluate(d, b_per_org, theta_per_org, ref_b, seen_mask):
    """b_per_org: list of K b-vectors (same vector repeated for shared-b methods)."""
    ys, ps, per = [], [], []
    for k, r in enumerate(d.test):
        p = predict(r, b_per_org[k], theta_per_org[k])
        ys.append(r.y); ps.append(p); per.append(auc(r.y, p))
    y, p = np.concatenate(ys), np.concatenate(ps)
    out = {"auc": auc(y, p), "brier": brier(y, p), "logloss": logloss(y, p),
           "auc_per_org": per, "auc_org_min": float(np.nanmin(per)), "auc_org_sd": float(np.nanstd(per))}
    rm, ma, sp = [], [], []
    for b in b_per_org if len({id(x) for x in b_per_org}) > 1 else b_per_org[:1]:
        r_, m_ = centred_rmse_mae(b, ref_b, seen_mask)
        rm.append(r_); ma.append(m_); sp.append(spearmanr(b[seen_mask], ref_b[seen_mask]).statistic)
    out.update({"b_rmse": float(np.mean(rm)), "b_mae": float(np.mean(ma)), "b_spearman": float(np.mean(sp))})
    if "difficulty" in d.item_meta:
        lab = d.item_meta["difficulty"].map({"easy": 0, "normal": 1, "hard": 2}).to_numpy()
        ok = ~np.isnan(lab.astype(float)) & seen_mask
        out["expert_spearman"] = float(np.mean([spearmanr(b[ok], lab[ok]).statistic
                                                 for b in (b_per_org if len({id(x) for x in b_per_org}) > 1 else b_per_org[:1])]))
        out["expert_n_items"] = int(ok.sum())
    return out


def run_one(d, seed, log):
    K = len(d.train)
    seen = np.zeros(d.n_items, bool)
    for r in d.train:
        seen[np.unique(r.i)] = True
    rows = []
    base = {"dataset": d.name, "seed": seed, "K": K, "n_users": d.n_users, "n_items": d.n_items,
            "n_train": int(sum(len(r.y) for r in d.train)), "n_test": int(sum(len(r.y) for r in d.test))}

    t = time.perf_counter()
    b_c, th_c, it = fit_central(pooled(d.train, d.n_users, d.n_items), iters=H["FED"]["max_rounds"], tol=H["FED"]["tol"])
    sec_c = time.perf_counter() - t
    ref = d.true_b if d.true_b is not None else b_c
    ref_name = "truth" if d.true_b is not None else "CEN"
    base["reference"] = ref_name

    def add(method, bs, ths, **extra):
        row = {**base, "method": method, **evaluate(d, bs, ths, ref, seen), **extra}
        rows.append(row)
        log(f"  {d.name} s={seed} {method:8s} {extra.get('eps', '')} auc={row['auc']:.4f} b_rmse={row['b_rmse']:.4f} "
            f"rho={row['b_spearman']:.3f}" + (f" expert={row['expert_spearman']:.3f}" if 'expert_spearman' in row else ""))

    add("CEN", [b_c] * K, [th_c] * K, seconds=sec_c, rounds=it)

    t = time.perf_counter()
    b_loc, th_loc = [], []
    for r in d.train:
        b_k, th_k, _ = fit_central(r, iters=H["FED"]["max_rounds"], tol=H["FED"]["tol"])
        b_loc.append(b_k); th_loc.append(th_k)
    add("LOC", b_loc, th_loc, seconds=time.perf_counter() - t)

    t = time.perf_counter()
    b_cnt = fit_counts(d.train, d.n_items)
    th_cnt = [theta_newton(r, b_cnt, np.zeros(d.n_users), 10) for r in d.train]
    add("CNT", [b_cnt] * K, th_cnt, seconds=time.perf_counter() - t, bytes_up=2 * d.n_items * 8 * K)

    b_f, th_f, flog = fit_federated(d.train, d.n_items, rounds=H["FED"]["max_rounds"], local_steps=H["FED"]["local_steps"],
                                    damping=H["FED"]["damping"], tol=H["FED"]["tol"])
    add("FED", [b_f] * K, th_f, seconds=flog.seconds, rounds=flog.rounds, bytes_up=flog.bytes_up,
        max_abs_diff_vs_CEN=float(np.max(np.abs(b_f - b_c))))

    hp = H["DP-FED"]
    for eps in EPS_GRID:
        for name, T in (("DP-FED", hp["rounds"]), ("DP-CNT", 0)):
            b_d, th_d, dlog, sigma = fit_dp_federated(d.train, d.n_items, eps=eps, delta=DELTA, rounds=T, clip=hp["clip"],
                                                      clip_count=hp["clip_count"], lr=hp["lr"], seed=seed + 7919)
            add(name, [b_d] * K, th_d, eps=eps, delta=DELTA, sigma=sigma, rounds=T, seconds=dlog.seconds,
                bytes_up=dlog.bytes_up + 2 * d.n_items * 8 * K)
    return rows


def main(which):
    out = RESULTS / "expAB" / f"{which}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out, "w")
    logf = open(RESULTS / "expAB" / f"{which}.log", "w")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    for seed in SEEDS:
        if which == "syn":
            ds = [make_synthetic(K, iid, seed) for K in (3, 10) for iid in (True, False)]
        elif which == "assist09":
            ds = [load_assist09(seed)]
        else:
            ds = [load_junyi(seed)]
        for d in ds:
            log(f"[{time.strftime('%H:%M:%S')}] {d.name} seed={seed} users={d.n_users} items={d.n_items} orgs={d.orgs}")
            for row in run_one(d, seed, log):
                fh.write(json.dumps(row, default=float) + "\n"); fh.flush()
    log("done")


if __name__ == "__main__":
    main(sys.argv[1])
