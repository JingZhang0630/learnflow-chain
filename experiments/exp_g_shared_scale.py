"""Experiment G (exploratory): where does the shared difficulty scale help?

syn        Synthetic non-IID K=10, leave-one-institution-out: difficulties fitted on the other K-1 institutions,
           abilities on the held-out institution's own training responses, prediction of its test responses.
syn_unseen Synthetic non-IID K=10: test responses on items an institution never saw in training. No such items occur
           naturally in this design (each institution's training responses cover its whole item window), so a random
           20% of an institution's items are withheld from that institution's training only.
junyi_loo  Junyi, leave-one-city-out, all seeds: CNT and LOC columns. Its FED column is warm-started from the all-city fit,
           which contains the held-out city, and reaches the round cap; the paper does not use it.
junyi_loo_cold  Junyi, leave-one-city-out: FED started from zeros, using nothing of the held-out city, stopping at 1e-4.
           An optional second argument gives seeds, so that seeds can run in parallel; rows are appended.
tests      Paired seed-level tests on the frozen Junyi runs.

usage: python exp_g_shared_scale.py {syn|syn_unseen|junyi_loo|tests}
"""
import json, sys, time
import numpy as np
from scipy.stats import wilcoxon

from common import *
sys.path.insert(0, str(ROOT / "src"))
from lfchain.data.datasets import load_junyi, make_synthetic
from lfchain.rasch import Responses, fit_central, fit_counts, fit_federated, theta_newton, predict, auc

H = json.loads((RESULTS / "hparams_frozen.json").read_text())
MAXR = H["FED"]["max_rounds"]; TOL = H["FED"]["tol"]
TOL_LOO = 1e-4   # leave-one-city-out only: at 1e-6 each Junyi fit reaches the 500-round cap (about 10 minutes per city)


def fit_fed(parts, n_items, init_b=None, tol=TOL):
    return fit_federated(parts, n_items, rounds=MAXR, local_steps=H["FED"]["local_steps"],
                         damping=H["FED"]["damping"], tol=tol, init_b=init_b)


def auc_on(r, b, theta, item_mask=None):
    p = predict(r, b, theta)
    y = r.y
    if item_mask is not None:
        keep = item_mask[r.i]
        p, y = p[keep], y[keep]
    return float(auc(y, p)) if len(y) and len(set(y)) > 1 else float("nan")


def abilities(train, b, steps=10):
    return theta_newton(train, b, np.zeros(train.n_users), steps)


def run_syn(seed, out, log):
    d = make_synthetic(10, False, seed)
    for k in range(len(d.train)):
        others = [r for j, r in enumerate(d.train) if j != k]
        b_fed, _, _ = fit_fed(others, d.n_items)
        b_cnt = fit_counts(others, d.n_items)
        b_loc, _, _ = fit_central(d.train[k], iters=MAXR, tol=TOL)
        r = d.test[k]
        row = {"seed": seed, "held_out_org": k,
               "auc_all_fed": auc_on(r, b_fed, abilities(d.train[k], b_fed)),
               "auc_all_cnt": auc_on(r, b_cnt, abilities(d.train[k], b_cnt)),
               "auc_all_loc": auc_on(r, b_loc, abilities(d.train[k], b_loc))}
        out.write(json.dumps(row, default=float) + "\n"); out.flush()
        log(f"  s={seed} org={k} fed={row['auc_all_fed']:.4f} cnt={row['auc_all_cnt']:.4f} loc={row['auc_all_loc']:.4f}")


def run_syn_unseen(seed, out, log):
    d = make_synthetic(10, False, seed)
    rng = np.random.default_rng(seed + 31)
    for k, tr in enumerate(d.train):
        local_items = np.unique(tr.i)
        withheld = rng.choice(local_items, size=int(round(0.2 * len(local_items))), replace=False)
        mask = np.zeros(d.n_items, bool); mask[withheld] = True
        keep = ~mask[tr.i]
        tr_k = Responses(tr.u[keep], tr.i[keep], tr.y[keep], tr.n_users, tr.n_items)
        parts = [tr_k if j == k else r for j, r in enumerate(d.train)]
        seen_elsewhere = np.zeros(d.n_items, bool)
        for j, r in enumerate(d.train):
            if j != k:
                seen_elsewhere[np.unique(r.i)] = True
        b_fed, _, _ = fit_fed(parts, d.n_items)
        b_cnt = fit_counts(parts, d.n_items)
        b_loc, _, _ = fit_central(tr_k, iters=MAXR, tol=TOL)    # withheld items stay at the prior b = 0
        r = d.test[k]
        err = lambda b: float(np.sqrt(np.mean(((b - b.mean()) - (d.true_b - d.true_b.mean()))[withheld] ** 2)))
        row = {"seed": seed, "org": k, "n_withheld_items": int(len(withheld)),
               "n_withheld_seen_elsewhere": int(seen_elsewhere[withheld].sum()),
               "n_test_on_withheld": int(mask[r.i].sum()),
               "auc_unseen_fed": auc_on(r, b_fed, abilities(tr_k, b_fed), mask),
               "auc_unseen_cnt": auc_on(r, b_cnt, abilities(tr_k, b_cnt), mask),
               "auc_unseen_loc": auc_on(r, b_loc, abilities(tr_k, b_loc), mask),
               "rmse_unseen_fed": err(b_fed), "rmse_unseen_cnt": err(b_cnt), "rmse_unseen_loc": err(b_loc)}
        out.write(json.dumps(row, default=float) + "\n"); out.flush()
        log(f"  s={seed} org={k} unseen fed={row['auc_unseen_fed']:.4f} cnt={row['auc_unseen_cnt']:.4f} "
            f"loc={row['auc_unseen_loc']:.4f}")


def run_junyi_loo(seed, out, log):
    d = load_junyi(seed)
    K = len(d.train)
    b_all = cached_central(d, seed)
    for k in range(K):
        t0 = time.perf_counter()
        others = [r for j, r in enumerate(d.train) if j != k]
        b_fed, _, flog = fit_fed(others, d.n_items, init_b=b_all, tol=TOL_LOO)
        b_cnt = fit_counts(others, d.n_items)
        b_loc, _, _ = fit_central(d.train[k], iters=MAXR, tol=TOL_LOO)
        r = d.test[k]
        seen_local = np.zeros(d.n_items, bool); seen_local[np.unique(d.train[k].i)] = True
        row = {"seed": seed, "held_out_org": d.orgs[k], "n_test": int(len(r.y)),
               "n_test_unseen_locally": int((~seen_local[r.i]).sum()), "fed_rounds": flog.rounds,
               "auc_all_fed": auc_on(r, b_fed, abilities(d.train[k], b_fed, 5)),
               "auc_all_cnt": auc_on(r, b_cnt, abilities(d.train[k], b_cnt, 5)),
               "auc_all_loc": auc_on(r, b_loc, abilities(d.train[k], b_loc, 5))}
        out.write(json.dumps(row, default=float) + "\n"); out.flush()
        log(f"[{time.strftime('%H:%M:%S')}] s={seed} held_out={d.orgs[k]:6s} rounds={flog.rounds} {time.perf_counter() - t0:.0f}s "
            f"fed={row['auc_all_fed']:.4f} cnt={row['auc_all_cnt']:.4f} loc={row['auc_all_loc']:.4f}")


def run_junyi_loo_cold(seed, out, log):
    """FED for the leave-one-city-out comparison, from zeros, so that nothing of the held-out city enters the fit."""
    d = load_junyi(seed)
    for k in range(len(d.train)):
        t0 = time.perf_counter()
        others = [r for j, r in enumerate(d.train) if j != k]
        b_fed, _, flog = fit_fed(others, d.n_items, tol=TOL_LOO)
        r = d.test[k]
        row = {"seed": seed, "held_out_org": d.orgs[k], "fed_rounds": flog.rounds,
               "auc_all_fed": auc_on(r, b_fed, abilities(d.train[k], b_fed, 5))}
        out.write(json.dumps(row, default=float) + "\n"); out.flush()
        log(f"[{time.strftime('%H:%M:%S')}] s={seed} held_out={d.orgs[k]:6s} rounds={flog.rounds} {time.perf_counter() - t0:.0f}s fed={row['auc_all_fed']:.4f}")


def seed_tests(log):
    """Paired tests on the frozen Junyi runs (5 seeds, so the smallest attainable two-sided p is 0.0625)."""
    rows = [json.loads(l) for l in (RESULTS / "expAB" / "junyi.jsonl").read_text().splitlines() if l.strip()]
    by = {(r["seed"], r["method"]): r for r in rows if r["method"] in ("FED", "LOC", "CNT")}
    seeds = sorted({s for s, _ in by})
    res = {}
    for m1, m2 in (("FED", "LOC"), ("FED", "CNT")):
        a1 = [by[(s, m1)]["auc"] for s in seeds]; a2 = [by[(s, m2)]["auc"] for s in seeds]
        diffs = np.array(a1) - np.array(a2)
        res[f"{m1}_vs_{m2}"] = {"n_seeds": len(diffs), "mean_diff": float(diffs.mean()), "sd_diff": float(diffs.std(ddof=1)),
                                "wilcoxon_p": float(wilcoxon(a1, a2).pvalue)}
        log(json.dumps(res[f"{m1}_vs_{m2}"]))
    return res


def main(which, seed_arg=None):
    outdir = RESULTS / "expG"
    outdir.mkdir(parents=True, exist_ok=True)
    mode = "a" if which == "junyi_loo_cold" else "w"
    logf = open(outdir / f"{which}.log", mode)

    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    if which == "tests":
        (outdir / "seed_tests.json").write_text(json.dumps(seed_tests(log), indent=2))
        return
    out = open(outdir / f"{which}.jsonl", mode)
    fn = {"syn": run_syn, "syn_unseen": run_syn_unseen, "junyi_loo": run_junyi_loo, "junyi_loo_cold": run_junyi_loo_cold}[which]
    seeds = {"junyi_loo": list(reversed(SEEDS)), "junyi_loo_cold": [int(x) for x in seed_arg.split(",")] if seed_arg else SEEDS}.get(which, SEEDS)   # reversed: shares the CEN cache with E
    for seed in seeds:
        fn(seed, out, log)
    log("done")


if __name__ == "__main__":
    main(*sys.argv[1:3])
