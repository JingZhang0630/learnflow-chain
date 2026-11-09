"""Protocol §3: select FED / DP-FED hyper-parameters on D3 non-IID K=10, seed 20261018 only, at eps=2."""
import itertools, json, sys
from common import *
sys.path.insert(0, str(ROOT / "src"))
from lfchain.data.datasets import make_synthetic
from lfchain.rasch import fit_federated, fit_dp_federated

d = make_synthetic(K=10, iid=False, seed=MASTER)
rows = []
for ls in (1, 3):
    b, _, log = fit_federated(d.train, d.n_items, rounds=500, local_steps=ls, tol=1e-6)
    rmse, mae = centred_rmse_mae(b, d.true_b)
    rows.append({"method": "FED", "local_steps": ls, "rounds_used": log.rounds, "rmse": rmse, "mae": mae})
fed = min([r for r in rows if r["method"] == "FED"], key=lambda r: (round(r["rmse"], 6), r["rounds_used"]))
for T, C, Cn, lr in itertools.product((0, 5, 10, 20, 50), (0.5, 1, 2, 4), (4, 8, 16), (0.5, 1.0)):
    b, _, log, sigma = fit_dp_federated(d.train, d.n_items, eps=2.0, delta=DELTA, rounds=T, clip=C, clip_count=Cn,
                                        lr=lr, seed=MASTER)
    rmse, mae = centred_rmse_mae(b, d.true_b)
    rows.append({"method": "DP-FED", "T": T, "clip": C, "clip_count": Cn, "lr": lr, "sigma": sigma, "rmse": rmse, "mae": mae})
dpf = min([r for r in rows if r["method"] == "DP-FED"], key=lambda r: r["rmse"])
frozen = {"selection_data": "SYN-K10-nonIID seed 20261018", "criterion": "lowest b-RMSE at eps=2 (DP-FED)",
          "FED": {"local_steps": fed["local_steps"], "max_rounds": 500, "tol": 1e-6, "damping": 1.0},
          "DP-FED": {"rounds": dpf["T"], "clip": dpf["clip"], "clip_count": dpf["clip_count"], "lr": dpf["lr"], "local_steps": 1},
          "lambda": 1.0, "delta": DELTA}
write_json("hparam_search.json", rows)
write_json("hparams_frozen.json", frozen)
print(json.dumps(frozen, indent=1))
print("top-5 DP-FED:", sorted([r for r in rows if r["method"] == "DP-FED"], key=lambda r: r["rmse"])[:5])
