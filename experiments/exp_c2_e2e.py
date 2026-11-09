"""Experiment C2-E2E: mid-task revocation end to end on the deployed Fabric network.

A DP-FED task (synthetic non-IID data, frozen hyper-parameters, eps = 2) runs with every round gated by endorsed
AdmitRound transactions. One grant is revoked during the task. Afterwards the admission log is anchored as a Merkle
batch and an auditor, using only the ledger and the off-chain log, verifies each record's proof and replays each
decision against the ledger state at that record's block.

Requires the test-network with learnflow_audit and fabric/gateway/server.js.
usage: python exp_c2_e2e.py [--timestamp-test]
"""
import hashlib, json, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import numpy as np

from common import *
sys.path.insert(0, str(ROOT / "src"))
from lfchain import merkle
from lfchain.data.datasets import make_synthetic
from lfchain.dp import BudgetLedger
from lfchain.ledger import FabricLedger, canonical
from lfchain.policy import Grant, PolicyState, Principal, Request, Task, decide
from lfchain.rasch import fit_dp_federated, fit_federated

H = json.loads((RESULTS / "hparams_frozen.json").read_text())["DP-FED"]
L = {"org1": FabricLedger(org="org1"), "org2": FabricLedger(org="org2")}
PURPOSE = "difficulty_calibration"
EPS = 2.0


def setup(run, d, stamp):
    K = len(d.train)
    orgs = [f"{run}-S{k}" for k in range(K)]
    msp = {o: ("org1" if k % 2 == 0 else "org2") for k, o in enumerate(orgs)}
    task_id = f"{run}-T"
    with ThreadPoolExecutor(K) as ex:
        list(ex.map(lambda o: L[msp[o]].submit("RegisterInstitution", o), orgs))

    def register(k):
        r = d.train[k]
        salt = merkle.new_leaf_salt()
        records = np.stack([r.u, r.i, r.y.astype(np.int64)], axis=1).astype(np.int64)
        tree = merkle.MerkleTree.from_leaves((rec.tobytes() for rec in records), salt=salt)
        L[msp[orgs[k]]].register_dataset_version(f"{orgs[k]}-dsv1", orgs[k], "v1", tree.root.hex(), len(records))
    with ThreadPoolExecutor(K) as ex:
        list(ex.map(register, range(K)))
    L["org1"].create_task({"task_id": task_id, "purpose": PURPOSE, "participants": orgs, "output_type": "aggregate",
                           "state": "draft", "algorithm_version": "dp-fed-v2"})
    L["org1"].set_task_state(task_id, "approved")

    def grant(k):
        g = {"grant_id": f"{run}-G{k}", "org_id": orgs[k], "resource_id": f"{orgs[k]}-dsv1", "task_id": task_id,
             "purpose": PURPOSE, "expires_at": stamp + 86400, "revoked_at": None, "granted_by": f"{orgs[k]}-admin"}
        L[msp[orgs[k]]].grant_access(g)
    with ThreadPoolExecutor(K) as ex:
        list(ex.map(grant, range(K)))
    L["org1"].set_task_state(task_id, "running")
    return orgs, msp, task_id


def admit(org, msp, task_id, round_):
    req = asdict(Request(Principal(f"{org}-admin", org, "data_admin"), "run_round", task_id, f"{org}-dsv1", org, "aggregate", 0.0))
    t0 = time.perf_counter()
    try:
        r = L[msp[org]].submit("AdmitRound", json.dumps(req))
        dec = json.loads(r["result"])
        rec = {"round": round_, "org": org, "tx_id": r["tx_id"], "block": r["block"], "allow": dec["allow"], "reason": dec["reason"],
               "grant_id": dec["grant_id"], "tau": dec["tau"], "committed": True}
    except RuntimeError as e:                 # endorsement refused or transaction invalidated: fail closed
        rec = {"round": round_, "org": org, "tx_id": None, "block": None, "allow": False, "reason": "NOT_COMMITTED",
               "grant_id": None, "tau": None, "committed": False, "error": str(e)[:160]}
    rec["latency_s"] = time.perf_counter() - t0
    return rec


def run_task(seed, K, T, revoke_round, race, stamp):
    run = f"e2e{stamp}k{K}r{revoke_round}{'race' if race else ''}s{seed}"
    d = make_synthetic(K, False, seed)
    t0 = time.perf_counter()
    orgs, msp, task_id = setup(run, d, stamp)
    setup_s = time.perf_counter() - t0
    rng = np.random.default_rng(seed)
    victim = int(rng.integers(K))
    active = set(range(K))
    log, trace, gate_times = [], {}, []
    revoke = {}
    budget = BudgetLedger(cap_eps=EPS, delta=DELTA)

    def do_revoke():
        t = time.perf_counter()
        r = L[msp[orgs[victim]]].revoke_access(f"{run}-G{victim}", time.time())
        revoke.update({"tx_id": r["tx_id"], "block": r["block"], "commit_s": time.perf_counter() - t, "round": revoke_round})

    def gate(t, k):
        if k == 0:
            g0 = time.perf_counter()
            if t == revoke_round and not race:
                do_revoke()
            with ThreadPoolExecutor(K + 1) as ex:
                fr = ex.submit(do_revoke) if t == revoke_round and race else None
                recs = list(ex.map(lambda j: admit(orgs[j], msp, task_id, t), sorted(active)))
                if fr:
                    fr.result()
            log.extend(recs)
            trace[t] = {orgs.index(r["org"]) for r in recs if r["allow"]}
            active.intersection_update(trace[t])       # a denied client stops asking
            gate_times.append(time.perf_counter() - g0)
        return k in trace[t]

    t1 = time.perf_counter()
    b, th, flog, sigma = fit_dp_federated(d.train, d.n_items, eps=EPS, delta=DELTA, rounds=H["rounds"] if T is None else T, clip=H["clip"],
                                          clip_count=H["clip_count"], lr=H["lr"], seed=seed + 7919, gate=gate, ledger=budget,
                                          task_id=task_id, org_ids=orgs)
    task_s = time.perf_counter() - t1
    rounds = flog.rounds
    t2 = time.perf_counter()
    b_off, _, _, _ = fit_dp_federated(d.train, d.n_items, eps=EPS, delta=DELTA, rounds=rounds, clip=H["clip"], clip_count=H["clip_count"],
                                      lr=H["lr"], seed=seed + 7919)
    offline_s = time.perf_counter() - t2
    rmse = lambda x: centred_rmse_mae(x, d.true_b)[0]
    # Non-private FED with and without the recorded admission trace: the effect of the exclusion without noise.
    f_gated, _, _ = fit_federated(d.train, d.n_items, rounds=rounds, tol=0, gate=lambda t, k: k in trace[t])
    f_all, _, _ = fit_federated(d.train, d.n_items, rounds=rounds, tol=0)
    # finish the task and anchor the admission log
    L["org1"].set_task_state(task_id, "completed")
    digest = hashlib.sha256(b.tobytes()).hexdigest()
    L[msp[orgs[0]]].record_calibration(task_id, digest, [f"{o}-dsv1" for o in orgs], "dp-fed-v2")
    committed = [r for r in log if r["committed"]]
    salt = merkle.new_leaf_salt()
    audit_keys = ("round", "org", "tx_id", "block", "allow", "reason", "grant_id", "tau")
    leaves = [canonical({k: r[k] for k in audit_keys}) for r in committed]
    tree = merkle.MerkleTree.from_leaves(leaves, salt=salt)
    batch_id = f"{run}-AUDIT"
    L["org1"].anchor_batch(batch_id, tree.root.hex(), len(leaves), "admission_log")
    audit = replay(task_id, batch_id, tree, salt, committed)
    excluded_from = min((t for t in trace if victim not in trace[t]), default=None)
    latencies = [r["latency_s"] for r in committed]
    before = [x for t, x in enumerate(gate_times) if t < revoke_round]
    after = [x for t, x in enumerate(gate_times) if t > revoke_round]
    return {"run": run, "seed": seed, "K": K, "rounds": rounds, "race": race, "victim": orgs[victim],
            "revoke_round": revoke_round, "revocation_commit_s": revoke["commit_s"], "revocation_block": revoke["block"],
            "excluded_from_round": excluded_from, "rounds_to_effect": None if excluded_from is None else excluded_from - revoke_round,
            "victim_participations": sum(victim in trace[t] for t in trace),
            "victim_round_outcome_at_revocation": next(({"allow": r["allow"], "reason": r["reason"], "committed": r["committed"], "block": r["block"]}
                                                        for r in log if r["round"] == revoke_round and r["org"] == orgs[victim]), None),
            "n_admission_records": len(log), "n_committed_records": len(committed), "n_denied": sum(not r["allow"] for r in log),
            "n_uncommitted": sum(not r["committed"] for r in log),
            "anchored_batch_records": len(leaves), "audit": audit,
            "setup_s": setup_s, "task_s": task_s, "offline_s": offline_s, "gate_total_s": sum(gate_times),
            "gate_per_round_s_mean": float(np.mean(gate_times)), "gate_per_round_s_before": float(np.mean(before)) if before else None,
            "gate_per_round_s_after": float(np.mean(after)) if after else None,
            "admission_latency_s": {"mean": float(np.mean(latencies)), "p50": float(np.median(latencies)), "p95": float(np.percentile(latencies, 95))},
            "dp_rmse_uninterrupted": rmse(b_off), "dp_rmse_with_revocation": rmse(b), "dp_max_abs_change": float(np.max(np.abs(b - b_off))),
            "fed_rmse_uninterrupted": rmse(f_all), "fed_rmse_with_revocation": rmse(f_gated), "fed_max_abs_change": float(np.max(np.abs(f_gated - f_all))),
            "sigma": sigma, "budget_records": len(budget.records), "uptime": subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()}


def tx_block(cache, tx_id):
    if tx_id not in cache:
        cache[tx_id] = L["org1"]._post("/txinfo", {"org": "org1", "tx_id": tx_id})
    return cache[tx_id]


def replay(task_id, batch_id, tree, salt, committed):
    """Auditor: verify every record against the anchored root, then recompute its decision from ledger history alone."""
    cache = {}
    proofs_ok = chain_ok = meta_ok = decisions_ok = 0
    audit_keys = ("round", "org", "tx_id", "block", "allow", "reason", "grant_id", "tau")
    root = L["org2"].evaluate("GetAnchor", batch_id)["root"]
    assert root == tree.root.hex()
    task_hist = L["org2"].evaluate("QueryAuditHistory", f"task:{task_id}")
    grant_hist = {}
    for j, r in enumerate(committed):
        leaf = merkle.leaf_hash(canonical({k: r[k] for k in audit_keys}), salt=salt)
        res = L["org2"].evaluate("VerifyAuditProof", batch_id, leaf.hex(), json.dumps(merkle.proof_to_json(tree.proof(j))))
        proofs_ok += int(res["match"])
        info = tx_block(cache, r["tx_id"])
        chain_ok += int(info["validation_code"] == 0)
        meta_ok += int(info["block"] == r["block"] and abs(info["timestamp"] - r["tau"]) < 1e-3)
        gid = r["grant_id"]
        if gid is None:
            continue
        if gid not in grant_hist:
            grant_hist[gid] = L["org2"].evaluate("QueryAuditHistory", f"grant:{gid}")
        # ledger state as of the block before this admission's block
        task_v = latest_before(task_hist, r["block"], cache)
        grant_v = latest_before(grant_hist[gid], r["block"], cache)
        st = PolicyState()
        st.tasks[task_id] = Task(task_id, task_v["purpose"], task_v["state"], tuple(task_v["participants"]), task_v["output_type"])
        st.grants[gid] = Grant(gid, grant_v["org_id"], grant_v["resource_id"], grant_v["task_id"], grant_v["purpose"], grant_v["expires_at"], grant_v["revoked_at"])
        req = Request(Principal(f"{r['org']}-admin", r["org"], "data_admin"), "run_round", task_id, f"{r['org']}-dsv1", r["org"], "aggregate", r["tau"])
        d = decide(req, st)
        decisions_ok += int(d.allow == r["allow"] and d.reason == r["reason"])
    return {"records": len(committed), "proofs_verified": proofs_ok, "transactions_valid": chain_ok, "block_and_time_match": meta_ok,
            "decisions_reproduced": decisions_ok}


def latest_before(history, block, cache):
    best = None
    for h in history:                       # newest first
        info = tx_block(cache, h["tx_id"])
        if info["block"] < block and (best is None or info["block"] > best[0]):
            best = (info["block"], h["value"])
    return best[1]


def timestamp_test(stamp):
    """Forged client timestamps against the endorsers' clock check; a valid grant, so a refusal can only be the check."""
    run = f"ts{stamp}"
    org = f"{run}-S0"
    L["org1"].submit("RegisterInstitution", org)
    task_id = f"{run}-T"
    L["org1"].create_task({"task_id": task_id, "purpose": PURPOSE, "participants": [org], "output_type": "aggregate", "state": "draft"})
    L["org1"].set_task_state(task_id, "approved")
    L["org1"].grant_access({"grant_id": f"{run}-G", "org_id": org, "resource_id": f"{org}-dsv1", "task_id": task_id, "purpose": PURPOSE,
                            "expires_at": stamp + 86400, "revoked_at": None})
    L["org1"].set_task_state(task_id, "running")
    req = asdict(Request(Principal("adm", org, "data_admin"), "run_round", task_id, f"{org}-dsv1", org, "aggregate", 0.0))
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(req, f)
    out = RESULTS / "expC" / "tx_timestamp_test.json"
    subprocess.run(["node", "test_tx_timestamp.js", f.name, str(out)], cwd=ROOT / "fabric" / "gateway", check=True, capture_output=True)
    rows = json.loads(out.read_text())
    write_json("expC/tx_timestamp_test.json", {"tolerance_s": 30, "rows": rows})
    print(json.dumps(rows, indent=0))


def main():
    stamp = int(time.time())
    if "--timestamp-test" in sys.argv:
        timestamp_test(stamp)
        return
    runs = []
    plan = [(MASTER, 10, 3, False), (MASTER + 1, 10, 7, False), (MASTER + 2, 10, 11, False), (MASTER + 3, 10, 15, False),
            (MASTER + 4, 10, 9, True), (MASTER, 3, 7, False), (MASTER + 1, 3, 11, False)]
    for seed, K, rr, race in plan:
        r = run_task(seed, K, None, rr, race, stamp)
        runs.append(r)
        print(json.dumps({k: r[k] for k in ("run", "revocation_commit_s", "excluded_from_round", "rounds_to_effect", "victim_participations",
                                            "n_admission_records", "n_uncommitted", "gate_per_round_s_before", "gate_per_round_s_after",
                                            "gate_total_s", "task_s", "offline_s", "dp_rmse_uninterrupted", "dp_rmse_with_revocation",
                                            "fed_rmse_uninterrupted", "fed_rmse_with_revocation", "audit", "victim_round_outcome_at_revocation", "uptime")}, default=float), flush=True)
        write_json("expC/c2_e2e.json", {"fabric": "2.5.14 test-network, 2 orgs, Raft, BatchTimeout 2s, MaxMessageCount 10", "runs": runs})


if __name__ == "__main__":
    main()
