"""Experiment C (protocol §5): C1 backend-vs-chaincode differential test on Fabric, write-side negative tests and
revocation-to-effect latency; C2 enforcement models under mid-task events; C3 tamper detection.

Requires the Fabric test network with learnflow_audit and fabric/gateway/server.js on 127.0.0.1:8800.
usage: python exp_c_access.py {c1|c2|c3|all}
"""
import hashlib, json, os, random, sys, tempfile, time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from common import *
sys.path.insert(0, str(ROOT / "src"))
from lfchain import merkle
from lfchain.ledger import FabricLedger
from lfchain.policy import (ACTIONS, OUTPUT_TYPES, PURPOSES, ROLES, TASK_STATES, Grant, PolicyState, Principal,
                            Request, Task, decide)

RUN = f"c{int(time.time())}"          # namespace so reruns never collide with existing on-chain keys
ORGS = [f"{RUN}-S{k}" for k in range(10)]
L = {"org1": FabricLedger(org="org1"), "org2": FabricLedger(org="org2")}
MSP_OF = {o: ("org1" if k < 5 else "org2") for k, o in enumerate(ORGS)}


def lg(org_id):
    return L[MSP_OF[org_id]]


def build_state(rng):
    """Create the same tasks/grants on-chain and in a local PolicyState."""
    st = PolicyState()
    for o in ORGS:
        lg(o).submit("RegisterInstitution", o)
    path = {"draft": [], "approved": ["approved"], "running": ["approved", "running"],
            "completed": ["approved", "running", "completed"], "cancelled": ["cancelled"]}
    tasks = []
    for j in range(25):
        state = TASK_STATES[j % 5]
        parts = tuple(sorted(rng.sample(ORGS, rng.randint(2, 6))))
        t = Task(f"{RUN}-T{j}", PURPOSES[j % 2] if j % 7 else PURPOSES[2], "draft", parts,
                 OUTPUT_TYPES[0] if j % 3 else OUTPUT_TYPES[1])
        L["org1"].create_task({"task_id": t.task_id, "purpose": t.purpose, "participants": list(parts),
                               "output_type": t.output_type, "state": "draft"})
        for s in path[state]:
            L["org1"].set_task_state(t.task_id, s)
        t.state = state
        st.tasks[t.task_id] = t
        tasks.append(t)
    grants = []
    gid = 0
    for t in tasks:
        for o in ORGS:
            if rng.random() < 0.6:
                for _rep in range(2 if rng.random() < 0.1 else 1):   # some re-grants: latest wins on both sides
                    g = Grant(f"{RUN}-G{gid}", o, f"{o}-dsv1", t.task_id,
                              t.purpose if rng.random() < 0.8 else rng.choice(PURPOSES),
                              expires_at=float(rng.randint(100, 1000)), revoked_at=None, granted_by=f"{o}-admin")
                    gid += 1
                    lg(o).grant_access(asdict(g))
                    st.grants[g.grant_id] = g
                    grants.append(g)
    for g in rng.sample(grants, len(grants) // 5):
        g.revoked_at = float(rng.randint(100, 1000))
        lg(g.org_id).revoke_access(g.grant_id, g.revoked_at)
    return st, tasks


def random_request(rng, tasks):
    org = rng.choice(ORGS + [f"{RUN}-unknown"])
    t = rng.choice(tasks + [None])
    res_org = org if rng.random() < 0.7 else rng.choice(ORGS)
    return Request(Principal(f"p{rng.randint(0, 999)}", org, rng.choice(ROLES), rng.random() > 0.05),
                   rng.choice(ACTIONS), t.task_id if t else f"{RUN}-Tnone", f"{res_org}-dsv1", res_org,
                   rng.choice(OUTPUT_TYPES), float(rng.randint(0, 1100)))


def c1():
    rng = random.Random(MASTER)
    t0 = time.time()
    st, tasks = build_state(rng)
    setup_s = time.time() - t0
    reqs = [random_request(rng, tasks) for _ in range(10_000)]
    local = [decide(r, st) for r in reqs]
    t = time.perf_counter()
    for r in reqs:
        decide(r, st)
    pdp_us = (time.perf_counter() - t) / len(reqs) * 1e6
    chain, lat = [], []
    for j in range(0, len(reqs), 200):
        batch = [asdict(r) for r in reqs[j:j + 200]]
        tt = time.perf_counter()
        chain += L["org1"].evaluate("CheckAccessBatch", json.dumps(batch))
        lat.append((time.perf_counter() - tt) * 1000)
    single = []
    for r in reqs[:200]:
        tt = time.perf_counter(); L["org2"].check_access(asdict(r)); single.append((time.perf_counter() - tt) * 1000)
    agree = [(a.allow, a.reason, a.grant_id) == (b["allow"], b["reason"], b["grant_id"]) for a, b in zip(local, chain)]
    reasons = {}
    for a in local:
        reasons[a.reason] = reasons.get(a.reason, 0) + 1
    mism = [{"req": asdict(reqs[i]), "local": local[i].to_dict(), "chain": chain[i]} for i, ok in enumerate(agree) if not ok][:20]
    # Supplementary boundary set (not in protocol §5): every grant, every grant-bound action, at the expiry and
    # revocation instants and one second either side, so that the grant branches are exercised densely.
    bnd = []
    for g in st.grants.values():
        t_ = st.tasks[g.task_id]
        times = {g.expires_at - 1, g.expires_at, g.expires_at + 1, 0.0}
        if g.revoked_at is not None:
            times |= {g.revoked_at - 1, g.revoked_at, g.revoked_at + 1}
        for now in sorted(times):
            for act, role in (("join_task", "data_admin"), ("run_round", "data_admin"), ("read_result", "analyst")):
                bnd.append(Request(Principal("pb", g.org_id, role), act, g.task_id, g.resource_id, g.org_id, t_.output_type, float(now)))
    bloc = [decide(r, st) for r in bnd]
    bchain = []
    for j in range(0, len(bnd), 200):
        bchain += L["org2"].evaluate("CheckAccessBatch", json.dumps([asdict(r) for r in bnd[j:j + 200]]))
    bagree = [(a.allow, a.reason, a.grant_id) == (b["allow"], b["reason"], b["grant_id"]) for a, b in zip(bloc, bchain)]
    breasons = {}
    for a in bloc:
        breasons[a.reason] = breasons.get(a.reason, 0) + 1
    out = {"run": RUN, "n_requests": len(reqs), "agreement": float(np.mean(agree)), "n_disagree": int(len(agree) - sum(agree)),
           "reason_counts": reasons, "allow_rate": float(np.mean([a.allow for a in local])),
           "n_tasks": len(tasks), "n_grants": len(st.grants), "n_revoked": sum(g.revoked_at is not None for g in st.grants.values()),
           "setup_seconds": setup_s, "pdp_us_per_decision": pdp_us,
           "chain_batch200_ms": {"mean": float(np.mean(lat)), "p95": float(np.percentile(lat, 95))},
           "chain_single_check_ms": {"mean": float(np.mean(single)), "p50": float(np.median(single)), "p95": float(np.percentile(single, 95))},
           "mismatches": mism,
           "boundary": {"n_requests": len(bnd), "agreement": float(np.mean(bagree)), "n_disagree": int(len(bagree) - sum(bagree)),
                        "reason_counts": breasons}}
    write_json("expC/c1_differential.json", out)
    out["negative_write_tests"] = negative_writes(st, tasks)
    write_json("expC/c1_differential.json", out)
    out["revocation_latency"] = revocation_latency(rng, tasks)
    write_json("expC/c1_differential.json", out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("mismatches",)}, indent=1, default=float))


def expect_fail(fn, code):
    try:
        fn()
        return {"expected": code, "observed": "SUCCEEDED", "pass": False}
    except Exception as e:
        return {"expected": code, "observed": code if code in str(e) else str(e)[-120:], "pass": code in str(e)}


def negative_writes(st, tasks):
    o1, o6 = ORGS[0], ORGS[6]               # o1 under Org1MSP, o6 under Org2MSP
    g = next(g for g in st.grants.values() if g.org_id == o1 and g.revoked_at is None)
    gr = next(g for g in st.grants.values() if g.revoked_at is not None)
    done = next(t for t in tasks if t.state == "completed")
    running = next(t for t in tasks if t.state == "running")
    L["org1"].register_dataset_version(f"{o1}-dsv1", o1, "v1", "ab" * 32, 10)
    cases = {
        "foreign_msp_registers_dataset": expect_fail(lambda: L["org2"].register_dataset_version(f"{o1}-dsvX", o1, "v1", "ab" * 32, 1), "CALLER_NOT_INSTITUTION"),
        "duplicate_dataset_version": expect_fail(lambda: L["org1"].register_dataset_version(f"{o1}-dsv1", o1, "v1", "ab" * 32, 10), "DATASET_VERSION_EXISTS"),
        "foreign_msp_revokes_grant": expect_fail(lambda: L["org2"].revoke_access(g.grant_id, 5), "CALLER_NOT_INSTITUTION"),
        "foreign_msp_grants_for_other_institution": expect_fail(lambda: L["org1"].grant_access({"grant_id": f"{RUN}-Gx", "org_id": o6, "resource_id": f"{o6}-dsv1", "task_id": running.task_id, "purpose": running.purpose, "expires_at": 10}), "CALLER_NOT_INSTITUTION"),
        "double_revoke": expect_fail(lambda: lg(gr.org_id).revoke_access(gr.grant_id, 5), "ALREADY_REVOKED"),
        "replayed_grant_id": expect_fail(lambda: lg(g.org_id).grant_access(asdict(g)), "GRANT_EXISTS"),
        "illegal_transition_completed_to_running": expect_fail(lambda: L["org1"].set_task_state(done.task_id, "running"), "ILLEGAL_TRANSITION"),
        "new_task_not_draft": expect_fail(lambda: L["org1"].create_task({"task_id": f"{RUN}-Tbad", "purpose": "difficulty_calibration", "participants": [o1], "state": "running"}), "NEW_TASK_MUST_BE_DRAFT"),
        "calibration_on_unfinished_task": expect_fail(lambda: L["org1"].record_calibration(running.task_id, "cd" * 32, [f"{o1}-dsv1"], "rasch-fed-1"), "TASK_NOT_COMPLETED"),
        "malformed_digest": expect_fail(lambda: L["org1"].anchor_batch(f"{RUN}-Abad", "xyz", 1, "audit"), "BAD_DIGEST"),
        "foreign_msp_requests_admission": expect_fail(lambda: L["org2"].submit("AdmitRound", json.dumps(asdict(Request(Principal("p", o1, "data_admin"), "run_round", running.task_id, f"{o1}-dsv1", o1)))), "CALLER_NOT_INSTITUTION"),
        "replayed_anchor_id": None,
    }
    L["org1"].anchor_batch(f"{RUN}-A1", "ef" * 32, 1, "audit")
    cases["replayed_anchor_id"] = expect_fail(lambda: L["org1"].anchor_batch(f"{RUN}-A1", "ef" * 32, 1, "audit"), "ANCHOR_EXISTS")
    return cases


def revocation_latency(rng, tasks, n=30):
    """Seconds from submitting RevokeAnalysisAccess to the first denied CheckAccess on the other org's peer."""
    running = [t for t in tasks if t.state == "running"]
    out = []
    for j in range(n):
        t = rng.choice(running)
        o = rng.choice(list(t.participants))
        g = {"grant_id": f"{RUN}-RL{j}", "org_id": o, "resource_id": f"{o}-rl{j}", "task_id": t.task_id,
             "purpose": t.purpose, "expires_at": 1e12, "revoked_at": None}
        lg(o).grant_access(g)
        req = asdict(Request(Principal("p", o, "data_admin"), "run_round", t.task_id, f"{o}-rl{j}", o, "aggregate", 1e9))
        other = L["org2"] if MSP_OF[o] == "org1" else L["org1"]
        t_vis = time.perf_counter()
        while not other.check_access(req)["allow"]:      # wait until the grant's block reached the other peer
            if time.perf_counter() - t_vis > 60:
                raise RuntimeError("grant never became visible on the other peer")
        t0 = time.perf_counter()
        r = lg(o).revoke_access(g["grant_id"], 0)
        t_commit = time.perf_counter() - t0
        polls = 0
        while other.check_access(req)["allow"]:
            polls += 1
        out.append({"commit_s": t_commit, "effect_s": time.perf_counter() - t0, "extra_polls": polls})
    eff = [x["effect_s"] for x in out]
    return {"n": n, "effect_s_mean": float(np.mean(eff)), "effect_s_p50": float(np.median(eff)),
            "effect_s_p95": float(np.percentile(eff, 95)), "effect_s_max": float(np.max(eff)),
            "commit_s_mean": float(np.mean([x["commit_s"] for x in out])),
            "max_extra_polls": int(max(x["extra_polls"] for x in out)), "samples": out}


# ---------------- C2 -----------------
def c2(n_tasks=200, K=10, rounds=20):
    rng = random.Random(MASTER + 2)
    models = {"login_only_rbac": {"viol": 0, "false_deny": 0, "lat": []},
              "join_time_check": {"viol": 0, "false_deny": 0, "lat": []},
              "per_round_reeval": {"viol": 0, "false_deny": 0, "lat": []}}
    n_events = n_auth_rounds = 0
    for j in range(n_tasks):
        st = PolicyState()
        orgs = [f"S{k}" for k in range(K)]
        t = Task(f"T{j}", "difficulty_calibration", "running", tuple(orgs))
        st.tasks[t.task_id] = t
        for o in orgs:
            st.grants[f"g{o}"] = Grant(f"g{o}", o, f"{o}-d", t.task_id, t.purpose, expires_at=1e9)
        revoke_at, expire_at = {}, {}
        for o in rng.sample(orgs, rng.randint(1, 4)):           # mid-task events
            at = float(rng.randint(1, rounds - 1))
            if rng.random() < 0.5:
                revoke_at[o] = at
            else:
                st.grants[f"g{o}"].expires_at = at
                expire_at[o] = at
            n_events += 1
        for o in orgs:
            g = st.grants[f"g{o}"]

            def at_round(r):
                # revocation is a state change: it enters the policy state at the round it happens
                g.revoked_at = revoke_at[o] if o in revoke_at and r >= revoke_at[o] else None
                return decide(Request(Principal("adm", o, "data_admin"), "run_round", t.task_id, f"{o}-d", o, "aggregate", float(r)), st).allow

            truth = [at_round(r) for r in range(rounds)]
            n_auth_rounds += sum(truth)
            event_round = revoke_at.get(o, expire_at.get(o))
            join_ok = truth[0]
            for name in models:
                allowed_at = {"login_only_rbac": lambda r: True, "join_time_check": lambda r: join_ok, "per_round_reeval": at_round}[name]
                for r in range(rounds):
                    allowed = allowed_at(r)
                    models[name]["viol"] += int(allowed and not truth[r])
                    models[name]["false_deny"] += int((not allowed) and truth[r])
                if event_round is not None:
                    first_block = next((r for r in range(int(event_round), rounds) if not allowed_at(r)), None)
                    models[name]["lat"].append(None if first_block is None else first_block - event_round)
    res = {}
    for name, m in models.items():
        lat = [x for x in m["lat"] if x is not None]
        res[name] = {"unauthorized_round_participations": m["viol"], "false_denials": m["false_deny"],
                     "events_never_enforced": sum(x is None for x in m["lat"]), "events": len(m["lat"]),
                     "mean_rounds_to_effect": float(np.mean(lat)) if lat else None}
    out = {"n_tasks": n_tasks, "K": K, "rounds": rounds, "n_mid_task_events": n_events,
           "authorised_round_participations": n_auth_rounds, "models": res}
    write_json("expC/c2_enforcement_models.json", out)
    print(json.dumps(out, indent=1))


# ---------------- C3 -----------------
def c3(n=1000):
    rng = np.random.default_rng(MASTER + 3)
    tmp = Path(tempfile.mkdtemp(prefix="lfc_c3_", dir=os.environ.get("TMPDIR")))
    sizes = rng.choice([1 << 10, 1 << 14, 1 << 17, 1 << 20, 1 << 23], n)
    paths = []
    for j, s in enumerate(sizes):
        p = tmp / f"artifact_{j:04d}.bin"
        p.write_bytes(rng.bytes(int(s)))
        paths.append(p)
    digests = [bytes.fromhex(merkle.file_digest(p)) for p in paths]
    tree = merkle.MerkleTree.from_leaves(digests)
    L["org1"].anchor_batch(f"{RUN}-C3", tree.root.hex(), n, "result_files")
    clean_ok, detect, t_hash, t_chain = 0, 0, [], []
    for j, p in enumerate(paths):
        proof = merkle.proof_to_json(tree.proof(j))
        t = time.perf_counter(); d = bytes.fromhex(merkle.file_digest(p)); t_hash.append((time.perf_counter() - t, int(sizes[j])))
        t = time.perf_counter()
        r = L["org2"].evaluate("VerifyAuditProof", f"{RUN}-C3", merkle.leaf_hash(d).hex(), json.dumps(proof))
        t_chain.append(time.perf_counter() - t)
        clean_ok += int(r["match"])
        data = bytearray(p.read_bytes()); pos = int(rng.integers(len(data))); data[pos] ^= 1 << int(rng.integers(8)); p.write_bytes(data)
        d2 = bytes.fromhex(merkle.file_digest(p))
        r2 = L["org2"].evaluate("VerifyAuditProof", f"{RUN}-C3", merkle.leaf_hash(d2).hex(), json.dumps(proof))
        detect += int(not r2["match"])
    by_size = {}
    for s_, sz in t_hash:
        by_size.setdefault(sz, []).append(s_ * 1000)
    out = {"n_artifacts": n, "untampered_verified": clean_ok, "tampered_detected": detect,
           "detection_rate": detect / n, "false_alarm_rate": 1 - clean_ok / n, "tree_depth": tree.depth,
           "proof_len": len(tree.proof(0)), "hash_ms_by_size": {str(k): float(np.mean(v)) for k, v in sorted(by_size.items())},
           "chain_verify_ms": {"mean": float(np.mean(t_chain) * 1000), "p95": float(np.percentile(t_chain, 95) * 1000)}}
    for p in paths:
        p.unlink()
    tmp.rmdir()
    write_json("expC/c3_tamper.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("c2", "all"):
        c2()
    if which in ("c1", "all"):
        c1()
    if which in ("c3", "all"):
        c3()
