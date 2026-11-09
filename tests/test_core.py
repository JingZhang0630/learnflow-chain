import math

import numpy as np
import pytest

from lfchain import merkle, dp
from lfchain.policy import (Principal, Task, Grant, Request, PolicyState, decide, transition)
from lfchain.rasch import fit_central, fit_federated, fit_dp_federated, Responses
from lfchain.data.datasets import make_synthetic
from lfchain.ledger import MockLedger


# ---------- Merkle ----------
@pytest.mark.parametrize("n", [1, 2, 3, 7, 8, 100, 1025])
def test_merkle_proofs_roundtrip(n):
    leaves = [f"rec{j}".encode() for j in range(n)]
    t = merkle.MerkleTree.from_leaves(leaves)
    for j in {0, n // 2, n - 1}:
        pr = t.proof(j)
        assert len(pr) <= math.ceil(math.log2(n)) if n > 1 else len(pr) == 0
        assert merkle.verify_proof(merkle.leaf_hash(leaves[j]), pr, t.root)
        assert not merkle.verify_proof(merkle.leaf_hash(b"tampered"), pr, t.root)


def test_merkle_domain_separation():
    a, b = b"a", b"b"
    t = merkle.MerkleTree.from_leaves([a, b])
    # an internal node must not verify as a leaf
    assert merkle.leaf_hash(merkle.leaf_hash(a) + merkle.leaf_hash(b)) != t.root


# ---------- DP accounting ----------
def test_eps_rho_inverse():
    for eps in (0.5, 1, 2, 8):
        rho = dp.eps_to_rho(eps, 1e-5)
        assert abs(dp.rho_to_eps(rho, 1e-5) - eps) < 1e-9


def test_budget_fail_closed():
    led = dp.BudgetLedger(cap_eps=2.0, delta=1e-5)
    s = dp.sigma_for(2.0, 1e-5, 10)
    led.charge("A", "t1", s, releases=10)
    with pytest.raises(dp.BudgetExceeded):
        led.charge("A", "t2", s, releases=1)
    led.charge("B", "t2", s, releases=10)  # other institution unaffected


# ---------- Policy ----------
def _state(now=100.0):
    st = PolicyState()
    st.tasks["T"] = Task("T", "difficulty_calibration", "running", ("A", "B"))
    st.grants["gA"] = Grant("gA", "A", "dsA", "T", "difficulty_calibration", expires_at=now + 50)
    st.grants["gB"] = Grant("gB", "B", "dsB", "T", "student_profiling", expires_at=now + 50)
    return st


def _req(org="A", role="data_admin", action="run_round", res="dsA", now=100.0, valid=True, out="aggregate"):
    return Request(Principal("p", org, role, valid), action, "T", res, org if action != "read_result" else "A", out, now)


def test_policy_allow_and_reasons():
    st = _state()
    assert decide(_req(), st).allow
    assert decide(_req(valid=False), st).reason == "IDENTITY_INVALID"
    assert decide(_req(role="student"), st).reason == "ROLE_FORBIDS_ACTION"
    assert decide(_req(org="B", res="dsB"), st).reason == "PURPOSE_MISMATCH"
    assert decide(_req(now=151.0), st).reason == "GRANT_EXPIRED"
    # Revocation is a state flag: denied at any evaluation time, including times before the recorded instant.
    assert decide(_req(now=119.0), st).allow
    st.grants["gA"].revoked_at = 120.0
    assert decide(_req(now=119.0), st).reason == "GRANT_REVOKED"
    assert decide(_req(now=120.0), st).reason == "GRANT_REVOKED"
    r = Request(Principal("p", "B", "data_admin"), "run_round", "T", "dsA", "A", now=100)
    assert decide(r, st).reason == "OUT_OF_SCOPE"


def test_task_state_machine():
    t = Task("T", "difficulty_calibration", "draft", ("A",))
    transition(t, "approved"); transition(t, "running"); transition(t, "completed")
    with pytest.raises(ValueError):
        transition(t, "running")


# ---------- Rasch ----------
def test_federated_equals_central():
    d = make_synthetic(K=3, iid=False, seed=1)
    pooled = Responses(np.concatenate([r.u for r in d.train]), np.concatenate([r.i for r in d.train]),
                       np.concatenate([r.y for r in d.train]), d.n_users, d.n_items)
    b_c, _, _ = fit_central(pooled, iters=60, tol=0)
    b_f, _, _ = fit_federated(d.train, d.n_items, rounds=60, tol=0)
    assert np.max(np.abs(b_c - b_f)) < 1e-9


def test_central_recovers_truth():
    d = make_synthetic(K=3, iid=True, seed=2)
    b, _, _ = fit_federated(d.train, d.n_items, rounds=100)
    err = (b - b.mean()) - (d.true_b - d.true_b.mean())
    assert np.sqrt(np.mean(err ** 2)) < 0.45


def test_dp_federated_runs_and_charges():
    d = make_synthetic(K=3, iid=True, seed=3)
    led = dp.BudgetLedger(cap_eps=4.0, delta=1e-5)
    b, _, log, sigma = fit_dp_federated(d.train, d.n_items, eps=4.0, delta=1e-5, rounds=10, clip=2.0,
                                        clip_count=8.0, lr=0.5, seed=0, ledger=led)
    assert np.all(np.isfinite(b)) and log.rounds == 10
    assert len(led.records) == 3
    with pytest.raises(dp.BudgetExceeded):
        fit_dp_federated(d.train, d.n_items, eps=4.0, delta=1e-5, rounds=10, clip=2.0, clip_count=8.0,
                         lr=0.5, seed=0, ledger=led)


def test_gate_blocks_participation():
    d = make_synthetic(K=3, iid=True, seed=4)
    b, _, log = fit_federated(d.train, d.n_items, rounds=5, tol=0, gate=lambda t, k: not (k == 2 and t >= 2))
    assert [p[2] for p in log.participation] == [True, True, False, False, False]


# ---------- Mock ledger ----------
def test_mock_ledger_chain_and_tamper():
    L = MockLedger()
    L.register_dataset_version("d1", "A", "v1", "ab" * 32, 10)
    L.anchor_batch("b1", "cd" * 32, 100, "audit")
    assert L.verify_chain()
    with pytest.raises(ValueError):
        L.register_dataset_version("d1", "A", "v1", "ab" * 32, 10)
    L.blocks[0]["args"][3] = "ef" * 32
    assert not L.verify_chain()


# ---------- Revision R-M1: salted leaves ----------
def test_merkle_salted_leaves():
    from lfchain.merkle import MerkleTree, new_leaf_salt, leaf_hash, verify_proof
    data = [b"record-%d" % i for i in range(8)]
    salt = new_leaf_salt()
    t_unsalted = MerkleTree.from_leaves(data)
    t_salted = MerkleTree.from_leaves(data, salt)
    assert t_salted.root != t_unsalted.root
    # same salt reproduces the same root; different salt differs again
    assert MerkleTree.from_leaves(data, salt).root == t_salted.root
    assert MerkleTree.from_leaves(data, new_leaf_salt()).root != t_salted.root
    # proofs verify against the salted root with the salted leaf digest
    for i in range(8):
        assert verify_proof(leaf_hash(data[i], salt), t_salted.proof(i), t_salted.root)
    # a candidate record without the salt cannot match the leaf
    assert leaf_hash(data[0]) != leaf_hash(data[0], salt)
