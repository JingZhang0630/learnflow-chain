"""Experiment D, big-data pipeline (protocol §6): Merkle manifests over the full Junyi log.

Each event row is serialised to a fixed 29-byte little-endian record
(ts:int64, user:int32, item:int32, correct:int8, sec:float32, attempts:int16, hints:int16, upid:int32; see RECORD), hashed as a Merkle leaf, and grouped by institution (user_city; missing city -> "unknown").
Each institution's root is registered on Fabric as a DatasetVersion. Leaf hashes are also written to
data/cache/junyi_leaves.bin for the anchoring benchmark.

usage: python exp_d_manifest.py [--no-chain]
"""
import hashlib, json, sys, time
import numpy as np
import pandas as pd

from common import *
sys.path.insert(0, str(ROOT / "src"))
from lfchain import merkle
from lfchain.data.datasets import CACHE

RECORD = np.dtype([("ts", "<i8"), ("user", "<i4"), ("item", "<i4"), ("correct", "i1"), ("sec", "<f4"),
                   ("attempts", "<i2"), ("hints", "<i2"), ("upid", "<i4")])


def main(use_chain=True):
    t0 = time.perf_counter()
    log = pd.read_parquet(CACHE / "junyi_log.parquet")
    users = pd.read_parquet(CACHE / "junyi_users.parquet")[["user", "user_city"]]
    city = users.set_index("user").user_city.reindex(log.user.to_numpy()).fillna("unknown").to_numpy()
    t_load = time.perf_counter() - t0
    rec = np.empty(len(log), RECORD)
    for f in RECORD.names:
        rec[f] = log[f].to_numpy()
    raw = rec.tobytes()
    w = RECORD.itemsize
    t = time.perf_counter()
    leaf = bytearray(32 * len(log))
    sha = hashlib.sha256
    for j in range(len(log)):
        leaf[32 * j:32 * j + 32] = sha(b"\x00" + raw[w * j:w * j + w]).digest()
    t_leaf = time.perf_counter() - t
    (CACHE / "junyi_leaves.bin").write_bytes(leaf)
    leaves = np.frombuffer(bytes(leaf), dtype="S32")
    orgs = {}
    big_tree, big = None, None
    t = time.perf_counter()
    order = np.argsort(city, kind="stable")
    cs = city[order]
    bounds = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        name = str(cs[a])
        tt = time.perf_counter()
        tree = merkle.MerkleTree.from_leaf_hashes(list(leaves[order[a:b]]))
        orgs[name] = {"n_records": int(b - a), "root": tree.root.hex(), "depth": tree.depth,
                      "build_s": time.perf_counter() - tt}
        if big_tree is None or tree.size > big_tree.size:
            big_tree, big = tree, name
        del tree
    t_tree = time.perf_counter() - t
    # proof size / verification time on the largest institution
    tree = big_tree
    rng = np.random.default_rng(MASTER)
    idx = rng.integers(0, tree.size, 1000)
    t = time.perf_counter()
    ok = all(merkle.verify_proof(tree.levels[0][j], tree.proof(int(j)), tree.root) for j in idx)
    t_ver = (time.perf_counter() - t) / len(idx)
    chain = {}
    if use_chain:
        from lfchain.ledger import FabricLedger
        L = [FabricLedger(org="org1"), FabricLedger(org="org2")]
        stamp = int(time.time())
        for k, (name, o) in enumerate(sorted(orgs.items())):
            inst = f"junyi-{name}-{stamp}"
            led = L[k % 2]
            led.submit("RegisterInstitution", inst)
            r = led.register_dataset_version(f"{inst}-v1", inst, "junyi-log-v1", o["root"], o["n_records"])
            chain[name] = {"institution": inst, "tx_id": r["tx_id"], "block": r["block"], "latency_ms": r["latency_ms"]}
    n = len(log)
    out = {"n_records": n, "record_bytes": w, "raw_bytes": n * w, "n_institutions": len(orgs),
           "load_s": t_load, "leaf_hash_s": t_leaf, "leaf_hash_records_per_s": n / t_leaf,
           "tree_build_s": t_tree, "tree_records_per_s": n / t_tree,
           "largest_institution": big, "largest_depth": orgs[big]["depth"],
           "proof_bytes": len(tree.proof(0)) * 33, "proof_verify_us": t_ver * 1e6, "proofs_valid": bool(ok),
           "onchain_bytes_per_institution_root": 32, "institutions": orgs, "chain": chain,
           "note": "single-threaded CPython hashlib; Junyi log as cached by lfchain.data.prep_junyi"}
    write_json("expD/manifest_pipeline.json", out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("institutions", "chain")}, indent=1))


if __name__ == "__main__":
    main(use_chain="--no-chain" not in sys.argv)
