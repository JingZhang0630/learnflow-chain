"""AuditLedger interface with a hash-chained in-process MockLedger and a Hyperledger Fabric adapter.

Business code depends only on AuditLedger. MockLedger is used for unit tests and the C2 simulation;
FabricLedger talks to the Node.js gateway in fabric/gateway (fabric-gateway SDK) over localhost HTTP.
Only digests, identifiers, codes and states are ever passed to a ledger — never raw learning events.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


class AuditLedger(ABC):
    @abstractmethod
    def submit(self, fn: str, *args: Any) -> Dict: ...

    @abstractmethod
    def evaluate(self, fn: str, *args: Any) -> Any: ...

    # Convenience wrappers mirroring the chaincode interface.
    def register_dataset_version(self, dsv_id, org_id, schema_version, manifest_root, n_records):
        return self.submit("RegisterDatasetVersion", dsv_id, org_id, schema_version, manifest_root, str(n_records))

    def create_task(self, task: dict):
        return self.submit("CreateAnalysisTask", json.dumps(task, sort_keys=True))

    def set_task_state(self, task_id, state):
        return self.submit("SetTaskState", task_id, state)

    def grant_access(self, grant: dict):
        return self.submit("GrantAnalysisAccess", json.dumps(grant, sort_keys=True))

    def revoke_access(self, grant_id, revoked_at):
        return self.submit("RevokeAnalysisAccess", grant_id, str(revoked_at))

    def check_access(self, request: dict):
        return self.evaluate("CheckAccess", json.dumps(request, sort_keys=True))

    def record_calibration(self, task_id, result_digest, dsv_ids, algorithm_version):
        return self.submit("RecordCalibration", task_id, result_digest, json.dumps(sorted(dsv_ids)), algorithm_version)

    def anchor_batch(self, batch_id, root_hex, n_records, record_type):
        return self.submit("AnchorBatch", batch_id, root_hex, str(n_records), record_type)

    def anchor_record(self, record_id, digest_hex, record_type):
        return self.submit("AnchorRecord", record_id, digest_hex, record_type)

    def get_anchor(self, anchor_id):
        return self.evaluate("GetAnchor", anchor_id)

    def history(self, key):
        return self.evaluate("QueryAuditHistory", key)


class MockLedger(AuditLedger):
    """Append-only, hash-chained log with the same state semantics as the chaincode (subset)."""

    def __init__(self):
        self.blocks: List[dict] = []
        self.state: Dict[str, dict] = {}
        self.prev = b"\x00" * 32

    def _append(self, fn, args, key, value):
        rec = {"n": len(self.blocks), "fn": fn, "args": list(map(str, args)), "ts": time.time(), "prev": self.prev.hex()}
        h = hashlib.sha256(canonical(rec)).digest()
        rec["hash"] = h.hex()
        self.prev = h
        self.blocks.append(rec)
        if key is not None:
            self.state[key] = value
        return {"tx_id": rec["hash"], "block": rec["n"]}

    def submit(self, fn, *args):
        if fn == "RegisterDatasetVersion":
            key = f"dsv:{args[0]}"
            if key in self.state:
                raise ValueError("DATASET_VERSION_EXISTS")
            return self._append(fn, args, key, {"org": args[1], "root": args[3], "n": int(args[4])})
        if fn in ("CreateAnalysisTask", "GrantAnalysisAccess"):
            obj = json.loads(args[0])
            key = ("task:" + obj["task_id"]) if fn == "CreateAnalysisTask" else ("grant:" + obj["grant_id"])
            if key in self.state:
                raise ValueError("EXISTS")
            return self._append(fn, args, key, obj)
        if fn == "SetTaskState":
            t = dict(self.state[f"task:{args[0]}"]); t["state"] = args[1]
            return self._append(fn, args, f"task:{args[0]}", t)
        if fn == "RevokeAnalysisAccess":
            g = dict(self.state[f"grant:{args[0]}"])
            if g.get("revoked_at") is not None:
                raise ValueError("ALREADY_REVOKED")
            g["revoked_at"] = float(args[1])
            return self._append(fn, args, f"grant:{args[0]}", g)
        if fn == "AnchorBatch":
            return self._append(fn, args, f"anchor:{args[0]}", {"root": args[1], "n": int(args[2]), "type": args[3]})
        if fn == "AnchorRecord":
            return self._append(fn, args, f"anchor:{args[0]}", {"root": args[1], "n": 1, "type": args[2]})
        if fn == "RecordCalibration":
            return self._append(fn, args, f"calib:{args[0]}", {"digest": args[1], "dsv": json.loads(args[2]), "alg": args[3]})
        raise ValueError(f"unknown fn {fn}")

    def evaluate(self, fn, *args):
        if fn == "GetAnchor":
            return self.state.get(f"anchor:{args[0]}")
        if fn == "QueryAuditHistory":
            return [b for b in self.blocks if args[0] in b["args"]]
        if fn == "CheckAccess":
            raise NotImplementedError("MockLedger delegates decisions to lfchain.policy.decide")
        raise ValueError(fn)

    def verify_chain(self) -> bool:
        prev = b"\x00" * 32
        for rec in self.blocks:
            body = {k: v for k, v in rec.items() if k != "hash"}
            if body["prev"] != prev.hex() or hashlib.sha256(canonical(body)).hexdigest() != rec["hash"]:
                return False
            prev = bytes.fromhex(rec["hash"])
        return True


class FabricLedger(AuditLedger):
    def __init__(self, url: str = "http://127.0.0.1:8800", org: str = "org1"):
        self.url = url
        self.org = org

    def _post(self, path, payload):
        req = urllib.request.Request(self.url + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            out = json.loads(resp.read())
        if "error" in out:
            raise RuntimeError(out["error"])
        return out

    def submit(self, fn, *args):
        return self._post("/submit", {"org": self.org, "fn": fn, "args": [str(a) for a in args]})

    def evaluate(self, fn, *args):
        return self._post("/evaluate", {"org": self.org, "fn": fn, "args": [str(a) for a in args]})["result"]
