"""Experiment D, anchoring performance on Fabric (protocol §6).

Runs fabric/gateway/bench.js for per-record and Merkle-batch strategies, 3 repetitions each, while sampling
`docker stats` once per second and measuring the peer's block-store growth.
usage: python exp_d_bench.py [duration_s]
"""
import json, re, subprocess, sys, threading, time
import numpy as np

from common import *

GW = ROOT / "fabric" / "gateway"
LEAVES = ROOT / "data" / "cache" / "junyi_leaves.bin"
CONTAINERS = ["peer0.org1.example.com", "peer0.org2.example.com", "orderer.example.com",
              "peer0org1_learnflow_audit_ccaas", "peer0org2_learnflow_audit_ccaas"]
CHAIN_DIR = "/var/hyperledger/production/ledgersData/chains/chains/mychannel"


def ledger_bytes():
    out = subprocess.run(["docker", "exec", "peer0.org1.example.com", "sh", "-c", f"du -sk {CHAIN_DIR}"],
                         capture_output=True, text=True).stdout.split()
    return int(out[0]) * 1024 if out else None


def to_mib(s):
    m = re.match(r"([\d.]+)\s*([KMG]i?B)", s)
    v, u = float(m.group(1)), m.group(2)
    return v * {"KiB": 1 / 1024, "MiB": 1, "GiB": 1024, "kB": 1 / 1000, "MB": 1, "GB": 1000, "B": 1 / 2**20}.get(u, 1)


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop = False
        self.samples = []

    def run(self):
        while not self.stop:
            out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.Name}}|{{.CPUPerc}}|{{.MemUsage}}"],
                                 capture_output=True, text=True).stdout
            now = time.time()
            for line in out.strip().splitlines():
                name, cpu, mem = line.split("|")
                self.samples.append({"t": now, "name": name, "cpu": float(cpu.rstrip("%") or 0), "mem_mib": to_mib(mem.split("/")[0].strip())})


def fabric_config():
    cfg = (ROOT / "fabric/fabric-samples/test-network/configtx/configtx.yaml").read_text()
    pick = lambda k: (re.search(rf"^\s*{k}:\s*(.+)$", cfg, re.M) or [None, None])[1]
    return {"BatchTimeout": pick("BatchTimeout"), "MaxMessageCount": pick("MaxMessageCount"),
            "AbsoluteMaxBytes": pick("AbsoluteMaxBytes"), "PreferredMaxBytes": pick("PreferredMaxBytes"),
            "OrdererType": pick("OrdererType"), "endorsement": "default lifecycle policy (MAJORITY Endorsement, 2 orgs)",
            "peers": 2, "orderers": 1}


def main(duration=120, warmup=15, conc=16, reps=3):
    configs = [("record", 1)] + [("batch", b) for b in (10, 100, 500, 1000)]
    rows = []
    offset = 0
    for rep in range(reps):
        for strategy, B in configs:
            tag = f"d{int(time.time())}r{rep}"
            outp = RESULTS / "expD" / f"bench_{strategy}_{B}_r{rep}.json"
            outp.parent.mkdir(parents=True, exist_ok=True)
            before = ledger_bytes()
            s = Sampler(); s.start()
            t0 = time.time()
            subprocess.run(["node", "bench.js", "--strategy", strategy, "--batch", str(B), "--duration", str(duration),
                            "--warmup", str(warmup), "--concurrency", str(conc), "--leaves", str(LEAVES),
                            "--tag", tag, "--offset", str(offset), "--out", str(outp)], cwd=GW, check=True)
            s.stop = True; s.join()
            time.sleep(3)
            after = ledger_bytes()
            r = json.loads(outp.read_text())
            win = [x for x in s.samples if x["t"] >= t0 + warmup]
            res = {}
            for c in CONTAINERS + [n for n in {x["name"] for x in win} if n.startswith("mosaic")]:
                xs = [x for x in win if x["name"] == c]
                if xs:
                    res[c] = {"cpu_mean": float(np.mean([x["cpu"] for x in xs])), "mem_mib_max": float(max(x["mem_mib"] for x in xs))}
            r.update({"rep": rep, "ledger_bytes_before": before, "ledger_bytes_after": after,
                      "ledger_growth_bytes": (after - before) if before and after else None, "resources": res,
                      "ledger_bytes_per_record": ((after - before) / r["records_all"]) if before and after and r.get("records_all") else None,
                      "ledger_bytes_per_tx": ((after - before) / r["committed_tx_all"]) if before and after and r.get("committed_tx_all") else None})
            offset += r["records"] * 2 + 1000
            outp.write_text(json.dumps(r, indent=2))
            rows.append(r)
            print(f"{strategy:6s} B={B:5d} rep={rep} tx/s={r['tx_per_s']:.1f} rec/s={r['records_per_s']:.0f} "
                  f"p50={r['lat_p50_ms']:.0f}ms p95={r['lat_p95_ms']:.0f}ms err={r['error_rate']:.3f}", flush=True)
    write_json("expD/bench_summary.json", {"fabric": fabric_config(), "duration_s": duration, "warmup_s": warmup,
                                           "concurrency": conc, "reps": reps, "runs": rows})


if __name__ == "__main__":
    main(duration=int(sys.argv[1]) if len(sys.argv) > 1 else 120)
