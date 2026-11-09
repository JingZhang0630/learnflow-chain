"""Shared helpers for experiment scripts."""
import json, platform, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
if not RESULTS.exists():
    RESULTS.mkdir(parents=True)
MASTER = 20261018
SEEDS = [MASTER + r for r in range(5)]
DELTA = 1e-5


def centred_rmse_mae(b, ref, mask=None):
    if mask is not None:
        b, ref = b[mask], ref[mask]
    e = (b - b.mean()) - (ref - ref.mean())
    return float(np.sqrt(np.mean(e ** 2))), float(np.mean(np.abs(e)))


def write_json(name, obj):
    p = RESULTS / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=float))
    return p


def record_env():
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as e:
            return f"unavailable: {e}"
    import numpy, scipy, pandas
    env = {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cpu": run(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "logical_cpus": run(["sysctl", "-n", "hw.ncpu"]),
        "ram_bytes": run(["sysctl", "-n", "hw.memsize"]),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": numpy.__version__, "scipy": scipy.__version__, "pandas": pandas.__version__,
        "docker": run(["docker", "version", "--format", "{{.Server.Version}}"]),
        "fabric_peer": run([str(ROOT / "fabric/fabric-samples/bin/peer"), "version"]).splitlines()[:3],
        "fabric_samples_commit": run(["git", "-C", str(ROOT / "fabric/fabric-samples"), "rev-parse", "HEAD"]),
        "node": run(["node", "--version"]),
        "repo_commit": run(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
    }
    write_json("env.json", env)
    return env


def cached_central(d, seed):
    """Pooled (CEN) difficulties for dataset d and split seed, cached because one Junyi fit takes minutes."""
    sys.path.insert(0, str(ROOT / "src"))
    from lfchain.rasch import Responses, fit_central
    path = ROOT / "data" / "cache" / f"b_cen_{d.name}_{seed}.npy"
    if path.exists():
        return np.load(path)
    pooled = Responses(np.concatenate([r.u for r in d.train]), np.concatenate([r.i for r in d.train]),
                       np.concatenate([r.y for r in d.train]), d.n_users, d.n_items)
    b, _, _ = fit_central(pooled, iters=500, tol=1e-6)
    tmp = path.with_suffix(f".{np.random.default_rng().integers(1 << 30)}.npy")
    np.save(tmp, b)
    tmp.replace(path)
    return b
