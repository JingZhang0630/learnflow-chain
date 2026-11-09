"""zCDP accounting for the Gaussian mechanism and a fail-closed per-institution budget ledger.

A Gaussian mechanism with L2 sensitivity Δ and noise N(0, (σΔ)²) is ρ = 1/(2σ²)-zCDP (Bun & Steinke 2016).
ρ composes additively; ρ-zCDP implies (ρ + 2·sqrt(ρ·ln(1/δ)), δ)-DP.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List


def rho_to_eps(rho: float, delta: float) -> float:
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


def eps_to_rho(eps: float, delta: float) -> float:
    """Largest ρ whose (ε,δ) conversion equals eps (solve ρ + 2√(ρL) = ε, L = ln 1/δ)."""
    L = math.log(1.0 / delta)
    s = -math.sqrt(L) + math.sqrt(L + eps)  # s = √ρ
    return s * s


def sigma_for(eps: float, delta: float, n_releases: int) -> float:
    """Noise multiplier σ so that n_releases Gaussian releases (each sensitivity-normalised) meet (ε,δ)."""
    rho = eps_to_rho(eps, delta)
    return math.sqrt(n_releases / (2.0 * rho))


class BudgetExceeded(Exception):
    pass


@dataclass
class BudgetLedger:
    """Per-institution zCDP budget. charge() refuses (raises) before any release that would overspend."""

    cap_eps: float
    delta: float
    spent_rho: Dict[str, float] = field(default_factory=dict)
    records: List[dict] = field(default_factory=list)

    @property
    def cap_rho(self) -> float:
        return eps_to_rho(self.cap_eps, self.delta)

    def remaining_rho(self, org: str) -> float:
        return self.cap_rho - self.spent_rho.get(org, 0.0)

    def charge(self, org: str, task_id: str, sigma: float, releases: int = 1, **meta) -> None:
        rho = releases / (2.0 * sigma * sigma)
        if rho > self.remaining_rho(org) + 1e-12:
            raise BudgetExceeded(f"{org}: requested rho={rho:.4g} > remaining {self.remaining_rho(org):.4g}")
        self.spent_rho[org] = self.spent_rho.get(org, 0.0) + rho
        self.records.append({"org": org, "task_id": task_id, "mechanism": "gaussian", "sigma": sigma,
                             "releases": releases, "rho": rho, "delta": self.delta,
                             "eps_spent_total": rho_to_eps(self.spent_rho[org], self.delta), **meta})
