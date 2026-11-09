"""Task-bound, purpose-constrained access control (RBAC + ABAC).

The decision predicate is the one frozen in docs/experiment_protocol.md §5:

    allow = identity valid ∧ role permits action ∧ resource in grantor's scope ∧ task purpose = grant purpose
            ∧ grant not revoked ∧ now < expires_at ∧ task state permits action ∧ output type allowed for role

Revocation is a state flag: once a grant is revoked in the current state every grant-bound action is denied,
whatever evaluation time is supplied. Only expiry depends on time; on the ledger that time is a transaction
timestamp accepted by the endorsers within a tolerance of their own clocks (chaincode AdmitRound).

The same predicate, with the same reason codes and the same evaluation order, is implemented in
chaincode/learnflow_audit/lib/policy.js; Experiment C1 checks the two implementations against each other.
Decisions are pure functions of (request, state) so that they can be replayed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional

ROLES = ("student", "teacher", "data_admin", "analyst", "auditor")
ACTIONS = ("join_task", "run_round", "read_result", "grant", "revoke", "read_audit", "read_raw")
PURPOSES = ("difficulty_calibration", "mastery_analysis", "student_profiling", "commercial")
TASK_STATES = ("draft", "approved", "running", "completed", "cancelled")
OUTPUT_TYPES = ("aggregate", "model", "per_student")

# RBAC: base capabilities per role.
ROLE_ACTIONS: Dict[str, frozenset] = {
    "student": frozenset(),
    "teacher": frozenset({"read_result"}),
    "data_admin": frozenset({"grant", "revoke", "join_task", "run_round", "read_result"}),
    "analyst": frozenset({"run_round", "read_result"}),
    "auditor": frozenset({"read_audit"}),
}
# Output minimisation: which output types each role may receive.
ROLE_OUTPUTS: Dict[str, frozenset] = {
    "student": frozenset(),
    "teacher": frozenset({"aggregate"}),
    "data_admin": frozenset({"aggregate", "model"}),
    "analyst": frozenset({"aggregate", "model"}),
    "auditor": frozenset(),
}
# Task state machine: which actions are legal in which state.
STATE_ACTIONS: Dict[str, frozenset] = {
    "draft": frozenset({"grant", "revoke", "read_audit"}),
    "approved": frozenset({"join_task", "grant", "revoke", "read_audit"}),
    "running": frozenset({"join_task", "run_round", "grant", "revoke", "read_audit"}),
    "completed": frozenset({"read_result", "revoke", "read_audit"}),
    "cancelled": frozenset({"revoke", "read_audit"}),
}
# Actions that touch an institution's data or results and therefore need a purpose-bound grant.
GRANT_BOUND = frozenset({"join_task", "run_round", "read_result"})
TASK_TRANSITIONS = {
    "draft": {"approved", "cancelled"},
    "approved": {"running", "cancelled"},
    "running": {"completed", "cancelled"},
    "completed": set(),
    "cancelled": set(),
}


@dataclass
class Principal:
    principal_id: str
    org_id: str
    role: str
    valid: bool = True  # certificate/identity not revoked


@dataclass
class Task:
    task_id: str
    purpose: str
    state: str
    participants: tuple  # org_ids
    output_type: str = "aggregate"


@dataclass
class Grant:
    grant_id: str
    org_id: str          # data-holding institution that granted
    resource_id: str     # dataset version id owned by org_id
    task_id: str
    purpose: str
    expires_at: float
    revoked_at: Optional[float] = None
    granted_by: str = ""


@dataclass
class Request:
    principal: Principal
    action: str
    task_id: str
    resource_id: str     # dataset version (for data actions) or task id (for audit)
    resource_org: str    # institution that owns the resource
    output_type: str = "aggregate"
    now: float = 0.0


@dataclass
class Decision:
    allow: bool
    reason: str
    grant_id: Optional[str] = None

    def to_dict(self):
        return asdict(self)


class PolicyState:
    """In-memory policy information point (tasks + grants)."""

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.grants: Dict[str, Grant] = {}

    def find_grant(self, task_id, resource_id, org_id) -> Optional[Grant]:
        # Latest grant wins (grants are append-only; re-granting creates a new id).
        best = None
        for g in self.grants.values():
            if g.task_id == task_id and g.resource_id == resource_id and g.org_id == org_id:
                best = g
        return best


def decide(req: Request, st: PolicyState) -> Decision:
    p = req.principal
    if not p.valid:
        return Decision(False, "IDENTITY_INVALID")
    if p.role not in ROLE_ACTIONS or req.action not in ROLE_ACTIONS[p.role]:
        return Decision(False, "ROLE_FORBIDS_ACTION")
    task = st.tasks.get(req.task_id)
    if task is None:
        return Decision(False, "TASK_UNKNOWN")
    # Scope: a principal acts only on its own institution's resources, except auditors (metadata only)
    # and analysts reading task-level results.
    if req.action in ("grant", "revoke", "join_task", "run_round") and p.org_id != req.resource_org:
        return Decision(False, "OUT_OF_SCOPE")
    if req.action in ("join_task", "run_round") and req.resource_org not in task.participants:
        return Decision(False, "NOT_PARTICIPANT")
    if req.action not in STATE_ACTIONS[task.state]:
        return Decision(False, "TASK_STATE_FORBIDS")
    grant_id = None
    if req.action in GRANT_BOUND:
        g = st.find_grant(task.task_id, req.resource_id, req.resource_org)
        if g is None:
            return Decision(False, "NO_GRANT")
        if g.purpose != task.purpose:
            return Decision(False, "PURPOSE_MISMATCH", g.grant_id)
        if g.revoked_at is not None:
            return Decision(False, "GRANT_REVOKED", g.grant_id)
        if req.now >= g.expires_at:
            return Decision(False, "GRANT_EXPIRED", g.grant_id)
        grant_id = g.grant_id
    if req.action == "read_result" and req.output_type not in ROLE_OUTPUTS[p.role]:
        return Decision(False, "OUTPUT_NOT_ALLOWED", grant_id)
    if req.action == "read_result" and req.output_type != task.output_type:
        return Decision(False, "OUTPUT_NOT_ALLOWED", grant_id)
    return Decision(True, "ALLOW", grant_id)


def transition(task: Task, new_state: str) -> None:
    if new_state not in TASK_TRANSITIONS[task.state]:
        raise ValueError(f"illegal transition {task.state} -> {new_state}")
    task.state = new_state


def request_to_json(req: Request) -> dict:
    d = asdict(req)
    return d


def state_to_json(st: PolicyState) -> dict:
    return {"tasks": [asdict(t) for t in st.tasks.values()], "grants": [asdict(g) for g in st.grants.values()]}
