'use strict';
// Mirror of src/lfchain/policy.py::decide — same predicate, same evaluation order, same reason codes.
const ROLE_ACTIONS = {
  student: [],
  teacher: ['read_result'],
  data_admin: ['grant', 'revoke', 'join_task', 'run_round', 'read_result'],
  analyst: ['run_round', 'read_result'],
  auditor: ['read_audit'],
};
const ROLE_OUTPUTS = {
  student: [], teacher: ['aggregate'], data_admin: ['aggregate', 'model'], analyst: ['aggregate', 'model'], auditor: [],
};
const STATE_ACTIONS = {
  draft: ['grant', 'revoke', 'read_audit'],
  approved: ['join_task', 'grant', 'revoke', 'read_audit'],
  running: ['join_task', 'run_round', 'grant', 'revoke', 'read_audit'],
  completed: ['read_result', 'revoke', 'read_audit'],
  cancelled: ['revoke', 'read_audit'],
};
const GRANT_BOUND = ['join_task', 'run_round', 'read_result'];
const TASK_TRANSITIONS = {
  draft: ['approved', 'cancelled'], approved: ['running', 'cancelled'], running: ['completed', 'cancelled'],
  completed: [], cancelled: [],
};

function dec(allow, reason, grantId) { return { allow, reason, grant_id: grantId === undefined ? null : grantId }; }

// req: {principal:{principal_id,org_id,role,valid}, action, task_id, resource_id, resource_org, output_type, now}
// task: stored task or null; grant: latest grant for (task, resource, org) or null
function decide(req, task, grant) {
  const p = req.principal;
  if (!p.valid) { return dec(false, 'IDENTITY_INVALID'); }
  if (!(p.role in ROLE_ACTIONS) || !ROLE_ACTIONS[p.role].includes(req.action)) { return dec(false, 'ROLE_FORBIDS_ACTION'); }
  if (!task) { return dec(false, 'TASK_UNKNOWN'); }
  if (['grant', 'revoke', 'join_task', 'run_round'].includes(req.action) && p.org_id !== req.resource_org) { return dec(false, 'OUT_OF_SCOPE'); }
  if (['join_task', 'run_round'].includes(req.action) && !task.participants.includes(req.resource_org)) { return dec(false, 'NOT_PARTICIPANT'); }
  if (!STATE_ACTIONS[task.state].includes(req.action)) { return dec(false, 'TASK_STATE_FORBIDS'); }
  let grantId = null;
  if (GRANT_BOUND.includes(req.action)) {
    if (!grant) { return dec(false, 'NO_GRANT'); }
    if (grant.purpose !== task.purpose) { return dec(false, 'PURPOSE_MISMATCH', grant.grant_id); }
    // Revocation is a state flag: denied whatever evaluation time is supplied. Only expiry depends on time.
    if (grant.revoked_at !== null && grant.revoked_at !== undefined) { return dec(false, 'GRANT_REVOKED', grant.grant_id); }
    if (req.now >= grant.expires_at) { return dec(false, 'GRANT_EXPIRED', grant.grant_id); }
    grantId = grant.grant_id;
  }
  if (req.action === 'read_result' && !ROLE_OUTPUTS[p.role].includes(req.output_type)) { return dec(false, 'OUTPUT_NOT_ALLOWED', grantId); }
  if (req.action === 'read_result' && req.output_type !== task.output_type) { return dec(false, 'OUTPUT_NOT_ALLOWED', grantId); }
  return dec(true, 'ALLOW', grantId);
}

module.exports = { decide, TASK_TRANSITIONS };
