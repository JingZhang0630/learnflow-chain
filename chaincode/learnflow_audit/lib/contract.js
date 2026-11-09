'use strict';
// LearnFlow-Chain audit chaincode. Stores only identifiers, codes, states, digests and receipts —
// never raw learning events, personal identifiers or model updates.
const { Contract } = require('fabric-contract-api');
const { decide, TASK_TRANSITIONS } = require('./policy');
const { verifyProof } = require('./merkle');

// Largest accepted difference (seconds) between a proposal's client-written timestamp and an endorser's clock.
const MAX_TX_SKEW_SECONDS = Number(process.env.LFCHAIN_MAX_TX_SKEW_SECONDS || 30);
const HEX64 = /^[0-9a-f]{64}$/;
const ID = /^[A-Za-z0-9_.:-]{1,128}$/;

function txSeconds(ctx) {
  const ts = ctx.stub.getTxTimestamp();
  return Number(ts.seconds.low !== undefined ? ts.seconds.low : ts.seconds) + ts.nanos / 1e9;
}

class LearnFlowAudit extends Contract {
  constructor() { super('LearnFlowAudit'); }

  async _get(ctx, key) {
    const b = await ctx.stub.getState(key);
    return b && b.length ? JSON.parse(b.toString()) : null;
  }

  async _put(ctx, key, obj) {
    await ctx.stub.putState(key, Buffer.from(JSON.stringify(obj)));
  }

  _check(cond, code) { if (!cond) { throw new Error(code); } }

  // Institutions are bound to the MSP that registered them; later writes for that institution must come from it.
  async RegisterInstitution(ctx, orgId) {
    this._check(ID.test(orgId), 'BAD_ID');
    const key = `inst:${orgId}`;
    this._check(!(await this._get(ctx, key)), 'INSTITUTION_EXISTS');
    await this._put(ctx, key, { org_id: orgId, msp: ctx.clientIdentity.getMSPID() });
    return 'OK';
  }

  async _assertInstitutionCaller(ctx, orgId) {
    const inst = await this._get(ctx, `inst:${orgId}`);
    this._check(inst, 'INSTITUTION_UNKNOWN');
    this._check(inst.msp === ctx.clientIdentity.getMSPID(), 'CALLER_NOT_INSTITUTION');
  }

  async RegisterDatasetVersion(ctx, dsvId, orgId, schemaVersion, manifestRoot, nRecords) {
    this._check(ID.test(dsvId), 'BAD_ID');
    this._check(HEX64.test(manifestRoot), 'BAD_DIGEST');
    await this._assertInstitutionCaller(ctx, orgId);
    const key = `dsv:${dsvId}`;
    this._check(!(await this._get(ctx, key)), 'DATASET_VERSION_EXISTS');
    await this._put(ctx, key, { dsv_id: dsvId, org_id: orgId, schema_version: schemaVersion, root: manifestRoot,
      n: Number(nRecords), tx_time: txSeconds(ctx) });
    return ctx.stub.getTxID();
  }

  async CreateAnalysisTask(ctx, taskJson) {
    const t = JSON.parse(taskJson);
    this._check(ID.test(t.task_id) && t.purpose && Array.isArray(t.participants) && t.participants.length > 0, 'BAD_TASK');
    this._check(t.state === 'draft' || t.state === undefined, 'NEW_TASK_MUST_BE_DRAFT');
    const key = `task:${t.task_id}`;
    this._check(!(await this._get(ctx, key)), 'TASK_EXISTS');
    await this._put(ctx, key, { task_id: t.task_id, purpose: t.purpose, state: t.state || 'draft',
      participants: t.participants, output_type: t.output_type || 'aggregate', config_hash: t.config_hash || null,
      algorithm_version: t.algorithm_version || null, tx_time: txSeconds(ctx) });
    return ctx.stub.getTxID();
  }

  async SetTaskState(ctx, taskId, newState) {
    const key = `task:${taskId}`;
    const t = await this._get(ctx, key);
    this._check(t, 'TASK_UNKNOWN');
    this._check((TASK_TRANSITIONS[t.state] || []).includes(newState), 'ILLEGAL_TRANSITION');
    t.state = newState;
    await this._put(ctx, key, t);
    return ctx.stub.getTxID();
  }

  async GrantAnalysisAccess(ctx, grantJson) {
    const g = JSON.parse(grantJson);
    this._check(ID.test(g.grant_id) && ID.test(g.org_id) && g.purpose && Number.isFinite(Number(g.expires_at)), 'BAD_GRANT');
    await this._assertInstitutionCaller(ctx, g.org_id);
    const key = `grant:${g.grant_id}`;
    this._check(!(await this._get(ctx, key)), 'GRANT_EXISTS');
    const rec = { grant_id: g.grant_id, org_id: g.org_id, resource_id: g.resource_id, task_id: g.task_id,
      purpose: g.purpose, expires_at: Number(g.expires_at), revoked_at: g.revoked_at === null || g.revoked_at === undefined ? null : Number(g.revoked_at),
      granted_by: g.granted_by || '', tx_time: txSeconds(ctx) };
    await this._put(ctx, key, rec);
    await this._put(ctx, `gidx:${g.task_id}:${g.org_id}:${g.resource_id}`, { grant_id: g.grant_id });
    return ctx.stub.getTxID();
  }

  async RevokeAnalysisAccess(ctx, grantId, revokedAt) {
    const key = `grant:${grantId}`;
    const g = await this._get(ctx, key);
    this._check(g, 'GRANT_UNKNOWN');
    await this._assertInstitutionCaller(ctx, g.org_id);
    this._check(g.revoked_at === null, 'ALREADY_REVOKED');
    g.revoked_at = Number(revokedAt);
    g.revoke_tx_time = txSeconds(ctx);
    await this._put(ctx, key, g);
    return ctx.stub.getTxID();
  }

  // Admission of a grant-bound action, submitted as an endorsed transaction. The evaluation time is the
  // transaction timestamp, which the client writes into its signed proposal: neither Fabric's endorsers nor its
  // ordering service compare it with a clock, so each endorser does that here and refuses a proposal outside the
  // tolerance. The clock is used only to accept or refuse, never in the returned decision or in written state.
  // The read set of the task and grant keys is validated at commit, so an admission that read a grant revoked in
  // an earlier block is invalidated (MVCC conflict) rather than admitted.
  async AdmitRound(ctx, requestJson) {
    const tau = txSeconds(ctx);
    const skew = Math.abs(Date.now() / 1000 - tau);
    this._check(skew <= MAX_TX_SKEW_SECONDS, 'TX_TIMESTAMP_OUT_OF_TOLERANCE');
    const req = JSON.parse(requestJson);
    this._check(req.principal && ID.test(req.resource_org), 'BAD_REQUEST');
    await this._assertInstitutionCaller(ctx, req.resource_org);
    req.now = tau;                                   // caller-supplied time is ignored
    const task = await this._get(ctx, `task:${req.task_id}`);
    let grant = null;
    const idx = await this._get(ctx, `gidx:${req.task_id}:${req.resource_org}:${req.resource_id}`);
    if (idx) { grant = await this._get(ctx, `grant:${idx.grant_id}`); }
    return JSON.stringify({ ...decide(req, task, grant), tau, tx_id: ctx.stub.getTxID() });
  }

  // Read-only policy decision over on-chain task/grant state at a caller-chosen time. Used for replay, the
  // differential test and latency measurement; a single-peer query is not endorsed, so it never admits anything.
  async CheckAccess(ctx, requestJson) {
    const req = JSON.parse(requestJson);
    const task = await this._get(ctx, `task:${req.task_id}`);
    let grant = null;
    const idx = await this._get(ctx, `gidx:${req.task_id}:${req.resource_org}:${req.resource_id}`);
    if (idx) { grant = await this._get(ctx, `grant:${idx.grant_id}`); }
    return JSON.stringify(decide(req, task, grant));
  }

  async CheckAccessBatch(ctx, requestsJson) {
    const out = [];
    for (const req of JSON.parse(requestsJson)) {
      out.push(JSON.parse(await this.CheckAccess(ctx, JSON.stringify(req))));
    }
    return JSON.stringify(out);
  }

  async RecordCalibration(ctx, taskId, resultDigest, dsvIdsJson, algorithmVersion) {
    this._check(HEX64.test(resultDigest), 'BAD_DIGEST');
    const t = await this._get(ctx, `task:${taskId}`);
    this._check(t, 'TASK_UNKNOWN');
    this._check(t.state === 'completed', 'TASK_NOT_COMPLETED');
    const dsv = JSON.parse(dsvIdsJson);
    for (const d of dsv) {
      const rec = await this._get(ctx, `dsv:${d}`);
      this._check(rec, 'DATASET_VERSION_UNKNOWN');
      this._check(t.participants.includes(rec.org_id), 'DATASET_NOT_IN_TASK');
    }
    const key = `calib:${taskId}`;
    this._check(!(await this._get(ctx, key)), 'CALIBRATION_EXISTS');
    await this._put(ctx, key, { task_id: taskId, digest: resultDigest, dsv_ids: dsv, algorithm_version: algorithmVersion, tx_time: txSeconds(ctx) });
    return ctx.stub.getTxID();
  }

  async RecordModelVersion(ctx, modelVersionId, taskId, artifactDigest, algorithmVersion) {
    this._check(HEX64.test(artifactDigest), 'BAD_DIGEST');
    this._check(await this._get(ctx, `task:${taskId}`), 'TASK_UNKNOWN');
    const key = `model:${modelVersionId}`;
    this._check(!(await this._get(ctx, key)), 'MODEL_VERSION_EXISTS');
    await this._put(ctx, key, { model_version_id: modelVersionId, task_id: taskId, digest: artifactDigest, algorithm_version: algorithmVersion, tx_time: txSeconds(ctx) });
    return ctx.stub.getTxID();
  }

  async AnchorBatch(ctx, batchId, rootHex, nRecords, recordType) {
    this._check(ID.test(batchId), 'BAD_ID');
    this._check(HEX64.test(rootHex), 'BAD_DIGEST');
    const key = `anchor:${batchId}`;
    this._check(!(await this._get(ctx, key)), 'ANCHOR_EXISTS');
    await this._put(ctx, key, { root: rootHex, n: Number(nRecords), type: recordType, tx_time: txSeconds(ctx) });
    return ctx.stub.getTxID();
  }

  async AnchorRecord(ctx, recordId, digestHex, recordType) {
    return this.AnchorBatch(ctx, recordId, digestHex, '1', recordType);
  }

  async GetAnchor(ctx, anchorId) {
    return JSON.stringify(await this._get(ctx, `anchor:${anchorId}`));
  }

  async VerifyAuditProof(ctx, anchorId, leafDigestHex, proofJson) {
    const a = await this._get(ctx, `anchor:${anchorId}`);
    if (!a) { return JSON.stringify({ match: false, reason: 'ANCHOR_UNKNOWN' }); }
    const ok = verifyProof(leafDigestHex, JSON.parse(proofJson), a.root);
    return JSON.stringify({ match: ok, reason: ok ? 'MATCH' : 'MISMATCH', anchor_tx_time: a.tx_time });
  }

  async QueryAuditHistory(ctx, key) {
    this._check(/^(task|grant|dsv|calib|model|anchor|inst):[A-Za-z0-9_.:-]{1,128}$/.test(key), 'BAD_KEY');
    const it = await ctx.stub.getHistoryForKey(key);
    const out = [];
    let r = await it.next();
    while (!r.done) {
      const m = r.value;
      out.push({ tx_id: m.txId, is_delete: m.isDelete, seconds: Number(m.timestamp.seconds.low !== undefined ? m.timestamp.seconds.low : m.timestamp.seconds), value: m.value && m.value.length ? JSON.parse(Buffer.from(m.value).toString()) : null });
      if (out.length >= 1000) { break; }
      r = await it.next();
    }
    await it.close();
    return JSON.stringify(out);
  }
}

module.exports = LearnFlowAudit;
