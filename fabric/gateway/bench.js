'use strict';
// Closed-loop load generator for Experiment D (protocol §6).
// node bench.js --strategy record|batch --batch 100 --duration 120 --warmup 15 --concurrency 16
//               --leaves ../../data/cache/junyi_leaves.bin --tag run0 --out result.json
const fs = require('fs');
const crypto = require('crypto');
const { newConnection } = require('./fabric');

const argv = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, arr) => {
  if (a.startsWith('--')) { acc.push([a.slice(2), arr[i + 1]]); } return acc;
}, []));
const strategy = argv.strategy || 'batch';
const B = strategy === 'record' ? 1 : Number(argv.batch || 100);
const duration = Number(argv.duration || 120) * 1000;
const warmup = Number(argv.warmup || 15) * 1000;
const conc = Number(argv.concurrency || 16);
const tag = argv.tag || 'run';
const leavesFile = argv.leaves;
const offset0 = Number(argv.offset || 0);

const fd = fs.openSync(leavesFile, 'r');
const nLeaves = fs.fstatSync(fd).size / 32;
let cursor = offset0 % nLeaves;
function nextLeaves(n) {
  const buf = Buffer.alloc(32 * n);
  let got = 0;
  while (got < n) {
    const take = Math.min(n - got, nLeaves - cursor);
    fs.readSync(fd, buf, got * 32, take * 32, cursor * 32);
    got += take; cursor = (cursor + take) % nLeaves;
  }
  const out = [];
  for (let j = 0; j < n; j++) { out.push(buf.subarray(j * 32, j * 32 + 32)); }
  return out;
}
const nodeHash = (l, r) => crypto.createHash('sha256').update(Buffer.from([1])).update(l).update(r).digest();
function merkleRoot(level) {
  while (level.length > 1) {
    const nxt = [];
    for (let j = 0; j + 1 < level.length; j += 2) { nxt.push(nodeHash(level[j], level[j + 1])); }
    if (level.length % 2) { nxt.push(level[level.length - 1]); }
    level = nxt;
  }
  return level[0];
}

async function main() {
  const conns = [newConnection('org1'), newConnection('org2')];
  const lat = []; let ok = 0, err = 0, records = 0, seq = 0, recordsAll = 0, okAll = 0;
  const errors = {};
  const tStart = Date.now();
  const tMeasure = tStart + warmup;
  const tEnd = tMeasure + duration;
  let measuredStart = null;

  async function worker(w) {
    const c = conns[w % 2].contract;
    while (Date.now() < tEnd) {
      const leaves = nextLeaves(B);
      const id = `${tag}-${strategy}-${B}-w${w}-${seq++}`;
      const t0 = process.hrtime.bigint();
      const inWindow = Date.now() >= tMeasure;
      try {
        let commit;
        if (strategy === 'record') {
          commit = await c.submitAsync('AnchorRecord', { arguments: [id, leaves[0].toString('hex'), 'event'] });
        } else {
          const root = merkleRoot(leaves).toString('hex');
          commit = await c.submitAsync('AnchorBatch', { arguments: [id, root, String(B), 'event_batch'] });
        }
        const st = await commit.getStatus();
        const ms = Number(process.hrtime.bigint() - t0) / 1e6;
        if (!st.successful) { throw new Error(`code ${st.code}`); }
        okAll += 1; recordsAll += B;
        if (inWindow && Date.now() <= tEnd) { lat.push(ms); ok += 1; records += B; if (measuredStart === null) { measuredStart = Date.now(); } }
      } catch (e) {
        if (inWindow) { err += 1; const k = String(e.message).slice(0, 80); errors[k] = (errors[k] || 0) + 1; }
      }
    }
  }
  await Promise.all(Array.from({ length: conc }, (_, w) => worker(w)));
  lat.sort((a, b) => a - b);
  const q = (p) => (lat.length ? lat[Math.min(lat.length - 1, Math.floor(p * lat.length))] : null);
  const secs = duration / 1000;
  const out = { tag, strategy, batch: B, concurrency: conc, duration_s: secs, warmup_s: warmup / 1000,
    committed_tx: ok, failed_tx: err, records, committed_tx_all: okAll, records_all: recordsAll, tx_per_s: ok / secs, records_per_s: records / secs,
    lat_mean_ms: lat.reduce((a, b) => a + b, 0) / Math.max(1, lat.length), lat_p50_ms: q(0.5), lat_p95_ms: q(0.95),
    lat_p99_ms: q(0.99), error_rate: err / Math.max(1, ok + err), errors };
  fs.writeFileSync(argv.out, JSON.stringify(out, null, 2));
  console.log(JSON.stringify(out));
  for (const c of conns) { c.gateway.close(); c.client.close(); }
}
main().catch((e) => { console.error(e); process.exit(1); });
