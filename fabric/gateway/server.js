'use strict';
// Minimal localhost-only HTTP bridge used by lfchain.ledger.FabricLedger.
// POST /submit   {org, fn, args} -> {tx_id, block, result, latency_ms}
// POST /evaluate {org, fn, args} -> {result, latency_ms}
const http = require('http');
const { common, peer } = require('@hyperledger/fabric-protos');
const { newConnection, utf8, CHANNEL } = require('./fabric');

const PORT = Number(process.env.PORT || 8800);
const conns = { org1: newConnection('org1'), org2: newConnection('org2') };

async function handle(path, body) {
  const c = conns[body.org || 'org1'];
  if (!c) { throw new Error('unknown org'); }
  const t0 = process.hrtime.bigint();
  if (path === '/evaluate') {
    const r = await c.contract.evaluateTransaction(body.fn, ...body.args);
    const s = utf8.decode(r);
    let result; try { result = JSON.parse(s); } catch { result = s; }
    return { result, latency_ms: Number(process.hrtime.bigint() - t0) / 1e6 };
  }
  if (path === '/submit') {
    const commit = await c.contract.submitAsync(body.fn, { arguments: body.args });
    const result = utf8.decode(commit.getResult());
    const status = await commit.getStatus();
    if (!status.successful) { throw new Error(`tx ${status.transactionId} failed code ${status.code}`); }
    return { tx_id: status.transactionId, block: Number(status.blockNumber), result, latency_ms: Number(process.hrtime.bigint() - t0) / 1e6 };
  }
  if (path === '/txinfo') {
    // Auditor view of a committed transaction, read from the ledger through the query system chaincode.
    const qscc = c.network.getContract('qscc');
    const processed = peer.ProcessedTransaction.deserializeBinary(await qscc.evaluateTransaction('GetTransactionByID', CHANNEL, body.tx_id));
    const block = common.Block.deserializeBinary(await qscc.evaluateTransaction('GetBlockByTxID', CHANNEL, body.tx_id));
    const payload = common.Payload.deserializeBinary(processed.getTransactionenvelope().getPayload_asU8());
    const header = common.ChannelHeader.deserializeBinary(payload.getHeader().getChannelHeader_asU8());
    const ts = header.getTimestamp();
    return { tx_id: body.tx_id, validation_code: processed.getValidationcode(), block: Number(block.getHeader().getNumber()),
      timestamp: ts.getSeconds() + ts.getNanos() / 1e9, latency_ms: Number(process.hrtime.bigint() - t0) / 1e6 };
  }
  throw new Error('bad path');
}

http.createServer((req, res) => {
  let data = '';
  req.on('data', (d) => { data += d; });
  req.on('end', async () => {
    let out;
    try { out = await handle(req.url, JSON.parse(data || '{}')); } catch (e) {
      const details = e.details ? e.details.map((d) => d.message).join('; ') : '';
      out = { error: `${e.message}${details ? ' | ' + details : ''}` };
    }
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify(out));
  });
}).listen(PORT, '127.0.0.1', () => console.log(`gateway listening on 127.0.0.1:${PORT}`));
