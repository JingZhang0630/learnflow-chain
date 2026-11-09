'use strict';
// Forged-timestamp test: submits AdmitRound proposals whose signed channel header carries a client-chosen timestamp,
// and records whether the endorsers accept them. usage: node test_tx_timestamp.js <requestJsonFile> <out.json>
// The request names an institution owned by Org1 with a valid grant on a running task.
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const grpc = require('@grpc/grpc-js');
const { connect, signers, hash } = require('@hyperledger/fabric-gateway');
const { common, gateway: gw, peer } = require('@hyperledger/fabric-protos');
const { Timestamp } = require('google-protobuf/google/protobuf/timestamp_pb');
const { CHANNEL, CHAINCODE, ORGS_DIR, firstFile, utf8 } = require('./fabric');

async function main() {
  const [requestFile, outFile] = process.argv.slice(2);
  const request = fs.readFileSync(requestFile, 'utf8');
  const base = path.join(ORGS_DIR, 'org1.example.com');
  const tls = fs.readFileSync(path.join(base, 'peers', 'peer0.org1.example.com', 'tls', 'ca.crt'));
  const client = new grpc.Client('localhost:7051', grpc.credentials.createSsl(tls), { 'grpc.ssl_target_name_override': 'peer0.org1.example.com' });
  const msp = path.join(base, 'users', 'User1@org1.example.com', 'msp');
  const credentials = fs.readFileSync(firstFile(path.join(msp, 'signcerts')));
  const key = crypto.createPrivateKey(fs.readFileSync(firstFile(path.join(msp, 'keystore'))));
  const sign = signers.newPrivateKeySigner(key);
  // No signer on the connection: proposals are built, altered, then signed explicitly.
  const gateway = connect({ client, identity: { mspId: 'Org1MSP', credentials }, hash: hash.sha256 });
  const contract = gateway.getNetwork(CHANNEL).getContract(CHAINCODE);
  const rows = [];
  for (const skew of [0, -10, 10, -25, 25, -40, 40, -3600, 3600, -86400 * 30]) {
    const unsigned = contract.newProposal('AdmitRound', { arguments: [request] });
    const pt = gw.ProposedTransaction.deserializeBinary(unsigned.getBytes());
    const sp = pt.getProposal();
    const prop = peer.Proposal.deserializeBinary(sp.getProposalBytes_asU8());
    const hdr = common.Header.deserializeBinary(prop.getHeader_asU8());
    const ch = common.ChannelHeader.deserializeBinary(hdr.getChannelHeader_asU8());
    const ts = new Timestamp();
    const when = Date.now() / 1000 + skew;
    ts.setSeconds(Math.floor(when)); ts.setNanos(0);
    ch.setTimestamp(ts);
    hdr.setChannelHeader(ch.serializeBinary());
    prop.setHeader(hdr.serializeBinary());
    const propBytes = prop.serializeBinary();
    sp.setProposalBytes(propBytes);
    pt.setProposal(sp);
    const signature = await sign(hash.sha256(propBytes));
    const signed = gateway.newSignedProposal(pt.serializeBinary(), signature);
    try {
      const tx = await signed.endorse();
      const r = JSON.parse(utf8.decode(tx.getResult()));
      rows.push({ skew_s: skew, endorsed: true, decision: r.reason, tau_minus_claimed: r.tau - Math.floor(when) });
    } catch (e) {
      const details = (e.details || []).map((d) => d.message).join('; ');
      rows.push({ skew_s: skew, endorsed: false, error: details.includes('TX_TIMESTAMP_OUT_OF_TOLERANCE') ? 'TX_TIMESTAMP_OUT_OF_TOLERANCE' : String(e.message + ' ' + details).slice(0, 200) });
    }
  }
  fs.writeFileSync(outFile, JSON.stringify(rows, null, 2));
  console.log(JSON.stringify(rows, null, 1));
  gateway.close(); client.close();
}
main().catch((e) => { console.error(e); process.exit(1); });
