'use strict';
// Shared connection helper for the fabric-samples test-network (Org1 / Org2, User1 identities).
const grpc = require('@grpc/grpc-js');
const { connect, signers, hash } = require('@hyperledger/fabric-gateway');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const ORGS_DIR = process.env.ORGS_DIR || path.resolve(__dirname, '../fabric-samples/test-network/organizations/peerOrganizations');
const CHANNEL = process.env.CHANNEL || 'mychannel';
const CHAINCODE = process.env.CHAINCODE || 'learnflow_audit';

const ORGS = {
  org1: { msp: 'Org1MSP', domain: 'org1.example.com', endpoint: 'localhost:7051' },
  org2: { msp: 'Org2MSP', domain: 'org2.example.com', endpoint: 'localhost:9051' },
};

function firstFile(dir) { return path.join(dir, fs.readdirSync(dir)[0]); }

function newConnection(orgKey) {
  const o = ORGS[orgKey];
  const base = path.join(ORGS_DIR, o.domain);
  const tlsCert = fs.readFileSync(path.join(base, 'peers', `peer0.${o.domain}`, 'tls', 'ca.crt'));
  const client = new grpc.Client(o.endpoint, grpc.credentials.createSsl(tlsCert), {
    'grpc.ssl_target_name_override': `peer0.${o.domain}`,
  });
  const msp = path.join(base, 'users', `User1@${o.domain}`, 'msp');
  const credentials = fs.readFileSync(firstFile(path.join(msp, 'signcerts')));
  const privateKey = crypto.createPrivateKey(fs.readFileSync(firstFile(path.join(msp, 'keystore'))));
  const gateway = connect({
    client,
    identity: { mspId: o.msp, credentials },
    signer: signers.newPrivateKeySigner(privateKey),
    hash: hash.sha256,
    evaluateOptions: () => ({ deadline: Date.now() + 30000 }),
    endorseOptions: () => ({ deadline: Date.now() + 60000 }),
    submitOptions: () => ({ deadline: Date.now() + 60000 }),
    commitStatusOptions: () => ({ deadline: Date.now() + 120000 }),
  });
  const network = gateway.getNetwork(CHANNEL);
  const contract = network.getContract(CHAINCODE);
  return { client, gateway, network, contract };
}

const utf8 = new TextDecoder();

module.exports = { newConnection, utf8, ORGS, CHANNEL, CHAINCODE, ORGS_DIR, firstFile };
