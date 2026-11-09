'use strict';
const crypto = require('crypto');
const sha = (...bufs) => crypto.createHash('sha256').update(Buffer.concat(bufs)).digest();
const leafHash = (data) => sha(Buffer.from([0]), data);
const nodeHash = (l, r) => sha(Buffer.from([1]), l, r);
// proof: [[siblingHex, siblingIsLeft], ...]; leafDigestHex is already leaf_hash(data)
function verifyProof(leafDigestHex, proof, rootHex) {
  let h = Buffer.from(leafDigestHex, 'hex');
  for (const [sib, left] of proof) {
    const s = Buffer.from(sib, 'hex');
    h = left ? nodeHash(s, h) : nodeHash(h, s);
  }
  return h.toString('hex') === rootHex;
}
module.exports = { leafHash, nodeHash, verifyProof };
