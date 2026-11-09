"""SHA-256 Merkle trees with domain separation (RFC 6962 style).

leaf hash  = H(0x00 || data)                 (no salt)
leaf hash  = H(0x00 || salt || data)         (salted, revision R-M1)
node hash  = H(0x01 || left || right)
An odd node at the end of a level is promoted unchanged (no duplication), so a proof is a list of
(sibling_hash, sibling_is_left) pairs whose length is at most ceil(log2 n).

Salted leaves (revision R-M1, reviewer M-minor on low-entropy leaf fields): each dataset version
draws a fresh random 128-bit salt at registration time and prepends it inside the leaf hash
domain, so a leaked proof cannot be combined with candidate records for offline membership
inference. The salt is stored with the manifest, never on the ledger. verify_proof is
salt-agnostic: callers pass the already-salted leaf digest.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

LEAF = b"\x00"
NODE = b"\x01"


def new_leaf_salt(nbytes: int = 16) -> bytes:
    """Fresh random salt for one dataset version (default 128 bits)."""
    return os.urandom(nbytes)


def leaf_hash(data: bytes, salt: bytes | None = None) -> bytes:
    if salt:
        return hashlib.sha256(LEAF + salt + data).digest()
    return hashlib.sha256(LEAF + data).digest()


def node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(NODE + left + right).digest()


def file_digest(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class MerkleTree:
    levels: List[List[bytes]]  # levels[0] = leaf hashes, levels[-1] = [root]

    @classmethod
    def from_leaves(cls, leaves: Iterable[bytes], salt: bytes | None = None) -> "MerkleTree":
        return cls.from_leaf_hashes([leaf_hash(x, salt) for x in leaves])

    @classmethod
    def from_leaf_hashes(cls, hashes: Sequence[bytes]) -> "MerkleTree":
        if not hashes:
            raise ValueError("empty tree")
        levels = [list(hashes)]
        cur = levels[0]
        while len(cur) > 1:
            nxt = [node_hash(cur[j], cur[j + 1]) for j in range(0, len(cur) - 1, 2)]
            if len(cur) % 2:
                nxt.append(cur[-1])
            levels.append(nxt)
            cur = nxt
        return cls(levels)

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    @property
    def size(self) -> int:
        return len(self.levels[0])

    @property
    def depth(self) -> int:
        return len(self.levels) - 1

    def proof(self, index: int) -> List[Tuple[bytes, bool]]:
        """Audit path for leaf `index`: list of (sibling, sibling_is_left)."""
        path = []
        for level in self.levels[:-1]:
            sib = index ^ 1
            if sib < len(level):
                path.append((level[sib], sib < index))
            index //= 2
        return path


def verify_proof(leaf_digest: bytes, proof: Sequence[Tuple[bytes, bool]], root: bytes) -> bool:
    h = leaf_digest
    for sib, sib_left in proof:
        h = node_hash(sib, h) if sib_left else node_hash(h, sib)
    return h == root


def proof_to_json(proof):
    return [[s.hex(), bool(left)] for s, left in proof]


def proof_from_json(obj):
    return [(bytes.fromhex(s), bool(left)) for s, left in obj]
