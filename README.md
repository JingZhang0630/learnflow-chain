# LearnFlow-Chain

Manuscript sources for **LearnFlow-Chain: purpose-bound, differentially private
item-difficulty calibration on a permissioned ledger**.

Item difficulty estimated from learning logs drives problem selection in adaptive
learning, but no single institution holds enough students per item. LearnFlow-Chain
calibrates item parameters across institutions without pooling raw learner records:
every federated round is admitted through an endorsed ledger transaction that
evaluates a purpose- and task-bound access predicate, student-level differential
privacy noise is added by each institution (no trusted aggregator), and the lineage
from dataset version to published result is anchored as identifiers, states and
Merkle roots.

## Contents

| Path | Description |
| --- | --- |
| `monet.tex` | Main manuscript (Springer Nature `sn-jnl` class, iicol, MathPhys numbered style) |
| `references.bib` | Bibliography |
| `sn-jnl.cls` | Springer Nature journal document class |
| `sn-mathphys-num.bst` | Springer Nature Math & Physical Sciences numbered bibliography style |
| `figures/anchoring.pdf` | Figure: Merkle anchoring |
| `figures/privacy_utility.pdf` | Figure: privacy–utility trade-off |
| `monet.pdf` | Compiled manuscript |

## Build

```bash
pdflatex monet && bibtex monet && pdflatex monet && pdflatex monet
```

Requires a TeX distribution with `pdflatex`, `bibtex`, `tikz` and `booktabs`.

## Status

Manuscript under submission. Please do not redistribute.
