# Deviations from the frozen protocol (v1.0, commit 0ef1e84)

## D-1 (2026-10-18) DP-FED estimator revised to v2 before any real-data run

- **What happened.** During hyper-parameter selection on D3 (protocol §3), the v1 DP-FED estimator diverged at
  ε = 2: best b-RMSE 2.66 against the true difficulties, worse than predicting b = 0 (RMSE ≈ 1). Cause: each
  institution adds its own noise (no secure aggregation, so noise grows with √K) and the noisy-count
  preconditioner can approach λ, which amplifies noisy steps. Raw search output is kept in
  `results/hparam_search_v1_dpfed_diverged.json`; the superseded selection in `results/hparams_v1_superseded.json`.
- **Change.** (i) Release 0 now publishes jointly clipped per-item (count, correct) sums and the server warm-starts
  b from them (T = 0 is therefore a DP count estimator, "DP-CNT"); (ii) the server step is damped by the public
  per-round noise scale τ = √K·σ·C; (iii) the grid for T was extended to {0, 5, 10, 20, 50}, and the count clip
  C_n (not named in protocol §3) is selected by the same rule. Privacy accounting is unchanged: T + 1 Gaussian
  releases, each with sensitivity equal to its clip norm, zCDP composition.
- **Amendment (same day, still before any D1/D2 run).** A diagnostic on D3 (seed 20261019+99, not the
  selection seed) showed the T = 0 warm start reaching RMSE 2.16 at ε = 2 because −logit of a noise-dominated ratio is
  extreme. The warm start now uses a noise-aware pseudo-count m = max(2, √K·σ·C_n): p0 = (c̃ + m/2)/(ñ + m).
  Diagnostic output (ε → ∞ with no clipping reproduces FED: RMSE 0.2763 vs 0.2756) is reported in the paper's appendix.
- **Data touched.** Only D3 (seed 20261018). No D1/D2 metric had been computed when the change was made.
- **Also:** `sigmoid` now clips its argument to [−35, 35] to avoid overflow warnings (no effect on finite results).

## A-1 (2026-10-18) Exploratory additions after the main runs (not deviations; reported as exploratory)

- C1 boundary request set (grant-bound actions at expiry/revocation instants ± 1 s) in addition to the 10,000 random requests.
- DP-FED component ablation (`experiments/exp_b_ablation.py`): pseudo-count and damping switched off individually and jointly,
  frozen hyper-parameters, ε ∈ {1, 2, 8}. Added after a pre-submission review asked for attribution of the D-1 components.

## R-1 (2026-10-28) Revision experiments added in response to the review (all exploratory; frozen runs untouched)

Added after the first-round review; every frozen-protocol number in the paper is unchanged. Scripts and raw outputs are committed:

- **E — trusted-aggregator baseline** (`experiments/exp_e_fedirtdp_baseline.py` → `results/expE/`): the same DP-FED estimator with
  FedIRT-DP-style central noise (one draw on the aggregated sum, √K less total noise) on synthetic non-IID K∈{3,10} (5 seeds)
  and Junyi (3 seeds). Quantifies the price of per-institution noise (reviewer M1). No frozen run was modified.
- **F — clipping rates and sensitivity** (`experiments/exp_f_clipping.py` → `results/expF/`): exact per-student clip fractions for
  C ∈ {0.25,0.5,1,2} and C_n ∈ {2,4,8} on synthetic non-IID K=10 and Junyi, plus DP-FED sensitivity at ε=2 over the same grid
  (5 seeds synthetic / 3 seeds Junyi) with per-round clip-rate tracking (reviewer M6).
- **G — shared-scale value** (`experiments/exp_g_shared_scale.py` → `results/expG/`): (i) leave-one-institution-out prediction on
  synthetic non-IID K=10, 5 seeds; (ii) leave-one-city-out prediction on Junyi (2 seeds); (iii) Wilcoxon paired tests over the
  frozen protocol's 5 Junyi seeds for the small Table 4/5 AUC differences (reviewer M5).
- **C2-E2E script** (`fabric/gateway/exp_c2_e2e_fabric.js`): not run in this revision, and superseded by R-2 (its calls did not match the chaincode interface).
- **Leaf salting** (`src/lfchain/merkle.py`, revision R-M1): dataset-version Merkle leaves now support a fresh random 128-bit salt,
  leaf = H(0x00 ‖ salt ‖ x); optional parameter, existing proofs and verification unchanged (reviewer minor on low-entropy leaves).

## R-2 (2026-11-06) Corrections and additions after the second-round review (frozen-protocol runs untouched)

Frozen results (A, B, D, C3) are unchanged. Everything below concerns exploratory experiments or the admission path; superseded outputs are in git history at commit 38d0ea3.

Errors found in R-1 and corrected:

- **E and G (Junyi): abilities fitted on test responses.** `evaluate` in `exp_e_fedirtdp_baseline.py` and `run_junyi_loo` in `exp_g_shared_scale.py` refitted abilities on the responses being predicted, which inflated AUC above the non-private reference (reviewer N4). Abilities are now fitted on training responses. E's DP-FED AUC now reproduces the frozen runs exactly; `make_paper_assets.py` asserts this. RMSE values were not affected.
- **F: same leak in the AUC, and wrong clip-rate denominators.** Clip rates divided by K x n_users instead of the students of each institution, understating them by a factor of about K. The reported 10.0% / 9.5% on Junyi and 0.2% on synthetic data were wrong; the frozen norms clip nearly all students. See `results/expF/clip_rates_*.json`.
- **Hand-written macros.** `paper/generated/numbers_revision.tex` and `tab_fedirtdp.tex` are now produced by `make_paper_assets.py`.

Admission path, changed in code and text (reviewer N1, N5):

- The R-1 text said the evaluation time was validated by the ordering service and that every calibration run was gated. Neither was true: Fabric does not check client timestamps, and the chaincode had no endorsed admission function.
- New chaincode function `AdmitRound`: submitted as a transaction, tau = transaction timestamp, each endorser refuses a timestamp more than 30 s from its clock, caller's MSP must own the institution. Revocation is now a state flag in `policy.py` and `policy.js`, independent of tau. C1 was rerun because of this; C2 gives identical counts under the new semantics.
- The experiments of Sections 7.2-7.3 were run offline and remain so; the text now says so.

Found while checking the final text:

- **G (Junyi): warm start contains the held-out city.** `run_junyi_loo` starts FED from the all-city fit, and all 50 fits reach the 500-round cap, so some of the held-out city's training data can remain in the estimate. `junyi_loo_cold` refits FED from zeros on all seeds; these fits stop at 1e-4 in about 215 rounds, and the paper uses them. On the seed compared first, FED AUC moved by at most 8e-5 per city, so the conclusions did not depend on the warm start. CNT and LOC columns stay from `junyi_loo` (LOC at 1e-4; the two cities also run at 1e-6 differ by less than 1e-6).
- **C1 timing.** Rerun with other jobs loading the host (load average about 13); the earlier quiet-host figures are at commit 38d0ea3 (single check 7.4 ms at the median). Disclosed in the paper; remeasure on a quiet host if the timing matters to a reviewer.

New experiments (exploratory):

- **C2-E2E** (`experiments/exp_c2_e2e.py` -> `results/expC/c2_e2e.json`): DP-FED gated by `AdmitRound` on Fabric 2.5.14, mid-task revocation, anchored admission log, auditor replay; measured while other jobs loaded the host (load average about 13), including this project's own Junyi reruns.
- **Forged timestamps** (`fabric/gateway/test_tx_timestamp.js` -> `results/expC/tx_timestamp_test.json`).
- **G**: leave-one-city-out on all five Junyi seeds, FED from zeros (see above; the first warm-started run is kept in `junyi_loo.jsonl` for CNT and LOC), stopping at 1e-4 instead of 1e-6 because each fit otherwise reaches the 500-round cap, city-level tests; `syn_unseen` withholds 20% of an institution's items from its own training (no such items occur naturally in D3).
- **H** (`experiments/exp_h_skill_aggregation.py` -> `results/expH/`): calibration per skill on the same ASSISTments responses and split.

The Fabric network for these runs ran under colima instead of Docker Desktop, same image versions.
