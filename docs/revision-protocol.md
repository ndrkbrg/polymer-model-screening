# Revision analysis protocol

Declared 20 October 2026, before revision-stage simulation outputs were generated. This plan is informed by the original results; it is not retrospective preregistration. The protocol and numerical settings are retained; editorial workflow notes are omitted from this code copy.

## Questions and independent units

The independent unit is a generated matrix/task seed. Cues, discovered states, time samples, delays and Gibbs samples within a seed are dependent measurements. Conclusions concern these effective numerical models and protocols, not material calibration or experimental validation.

1. Does changing Langevin response from bistable (a=0.3) to graded (a=-1) improve prediction at the same noise strength? Does that difference depend on noise?
2. How sensitive is the graded reference to topology, relaxation time and input coupling in each model family?
3. Do observed attractor counts, target-set coverage and recovery stabilize as discovery effort increases? How do top-five selection and thresholds affect interpretation?
4. How does the continuous diagonal-symmetry residual relate to recovery and convergence after non-factorizable perturbation? Does a binary interpretation depend on the cutoff?
5. What do the existing reservoir distributions and joint RBM validity/coverage/likelihood/duplicate/mixing metrics support?

## Preservation and provenance

Original code and saved `data/original/` outputs are inputs, not replacement targets. Record input and source hashes before runs. New raw records, configuration, matrix/input checkpoints, summaries and figures go here. The original run manifest records an earlier dirty checkout; matching saved-file hashes establishes local integrity, not independent historical provenance.

Reuse original full-profile outputs after checking uniqueness, seed counts, finite values and metric definitions. Separate 30-seed groups from 10-seed sensitivity groups. Do not rerun RBMs to replace inconvenient outcomes.

## Reservoir design

- Both PAA and NVPVPE; N=40, spread=1, salt=0.01 mol/L, width=1, grid=40, temperature=40 C and nominal volume parameter=0.35 where used by the constructor. Spectral radius=0.9.
- New seed labels 30–59 (30 per family): response a={0.3,-1} crossed with D={0,0.1,0.3}, with b=1, gain=0.5, dt=0.02, tau=1 and all nodes driven at input scale=1.
- Explicitly zero every new matrix diagonal before rescaling. Native, symmetrized, shuffled-off-diagonal and dense Gaussian matrices are computational controls. The original random tanh control retained its Gaussian diagonal; report that distinction without revising its saved results.
- Secondary one-factor sensitivity around graded/D=0.1/native, seeds 30–39 in both families: topology={symmetric,shuffled,random}, tau={0.5,2}, input scale={0.5,2}, driven fraction=0.5. These are conditional ablations, not a full interaction experiment or optimization.
- Separate deterministic streams for geometry, shuffle, random matrix, input weights, input mask, task, dynamic noise and memory input. Share matrices, tasks, input base draws and standardized noise innovations across appropriate conditions within a seed. A fixed permutation selects the driven half, nested in the full mask. No power renormalization after masking: this tests reduced direct coverage and total input power together.
- NARMA-10 length=1500, washout=200, train indices 200–849, test 850–1499 (zero-based), no reset at the split. Features: state, current input, intercept. Ridge=1e-6 including intercept. Use the original finite-target rejection rule, record actual task seed, and never select by model score.
- Memory: independent uniform[-1,1] input, length=2000, delays 1–40, state plus intercept without current input; existing delay-specific chronological split after washout+k, ridge=1e-6, sum of held-out squared correlations. Save all 40 scores.
- Record train/test NRMSE, counts, memory capacity, delay scores, RNG identifiers and failure status. Retain failed runs with missing scores and reasons. Do not clip scores, substitute favorable seeds or suppress failures.

Primary contrasts in each family for NRMSE and memory: graded minus bistable at each D, plus interaction `[graded-bistable at D=0.3] - [graded-bistable at D=0.1]`. Report paired differences and 95% percentile-bootstrap intervals using 10,000 seed resamples. Two-sided Wilcoxon tests use the recorded SciPy implementation; Holm adjustment spans all 16 contrasts. Intervals are pointwise. Secondary ablations have descriptive paired pointwise intervals, without significance-based selection. Report complete/eligible pairs and failures.

## Attractor design

- Both families, N=30, spread=1, grid=40, zero diagonal, seeds 30–39, original geometry seed convention. Fixed random asynchronous order per trajectory, tie retention, fixed/cycle/80-sweep-timeout distinction.
- One nested 1024-start bank; prefixes {128,256,512,1024}. Retain all zero-threshold fixed points and counts; deterministic tie ordering. Report distinct count, convergence, top-five empirical coverage and new discoveries. An independent 256-start bank estimates the fraction of outcomes represented by each prefix.
- At each prefix compare top 5, top 10 and all targets: 20% corruption, five cues per target. Cue RNG depends on seed and target pattern, never discovery budget or rank. Report uniform-target and discovery-count-weighted recovery separately. Stabilization does not prove exhaustive discovery.
- For final top-five targets, threshold fractions {-0.1,-0.05,0,0.05,0.1} multiply median absolute row sum. Outer values extend the original operational ±0.05 range twofold. Check target fixed-point retention under each threshold; no physical threshold calibration is claimed.
- One nested off-diagonal Gaussian perturbation direction, Frobenius norm equal to W; amplitudes {0,0.0001,0.001,0.01,0.05,0.1,0.2}. Report continuous fitted diagonal-symmetry residual, recovery of original top-five targets, target retention, and fixed/cycle/timeout rates on shared cues.
- Residual exceedance cutoffs {1e-8,1e-4,1e-3,0.01,0.05,0.1,0.5}. This log-ratio-fit diagnostic uses same-sign reciprocal edges; it is not a physical threshold or a proof about all trajectories. Show continuous residual/outcome relationships without attributing causality to cutoff crossings.
- Bootstrap over matrix seeds and retain target-level denominators. If discovery does not stabilize, report that limit rather than increasing effort opportunistically.

## Existing RBM and route results

Report validity, valid-mode coverage, target-to-model KL, exact mean target NLL, duplicate fraction and stored `mixing_time` jointly. The last is a truncated positive-autocorrelation estimate for mean visible activity from one finite chain (lag cap; constant signal returns chain length), not a validated global mixing time. It can miss trapped modes with fluctuating activity. Duplicate rates depend on finite support and sample budget. Keep these limitations visible.

Route metrics remain descriptive under their different perturbation processes. Since A/B removed universal cross-route superiority, additional unmatched route simulations are unnecessary.

## Verification and completion

Before full runs test seed pairing, train/test boundaries, nested discovery, target-stable cues, the fast trajectory routine against the original, denominators and failure preservation. A one-seed pilot is for execution/runtime only, not selecting settings. Record config/code hashes and dependency versions; resume only when they agree. Reproduce summaries and figures from raw outputs.

