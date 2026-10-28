# Parameter provenance

Historical values are retained; Phase C extensions are recorded in the sensitivity column. Human scientific verification remains pending.

| ID | Parameter | Value and unit | Evidence | Sensitivity or limit |
|---|---|---|---|---|
| P001 | NVP–VPE association | Qualitative mechanism (not applicable) | E | Does not validate an overlap kernel or numerical temperature map |
| P002 | PAA temperature/pH dependence | Condition-dependent UCST behavior (not applicable) | E | Different electrolyte regime; no transferred phase diagram |
| P003 | PAA interaction | Reciprocal screened potential (reduced potential per elementary charge) | P | Periodic boundary and Gaussian charge clouds are idealizations |
| P004 | Electrostatic constants | 0.71; 0.304/sqrt(I) (nm; I in mol/L) | P | Fixed solvent/temperature constants; no specific-ion calibration |
| P005 | PAA monomer count and fixed ionization | 1000; 0.10 (count; dimensionless) | H | Neither value swept in the Q1 task grid; no charge regulation |
| P006 | Radius scale and PAA volume swelling ratio | 8; 1.5–6.0 (nm; dimensionless) | H | No joint molecular mass–radius fit; range not a measured swelling curve |
| P007 | Gaussian charge-cloud width | R_i/2 (nm) | H | Not independently calibrated or swept |
| P008 | Salt input | 0.01; structural sweep 0.005 and 0.05 (mol/L) | H | Salt sweep evaluates structural diagnostics, not all computing tasks; salt identity not resolved in PAA |
| P009 | Nominal volume parameter | 0.35; structural sweep 0.25 and 0.50 (dimensionless) | H | Not polymer mass fraction; not sum of individual coil volumes for heterogeneous radii |
| P010 | Reduced local drive and its standard deviation | u=spread*Normal(0,1); spread=0,0.25,0.5,1,1.5,2 (reduced drive) | H | Does not specify molecular-weight dispersity; used in structural, recovery and storage tests |
| P011 | Sigmoid response width | 1; structural/RC sweep 0.5,1,2 (reduced drive) | H | RC width/timing sensitivity is PAA-only; not a measured transition width |
| P012 | NVP–VPE calculation temperature | 40 (degrees C) | H | No temperature sweep or sample fit |
| P013 | NVP–VPE transition proxy | 47 - 0.6*c_s[mM] (degrees C; slope degrees C per mM) | H | Qualitative only; not a calibrated phase boundary |
| P014 | Temperature-to-drive factor | 0.25*(T-T0) (reduced drive; factor per degree C) | H | Magnitude and linear form not fitted or independently swept |
| P015 | Collapse/source fraction and swelling map | 0.05–1.0; Q=6-4.5*collapse (dimensionless) | H | Receiver is sigmoid slope; no measured input–output law |
| P016 | Contact extension and kernel | 1; max(0,R_i+R_j+delta_c-d_ij) off diagonal (nm) | H | No periodic wrapping or calibrated energy prefactor; range not swept |
| P017 | Coil counts by task | Coupling 30; recovery 20/30/40; reservoir 40; storage 16 (count) | C | Reservoir sizes 40/80/120/200 tested for PAA only; not a finite-size proof |
| P018 | Electrostatic grid by task | Coupling 44; structural 32/44/64; recovery 40; principal reservoir 40; storage 32 (points per axis) | C | RC helper uses 28 for N<40, absent from full-profile RC size grid; no grid in contact model |
| P019 | Replication units | Main tasks 0–29; sensitivity 0–9; coupling 100–102; structural robustness 0–1 (seed indices) | C | Original cohorts retained; revision main seeds 30-59 and sensitivity 30-39 remain separate (SI S10). |
| P020 | Native-attractor search | 128 starts; retain up to 5 largest observed basins; 80 sweeps (counts) | C | Revision budgets 128/256/512/1024 and top-five/top-ten/all sensitivity remain incomplete (SI S13). |
| P021 | Recovery perturbations | 10/20/30/40% flips; 5 cues/target; thresholds -0.05/0/+0.05 times median field (fractions; reduced field) | C | Revision threshold range extends to +/-0.1; original targets and retention tracked (SI S13). |
| P022 | Symmetrization diagnostic | Dense perturbation 1/5/10/20%; residual cutoff 0.01 (dimensionless) | C | Revision continuous residual and cutoffs 1e-8 to 0.5; not a dynamical failure boundary (SI S13). |
| P023 | Reservoir spectral radius | 0.9; PAA sweep 0.5/0.7/0.9/1.1 (dimensionless) | C | Not a material operation or sufficient proof of echo-state behavior |
| P024 | Langevin configurations | Bistable: a=0.3, b=1, g=0.5, D=0.3; graded: a=-1, b=1, g=0.5, D=0.1 (reduced computational units) | H/C | Original response/noise confound retained; matched revision D=0/0.1/0.3 in both families (SI S11). |
| P025 | Langevin timing | dt=0.02; tau=1; PAA tau sweep 0.25/0.5/1/2/4 (reduced time) | H/C | Original PAA grid retained; revision tau=0.5/2 for both families; no seconds calibration (SI S12). |
| P026 | ESN settings | gain=1; leak=1; additive noise variance=0.001 (SD=sqrt(0.001)) (dimensionless) | C | ESN noise parameter is not identical to Langevin diffusion D |
| P027 | Input coupling | Uniform[-1,1]; one input; all units driven (dimensionless) | C | Revision amplitude 0.5/2 and half-mask screens in both families; mask also changes power (SI S12). |
| P028 | Task sequence and readout split | 1500;200; chronological 50/50 after washout (steps; fraction) | C | Verified 650/650 split; zero initial reset, continuity at split, state/input/intercept (SI S10). |
| P029 | Readout penalty and memory task | 1e-6; length 2000; maximum delay 40 (dimensionless; steps) | C | Not a measured material memory duration |
| P030 | Storage/control protocol | 128 starts; up to 10 symbols; 5 repeats; acceptance 0.90 (counts; fraction) | C | Partial transitions do not establish the declared gate |
| P031 | Generic network sizes and training | 3x3:9 visible/16 hidden/2500 epochs; 4x4:16/32/4000; learning rate 0.5 (counts; dimensionless) | C | Not polymer-specific; training budget does not prove convergence |
| P032 | Ranks and projections | Ranks 1/2/3/4/6/8/12; 12 projections; residual tolerance 1e-6 (counts; dimensionless) | C | Approximate intersections are retained and flagged |
| P033 | Sampling budgets | 2000 samples; 500 Gibbs steps before sampling; 5000-step mixing diagnostic (counts/steps) | C | Activity-autocorrelation proxy is not global mixing; joint coverage/NLL/KL/duplicates in SI S14. |
| P034 | One-time error and quantization | Error SD fraction 0/0.05/0.10/0.20/0.40/0.60; levels 2/3/4/8/16 (dimensionless; levels) | H/C | Main cohort: 20% and 8 levels; no actual fabrication or common error axis with learning |
