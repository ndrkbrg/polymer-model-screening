# Local validation

Checked on Windows with Python 3.12 in a dedicated environment on 24 October 2026.
Machine-readable results and dependency versions are in [validation/report.json](validation/report.json).

- All 96 tests passed, including fresh quick simulations, resume checks, figure generation and protections for archived input paths. No tests were excluded.
- Regeneration matched 24 CSV tables/source files (166,212 cells) and 6,960 independent bank/cue/metric aggregates; seven main and six supplementary figures were generated.
- Fresh quick original-suite simulations passed five raw/summary checks. A new seed-30 revision pilot matched six archived raw tables (93,387 cells).
- AST comparison retained 201 numerical/helper function and class definitions unchanged. Ten orchestration/I/O functions and the reproduction wrapper were reviewed separately.
- All 211 CSV, checkpoint and reference-figure files are byte-identical to the validated source package. Scientific configuration values are unchanged.
- Ruff checks, formatting and pip dependency consistency passed.
- Candidate-file checks found no credential patterns, workstation paths or private manuscript/PDF files.

The pinned plotting dependencies emitted 240 deprecation warnings in the tests. They did not cause failures. Visual renderings can differ between environments; numerical figure-source tables were compared.

The full production datasets were not resimulated. The historical dirty checkout was not reconstructed. GitHub CI is configured for Windows and Linux but has not yet run remotely. Local checks do not establish scientific acceptance or laboratory validation.
