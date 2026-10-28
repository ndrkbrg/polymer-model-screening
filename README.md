# Polymer model screening

Reproducibility code and archived computational results for **Computational Screening of Effective Interaction Models for Stimuli-Responsive Polymer Solutions**.

The study compares effective interaction models, native-attractor recovery, reservoir computing, multistate storage and constrained restricted Boltzmann machines. PAA and NVPVPE are model families. Numerical parameters are not fitted to new laboratory measurements; experimental validation remains future work.

Repository: [ndrkbrg/polymer-model-screening](https://github.com/ndrkbrg/polymer-model-screening).

Code and software documentation use the **MIT License**; datasets and figures use **CC BY 4.0**. See [licensing scope](LICENSING.md), [publication information](docs/publication.md) and [local validation](VALIDATION.md). No archival DOI has been assigned.

## Install

Use Python 3.12 and the pinned environment. Run commands from this repository's root.

Windows PowerShell:

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe reproduce.py verify
.\.venv\Scripts\python.exe -m pytest -m "not integration"
~~~

Linux/macOS:

~~~bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python reproduce.py verify
.venv/bin/python -m pytest -m "not integration"
~~~

Below, python means the interpreter in that environment. An editable install is optional:

~~~text
python -m pip install --no-deps -e .
~~~

Archived datasets remain in the repository, outside the installed package.

## Reproduce the article

~~~text
python reproduce.py verify
python reproduce.py regenerate --out work/regenerated
python reproduce.py smoke --out work/smoke
~~~

- **verify** checks distributed files against MANIFEST.json.
- **regenerate** recomputes summaries, paired contrasts, independent bank/cue aggregates, seven main figures and six supplementary figures from archived raw data. It compares numerical CSV cells with saved outputs using relative tolerance 1e-9 and absolute tolerance 1e-11.
- **smoke** performs new quick original-suite simulations and a new revision pilot using seed 30 in both model families. Five original raw/summary pairs are checked; six revision raw tables are compared with archived seed-30 records.

Regeneration does not repeat the full simulations. The smoke run is not the complete manuscript dataset. Neither establishes experimental calibration or scientific acceptance.

Use a new output directory for each wrapper run. Outputs go under work/ or to an explicitly chosen external directory. Run commands reject paths that would overwrite archived repository inputs. Input checksums are checked again after regeneration and smoke runs.

## Contents

| Path | Purpose |
| --- | --- |
| coilnet/ | Numerical models, experiment runners and main-figure builders |
| configs/ | Original full/quick settings, revision settings and result-to-table assertions |
| data/original/ | Original raw/summary CSVs, figure source tables and historical manifests |
| data/revision/ | Revision raw/summary CSVs, 160 checkpoint files and historical run manifest |
| data/parameters/ | Parameter values, units and provenance |
| data/diagnostics/ | Earlier multibody diagnostic output |
| scripts/ | Analysis and independent aggregation checks |
| figures/main/ and figures/supplementary/ | Reference figures |
| tests/ | Scientific regression tests and workflow checks |
| docs/ | Data dictionary, CSV/column indexes and revision protocol |
| provenance/ | File mapping, exclusions and source-equivalence evidence |
| work/ | Ignored local simulations, regenerated figures and logs |

Original and revision cohorts use different protocols. Original main seeds are 0–29; revision main seeds are 30–59, with revision sensitivity/discovery seeds 30–39. Do not pool them as one experiment. See the [revision protocol](docs/revision-protocol.md), [data dictionary](docs/data-dictionary.md) and [CSV index](docs/DATASET_INDEX.json).

## Full simulations and further checks

~~~text
python -m coilnet.experiments.run_q1_upgrade --config configs/original_full.json --out work/new-original-full --jobs 2
python -m coilnet.experiments.run_revision_phase_c --config configs/revision.json --out work/new-revision-full --jobs 2
python -m scripts.validate_tables
python -m pytest
~~~

Full simulations can take substantially longer. They create a new computational cohort and do not replace archived results. The complete test suite also runs a fresh quick workflow, checks resume behavior and renders the article's main figures. Some module names retain historical phase identifiers for traceability.

## Provenance and limits

This repository was extracted from the validated 23 October 2026 reproducibility package without copying the original project's Git history. The archive hash is in provenance/extraction.json. Numerical datasets and checkpoint arrays are retained byte-for-byte; configuration metadata is sanitized without changing scientific settings.

The original production run used a different recorded environment and a dirty checkout. Its historical commit alone does not identify the full executed tree. Some original CSV hashes match historical records after LF-to-CRLF normalization; the audit reports these cases. Historical and cleaned files have separate checksums.

Legacy demos, material-selection tools, manuscripts, review correspondence, source PDFs, local environments and private editorial snapshots are excluded. Omitted historical sources are listed explicitly; integrity checks do not claim to reconstruct them. Retained numerical functions are compared with the validated package using Python ASTs, and numerical workflows are exercised separately.

Citation names follow the confirmed article byline. Copyright attribution is to the polymer-model-screening contributors. Third-party dependencies are installed separately and retain their own terms.
