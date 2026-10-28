# Development

Use Python 3.12 and install requirements-dev.txt. Run:

~~~text
python -m ruff check .
python -m ruff format --check .
python -m pytest
python reproduce.py verify
~~~

Keep archived data and checkpoints unchanged. Put fresh simulations and generated files under work/. Distinguish model/statistics changes from layout and documentation changes.

MANIFEST.json binds a reviewed release snapshot. Source or documentation changes should fail verification until reviewed and assigned a new manifest:

~~~text
python -m scripts.build_manifest
~~~

Creating a manifest does not validate scientific changes. Run the relevant tests and numerical comparisons. Historically mapped input changes require an explicit provenance update; do not relabel old results as outputs from new code.

Historical hashes and extraction mappings document the archived run and cleanup. They do not independently reconstruct the original dirty checkout.
