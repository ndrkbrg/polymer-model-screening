"""Check that public entry points cannot overwrite distributed scientific evidence."""

import pytest
from coilnet.paths import ROOT, output_directory
from coilnet.experiments.run_q1_upgrade import main as run_original
from coilnet.experiments.run_revision_phase_c import main as run_revision
from coilnet.paper.make_q1_figures import main as make_figures


@pytest.mark.parametrize(
    "relative", ["data/original", "data/revision/pilot", "configs", "figures/main"]
)
def test_distributed_locations_are_rejected(relative):
    with pytest.raises(ValueError, match="archived inputs"):
        output_directory(ROOT / relative)


def test_repo_parent_cannot_be_output():
    with pytest.raises(ValueError):
        output_directory(ROOT.parent)


def test_work_and_external_directories_are_accepted(tmp_path):
    assert output_directory(ROOT / "work/new") == (ROOT / "work/new").resolve()
    assert output_directory(tmp_path) == tmp_path.resolve()


def test_traversal_out_of_work_into_archived_inputs_is_rejected():
    with pytest.raises(ValueError):
        output_directory(ROOT / "work/../data/original")


@pytest.mark.parametrize("runner", [run_original, run_revision, make_figures])
def test_cli_guard_runs_before_any_archived_file_writes(runner):
    with pytest.raises(ValueError, match="archived inputs"):
        runner(["--out", str(ROOT / "data/original")])


def test_analysis_import_does_not_execute_or_write():
    from scripts import analyze_revision, check_aggregates

    assert callable(analyze_revision.main)
    assert callable(check_aggregates.main)


def test_empty_original_groups_fail_without_creating_output(tmp_path):
    destination = tmp_path / "empty-run"
    with pytest.raises(SystemExit):
        run_original(["--groups", "", "--out", str(destination)])
    assert not destination.exists()
