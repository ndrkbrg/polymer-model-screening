"""Repository paths and protection for archived scientific inputs."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "data/original"
REVISION = ROOT / "data/revision"
CONFIGS = ROOT / "configs"
WORK = ROOT / "work"


def output_directory(path):
    """Allow work/ or external outputs, but never overwrite distributed files."""
    path = Path(path).expanduser().resolve()
    root = ROOT.resolve()
    work = WORK.resolve()
    if path == root or root.is_relative_to(path):
        raise ValueError("Output cannot be the repository or its parent.")
    if path.is_relative_to(root) and not path.is_relative_to(work):
        raise ValueError("Use work/ or an external directory; archived inputs are read-only.")
    return path
