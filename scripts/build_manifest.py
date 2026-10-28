"""Create a new integrity manifest after reviewing a release snapshot."""

from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "work",
    "results",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
}
EXCLUDED_FILES = {"MANIFEST.json", ".DS_Store", "Thumbs.db"}


def release_files():
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_DIRS or part.endswith(".egg-info") for part in relative.parts):
            continue
        if relative.name in EXCLUDED_FILES or path.suffix in {".pyc", ".pyo"}:
            continue
        yield path


def main():
    manifest = {
        "algorithm": "SHA-256",
        "self_excluded": True,
        "files": {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in release_files()
        },
    }
    (ROOT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest contains {len(manifest['files'])} reviewed candidate files.")


if __name__ == "__main__":
    main()
