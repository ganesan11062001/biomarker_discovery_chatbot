"""Generate a root-level Posit Connect manifest.json for the FastAPI app.

Walks git-tracked files only (avoids .venv, data, outputs, etc.) and writes
manifest.json at the repo root with entrypoint api.main:app.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXCLUDE_PREFIXES = (
    "data/", "outputs/", "frontend/", "tests/", "r_scripts/",
    ".vscode/", ".claude/", ".github/", "claude", "skills/",
    "ui/",  # streamlit UI not needed for API
)
EXCLUDE_SUFFIXES = (".pyc", ".pyo")
EXCLUDE_NAMES = {".env", "manifest.json", ".rscignore"}


def tracked_files() -> list[str]:
    out = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True, encoding="utf-8"
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    files: dict[str, dict] = {}
    for rel in tracked_files():
        rel_posix = rel.replace("\\", "/")
        if rel_posix in EXCLUDE_NAMES:
            continue
        if any(rel_posix.startswith(p) for p in EXCLUDE_PREFIXES):
            continue
        if rel_posix.endswith(EXCLUDE_SUFFIXES):
            continue
        p = ROOT / rel
        if not p.is_file():
            continue
        files[rel_posix] = {"checksum": md5(p)}

    manifest = {
        "version": 1,
        "metadata": {
            "appmode": "python-fastapi",
            "entrypoint": "api.main:app",
        },
        "python": {
            "version": "3.12.4",
            "package_manager": {
                "name": "pip",
                "version": "24.0",
                "package_file": "requirements.txt",
            },
        },
        "environment": {"python": {"requires": "~=3.12.0"}},
        "files": dict(sorted(files.items())),
    }

    out_path = ROOT / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} with {len(files)} files")


if __name__ == "__main__":
    main()
