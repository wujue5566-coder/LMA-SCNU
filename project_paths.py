"""Portable project paths and directory-link helpers."""

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def _path_from_env(name, default):
    return Path(os.environ.get(name, str(default))).expanduser()


TEIKE_ROOT = _path_from_env("LMAFORMER_TEIKE_ROOT", REPO_ROOT / "Datasets" / "TeiKe")
TEIKE_DATA_ROOT = TEIKE_ROOT / "data"
LMADATASET_ROOT = REPO_ROOT / "Datasets" / "TeiKe_LMAFormer"


def create_directory_link(source, target):
    """Create a directory link without duplicating dataset files."""
    source = Path(source).resolve()
    target = Path(target)
    if target.exists() or target.is_symlink():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)

    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(target), str(source)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "could not create junction {} -> {}: {}".format(
                    target, source, (result.stderr or result.stdout).strip()
                )
            )
    else:
        try:
            target.symlink_to(source, target_is_directory=True)
        except OSError as exc:
            raise RuntimeError(
                "could not create symlink {} -> {}: {}".format(target, source, exc)
            ) from exc
    return True
