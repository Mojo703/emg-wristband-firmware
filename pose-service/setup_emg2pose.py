#!/usr/bin/env python3
"""Download and install the Meta emg2pose model and its UmeTrack dependency.

Usage:
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pip install -r requirements-emg2pose.txt
    python setup_emg2pose.py

This script:
  1. Clones the facebookresearch/emg2pose repository into vendor/emg2pose.
  2. Installs the emg2pose package and the UmeTrack submodule in editable mode.
  3. Downloads the pre-trained checkpoints into checkpoints/.

The model is licensed under CC-BY-NC-SA-4.0; see the upstream repository for
terms. The checkpoints are hosted by Meta's open-source S3 bucket.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("setup_emg2pose")

REPO_URL = "https://github.com/facebookresearch/emg2pose.git"
CHECKPOINT_URL = "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_model_checkpoints.tar.gz"

DEFAULT_VENDOR = "vendor"
DEFAULT_CHECKPOINTS = "checkpoints"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    logger.info("running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def clone_repo(vendor_dir: Path, recursive: bool = True) -> Path:
    repo_dir = vendor_dir / "emg2pose"
    if repo_dir.exists():
        logger.info("emg2pose repo already exists at %s", repo_dir)
        return repo_dir

    vendor_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone"]
    if recursive:
        cmd.append("--recursive")
    cmd.extend([REPO_URL, str(repo_dir)])
    run(cmd, cwd=vendor_dir.parent if vendor_dir.name == "vendor" else vendor_dir)
    return repo_dir


def install_packages(repo_dir: Path) -> None:
    # Install the emg2pose package itself.
    run([sys.executable, "-m", "pip", "install", "-e", "."], cwd=repo_dir)
    # Install the UmeTrack submodule.
    umetrack_dir = repo_dir / "emg2pose" / "UmeTrack"
    if umetrack_dir.exists():
        run([sys.executable, "-m", "pip", "install", "-e", str(umetrack_dir)])
    else:
        logger.warning("UmeTrack submodule not found at %s", umetrack_dir)


def download_checkpoint(checkpoints_dir: Path) -> None:
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    expected_ckpt = checkpoints_dir / "tracking_vemg2pose.ckpt"
    if expected_ckpt.exists():
        logger.info("checkpoint already present: %s", expected_ckpt)
        return

    logger.info("downloading checkpoints from %s", CHECKPOINT_URL)
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = Path(tmpdir) / "emg2pose_model_checkpoints.tar.gz"
        with urlopen(CHECKPOINT_URL) as response, open(tar_path, "wb") as f:
            shutil.copyfileobj(response, f)
        logger.info("extracting checkpoints to %s", checkpoints_dir)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=checkpoints_dir)

    # The tar extracts into a subdirectory; flatten it if needed.
    extracted_subdir = checkpoints_dir / "emg2pose_model_checkpoints"
    if extracted_subdir.exists():
        for child in extracted_subdir.iterdir():
            dest = checkpoints_dir / child.name
            if dest.exists():
                continue
            shutil.move(str(child), str(dest))
        extracted_subdir.rmdir()

    logger.info("checkpoints ready in %s", checkpoints_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Install Meta emg2pose and download checkpoints")
    parser.add_argument(
        "--vendor-dir",
        type=Path,
        default=Path(__file__).parent / DEFAULT_VENDOR,
        help="Directory to clone the emg2pose repository into",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path(__file__).parent / DEFAULT_CHECKPOINTS,
        help="Directory to download model checkpoints into",
    )
    parser.add_argument(
        "--skip-repo",
        action="store_true",
        help="Skip cloning the repository (use if already cloned)",
    )
    parser.add_argument(
        "--skip-checkpoint",
        action="store_true",
        help="Skip downloading the checkpoint (use if already present)",
    )
    args = parser.parse_args()

    repo_dir = None
    if not args.skip_repo:
        repo_dir = clone_repo(args.vendor_dir)

    if repo_dir and not args.skip_repo:
        install_packages(repo_dir)

    if not args.skip_checkpoint:
        download_checkpoint(args.checkpoints_dir)

    logger.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
