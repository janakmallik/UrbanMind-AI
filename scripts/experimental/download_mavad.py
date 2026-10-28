"""Download, extract, and inspect MAVAD dataset.

NOTE: This script is preserved for historical/exploratory reference only.
The project's current dataset is Urbansas (which includes MAVD, a
Madrid-based subset), not MAVAD (Malta cameras). The two are different
datasets with different content and label schemes.

MAVAD's public release was investigated during Week 9b and found to be
video-only (no audio) despite metadata claims. The project pivoted to
Urbansas, which is genuinely paired audio+video. See docs/experiments.md
for the Week 9b entry that documents this.

Usage (if MAVAD is ever needed again):
    uv run python scripts/experimental/download_mavad.py --camera scrapyard
    uv run python scripts/experimental/download_mavad.py --inspect-only
"""

from __future__ import annotations

import argparse
import os
import time
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://zenodo.org/records/7950008/files"

FILES = {
    "scrapyard": {
        "name": "Masked_Zejtun_Scrapyard.zip",
        "size_gb": 1.82,
        "extract_to": "scrapyard",
    },
    "field": {"name": "Masked_Zejtun_Field.zip", "size_gb": 4.66, "extract_to": "zejtun_field"},
    "mgarr": {"name": "Masked_Mgarr.zip", "size_gb": 5.50, "extract_to": "mgarr"},
    "metadata": {"name": "MARVEL_AV_AD_Metadata.pdf", "size_gb": 0.0001, "extract_to": None},
}


def get_target_dir() -> Path:
    """Resolve MAVAD root from env vars or fallback."""
    env = os.environ.get("MAVAD_DIR")
    if env:
        return Path(env).expanduser().resolve()
    root = os.environ.get("URBANINTEL_DATA_ROOT")
    if root:
        return (Path(root).expanduser() / "mavad").resolve()
    return (Path.home() / "ml-data" / "mavad").resolve()


def download_file(url: str, dest: Path) -> None:
    """Download with progress and resumable if partially exists."""
    existing = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}

    print(f"  Downloading: {dest.name}")
    start = time.time()
    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        mode = "ab" if existing > 0 else "wb"
        total = existing + int(r.headers.get("content-length", 0))
        downloaded = existing
        with open(dest, mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = 100 * downloaded / total
                        mb = downloaded / 1e6
                        speed = mb / max(time.time() - start, 0.001)
                        print(
                            f"\r    {pct:5.1f}%  {mb:7.1f} MB  {speed:5.1f} MB/s",
                            end="",
                            flush=True,
                        )
    print()
    elapsed = time.time() - start
    mb = dest.stat().st_size / 1e6
    print(f"    done: {mb:.1f} MB in {elapsed:.0f}s ({mb / max(elapsed, 0.001):.2f} MB/s)")


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract zip, skipping if already extracted."""
    if dest_dir.exists() and any(dest_dir.rglob("*")):
        count = sum(1 for _ in dest_dir.rglob("*") if _.is_file())
        print(f"  [skip] {dest_dir.name} already has {count} files")
        return

    print(f"  Extracting: {zip_path.name} -> {dest_dir.name}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
    count = sum(1 for _ in dest_dir.rglob("*") if _.is_file())
    print(f"    done: {count} files")


def inspect_dir(dest_dir: Path) -> None:
    """Print structure, file type counts, sample filenames."""
    print(f"\n  === {dest_dir.name} ===")

    # Subdirectories
    dirs = [d for d in dest_dir.rglob("*") if d.is_dir()]
    print(f"  Subdirectories: {len(dirs)}")
    for d in dirs[:10]:
        print(f"    {d.relative_to(dest_dir)}")

    # Extensions
    exts: dict[str, int] = {}
    for f in dest_dir.rglob("*"):
        if f.is_file():
            ext = f.suffix.lower() or "(none)"
            exts[ext] = exts.get(ext, 0) + 1
    print("  File types:")
    for ext, count in sorted(exts.items(), key=lambda x: -x[1]):
        print(f"    {ext}: {count}")

    # Sample filenames
    files = [f for f in dest_dir.rglob("*") if f.is_file()]
    print(f"  Sample (first 10 of {len(files)}):")
    for f in files[:10]:
        print(f"    {f.name}")

    # Total size
    total_bytes = sum(f.stat().st_size for f in files)
    print(f"  Total: {total_bytes / 1e9:.2f} GB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", choices=list(FILES.keys()) + ["all"], default="all")
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--no-inspect", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()

    target = get_target_dir()
    target.mkdir(parents=True, exist_ok=True)
    print(f"Target: {target}")

    cameras = list(FILES.keys()) if args.camera == "all" else [args.camera]

    # Phase 1 — Download
    if not args.inspect_only:
        print("\n--- Download ---")
        for key in cameras:
            spec = FILES[key]
            url = f"{BASE_URL}/{spec['name']}?download=1"
            dest = target / spec["name"]
            if dest.exists() and dest.stat().st_size > 0:
                mb = dest.stat().st_size / 1e6
                print(f"  [skip] {spec['name']} ({mb:.1f} MB)")
                continue
            download_file(url, dest)

    # Phase 2 — Extract
    if not args.no_extract and not args.inspect_only:
        print("\n--- Extract ---")
        for key in cameras:
            spec = FILES[key]
            if spec["extract_to"] is None:
                continue
            zip_path = target / spec["name"]
            dest_dir = target / spec["extract_to"]
            if not zip_path.exists():
                print(f"  [skip] {spec['name']} not downloaded")
                continue
            extract_zip(zip_path, dest_dir)

    # Phase 3 — Inspect
    if not args.no_inspect:
        print("\n--- Inspect ---")
        for key in cameras:
            spec = FILES[key]
            if spec["extract_to"] is None:
                continue
            dest_dir = target / spec["extract_to"]
            if not dest_dir.exists():
                print(f"  [skip] {spec['extract_to']} not extracted")
                continue
            inspect_dir(dest_dir)

    print(f"\nDone. Data at: {target}")


if __name__ == "__main__":
    main()
