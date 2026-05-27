#!/usr/bin/env python3
"""Retroactively apply a distance-to-trench filter to training_data_global.csv.

Intended for runs where data was already extracted without the distance filter
baked into coregister_combined_point_data.  Positive rows beyond the buffer
distance are dropped; unlabelled rows are kept.  Overwrites in place with a
.bak backup.

Usage (from repo root):
  python clip_trench_distance.py config/all_features_9deg.yml
  python clip_trench_distance.py config/all_features_9deg.yml --dry-run
"""

import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from lib.paths import PathConfigManager

DIST_COL = "distance_to_trench (km)"


def _filter_file(path: Path, cutoff_km: float, positive_only: bool, dry_run: bool) -> None:
    if not path.is_file():
        print(f"  SKIP   {path.name}  (file not found)")
        return

    df = pd.read_csv(path)

    if DIST_COL not in df.columns:
        print(
            f"  SKIP   {path.name}  (no '{DIST_COL}' column — "
            "likely a pre-coregistration intermediate; filter via the pipeline instead)"
        )
        return

    n_before = len(df)

    if positive_only and "label" in df.columns:
        # Keep unlabelled/negative rows always; filter positive rows by distance
        is_positive = df["label"] == "positive"
        within_range = df[DIST_COL] <= cutoff_km
        keep = ~is_positive | within_range
        n_pos_dropped = int((is_positive & ~within_range).sum())
        df_out = df[keep]
        detail = f"{n_pos_dropped} positive rows dropped"
    else:
        # No label column (e.g. grid_data): filter all rows
        keep = df[DIST_COL] <= cutoff_km
        df_out = df[keep]
        detail = f"{n_before - len(df_out)} rows dropped"

    n_after = len(df_out)
    print(f"  FILTER {path.name}  {n_before} → {n_after} rows  ({detail})")

    if not dry_run:
        backup = path.with_suffix(".csv.bak")
        shutil.copy2(path, backup)
        df_out.to_csv(path, index=False)
        print(f"         Backup saved: {backup.name}")


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        print(f"Usage: {sys.argv[0]} <config.yml> [--dry-run]")
        sys.exit(1)

    config_path = args[0]
    dry_run = "--dry-run" in args

    pcm = PathConfigManager(config_path)
    buffer_distance: float = pcm.config["study_zone_buffer"]
    cutoff_km = buffer_distance * 111.32

    print(f"Config:      {config_path}")
    print(f"Buffer:      {buffer_distance}°  →  cutoff = {cutoff_km:.1f} km")
    print(f"Dry run:     {dry_run}")
    print()

    _filter_file(pcm.TRAINING_DATA_PATH, cutoff_km, positive_only=True, dry_run=dry_run)

    print()
    if dry_run:
        print("Dry run complete — no files were modified.")
    else:
        print("Done.")


if __name__ == "__main__":
    main()
