"""
Export forecasts to CSV: by region and/or combined.
"""

import pandas as pd
from pathlib import Path

from .config import OUTPUT_DIR


def export_forecasts_to_csv(
    forecasts: dict,
    out_dir: Path = None,
    region_col_name: str = "Country",
    date_col: str = "ds",
    value_col: str = "yhat",
    combined_path: Path = None,
) -> list:
    """
    Write per-region forecast DataFrames to CSV and optionally one combined file.

    forecasts : dict
        region -> DataFrame with at least [date_col, value_col]
    out_dir : Path
        Directory for per-region files (forecast_<region>.csv).
    region_col_name : str
        Name of region column in combined CSV.
    combined_path : Path
        If set, write combined CSV with columns [date_col, value_col, region_col_name].
    Returns
    -------
    List of written file paths.
    """
    out_dir = Path(out_dir or OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for region, df in forecasts.items():
        path = out_dir / f"forecast_{region}.csv"
        df[[date_col, value_col]].to_csv(path, index=False)
        written.append(path)

    if combined_path is not None:
        combined_path = Path(combined_path)
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        parts = []
        for region, df in forecasts.items():
            part = df[[date_col, value_col]].copy()
            part[region_col_name] = region
            parts.append(part)
        combined = pd.concat(parts, ignore_index=True)
        combined.to_csv(combined_path, index=False)
        written.append(combined_path)

    return written
