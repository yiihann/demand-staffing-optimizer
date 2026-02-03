"""
Load raw event data, aggregate to daily counts, and compute MA7 for Prophet.

Expects a CSV with a date column and optional region column (e.g. country).
Produces Prophet-ready DataFrames with columns 'ds' and 'y'.
"""

import pandas as pd
from pathlib import Path
from .config import (
    DATE_COL,
    REGION_COL,
    MA7_WINDOW,
    DEFAULT_DATA_PATH,
    DEFAULT_SPLIT_DATE,
)


def load_raw(data_path: Path = None) -> pd.DataFrame:
    """Load raw CSV; ensure date column is datetime."""
    path = data_path or DEFAULT_DATA_PATH
    if not Path(path).exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df


def load_daily_counts(
    data_path: Path = None,
    date_col: str = DATE_COL,
    region_col: str = REGION_COL,
) -> pd.DataFrame:
    """
    Aggregate to daily counts. If region_col is present, counts are per region and date.
    Otherwise returns global daily counts with columns [date_col, 'SignUps'].
    """
    df = load_raw(data_path)
    if region_col and region_col in df.columns:
        counts = (
            df.groupby([region_col, date_col]).size().reset_index(name="SignUps")
        )
        counts = counts.sort_values([region_col, date_col])
    else:
        counts = df.groupby(date_col).size().reset_index(name="SignUps")
        counts = counts.sort_values(date_col)
    return counts


def _global_daily_to_ma7(data_path=None):
    """Internal: global daily counts -> MA7 series (ds, y)."""
    counts = load_daily_counts(data_path, region_col=None)
    counts[DATE_COL] = pd.to_datetime(counts[DATE_COL])
    counts["MA7"] = counts["SignUps"].rolling(window=MA7_WINDOW).mean()
    counts = counts.dropna(subset=["MA7"])
    return counts[[DATE_COL, "MA7"]].rename(columns={DATE_COL: "ds", "MA7": "y"})


def compute_ma7(
    counts: pd.DataFrame,
    date_col: str = DATE_COL,
    value_col: str = "SignUps",
    region_col: str = None,
    window: int = MA7_WINDOW,
) -> pd.DataFrame:
    """
    Compute 7-day moving average. Drops rows with NaN from the rolling window.
    If region_col is set, MA7 is computed within each region.
    """
    out = counts.copy()
    if region_col and region_col in out.columns:
        out["MA7"] = (
            out.groupby(region_col)[value_col].transform(
                lambda x: x.rolling(window=window).mean()
            )
        )
    else:
        out["MA7"] = out[value_col].rolling(window=window).mean()
    out = out.dropna(subset=["MA7"])
    return out


def get_prophet_data(
    counts: pd.DataFrame,
    date_col: str = DATE_COL,
    value_col: str = "MA7",
    region_col: str = None,
    region: str = None,
) -> pd.DataFrame:
    """
    Return a DataFrame with columns 'ds' and 'y' for Prophet.
    If region_col and region are given, filter to that region first.
    """
    df = counts.copy()
    if region_col and region is not None and region_col in df.columns:
        df = df[df[region_col] == region]
    df = df[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})
    return df.dropna()


def prepare_global_ma7(
    data_path: Path = None,
    split_date: str = None,
):
    """
    Load data, aggregate globally to daily counts, compute MA7, and split train/test.
    Returns (train_df, test_df) in Prophet format ('ds', 'y').
    """
    df = _global_daily_to_ma7(data_path=data_path)
    split_dt = pd.to_datetime(split_date or DEFAULT_SPLIT_DATE)
    train_df = df[df["ds"] <= split_dt]
    test_df = df[df["ds"] > split_dt]
    return train_df, test_df


def prepare_per_region_ma7(
    data_path: Path = None,
    split_date: str = None,
):
    """
    Load data, aggregate by region and date, compute MA7 per region.
    Returns dict: region -> (train_df, test_df) in Prophet format.
    """
    from .config import DEFAULT_SPLIT_DATE

    split_date = pd.to_datetime(split_date or DEFAULT_SPLIT_DATE)
    counts = load_daily_counts(data_path, region_col=REGION_COL)
    counts = compute_ma7(counts, date_col=DATE_COL, value_col="SignUps", region_col=REGION_COL)
    result = {}
    for region in counts[REGION_COL].unique():
        df = get_prophet_data(
            counts, date_col=DATE_COL, value_col="MA7", region_col=REGION_COL, region=region
        )
        df["ds"] = pd.to_datetime(df["ds"])
        train_df = df[df["ds"] <= split_date]
        test_df = df[df["ds"] > split_date]
        result[region] = (train_df, test_df)
    return result
