"""
Configuration: paths, split date, and default Prophet hyperparameters.

Tuned for forecasting a 7-day moving average (MA7) of daily demand counts
with weekly and yearly seasonality and moderate trend flexibility.
"""

from pathlib import Path

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Default input file: daily events with columns [date_col, (optional) region_col]
DEFAULT_DATA_PATH = DATA_DIR / "AdvSignUp.csv"
DATE_COL = "Sign_Up_Date"
VALUE_COL = "SignUps"  # after aggregation
REGION_COL = "Country"  # optional; if None, global aggregation only

# Train/test split (forecast method was validated with train before this date)
DEFAULT_SPLIT_DATE = "2024-06-24"

# Prophet baseline (global tuning from MA7 grid search; report: MAPE ~19–20%)
PROPHET_BASELINE_PARAMS = {
    "changepoint_prior_scale": 0.02,
    "seasonality_prior_scale": 0.18,
    "changepoint_range": 0.8,
    "seasonality_mode": "additive",
    "weekly_seasonality": True,
    "yearly_seasonality": True,
    "daily_seasonality": False,
}

# Refined grid for country-level tuning (narrow around baseline)
PROPHET_GRID_REFINED = {
    "changepoint_prior_scale": [0.015, 0.02, 0.025],
    "seasonality_prior_scale": [0.16, 0.18, 0.20],
    "changepoint_range": [0.75, 0.8, 0.85],
    "seasonality_mode": ["additive"],
}

# MA7 window
MA7_WINDOW = 7
