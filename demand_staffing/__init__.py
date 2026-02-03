"""
Demand forecasting and staffing optimization for capacity planning.

Uses a 7-day moving average (MA7) of daily counts and Prophet for time-series
forecasting, with optional per-region calibration and grid search. Includes
GA-based staffing optimization to maximize net profit under real-world constraints.
"""

from .config import DEFAULT_SPLIT_DATE, PROPHET_BASELINE_PARAMS
from .data_loader import (
    load_daily_counts,
    compute_ma7,
    get_prophet_data,
    prepare_global_ma7,
    prepare_per_region_ma7,
)
from .forecast import (
    build_prophet_model,
    fit_and_forecast,
    grid_search_prophet,
    evaluate_mape,
)
from .export import export_forecasts_to_csv
from .optimization import (
    load_forecast_weekly,
    load_salary_and_staffing,
    ga_optimize,
    simulate,
    staffing_plan_from_chromosome,
)

__all__ = [
    "DEFAULT_SPLIT_DATE",
    "PROPHET_BASELINE_PARAMS",
    "load_daily_counts",
    "compute_ma7",
    "get_prophet_data",
    "prepare_global_ma7",
    "prepare_per_region_ma7",
    "build_prophet_model",
    "fit_and_forecast",
    "grid_search_prophet",
    "evaluate_mape",
    "export_forecasts_to_csv",
    "load_forecast_weekly",
    "load_salary_and_staffing",
    "ga_optimize",
    "simulate",
    "staffing_plan_from_chromosome",
]
