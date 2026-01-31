"""
Prophet model for MA7 forecasting: build, fit, grid search, and evaluate.

Uses weekly + yearly seasonality and optional custom monthly seasonality.
Country-specific holiday effects can be added when forecasting by region.
"""

import itertools
import numpy as np
import pandas as pd
from prophet import Prophet

from .config import (
    PROPHET_BASELINE_PARAMS,
    PROPHET_GRID_REFINED,
)


def build_prophet_model(
    changepoint_prior_scale: float = None,
    seasonality_prior_scale: float = None,
    changepoint_range: float = None,
    seasonality_mode: str = None,
    add_weekly_monthly: bool = True,
    country_holidays: str = None,
    **kwargs,
) -> Prophet:
    """
    Build a Prophet model with MA7-oriented defaults.

    Parameters
    ----------
    changepoint_prior_scale, seasonality_prior_scale, changepoint_range, seasonality_mode
        Override baseline; defaults from PROPHET_BASELINE_PARAMS.
    add_weekly_monthly : bool
        If True, add custom weekly (period=7) and monthly (period=31) seasonality.
    country_holidays : str
        ISO 2-letter country code for built-in holidays (e.g. 'US', 'DE').
    **kwargs
        Passed to Prophet().
    """
    params = {**PROPHET_BASELINE_PARAMS, **kwargs}
    if changepoint_prior_scale is not None:
        params["changepoint_prior_scale"] = changepoint_prior_scale
    if seasonality_prior_scale is not None:
        params["seasonality_prior_scale"] = seasonality_prior_scale
    if changepoint_range is not None:
        params["changepoint_range"] = changepoint_range
    if seasonality_mode is not None:
        params["seasonality_mode"] = seasonality_mode

    model = Prophet(
        weekly_seasonality=params.get("weekly_seasonality", True),
        yearly_seasonality=params.get("yearly_seasonality", True),
        daily_seasonality=params.get("daily_seasonality", False),
        changepoint_prior_scale=params["changepoint_prior_scale"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
        changepoint_range=params["changepoint_range"],
        seasonality_mode=params["seasonality_mode"],
    )
    if add_weekly_monthly:
        model.add_seasonality(name="weekly", period=7, fourier_order=3)
        model.add_seasonality(name="monthly", period=31, fourier_order=5)
    if country_holidays:
        try:
            model.add_country_holidays(country_name=country_holidays)
        except Exception:
            pass
    return model


def evaluate_mape(
    forecast: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str = "ds",
    value_col: str = "y",
    pred_col: str = "yhat",
) -> float:
    """
    Compute MAPE between forecast and test. Aligns on dates; ignores zeros in denominator.
    """
    fc = forecast[[date_col, pred_col]].set_index(date_col)
    test = test_df[[date_col, value_col]].set_index(date_col)
    aligned = fc.reindex(test.index, method="nearest")
    denom = test[value_col].replace(0, np.nan)
    return float(np.nanmean(np.abs(aligned[pred_col] - test[value_col]) / denom))


def grid_search_prophet(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    param_grid: dict = None,
    verbose: bool = True,
):
    """
    Grid search over Prophet hyperparameters; minimize MAPE on test set.

    param_grid : dict
        Keys: 'changepoint_prior_scale', 'seasonality_prior_scale',
              'changepoint_range', 'seasonality_mode' (lists).
    Returns
    -------
    best_params : dict
    best_mape : float
    results : list of (params_dict, mape)
    """
    param_grid = param_grid or PROPHET_GRID_REFINED
    keys = [
        "changepoint_prior_scale",
        "seasonality_prior_scale",
        "changepoint_range",
        "seasonality_mode",
    ]
    best_mape = float("inf")
    best_params = None
    results = []

    for values in itertools.product(
        param_grid["changepoint_prior_scale"],
        param_grid["seasonality_prior_scale"],
        param_grid["changepoint_range"],
        param_grid["seasonality_mode"],
    ):
        params = dict(zip(keys, values))
        if verbose:
            print(f"Testing {params}")
        model = build_prophet_model(**params)
        model.fit(train_df)
        n_test = len(test_df)
        future = model.make_future_dataframe(periods=n_test, freq="D")
        forecast = model.predict(future)
        mape = evaluate_mape(forecast, test_df)
        results.append((params, mape))
        if verbose:
            print(f"  MAPE: {mape:.2%}")
        if mape < best_mape:
            best_mape = mape
            best_params = params
    if verbose:
        print("Best params:", best_params, "Best MAPE:", best_mape)
    return best_params, best_mape, results


def fit_and_forecast(
    train_df: pd.DataFrame,
    periods: int,
    freq: str = "D",
    params: dict = None,
    country_holidays: str = None,
) -> tuple:
    """
    Fit Prophet on train_df and forecast `periods` ahead.
    Returns (fitted_model, forecast_DataFrame).
    """
    params = params or {}
    model = build_prophet_model(
        changepoint_prior_scale=params.get("changepoint_prior_scale"),
        seasonality_prior_scale=params.get("seasonality_prior_scale"),
        changepoint_range=params.get("changepoint_range"),
        seasonality_mode=params.get("seasonality_mode"),
        country_holidays=country_holidays,
    )
    model.fit(train_df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return model, forecast
