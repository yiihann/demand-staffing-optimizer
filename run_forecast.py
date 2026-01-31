#!/usr/bin/env python3
"""
Run demand forecasting: MA7 + Prophet.

Usage:
  python run_forecast.py [--data PATH] [--split-date DATE] [--global | --by-country]
                        [--grid-search] [--periods N] [--out-dir DIR]

  --by-country     One Prophet model per country/region (default: single global model).
  --grid-search   Run hyperparameter grid search (global model only).
  --periods N     Forecast N days ahead (default: 365).
"""

import argparse
from pathlib import Path

from demand_forecasting.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_SPLIT_DATE,
    OUTPUT_DIR,
    PROPHET_BASELINE_PARAMS,
)
from demand_forecasting.data_loader import (
    prepare_global_ma7,
    prepare_per_region_ma7,
)
from demand_forecasting.forecast import (
    fit_and_forecast,
    grid_search_prophet,
    evaluate_mape,
)
from demand_forecasting.export import export_forecasts_to_csv


# ISO 2-letter codes for Prophet built-in country holidays
COUNTRY_HOLIDAYS = {
    "Argentina": "AR",
    "Australia": "AU",
    "Brazil": "BR",
    "Canada": "CA",
    "China": "CN",
    "France": "FR",
    "Germany": "DE",
    "India": "IN",
    "Indonesia": "ID",
    "Italy": "IT",
    "Japan": "JP",
    "Mexico": "MX",
    "Poland": "PL",
    "South Africa": "ZA",
    "South Korea": "KR",
    "Spain": "ES",
    "Thailand": "TH",
    "Turkey": "TR",
    "UAE": "AE",
    "UK": "GB",
    "USA": "US",
    "Vietnam": "VN",
}


def main():
    p = argparse.ArgumentParser(description="Demand forecasting: MA7 + Prophet")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Input CSV path")
    p.add_argument("--split-date", default=DEFAULT_SPLIT_DATE, help="Train/test split date")
    p.add_argument("--by-country", action="store_true", help="One model per country")
    p.add_argument("--grid-search", action="store_true", help="Run hyperparameter grid search (global)")
    p.add_argument("--periods", type=int, default=365, help="Forecast horizon (days)")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR, help="Output directory for CSVs")
    args = p.parse_args()

    if not args.by_country:
        train_df, test_df = prepare_global_ma7(data_path=args.data, split_date=args.split_date)
        if args.grid_search:
            best_params, best_mape, _ = grid_search_prophet(train_df, test_df)
            params = best_params
            print("Best MAPE:", best_mape)
        else:
            params = PROPHET_BASELINE_PARAMS
        model, forecast = fit_and_forecast(train_df, periods=args.periods, params=params)
        mape = evaluate_mape(forecast, test_df)
        print("Test MAPE (global):", mape)
        # Export: single "global" region
        out_dir = args.out_dir / "forecasts"
        out_dir.mkdir(parents=True, exist_ok=True)
        forecast[["ds", "yhat"]].to_csv(out_dir / "forecast_global.csv", index=False)
        print("Wrote", out_dir / "forecast_global.csv")
        return

    # Per-country
    region_data = prepare_per_region_ma7(data_path=args.data, split_date=args.split_date)
    forecasts = {}
    for region, (train_df, test_df) in region_data.items():
        if len(train_df) < 2:
            continue
        holidays = COUNTRY_HOLIDAYS.get(region)
        model, forecast = fit_and_forecast(
            train_df,
            periods=args.periods,
            params=PROPHET_BASELINE_PARAMS,
            country_holidays=holidays,
        )
        mape = evaluate_mape(forecast, test_df)
        print(f"{region}: MAPE = {mape:.2%}")
        forecasts[region] = forecast

    out_dir = args.out_dir / "forecasts"
    export_forecasts_to_csv(
        forecasts,
        out_dir=out_dir,
        region_col_name="Country",
        combined_path=out_dir / "combined_predictions.csv",
    )
    print("Wrote", out_dir)


if __name__ == "__main__":
    main()
