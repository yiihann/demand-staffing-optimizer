# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup
```bash
pip install -r requirements.txt
```

### Run forecasting
```bash
# Global model
python run_forecast.py

# Per-country with holidays, grid search, custom output
python run_forecast.py --by-country --grid-search --out-dir ./outputs
```

Key flags: `--data PATH`, `--split-date DATE`, `--by-country`, `--grid-search`, `--periods N`, `--out-dir DIR`

### Run GA optimization
```bash
python run_optimization.py --forecast outputs/forecasts/combined_predictions.csv

# Tune GA parameters
python run_optimization.py --weeks 52 --pop 100 --gen 200 --mut 0.2 --seed 42
```

Key flags: `--forecast PATH`, `--weeks N`, `--pop N`, `--gen N`, `--mut RATE`, `--seed N`, `--out-dir DIR`

There are no automated tests.

## Architecture

Two-stage pipeline with separate entry scripts:

1. **Forecasting** (`run_forecast.py` → `demand_staffing/`)
   - `data_loader.py`: Loads `data/AdvSignUp.csv`, aggregates to daily counts, applies 7-day moving average (MA7), formats for Prophet (`ds`, `y` columns)
   - `forecast.py`: Builds and fits Facebook Prophet models with custom weekly/monthly seasonality; supports country-specific holidays and grid search over `changepoint_prior_scale` / `seasonality_prior_scale`
   - `export.py`: Writes per-region `forecast_{country}.csv` and a combined `combined_predictions.csv`

2. **Optimization** (`run_optimization.py` → `demand_staffing/optimization.py`)
   - Reads the combined forecast CSV plus salary/staffing CSVs from `data/`
   - Runs a Genetic Algorithm (population of weekly hire/fire delta vectors) to maximize net profit
   - `simulate()` models agent ramp-up (4-week delay), FIFO backlog (60-day expiration), and financials (revenue uplift, salary, hire/fire costs)
   - Outputs `staffing_plan.csv` and `optimization_summary.txt` to `outputs/optimization/`

**All hyperparameters and file paths are centralized in `demand_staffing/config.py`.**

## Key Data Flow

```
data/AdvSignUp.csv
  → daily aggregation → MA7 smoothing → Prophet train/test split
  → [optional grid search] → Prophet fit → forecast CSVs

outputs/forecasts/combined_predictions.csv + salary/staffing CSVs
  → weekly demand aggregation → GA chromosome evolution
  → simulate() → staffing_plan.csv + optimization_summary.txt
```

## Important Constants (optimization.py)

| Constant | Value | Meaning |
|---|---|---|
| `WEEKS_RAMP` | 4 | Weeks before new hires are fully productive |
| `DAYS_BACKLOG_LIMIT` | 60 | FIFO backlog expiration (days) |
| `AGENTS_PER_SLOT` | 10 | Advertisers managed per agent |
| `EXPECTED_UPLIFT` | 0.135 | Incremental revenue factor (13.5%) |
| `FIRE_PENALTY_RATIO` | 0.40 | Firing cost as fraction of annual salary |
| `MAX_HIRE_PER_WEEK` | 10 | GA chromosome clamp per gene |
