# demand-staffing-optimizer

Demand forecasting and dynamic staffing optimization: **7-day moving average (MA7)** + **Prophet** for time-series demand, plus a **Genetic Algorithm (GA)** for hire/fire decisions. The method smooths daily noise, captures weekly and yearly seasonality, and supports optional per-region models; the GA then optimizes hire/fire decisions to maximize net profit (revenue − costs − lost revenue) under 60-day backlog and ramp constraints.

## Method

1. **Aggregation**  
   Raw event data (e.g. daily sign-ups) is aggregated to daily counts, optionally by region (e.g. country).

2. **MA7**  
   A 7-day moving average is computed to reduce outliers and improve forecast stability (MAPE on raw daily targets was high; predicting MA7 gives more reliable metrics).

3. **Prophet**  
   Prophet is fit on the MA7 series with:
   - Weekly and yearly seasonality
   - Custom monthly seasonality (Fourier terms)
   - Low trend flexibility (`changepoint_prior_scale=0.02`, `changepoint_range=0.8`)
   - Additive seasonality

4. **Calibration**  
   - **Global**: one model on all regions; optional grid search over Prophet hyperparameters.
   - **Per-region**: one Prophet model per region with optional built-in country holidays for better MAPE.

5. **Output**  
   Forecasts are written to CSV (per region and/or combined).

## Project layout (code only)

```
.
├── README-forecasting.md          # This file
├── requirements-forecasting.txt   # Python dependencies
├── run_forecast.py                # CLI: forecasting (MA7 + Prophet)
├── run_optimization.py            # CLI: GA staffing optimization
└── demand_forecasting/            # Package
    ├── __init__.py
    ├── config.py                  # Paths, split date, Prophet defaults
    ├── data_loader.py             # Load CSV, daily counts, MA7, train/test split
    ├── forecast.py                # Prophet build, grid search, MAPE
    ├── export.py                  # Export forecasts to CSV
    └── optimization.py            # GA: simulator, fitness, staffing plan
```

## Setup

```bash
pip install -r requirements-forecasting.txt
```

Place your input CSV under `data/` with at least:
- A date column (default: `Sign_Up_Date`)
- Optional region column (default: `Country`) for per-region forecasting

Default input path: `data/AdvSignUp.csv`.

## Usage

**Global model (single MA7 series, one Prophet):**
```bash
python run_forecast.py --data data/AdvSignUp.csv
```

**Global model with hyperparameter grid search:**
```bash
python run_forecast.py --grid-search
```

**Per-region models (one Prophet per country, with holidays where supported):**
```bash
python run_forecast.py --by-country
```

**Custom split date and forecast horizon:**
```bash
python run_forecast.py --split-date 2024-06-24 --periods 365 --out-dir outputs
```

Outputs are written under `outputs/forecasts/` (or `--out-dir`):  
`forecast_<region>.csv` and optionally `combined_predictions.csv`.

### Optimization (GA)

After forecasts exist (e.g. `results_prediction/combined_predictions.csv`), run the GA to get a staffing plan:

```bash
python run_optimization.py --forecast results_prediction/combined_predictions.csv
```

Optional: `--weeks 52 --pop 50 --gen 80 --mut 0.15 --seed 42`.  
Uses salary and existing staffing from `data/[BADSS case] Agent Salary...` and `...Existing Agent Staffing...`.  
Writes `outputs/optimization/staffing_plan.csv` (week, agents_total, agents_active, delta) and `optimization_summary.txt`.

**GA (per report):** Chromosome = weekly staffing deltas (hire/fire per week, max ±10). Fitness = net profit (revenue − salary − hire cost − fire cost − lost revenue). Constraints: 60-day backlog, 1-month ramp for new hires, labor efficiency 95%.

## References

- **Prophet**: [facebook/prophet](https://github.com/facebook/prophet)  
- **Idea**: Forecast a smoothed target (MA7) for stability; use Prophet for multiple seasonality and automatic trend changepoints.
