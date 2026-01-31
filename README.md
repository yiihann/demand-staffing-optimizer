# Demand–Staffing Optimizer

## Overview

This project implements a **demand forecasting and dynamic staffing optimization** pipeline for capacity planning. It combines **time-series forecasting** (7-day moving average + Prophet) with a **Genetic Algorithm (GA)** to produce hire/fire decisions that maximize net profit under real-world constraints: 60-day service backlog, 1-month agent ramp-up, and labor efficiency.

The system guides the user from raw event data (e.g. daily sign-ups by region) through **aggregation and MA7 smoothing**, **Prophet-based demand forecasts** (with optional per-region calibration and grid search), and **GA-driven staffing plans** that balance revenue, salary, hiring/firing costs, and lost revenue from unmet demand.

## Pipeline Architecture

The project is organized into a Python package and two entrypoint scripts:

- **`demand_forecasting/config.py`**  
  Paths, train/test split date, and default Prophet and GA parameters.

- **`demand_forecasting/data_loader.py`**  
  Loads raw CSV, aggregates to daily counts (global or by region), computes 7-day moving average (MA7), and prepares Prophet-ready train/test splits.  
  **Role**: Data ingestion and preprocessing.

- **`demand_forecasting/forecast.py`**  
  Builds Prophet models (weekly/yearly/monthly seasonality, additive), runs hyperparameter grid search, and evaluates MAPE.  
  **Role**: Time-series forecasting.

- **`demand_forecasting/export.py`**  
  Writes forecast CSVs per region and/or combined.  
  **Role**: Output of forecasting stage.

- **`demand_forecasting/optimization.py`**  
  Loads forecast and salary/staffing data, simulates capacity and backlog (FIFO, 60-day limit), and runs a Genetic Algorithm over weekly staffing deltas to maximize net profit (revenue − salary − hire cost − fire cost − lost revenue).  
  **Role**: Staffing optimization.

- **`run_forecast.py`**  
  CLI for the forecasting pipeline: global or per-region Prophet, optional grid search. Produces forecast CSVs.  
  **Usage**: Run this first to generate demand forecasts.

- **`run_optimization.py`**  
  CLI for the GA: reads forecast CSVs and salary/staffing CSVs, runs GA, and writes staffing plan and summary.  
  **Usage**: Run after forecasts exist.

## Pipeline Execution Flow

1. **Data**  
   Place input CSVs in `data/`: daily events (date, optional region), and optionally salary and existing staffing tables.

2. **Forecast**  
   `run_forecast.py` aggregates to daily counts, computes MA7, fits Prophet (global or per-region), and exports forecast CSVs (e.g. `combined_predictions.csv`).

3. **Optimization**  
   `run_optimization.py` loads the forecast and salary/staffing data, converts monthly demand to weekly, and runs the GA over weekly hire/fire deltas (max ±10 agents per week, 1-month ramp, 60-day backlog). Fitness is net profit.

4. **Output**  
   The GA outputs a staffing plan (week, agents_total, agents_active, delta) and a summary (revenue, costs, lost revenue, net profit).

## How to Run

1. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
   Key dependencies: `pandas`, `numpy`, `prophet`.

2. **Set Up the Environment**  
   - Place your input CSV in `data/` with a date column (default: `Sign_Up_Date`) and optional region column (default: `Country`).  
   - For optimization, place salary and existing staffing CSVs in `data/` (see repo for expected column names).

3. **Run Forecasting**  
   ```bash
   python run_forecast.py --by-country
   ```
   Optional: `--grid-search` for hyperparameter tuning; `--out-dir` to set output directory. Forecasts are written under `outputs/forecasts/` (or `results_prediction/` if configured).

4. **Run Optimization**  
   After forecasts exist, run:
   ```bash
   python run_optimization.py --forecast results_prediction/combined_predictions.csv
   ```
   Optional: `--weeks`, `--pop`, `--gen`, `--mut`, `--seed`. Outputs are written under `outputs/optimization/` (staffing plan CSV and summary).

## About

Demand forecasting (MA7 + Prophet) and GA-based staffing optimization for capacity planning. Built for a case competition; repurposed as a standalone Python pipeline.
