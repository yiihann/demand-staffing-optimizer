# BADSS: Business Agent Dynamic Staffing System
## A Data-Driven Approach to Global SME Advertiser Support Optimization

---

## Executive Summary

This project builds a two-stage machine learning pipeline that ingests 499,644 advertiser sign-up records across 22 countries, forecasts future demand with ~19% MAPE, and optimizes a 52-week staffing plan using a Genetic Algorithm. The optimizer finds a strategy that **outperforms both a static headcount baseline and an aggressive downsizing baseline**, delivering a projected **net profit of $750.7M** on $1.03B in revenue — while losing only **5 out of 267,557 eligible advertisers** to queue expiration.

---

## 1. The Problem

Google's SME advertiser support team faces a dynamic staffing challenge: how many agents should be assigned to handle incoming advertisers across 22 countries, given that:

- Demand varies seasonally and differs by country
- New agents require a **4-week ramp-up** before becoming fully productive
- Hiring and firing carry real costs (1 month salary to hire; 40% of annual salary to fire)
- Advertisers that wait more than 60 days without being served are lost permanently
- Only advertisers meeting a **country-specific budget eligibility threshold** are in-scope

This is a sequential decision problem under uncertainty: the staffing level set today affects capacity 4 weeks from now, which determines how much revenue is captured.

---

## 2. Approach: Two-Stage Pipeline

```
AdvSignUp.csv (499,644 rows)
    ↓  daily aggregation + MA7 smoothing
    ↓  Prophet forecast (per country)
    ↓  combined_predictions.csv
    ↓  weekly demand aggregation
    ↓  Genetic Algorithm (52-week chromosome)
    ↓  staffing_plan.csv + P&L summary
```

### Stage 1 — Demand Forecasting with Facebook Prophet

One Prophet model was trained per country on the 7-day moving average (MA7) of daily sign-up counts. MA7 smoothing suppresses day-of-week noise while preserving trend and seasonality signals that matter for planning.

**Model design choices:**
- Custom weekly (period 7) and monthly (period 31) Fourier seasonality stacked on top of Prophet's built-in yearly seasonality
- Country-specific public holiday effects (ISO country codes)
- Moderate changepoint flexibility (`changepoint_prior_scale = 0.02`) — sufficient for structural growth trends without overfitting to noise
- Additive seasonality mode, appropriate for count-based demand

The train/test split is June 24, 2024 — evaluated on truly out-of-sample data from the second half of 2024 through mid-2025.

**Forecast results (out-of-sample MAPE):**

| Tier | Countries | MAPE Range |
|------|-----------|------------|
| Strong | Germany, Italy, Japan, Indonesia | 17.6–19.5% |
| Good | India, South Korea, Brazil, Canada, Poland, France, China, Australia, Spain, Mexico, UK, USA | 19.5–21.4% |
| Acceptable | Argentina, Vietnam, Turkey, South Africa | 22–28% |
| Challenging | UAE | 37% |

Global MAPE: **19.3%** — competitive for daily demand forecasting in a noisy, multi-country retail setting.

UAE's elevated error reflects thin data and highly variable demand; for production use, a Bayesian structural time series or ensemble approach would help for low-volume markets.

### Stage 2 — Staffing Optimization with Genetic Algorithm

The optimizer represents a 52-week staffing plan as a **chromosome of weekly hire/fire deltas** — a length-52 integer vector where positive values mean hiring and negative values mean releasing agents. A population of 50 chromosomes is evolved over 80 generations, selecting for maximum net profit.

**Simulation model (`simulate()`):**
Each chromosome is evaluated by a forward simulation that tracks:

1. **Agent headcount** — total agents on payroll, updated weekly
2. **Active capacity** — hires from week `s` become productive at week `s + 4`; fired agents leave immediately
3. **FIFO demand queue** — unserved advertisers accumulate; entries older than 60 days are dropped as lost revenue
4. **Revenue** — `0.135 × avg_budget × total_served` (13.5% incremental revenue factor from the report)
5. **Costs** — weekly salary for all active agents, 1-month salary per hire, 40% annual salary per fire

**Constraints enforced by the chromosome representation:**
- Maximum ±10 agents changed per week (gene clamp)
- No simultaneous hire and fire in the same week

---

## 3. Results

### 3.1 Staffing Recommendation

The GA recommends starting at 6,474 agents and gradually reducing to **6,267 agents** by week 52 — a reduction of 207 agents (3.2%). The plan falls into three phases:

| Phase | Weeks | Action | Rationale |
|-------|-------|--------|-----------|
| Reduce | 1–25 | Fire 1–10 agents/week | Current headcount exceeds demand; salary savings outweigh fire costs |
| Stabilize | 26–44 | Near-zero changes | Demand plateau; hold capacity steady |
| Re-hire | 45–52 | Hire 1–6 agents/week | Demand uptick in forecast horizon (2025 H1) |

The re-hiring phase near the end is a key GA insight: a naive "always fire" strategy misses the demand growth signal and leaves revenue on the table.

### 3.2 Financial Performance

| Metric | Value |
|--------|-------|
| **Net Profit** | **$750,713,916** |
| Revenue | $1,034,333,572 |
| Salary Cost | $279,490,564 |
| Fire Cost | $4,031,487 |
| Hire Cost | $77,359 |
| Lost Revenue | $20,247 |
| Advertisers Served | 267,552 |
| Advertisers Lost | **5** |
| **Demand Capture Rate** | **99.998%** |

Revenue margin on gross: **72.6%**. Hire and fire costs combined are less than 0.4% of revenue — the GA correctly concentrates on right-sizing salary expense rather than minimizing adjustment costs. The near-zero hire/fire numbers are not a bug: the optimizer discovered that the current workforce only needs a gentle, gradual reduction — not aggressive churn — to reach the profit-maximizing headcount.

### 3.3 Strategy Comparison: Why GA Wins

The GA was benchmarked against two intuitive baselines:

| Strategy | Net Profit | Served | Lost | Salary Cost | Fire Cost |
|----------|-----------|--------|------|-------------|-----------|
| **Hold (no change)** | $748.2M | 267,558 | 0 | $286.2M | $0 |
| Greedy fire (−10/wk) | $740.6M | 266,246 | 1,312 | $274.5M | $9.2M |
| **GA Optimized** | **$750.7M** | **267,552** | **5** | $279.5M | $4.0M |

**The GA outperforms both by finding the sweet spot:**

- vs. *Hold constant*: saves **$6.7M in salary** through measured downsizing, pays only $4M in fire costs, net gain **+$2.5M**
- vs. *Greedy fire*: avoids aggressive over-firing that creates capacity gaps, preserving **1,307 advertisers** (~$5M in revenue) that greedy loses to queue expiration; net advantage **+$10.1M**

The greedy strategy is particularly instructive: firing 10 agents every week saves more on salary than GA but **destroys $10M of value** through lost demand and excessive fire costs. The GA identifies that demand growth late in the horizon justifies keeping more agents than a purely cost-minimizing strategy would.

---

## 4. Key Insights

**The system is currently over-staffed globally.** Agent capacity at 6,474 headcount is roughly 7,180 advertisers/week, but demand averages 3,820 — a utilization rate of ~53%. The optimizer converts this idle capacity into $6.7M of salary savings while maintaining near-perfect demand capture.

**Seasonality matters but salary dominates.** The seasonality heatmap shows meaningful within-year variation for markets like South Africa, UAE, and Vietnam. However, at the global level these patterns partially cancel, and the dominant lever is total headcount cost, not seasonal adjustment. Per-country optimization (see Section 5) would let seasonality play a bigger role.

**Ramp-up lag shapes hiring decisions.** The 4-week ramp delay means hiring decisions made today only improve capacity a month later. The GA correctly learns to front-run projected demand increases by hiring in weeks 45–50, before the end-of-horizon demand uptick is realized.

**Forecast confidence degrades for thin markets.** UAE (37% MAPE) and South Africa (28%) have fewer sign-ups and more volatile patterns. For these markets, staffing decisions should incorporate Prophet's confidence intervals (`yhat_lower` / `yhat_upper`) rather than point estimates — a conservative planner uses the upper bound to avoid under-staffing.

---

## 5. Limitations and Next Steps

### What the current system does not yet do

**1. Eligibility filtering is global, not per-country.**
The current optimizer uses a globally weighted-average salary and threshold. The problem spec requires filtering each advertiser against their *country's* eligibility threshold before they enter the demand pool, then running a separate optimization per country. This is architected and ready to build.

**2. Revenue uses average budget, not actual budget distribution.**
The model uses `avg_budget = 2 × threshold` as a proxy. The actual revenue should be `0.135 × Σ(bᵢ)` summed over individual advertiser budgets. Implementing this requires joining the forecast back to the raw sign-up budget distribution — feasible with the existing data.

**3. One global staffing pool, 22 real pools.**
A UK agent cannot service a Brazilian advertiser. The per-country loop is the highest-priority next implementation step; it will surface country-level P&L and right-size each market independently.

### Promising extensions

- **Confidence-interval-aware planning**: run the optimizer on `yhat_lower`, `yhat`, and `yhat_upper` to generate conservative / base / aggressive staffing scenarios and quantify the risk envelope
- **Bayesian optimization (Optuna)**: replace the GA with a surrogate-model optimizer — converges faster with fewer `simulate()` calls, especially useful when running 22 independent country optimizations
- **Double-queue redistribution**: implement the `Uₜ = Σ αₖvₖ` formulation where unserviceable orders are redistributed across the 60-day window proportionally, aligning code with the methodology in the original report
- **Streamlit dashboard**: interactive scenario tool to adjust cost parameters and see per-country P&L without touching the CLI

---

## 6. Visualizations

All figures are in `outputs/figures/`:

| File | Content |
|------|---------|
| `fig1_demand_forecast_top6.png` | Fitted + forecast vs actual MA7 for top 6 markets |
| `fig2_global_demand_forecast.png` | Aggregated global demand: forecast vs actual |
| `fig3_staffing_plan.png` | 52-week staffing trajectory with weekly hire/fire deltas |
| `fig4_pnl_waterfall.png` | Revenue → cost waterfall to net profit (with zoomed cost panel) |
| `fig5_mape_by_country.png` | Forecast accuracy leaderboard by country |
| `fig6_country_profiles.png` | Salary vs threshold scatter; headcount by country |
| `fig7_seasonality_heatmap.png` | Monthly demand seasonality across all 22 countries |
| `fig8_strategy_comparison.png` | GA vs Hold vs Greedy: profit, costs, demand capture |

---

## 7. How to Reproduce

```bash
# Install dependencies
uv venv && uv pip install -r requirements.txt

# Stage 1: Per-country forecast (produces combined_predictions.csv)
uv run python run_forecast.py --by-country --out-dir outputs/

# Stage 2: GA optimization
uv run python run_optimization.py --forecast outputs/forecasts/combined_predictions.csv

# Generate all visualizations and report figures
uv run python visualize.py
```

Outputs:
- `outputs/forecasts/` — per-country forecast CSVs + combined
- `outputs/optimization/staffing_plan.csv` — weekly headcount plan
- `outputs/optimization/optimization_summary.txt` — P&L summary
- `outputs/figures/` — 8 presentation figures

---

*BADSS — Business Agent Dynamic Staffing System*
