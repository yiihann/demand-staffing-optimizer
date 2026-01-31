#!/usr/bin/env python3
"""
Run GA staffing optimization (per Final Report).

Usage:
  python run_optimization.py [--forecast PATH] [--weeks N] [--pop N] [--gen N] [--out-dir DIR]

  Loads forecast (combined_predictions.csv), salary & staffing data, runs GA,
  and writes staffing_plan.csv + summary to outputs/optimization/.
"""

import argparse
from pathlib import Path

from demand_forecasting.config import OUTPUT_DIR, PROJECT_ROOT
from demand_forecasting.optimization import (
    load_forecast_weekly,
    load_salary_and_staffing,
    ga_optimize,
    simulate,
    staffing_plan_from_chromosome,
    EXPECTED_UPLIFT,
)


def main():
    p = argparse.ArgumentParser(description="GA staffing optimization")
    p.add_argument(
        "--forecast",
        type=Path,
        default=PROJECT_ROOT / "results_prediction" / "combined_predictions.csv",
        help="Path to combined forecast CSV (ds, yhat, Country)",
    )
    p.add_argument("--weeks", type=int, default=52, help="Optimization horizon (weeks)")
    p.add_argument("--pop", type=int, default=50, help="GA population size")
    p.add_argument("--gen", type=int, default=80, help="GA generations")
    p.add_argument("--mut", type=float, default=0.15, help="Mutation rate")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR / "optimization", help="Output directory")
    args = p.parse_args()

    if not args.forecast.exists():
        raise FileNotFoundError(
            f"Forecast not found: {args.forecast}. "
            "Run run_forecast.py --by-country first, or point --forecast to combined_predictions.csv"
        )

    weekly_demand = load_forecast_weekly(args.forecast, num_weeks=args.weeks)
    avg_salary, initial_agents, avg_threshold = load_salary_and_staffing()
    # Avg budget for eligible advertisers: use 2× threshold as proxy (report: concave truncated)
    avg_budget = 2.0 * avg_threshold

    print("Inputs:")
    print(f"  Initial agents: {initial_agents}")
    print(f"  Avg annual salary: ${avg_salary:,.0f}")
    print(f"  Avg eligibility threshold: ${avg_threshold:,.0f}")
    print(f"  Avg budget (2× threshold): ${avg_budget:,.0f}")
    print(f"  Weekly demand (first 4): {weekly_demand[:4].round(1).tolist()} ...")
    print("Running GA ...")

    best_deltas, best_profit, history = ga_optimize(
        weekly_demand,
        initial_agents,
        avg_salary,
        avg_budget,
        num_weeks=args.weeks,
        population_size=args.pop,
        generations=args.gen,
        mutation_rate=args.mut,
        random_seed=args.seed,
    )

    profit, breakdown = simulate(
        weekly_demand, initial_agents, best_deltas,
        avg_salary, avg_budget, expected_uplift=EXPECTED_UPLIFT,
    )

    print("\nBest net profit:", f"${profit:,.0f}")
    print("Breakdown:")
    for k, v in breakdown.items():
        if isinstance(v, float):
            print(f"  {k}: ${v:,.0f}")
        else:
            print(f"  {k}: {v}")

    plan = staffing_plan_from_chromosome(best_deltas, initial_agents)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.out_dir / "staffing_plan.csv"
    plan.to_csv(plan_path, index=False)
    print(f"\nWrote {plan_path}")

    summary_path = args.out_dir / "optimization_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Net profit: ${profit:,.0f}\n")
        f.write(f"Revenue: ${breakdown['revenue']:,.0f}\n")
        f.write(f"Salary cost: ${breakdown['salary_cost']:,.0f}\n")
        f.write(f"Hire cost: ${breakdown['hire_cost']:,.0f}\n")
        f.write(f"Fire cost: ${breakdown['fire_cost']:,.0f}\n")
        f.write(f"Lost revenue: ${breakdown['lost_revenue']:,.0f}\n")
        f.write(f"Total served: {breakdown['total_served']:,.0f}\n")
        f.write(f"Total lost: {breakdown['total_lost']:,.0f}\n")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
