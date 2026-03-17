#!/usr/bin/env python3
"""
BADSS Visualization Suite.
Generates 8 presentation-ready figures to outputs/figures/.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demand_staffing.optimization import simulate, staffing_plan_from_chromosome, EXPECTED_UPLIFT

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent
FORECASTS_DIR = ROOT / "outputs" / "forecasts"
OPT_DIR       = ROOT / "outputs" / "optimization"
DATA_DIR      = ROOT / "data"
FIGS_DIR      = ROOT / "outputs" / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Brand palette ─────────────────────────────────────────────────────────────
BLUE   = "#4285F4"
RED    = "#EA4335"
YELLOW = "#FBBC04"
GREEN  = "#34A853"
GRAY   = "#5F6368"
LGRAY  = "#E8EAED"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.facecolor": "#FAFAFA",
    "figure.facecolor": "white",
})

# ── Load data ─────────────────────────────────────────────────────────────────
combined     = pd.read_csv(FORECASTS_DIR / "combined_predictions.csv", parse_dates=["ds"])
staffing     = pd.read_csv(OPT_DIR / "staffing_plan.csv")
salary_df    = pd.read_csv(DATA_DIR / "[BADSS case] Agent Salary and Eligibility Threshold - data.csv")
staffing_meta = pd.read_csv(DATA_DIR / "[BADSS case] Existing Agent Staffing - data.csv")
raw          = pd.read_csv(DATA_DIR / "AdvSignUp.csv", parse_dates=["Sign_Up_Date"])

country_meta = salary_df.merge(staffing_meta, on="Country")
SPLIT = pd.Timestamp("2024-06-24")

MAPE = {
    "Argentina": 22.37, "Australia": 21.31, "Brazil": 20.68,
    "Canada": 20.07, "China": 21.25, "France": 20.74,
    "Germany": 17.63, "India": 19.81, "Indonesia": 19.44,
    "Italy": 18.95, "Japan": 18.88, "Mexico": 21.35,
    "Poland": 20.74, "South Africa": 27.90, "South Korea": 20.71,
    "Spain": 21.16, "Thailand": 21.65, "Turkey": 23.69,
    "UAE": 37.25, "UK": 21.62, "USA": 20.82, "Vietnam": 23.38,
}

# ── Fig 1: Demand Forecast — top 6 countries ──────────────────────────────────
print("Generating fig 1 ...")
TOP6 = ["USA", "UK", "Germany", "France", "Australia", "Canada"]
actual_by_country = (
    raw.groupby(["Country", "Sign_Up_Date"])
    .size()
    .reset_index(name="count")
)
actual_by_country["MA7"] = (
    actual_by_country.groupby("Country")["count"]
    .transform(lambda x: x.rolling(7).mean())
)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Demand Forecast: Top 6 Markets  (Prophet + MA7 smoothing)", fontsize=14, fontweight="bold")

for ax, country in zip(axes.flat, TOP6):
    fc  = combined[combined["Country"] == country].sort_values("ds")
    act = actual_by_country[actual_by_country["Country"] == country].sort_values("Sign_Up_Date")

    ax.plot(act["Sign_Up_Date"], act["MA7"], color=GRAY, lw=1.2, alpha=0.55, label="Actual MA7", zorder=1)
    ax.plot(fc[fc["ds"] <= SPLIT]["ds"], fc[fc["ds"] <= SPLIT]["yhat"],
            color=BLUE, lw=1.8, label="Fit", zorder=2)
    ax.plot(fc[fc["ds"] > SPLIT]["ds"], fc[fc["ds"] > SPLIT]["yhat"],
            color=RED, lw=1.8, label="Forecast", zorder=2)
    ax.axvline(SPLIT, color="black", ls=":", lw=1.2, alpha=0.6)

    ax.set_title(f"{country}   MAPE {MAPE[country]:.1f}%", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.set_ylabel("Daily signups (MA7)", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))

legend_handles = [
    plt.Line2D([0],[0], color=GRAY, lw=1.5, label="Actual MA7"),
    plt.Line2D([0],[0], color=BLUE, lw=1.8, label="Fitted"),
    plt.Line2D([0],[0], color=RED,  lw=1.8, label="Out-of-sample forecast"),
    plt.Line2D([0],[0], color="black", ls=":", lw=1.2, label="Train / test split (Jun 2024)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, -0.02), frameon=True)
plt.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(FIGS_DIR / "fig1_demand_forecast_top6.png", bbox_inches="tight")
plt.close()
print("  saved fig1_demand_forecast_top6.png")

# ── Fig 2: Global aggregated demand ───────────────────────────────────────────
print("Generating fig 2 ...")
global_fc = combined.groupby("ds")["yhat"].sum().reset_index()
global_act = raw.groupby("Sign_Up_Date").size().reset_index(name="count")
global_act["MA7"] = global_act["count"].rolling(7).mean()

fig, ax = plt.subplots(figsize=(13, 5))
ax.fill_between(global_fc["ds"], global_fc["yhat"], alpha=0.12, color=BLUE)
ax.plot(global_fc["ds"], global_fc["yhat"], color=BLUE, lw=2, label="Aggregate forecast (22 countries)")
ax.plot(global_act["Sign_Up_Date"], global_act["MA7"], color=GRAY, lw=1.5, alpha=0.65, label="Actual (MA7)")
ax.axvline(SPLIT, color="black", ls=":", lw=1.5, label="Train / test split")

# Shade forecast region
fc_start = global_fc[global_fc["ds"] > SPLIT]["ds"].min()
ax.axvspan(fc_start, global_fc["ds"].max(), alpha=0.05, color=RED, label="Forecast horizon")

ax.set_title("Global Daily Advertiser Demand — 22 Countries", fontsize=14, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Daily sign-ups (MA7, all countries)")
ax.legend(fontsize=10)
plt.tight_layout()
fig.savefig(FIGS_DIR / "fig2_global_demand_forecast.png", bbox_inches="tight")
plt.close()
print("  saved fig2_global_demand_forecast.png")

# ── Fig 3: Optimized staffing plan ────────────────────────────────────────────
print("Generating fig 3 ...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

weeks = staffing["week"]
ax1.plot(weeks, staffing["agents_total"],  color=BLUE,  lw=2.5, zorder=3, label="Total agents (on payroll)")
ax1.plot(weeks, staffing["agents_active"], color=GREEN, lw=2, ls="--", zorder=3, label="Active agents (fully ramped)")
ax1.fill_between(weeks, staffing["agents_active"], staffing["agents_total"],
                  alpha=0.18, color=YELLOW, label="Ramp-up buffer")
ax1.axhline(staffing["agents_total"].iloc[0], color=GRAY, ls=":", lw=1.2, alpha=0.6, label=f"Initial: {staffing['agents_total'].iloc[0]:,}")
ax1.axhline(staffing["agents_total"].iloc[-1], color=RED, ls=":", lw=1.2, alpha=0.6, label=f"Final: {staffing['agents_total'].iloc[-1]:,}")
ax1.set_ylabel("Agents", fontsize=11)
ax1.set_title("GA-Optimized 52-Week Global Staffing Plan", fontsize=14, fontweight="bold")
ax1.legend(fontsize=9, loc="upper right")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

deltas = staffing["delta"]
bar_cols = [GREEN if d >= 0 else RED for d in deltas]
ax2.bar(weeks, deltas, color=bar_cols, width=0.8, alpha=0.85)
ax2.axhline(0, color="black", lw=0.8)
ax2.set_ylabel("Weekly Δ", fontsize=10)
ax2.set_xlabel("Week", fontsize=11)
hire_p = mpatches.Patch(color=GREEN, alpha=0.85, label="Hire")
fire_p = mpatches.Patch(color=RED,   alpha=0.85, label="Fire / reduce")
ax2.legend(handles=[hire_p, fire_p], fontsize=9, loc="upper right")

plt.tight_layout()
fig.savefig(FIGS_DIR / "fig3_staffing_plan.png", bbox_inches="tight")
plt.close()
print("  saved fig3_staffing_plan.png")

# ── Fig 4: P&L Waterfall (split layout: main + cost zoom) ────────────────────
print("Generating fig 4 ...")
items = [
    ("Revenue",        1_034_333_572,  BLUE),
    ("Salary cost",   -279_490_564,   RED),
    ("Fire cost",      -4_031_487,    RED),
    ("Hire cost",        -77_359,     RED),
    ("Lost revenue",     -20_247,     RED),
    ("Net Profit",     750_713_916,   GREEN),
]
labels = [i[0] for i in items]
values = [i[1] for i in items]
colors = [i[2] for i in items]

bottoms = []
running = 0
for i, (lbl, val, _) in enumerate(items):
    if i == len(items) - 1:
        bottoms.append(0)
    else:
        bottoms.append(running if val >= 0 else running + val)
        running += val

fig, (ax, ax_zoom) = plt.subplots(1, 2, figsize=(15, 6),
                                    gridspec_kw={"width_ratios": [3, 1.4]})
fig.suptitle("52-Week P&L Waterfall  (global, all 22 countries)", fontsize=13, fontweight="bold")

# ── Main waterfall (Revenue, Salary, Net Profit only) ────────────────────────
main_items = [(0, "Revenue", 1_034_333_572, BLUE),
              (1, "Salary cost", -279_490_564, RED),
              (5, "Net Profit", 750_713_916, GREEN)]
main_bottoms = {0: 0, 1: 279_490_564 + 4_031_487 + 77_359 + 20_247, 5: 0}
# Recompute running bottoms for main items from original bottoms list
for idx, lbl, val, col in main_items:
    b = bottoms[idx]
    ax.bar(idx, abs(val), bottom=b, color=col, edgecolor="white", width=0.6, alpha=0.88)
    mid = b + abs(val) / 2
    sign = "+" if val >= 0 else ""
    ax.text(idx, mid, f"{sign}${abs(val)/1e6:.0f}M",
            ha="center", va="center", fontsize=11, fontweight="bold", color="white")

# Draw placeholder stubs for small costs with annotations
small_items = [(2, "Fire cost",    -4_031_487, RED),
               (3, "Hire cost",      -77_359,  RED),
               (4, "Lost revenue",   -20_247,  RED)]
for idx, lbl, val, col in small_items:
    b = bottoms[idx]
    # Enforce minimum visible height = 2% of revenue so bar is visible
    min_h = 1_034_333_572 * 0.02
    ax.bar(idx, min_h, bottom=b, color=col, edgecolor="white", width=0.6, alpha=0.88)
    ax.text(idx, b + min_h + 4e6, f"-${abs(val)/1e6:.1f}M",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color=col)
    ax.annotate("(see zoom →)", xy=(idx, b + min_h / 2), ha="center", va="center",
                fontsize=7, color="white", style="italic")

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
ax.set_ylabel("USD", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e9:.2f}B"))
ax.axhline(0, color="black", lw=0.8)

# ── Zoom panel: small cost items only ────────────────────────────────────────
zoom_labels = ["Fire cost\n$4.0M", "Hire cost\n$77K", "Lost revenue\n$20K"]
zoom_vals   = [4_031_487, 77_359, 20_247]
zoom_cols   = [RED, RED, RED]
zbars = ax_zoom.bar(zoom_labels, zoom_vals, color=zoom_cols, edgecolor="white",
                    width=0.5, alpha=0.88)
for bar, val in zip(zbars, zoom_vals):
    if val >= 1e6:
        label = f"${val/1e6:.2f}M"
    elif val >= 1e3:
        label = f"${val/1e3:.0f}K"
    else:
        label = f"${val:,.0f}"
    ax_zoom.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.03,
                 label, ha="center", va="bottom", fontsize=10, fontweight="bold", color=RED)

ax_zoom.set_title("Small cost items\n(zoomed — note scale)", fontsize=10, fontweight="bold")
ax_zoom.set_ylabel("USD", fontsize=10)
ax_zoom.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"${x/1e6:.1f}M" if x >= 1e6 else f"${x/1e3:.0f}K"))
ax_zoom.tick_params(axis="x", labelsize=9)

# Add total-cost note
total_small = sum(zoom_vals)
ax_zoom.set_title(f"Adjustment costs: ${total_small/1e6:.1f}M total\n(0.4% of revenue)", fontsize=10, fontweight="bold")

plt.tight_layout()
fig.savefig(FIGS_DIR / "fig4_pnl_waterfall.png", bbox_inches="tight")
plt.close()
print("  saved fig4_pnl_waterfall.png")

# ── Fig 5: MAPE by country ────────────────────────────────────────────────────
print("Generating fig 5 ...")
mape_s = pd.Series(MAPE).sort_values()
bar_colors = [RED if v > 25 else YELLOW if v > 22 else GREEN for v in mape_s.values]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(mape_s.index, mape_s.values, color=bar_colors, edgecolor="white", alpha=0.88, height=0.7)
ax.axvline(20, color=BLUE, ls="--", lw=1.8, alpha=0.8, label="Global benchmark (20%)")
for bar, val in zip(bars, mape_s.values):
    ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=9)

patches = [
    mpatches.Patch(color=GREEN,  alpha=0.88, label="≤ 22%  (good)"),
    mpatches.Patch(color=YELLOW, alpha=0.88, label="22–25%  (acceptable)"),
    mpatches.Patch(color=RED,    alpha=0.88, label="> 25%  (challenging)"),
]
ax.legend(handles=patches + [plt.Line2D([0],[0], color=BLUE, ls="--", lw=1.8, label="Global benchmark (20%)")],
          fontsize=9, loc="lower right")
ax.set_xlabel("MAPE (%)", fontsize=11)
ax.set_title("Forecast Accuracy by Country (MAPE, out-of-sample)", fontsize=13, fontweight="bold")
ax.set_xlim(0, mape_s.max() + 5)
plt.tight_layout()
fig.savefig(FIGS_DIR / "fig5_mape_by_country.png", bbox_inches="tight")
plt.close()
print("  saved fig5_mape_by_country.png")

# ── Fig 6: Country profiles ───────────────────────────────────────────────────
print("Generating fig 6 ...")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: salary vs threshold scatter, bubble = agent count
ax = axes[0]
for _, row in country_meta.iterrows():
    size = row["Existing_Agent_Count"] ** 1.1 * 0.35
    ax.scatter(row["Advertiser_Eligibility_Threshold_USD"] / 1e3,
               row["Annual_Agent_Salary_USD"] / 1e3,
               s=size, alpha=0.7, color=BLUE, edgecolors="white", lw=0.8)
    ax.annotate(row["Country"],
                (row["Advertiser_Eligibility_Threshold_USD"] / 1e3,
                 row["Annual_Agent_Salary_USD"] / 1e3),
                fontsize=7, ha="center", va="bottom")
ax.set_xlabel("Eligibility Threshold ($K USD)", fontsize=11)
ax.set_ylabel("Annual Agent Salary ($K USD)", fontsize=11)
ax.set_title("Salary vs Eligibility Threshold\n(bubble = existing agent count)", fontsize=11, fontweight="bold")

# Right: agent headcount bar, colored by salary tier
ax2 = axes[1]
cm_sorted = country_meta.sort_values("Existing_Agent_Count", ascending=True)
norm_sal = (cm_sorted["Annual_Agent_Salary_USD"] - cm_sorted["Annual_Agent_Salary_USD"].min())
norm_sal = norm_sal / norm_sal.max()
bar_cols = [plt.cm.Blues(0.35 + 0.55 * v) for v in norm_sal]
bars = ax2.barh(cm_sorted["Country"], cm_sorted["Existing_Agent_Count"],
                color=bar_cols, edgecolor="white", alpha=0.9)
for bar, cnt in zip(bars, cm_sorted["Existing_Agent_Count"]):
    ax2.text(bar.get_width() + 4, bar.get_y() + bar.get_height() / 2,
             str(cnt), va="center", fontsize=8)
sm = plt.cm.ScalarMappable(cmap="Blues",
                            norm=plt.Normalize(cm_sorted["Annual_Agent_Salary_USD"].min(),
                                               cm_sorted["Annual_Agent_Salary_USD"].max()))
sm.set_array([])
plt.colorbar(sm, ax=ax2, label="Annual salary (USD)", pad=0.01)
ax2.set_xlabel("Existing Agent Count", fontsize=11)
ax2.set_title("Agent Headcount by Country\n(color = salary level)", fontsize=11, fontweight="bold")

plt.tight_layout()
fig.savefig(FIGS_DIR / "fig6_country_profiles.png", bbox_inches="tight")
plt.close()
print("  saved fig6_country_profiles.png")

# ── Fig 7: Demand seasonality heatmap ─────────────────────────────────────────
print("Generating fig 7 ...")
combined["month"] = combined["ds"].dt.to_period("M")
monthly_demand = combined.groupby(["Country", "month"])["yhat"].sum().reset_index()
pivot = monthly_demand.pivot(index="Country", columns="month", values="yhat")
pivot_norm = pivot.div(pivot.max(axis=1), axis=0)
order = pivot.sum(axis=1).sort_values(ascending=False).index
pivot_norm = pivot_norm.loc[order]
col_labels = [str(c) if i % 2 == 0 else "" for i, c in enumerate(pivot_norm.columns)]

fig, ax = plt.subplots(figsize=(17, 7))
im = ax.imshow(pivot_norm.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(pivot_norm.index)))
ax.set_yticklabels(pivot_norm.index, fontsize=9)
plt.colorbar(im, ax=ax, label="Relative monthly demand (normalized per country)", fraction=0.02, pad=0.01)
ax.set_title("Monthly Demand Seasonality Heatmap  (normalized per country, sorted by total volume)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(FIGS_DIR / "fig7_seasonality_heatmap.png", bbox_inches="tight")
plt.close()
print("  saved fig7_seasonality_heatmap.png")

# ── Fig 8: GA vs Baselines (strategy comparison) ──────────────────────────────
print("Generating fig 8 ...")

# Load weekly demand (same as run_optimization.py)
from demand_staffing.optimization import load_forecast_weekly
weekly_demand = load_forecast_weekly(FORECASTS_DIR / "combined_predictions.csv", num_weeks=52)
initial_agents = int(staffing_meta["Existing_Agent_Count"].sum())  # 6474

# Weighted-average salary and avg_budget (matches run_optimization.py logic)
merge = staffing_meta.merge(salary_df, on="Country")
weighted_salary = (merge["Annual_Agent_Salary_USD"] * merge["Existing_Agent_Count"]).sum()
avg_salary = weighted_salary / initial_agents
avg_threshold = salary_df["Advertiser_Eligibility_Threshold_USD"].mean()
avg_budget = 2.0 * avg_threshold

# Strategies
strategies = {
    "Hold (no change)":     np.zeros(52, dtype=int),
    "Greedy fire (-10/wk)": np.full(52, -10, dtype=int),
    "GA Optimized":         staffing["delta"].values[1:].astype(int),  # skip week 0 (always 0)
}

results = {}
for name, deltas in strategies.items():
    profit, breakdown = simulate(weekly_demand, initial_agents, deltas, avg_salary, avg_budget,
                                  expected_uplift=EXPECTED_UPLIFT)
    results[name] = {"profit": profit, **breakdown}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Strategy Comparison: GA vs Baselines", fontsize=14, fontweight="bold")

colors_strategies = [GRAY, YELLOW, GREEN]
strategy_names = list(results.keys())

# Subplot 1: Net profit
ax = axes[0]
profits = [results[s]["profit"] for s in strategy_names]
bars = ax.bar(strategy_names, [p / 1e6 for p in profits], color=colors_strategies, edgecolor="white", alpha=0.88, width=0.5)
for bar, val in zip(bars, profits):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
            f"${val/1e6:.0f}M", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("Net Profit ($M)", fontsize=11)
ax.set_title("Net Profit (52 weeks)", fontsize=11, fontweight="bold")
ax.tick_params(axis="x", labelsize=9)

# Subplot 2: Salary + hire/fire costs
ax2 = axes[1]
x = np.arange(len(strategy_names))
width = 0.25
sal  = [results[s]["salary_cost"] / 1e6 for s in strategy_names]
hire = [results[s]["hire_cost"]   / 1e6 for s in strategy_names]
fire = [results[s]["fire_cost"]   / 1e6 for s in strategy_names]
ax2.bar(x - width, sal,  width, label="Salary",    color=BLUE,   alpha=0.85, edgecolor="white")
ax2.bar(x,         hire, width, label="Hire cost", color=GREEN,  alpha=0.85, edgecolor="white")
ax2.bar(x + width, fire, width, label="Fire cost", color=RED,    alpha=0.85, edgecolor="white")
ax2.set_xticks(x)
ax2.set_xticklabels(strategy_names, fontsize=9)
ax2.set_ylabel("Cost ($M)", fontsize=11)
ax2.set_title("Cost Breakdown", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9)

# Subplot 3: Demand served vs lost
ax3 = axes[2]
served = [results[s]["total_served"] for s in strategy_names]
lost   = [results[s]["total_lost"]   for s in strategy_names]
ax3.bar(x - width/2, served, width, label="Served",   color=GREEN, alpha=0.85, edgecolor="white")
ax3.bar(x + width/2, lost,   width, label="Lost",     color=RED,   alpha=0.85, edgecolor="white")
for xi, (s, l) in enumerate(zip(served, lost)):
    pct = 100 * s / (s + l) if (s + l) > 0 else 0
    ax3.text(xi, max(s, l) * 1.01, f"{pct:.2f}%\ncaptured", ha="center", fontsize=8)
ax3.set_xticks(x)
ax3.set_xticklabels(strategy_names, fontsize=9)
ax3.set_ylabel("Advertisers", fontsize=11)
ax3.set_title("Demand Capture", fontsize=11, fontweight="bold")
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax3.legend(fontsize=9)

plt.tight_layout()
fig.savefig(FIGS_DIR / "fig8_strategy_comparison.png", bbox_inches="tight")
plt.close()
print("  saved fig8_strategy_comparison.png")

# ── Print summary table for report ────────────────────────────────────────────
print("\n=== Strategy Comparison Summary ===")
for name, r in results.items():
    print(f"\n{name}:")
    print(f"  Net profit:   ${r['profit']:>15,.0f}")
    print(f"  Revenue:      ${r['revenue']:>15,.0f}")
    print(f"  Salary cost:  ${r['salary_cost']:>15,.0f}")
    print(f"  Fire cost:    ${r['fire_cost']:>15,.0f}")
    print(f"  Served:       {r['total_served']:>15,.0f}")
    print(f"  Lost:         {r['total_lost']:>15,.0f}")

print(f"\nAll figures saved to {FIGS_DIR}")
