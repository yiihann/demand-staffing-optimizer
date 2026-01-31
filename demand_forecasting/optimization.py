"""
Genetic Algorithm for dynamic staffing (per Final Report).

Objective: maximize net profit = revenue - staffing cost - hiring cost - firing cost - lost revenue.
Constraints: 60-day backlog, 1-month ramp for new hires, max ±10 agents per week, no hire+fire same week.
Chromosome: weekly staffing deltas (change in agents per week) over the horizon.
"""

import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import DATA_DIR, OUTPUT_DIR, PROJECT_ROOT


# --- Report parameters ---
WEEKS_RAMP = 4  # new hires become available after 1 month ≈ 4 weeks
DAYS_BACKLOG_LIMIT = 60
WEEKS_BACKLOG_LIMIT = 9  # 63 days > 60
AGENTS_PER_SLOT = 10  # one agent manages 10 advertisers
SUPPORT_DAYS = 60
EXPECTED_UPLIFT = 0.135  # report: expected incremental revenue factor
HIRE_COST_MONTHS = 1  # 1 month salary per hired agent
FIRE_PENALTY_RATIO = 0.40  # 40% of annual salary per fired agent
LABOR_EFFICIENCY = 0.95  # ~5% frictional loss
MAX_HIRE_PER_WEEK = 10
MAX_FIRE_PER_WEEK = 10


def load_forecast_weekly(forecast_path: Path, num_weeks: int = 52) -> np.ndarray:
    """
    Load combined forecast CSV (ds, yhat, Country), aggregate to global monthly,
    then convert to weekly demand (spread evenly over weeks in month).
    Returns array of length num_weeks: demand per week (eligible advertisers).
    """
    df = pd.read_csv(forecast_path)
    df["ds"] = pd.to_datetime(df["ds"])
    monthly = df.groupby("ds")["yhat"].sum().sort_index()
    weeks = []
    for ts in monthly.index:
        vol = float(monthly.loc[ts])
        days_in_month = (pd.Timestamp(ts) + pd.offsets.MonthEnd(0)).day
        n_weeks = max(1, round(days_in_month / 7))
        per_week = vol / n_weeks
        for _ in range(n_weeks):
            weeks.append(per_week)
    arr = np.array(weeks, dtype=float)
    if len(arr) < num_weeks:
        arr = np.resize(arr, num_weeks)
    else:
        arr = arr[:num_weeks]
    return arr


def load_salary_and_staffing(
    salary_path: Optional[Path] = None,
    staffing_path: Optional[Path] = None,
) -> tuple[float, int, float]:
    """
    Load salary and existing staffing CSVs. Returns (avg_annual_salary, total_initial_agents, avg_eligibility_threshold).
    """
    base = DATA_DIR
    salary_path = salary_path or base / "[BADSS case] Agent Salary and Eligibility Threshold - data.csv"
    staffing_path = staffing_path or base / "[BADSS case] Existing Agent Staffing - data.csv"
    salary_df = pd.read_csv(salary_path)
    staffing_df = pd.read_csv(staffing_path)
    # Merge on country
    merge = staffing_df.merge(
        salary_df,
        left_on="Country",
        right_on="Country",
        how="left",
    )
    merge = merge.dropna(subset=["Annual_Agent_Salary_USD", "Existing_Agent_Count"])
    total_agents = int(merge["Existing_Agent_Count"].sum())
    if total_agents == 0:
        avg_salary = 50000.0
        avg_threshold = 20000.0
    else:
        weighted_salary = (merge["Annual_Agent_Salary_USD"] * merge["Existing_Agent_Count"]).sum()
        avg_salary = weighted_salary / total_agents
        avg_threshold = merge["Advertiser_Eligibility_Threshold_USD"].mean()
    return avg_salary, total_agents, avg_threshold


def simulate(
    weekly_demand: np.ndarray,
    initial_agents: int,
    deltas: np.ndarray,
    avg_annual_salary: float,
    avg_budget: float,
    expected_uplift: float = EXPECTED_UPLIFT,
    weeks_ramp: int = WEEKS_RAMP,
    weeks_backlog_limit: int = WEEKS_BACKLOG_LIMIT,
    labor_efficiency: float = LABOR_EFFICIENCY,
) -> tuple[float, dict]:
    """
    Simulate staffing plan: apply weekly deltas (with ramp), compute capacity,
    serve demand, track backlog, and compute net profit.

    deltas : length T, net change in agents each week (hires - fires).
    Returns (net_profit, breakdown_dict).
    """
    T = min(len(weekly_demand), len(deltas))
    weekly_demand = weekly_demand[:T]
    deltas = deltas[:T]

    # Clamp deltas to [−MAX_FIRE, +MAX_HIRE]
    deltas = np.clip(deltas.astype(int), -MAX_FIRE_PER_WEEK, MAX_HIRE_PER_WEEK)

    # Effective agents: new hires available after weeks_ramp
    A = np.zeros(T + 1, dtype=float)
    A[0] = initial_agents
    for t in range(T):
        hire = max(0, deltas[t])
        fire = max(0, -deltas[t])
        A[t + 1] = A[t] + deltas[t]

    # Active agents (ramp): hire at week s becomes active at week s + weeks_ramp
    A_active = np.zeros(T + 1)
    A_active[0] = initial_agents
    for t in range(1, T + 1):
        # Hires at week s become active at week s + weeks_ramp
        hires_active_by_t = sum(
            max(0, int(deltas[s])) for s in range(max(0, t - weeks_ramp + 1))
        )
        fires_by_t = sum(max(0, int(-deltas[s])) for s in range(t))
        A_active[t] = initial_agents + hires_active_by_t - fires_by_t
    A_active = np.maximum(A_active, 0)

    # Capacity: each active agent can start 10 * (7/60) advertisers per week (60-day support)
    capacity_per_agent_per_week = AGENTS_PER_SLOT * (7 / SUPPORT_DAYS) * labor_efficiency
    capacity = A_active[1 : T + 1] * capacity_per_agent_per_week

    # FIFO backlog: demand enters queue; we serve up to capacity each week; age > 60 days → lost
    backlog = []  # list of (amount, weeks_waiting); weeks_waiting 0 = arrived this week
    total_served = 0
    total_lost = 0
    lost_revenue = 0.0

    for t in range(T):
        cap_t = capacity[t]
        new_demand = max(0, weekly_demand[t])
        backlog.append((new_demand, 0))
        backlog.sort(key=lambda x: x[1])  # oldest first
        remaining_cap = cap_t
        new_backlog = []
        for amt, weeks_old in backlog:
            if remaining_cap <= 0:
                if weeks_old + 1 > weeks_backlog_limit:
                    total_lost += amt
                    lost_revenue += amt * avg_budget * expected_uplift
                else:
                    new_backlog.append((amt, weeks_old + 1))
                continue
            serve = min(amt, remaining_cap)
            if serve > 0:
                total_served += serve
                remaining_cap -= serve
            left = amt - serve
            if left > 0:
                if weeks_old + 1 > weeks_backlog_limit:
                    total_lost += left
                    lost_revenue += left * avg_budget * expected_uplift
                else:
                    new_backlog.append((left, weeks_old + 1))
        backlog = new_backlog

    # Revenue from served
    revenue = total_served * avg_budget * expected_uplift

    # Costs
    salary_per_week = avg_annual_salary / 52
    salary_cost = float(np.sum(A_active[1 : T + 1]) * salary_per_week)
    hire_cost = float(np.sum(np.maximum(deltas, 0)) * (avg_annual_salary / 12) * HIRE_COST_MONTHS)
    fire_cost = float(np.sum(np.maximum(-deltas, 0)) * avg_annual_salary * FIRE_PENALTY_RATIO)

    net_profit = revenue - salary_cost - hire_cost - fire_cost - lost_revenue

    breakdown = {
        "revenue": revenue,
        "salary_cost": salary_cost,
        "hire_cost": hire_cost,
        "fire_cost": fire_cost,
        "lost_revenue": lost_revenue,
        "total_served": total_served,
        "total_lost": total_lost,
    }
    return net_profit, breakdown


def fitness(
    chromosome: np.ndarray,
    weekly_demand: np.ndarray,
    initial_agents: int,
    avg_annual_salary: float,
    avg_budget: float,
) -> float:
    """Fitness = net profit (to maximize)."""
    profit, _ = simulate(
        weekly_demand, initial_agents, chromosome,
        avg_annual_salary, avg_budget,
    )
    return profit


def ga_optimize(
    weekly_demand: np.ndarray,
    initial_agents: int,
    avg_annual_salary: float,
    avg_budget: float,
    num_weeks: int = 52,
    population_size: int = 50,
    generations: int = 80,
    mutation_rate: float = 0.15,
    tournament_size: int = 5,
    random_seed: Optional[int] = None,
) -> tuple[np.ndarray, float, list]:
    """
    Genetic algorithm: maximize net profit over staffing deltas.

    Returns (best_chromosome, best_fitness, fitness_history).
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    T = min(num_weeks, len(weekly_demand))
    weekly_demand = weekly_demand[:T]

    def eval_fit(chromosome: np.ndarray) -> float:
        return fitness(chromosome, weekly_demand, initial_agents, avg_annual_salary, avg_budget)

    # Initialize population: each gene in [-10, 10]
    population = [
        np.random.randint(-MAX_FIRE_PER_WEEK, MAX_HIRE_PER_WEEK + 1, size=T)
        for _ in range(population_size)
    ]
    fitness_history = []

    for gen in range(generations):
        scores = [eval_fit(ch) for ch in population]
        fitness_history.append(max(scores))

        # Tournament selection
        def select():
            idx = random.sample(range(population_size), tournament_size)
            winner = max(idx, key=lambda i: scores[i])
            return population[winner].copy()

        # Elitism: keep best
        best_idx = max(range(population_size), key=lambda i: scores[i])
        new_pop = [population[best_idx].copy()]

        while len(new_pop) < population_size:
            p1 = select()
            p2 = select()
            # Single-point crossover
            pt = random.randint(1, T - 1) if T > 1 else 0
            child = np.concatenate([p1[:pt], p2[pt:]])
            # Mutation
            for i in range(T):
                if random.random() < mutation_rate:
                    child[i] = random.randint(-MAX_FIRE_PER_WEEK, MAX_HIRE_PER_WEEK)
            new_pop.append(child)

        population = new_pop

    best_idx = max(range(population_size), key=lambda i: eval_fit(population[i]))
    best_chromosome = population[best_idx]
    best_fitness = eval_fit(best_chromosome)
    return best_chromosome, best_fitness, fitness_history


def staffing_plan_from_chromosome(
    deltas: np.ndarray,
    initial_agents: int,
    weeks_ramp: int = WEEKS_RAMP,
) -> pd.DataFrame:
    """Build a table: week, agents_total, agents_active, delta (hire/fire)."""
    T = len(deltas)
    A_total = np.zeros(T + 1)
    A_total[0] = initial_agents
    for t in range(T):
        A_total[t + 1] = A_total[t] + deltas[t]
    A_total = np.maximum(A_total, 0)

    A_active = np.zeros(T + 1)
    A_active[0] = initial_agents
    for t in range(1, T + 1):
        hires_active_by_t = sum(
            max(0, int(deltas[s])) for s in range(max(0, t - weeks_ramp + 1))
        )
        fires_by_t = sum(max(0, int(-deltas[s])) for s in range(t))
        A_active[t] = initial_agents + hires_active_by_t - fires_by_t
    A_active = np.maximum(A_active, 0)

    return pd.DataFrame({
        "week": np.arange(T + 1),
        "agents_total": A_total.astype(int),
        "agents_active": A_active.astype(int),
        "delta": np.concatenate([[0], deltas]).astype(int),
    })
