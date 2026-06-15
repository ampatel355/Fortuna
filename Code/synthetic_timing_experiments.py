"""Synthetic validation for the structure-preserving timing null.

The goal is to test calibration and power in worlds where the truth is known:
no timing skill, drift-only profitability, volatility clustering without skill,
and explicit entry-placement skill.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "Code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

os.environ.setdefault("MONTE_CARLO_SIMULATE_EXECUTION_COSTS", "0")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import monte_carlo
from artifact_provenance import write_dataframe_artifact

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SyntheticConfig:
    horizon_bars: int = int(os.environ.get("SYNTHETIC_HORIZON_BARS", "1250"))
    trade_count: int = int(os.environ.get("SYNTHETIC_TRADE_COUNT", "40"))
    replications: int = int(os.environ.get("SYNTHETIC_REPLICATIONS", "200"))
    power_replications: int = int(os.environ.get("SYNTHETIC_POWER_REPLICATIONS", "120"))
    simulations: int = int(os.environ.get("SYNTHETIC_SIMULATIONS", "2000"))
    base_seed: int = int(os.environ.get("SYNTHETIC_BASE_SEED", "9100"))
    alpha: float = float(os.environ.get("SYNTHETIC_ALPHA", "0.05"))


DEFAULT_POWER_SIGNAL_STRENGTHS = (0.0, 0.0005, 0.0010, 0.0020, 0.0035, 0.0050)


def parse_signal_strengths() -> tuple[float, ...]:
    """Parse the signal-strength grid used for the power curve."""
    configured = os.environ.get("SYNTHETIC_POWER_SIGNAL_STRENGTHS", "").strip()
    if not configured:
        return DEFAULT_POWER_SIGNAL_STRENGTHS
    return tuple(float(value.strip()) for value in configured.split(",") if value.strip())


def build_price_path(returns: np.ndarray, start_price: float = 100.0) -> np.ndarray:
    """Convert one-period returns into an open-price path."""
    prices = np.empty(len(returns) + 1, dtype=float)
    prices[0] = start_price
    prices[1:] = start_price * np.cumprod(1.0 + returns)
    return prices


def signal_event_starts(horizon_bars: int, trade_count: int, duration: int = 8) -> np.ndarray:
    """Return deterministic non-overlapping event starts for entry-skill worlds."""
    periods = horizon_bars - 1
    if trade_count <= 0:
        raise ValueError("Trade count must be positive.")
    if periods <= duration + 2:
        raise ValueError("Synthetic horizon is too short for signal events.")

    starts = np.linspace(20, periods - duration - 20, trade_count)
    starts = np.rint(starts).astype(np.int64)
    for index in range(1, len(starts)):
        starts[index] = max(starts[index], starts[index - 1] + duration + 1)
    if starts[-1] + duration > periods:
        raise ValueError("Signal events do not fit in the synthetic horizon.")
    return starts


def generate_returns(
    world: str,
    horizon_bars: int,
    rng: np.random.Generator,
    *,
    signal_strength: float = 0.0,
    trade_count: int = 40,
    event_starts: np.ndarray | None = None,
) -> np.ndarray:
    """Generate synthetic open-to-open returns for one world."""
    periods = horizon_bars - 1
    if world == "iid_no_skill":
        return rng.normal(loc=0.0, scale=0.010, size=periods)
    if world == "drift_no_skill":
        return rng.normal(loc=0.00045, scale=0.010, size=periods)
    if world == "structural_skill_absorbed":
        return rng.normal(loc=0.00055, scale=0.010, size=periods)
    if world == "vol_cluster_no_skill":
        states = np.empty(periods, dtype=np.int8)
        states[0] = rng.integers(0, 2)
        for idx in range(1, periods):
            stay_probability = 0.96 if states[idx - 1] == 0 else 0.90
            states[idx] = states[idx - 1] if rng.random() < stay_probability else 1 - states[idx - 1]
        volatility = np.where(states == 0, 0.006, 0.024)
        return rng.normal(loc=0.0, scale=volatility)
    if world == "entry_skill":
        returns = rng.normal(loc=0.0, scale=0.010, size=periods)
        starts = (
            np.asarray(event_starts, dtype=np.int64)
            if event_starts is not None
            else signal_event_starts(horizon_bars, trade_count)
        )
        if len(starts) != trade_count:
            raise ValueError("Entry-skill event count must match the configured trade count.")
        for start in starts:
            returns[start : start + 8] += signal_strength
        return returns
    raise ValueError(f"Unknown synthetic world: {world}")


def make_trade_structure(
    entry_indices: np.ndarray,
    durations: np.ndarray,
    max_open_index: int,
) -> monte_carlo.TradeStructure:
    """Build a TradeStructure from realized entries and durations."""
    exit_indices = entry_indices + durations
    internal_gap_sizes, external_slack = monte_carlo.calculate_gap_structure(
        entry_indices=entry_indices,
        exit_indices=exit_indices,
        max_open_index=max_open_index,
    )
    return monte_carlo.TradeStructure(
        entry_indices=entry_indices.astype(np.int64, copy=False),
        exit_indices=exit_indices.astype(np.int64, copy=False),
        durations=durations.astype(np.int64, copy=False),
        position_value_fractions=np.ones(len(durations), dtype=float),
        direction_signs=np.ones(len(durations), dtype=np.int8),
        asset_indices=np.zeros(len(durations), dtype=np.int64),
        transition_gap_floors=monte_carlo.calculate_transition_gap_floors(
            entry_indices=entry_indices,
            exit_indices=exit_indices,
        ),
        internal_gap_sizes=internal_gap_sizes,
        external_slack=external_slack,
    )


def sample_random_structure(
    config: SyntheticConfig,
    rng: np.random.Generator,
    *,
    duration_low: int = 4,
    duration_high: int = 13,
) -> monte_carlo.TradeStructure:
    """Sample a realized structure from the same schedule family used by the null."""
    max_open_index = config.horizon_bars - 1
    durations = rng.integers(duration_low, duration_high, size=config.trade_count, dtype=np.int64)
    minimum_span = int(durations.sum())
    slack = max_open_index - minimum_span
    if slack < config.trade_count:
        raise ValueError("Synthetic horizon is too short for the requested trade count.")

    gap_slots = rng.multinomial(slack, np.full(config.trade_count + 1, 1 / (config.trade_count + 1)))
    internal_gap_sizes = gap_slots[1:-1].astype(np.int64, copy=False)
    external_slack = int(gap_slots[0] + gap_slots[-1])
    base_structure = monte_carlo.TradeStructure(
        entry_indices=np.array([], dtype=np.int64),
        exit_indices=np.array([], dtype=np.int64),
        durations=durations,
        position_value_fractions=np.ones(config.trade_count, dtype=float),
        direction_signs=np.ones(config.trade_count, dtype=np.int8),
        asset_indices=np.zeros(config.trade_count, dtype=np.int64),
        transition_gap_floors=np.ones(config.trade_count - 1, dtype=np.int64),
        internal_gap_sizes=internal_gap_sizes,
        external_slack=external_slack,
    )
    entry_indices, _ = monte_carlo.build_structure_preserving_schedule_batch(
        trade_structure=base_structure,
        max_open_index=max_open_index,
        batch_size=1,
        rng=rng,
    )
    return make_trade_structure(
        entry_indices=entry_indices[0],
        durations=durations,
        max_open_index=max_open_index,
    )


def choose_skilled_structure(
    open_prices: np.ndarray,
    config: SyntheticConfig,
) -> monte_carlo.TradeStructure:
    """Choose the deterministic event windows that carry synthetic entry signal."""
    duration = 8
    max_open_index = len(open_prices) - 1
    entry_indices = signal_event_starts(len(open_prices), config.trade_count, duration)
    durations = np.full(config.trade_count, duration, dtype=np.int64)
    return make_trade_structure(
        entry_indices=entry_indices,
        durations=durations,
        max_open_index=max_open_index,
    )


def cumulative_return_for_structure(
    open_prices: np.ndarray,
    trade_structure: monte_carlo.TradeStructure,
) -> float:
    """Calculate long-only cumulative return for one realized synthetic schedule."""
    gross_returns = (
        open_prices[trade_structure.exit_indices] / open_prices[trade_structure.entry_indices]
    ) - 1.0
    log_returns = np.log1p(gross_returns * trade_structure.position_value_fractions)
    return float(np.expm1(log_returns.sum()))


def run_one(
    *,
    world: str,
    config: SyntheticConfig,
    replication: int,
    signal_strength: float = 0.0,
    experiment: str = "world_validation",
) -> dict[str, float | int | str | bool]:
    """Run one synthetic validation replication."""
    world_offsets = {
        "iid_no_skill": 11,
        "drift_no_skill": 23,
        "vol_cluster_no_skill": 37,
        "structural_skill_absorbed": 43,
        "entry_skill": 53,
    }
    signal_offset = int(round(signal_strength * 1_000_000))
    rng = np.random.default_rng(
        config.base_seed + replication * 1009 + world_offsets[world] + signal_offset
    )
    if world == "entry_skill":
        trade_structure = sample_random_structure(
            config,
            rng,
            duration_low=8,
            duration_high=9,
        )
        returns = generate_returns(
            world,
            config.horizon_bars,
            rng,
            signal_strength=signal_strength,
            trade_count=config.trade_count,
            event_starts=trade_structure.entry_indices,
        )
    elif world == "structural_skill_absorbed":
        returns = generate_returns(
            world,
            config.horizon_bars,
            rng,
            signal_strength=signal_strength,
            trade_count=config.trade_count,
        )
        trade_structure = sample_random_structure(
            config,
            rng,
            duration_low=16,
            duration_high=29,
        )
    else:
        returns = generate_returns(
            world,
            config.horizon_bars,
            rng,
            signal_strength=signal_strength,
            trade_count=config.trade_count,
        )
        trade_structure = sample_random_structure(config, rng)

    open_prices = build_price_path(returns)

    actual_cumulative_return = cumulative_return_for_structure(open_prices, trade_structure)
    calendar_dates = tuple(pd.date_range("2000-01-03", periods=len(open_prices), freq="B"))
    null_model_inputs = monte_carlo.NullModelInputs(
        open_price_matrix=open_prices.reshape(1, -1),
        trade_structure=trade_structure,
        null_model_name="synthetic_structure_preserving_random_timing",
        calendar_dates=calendar_dates,
        context_entry_candidate_pools=tuple(),
    )
    simulated_returns = monte_carlo.simulate_structure_preserving_cumulative_returns(
        null_model_inputs=null_model_inputs,
        simulation_count=config.simulations,
        rng=np.random.default_rng(config.base_seed + replication * 1009 + 17),
    ).to_numpy(dtype=float)

    p_value = monte_carlo.calculate_p_value(simulated_returns, actual_cumulative_return)
    percentile = monte_carlo.calculate_actual_percentile(
        simulated_returns,
        actual_cumulative_return,
    )
    simulated_mean = float(np.mean(simulated_returns))
    simulated_std = float(np.std(simulated_returns, ddof=0))
    rcsi = actual_cumulative_return - simulated_mean
    rcsi_z = rcsi / simulated_std if simulated_std > 0 else np.nan

    return {
        "experiment": experiment,
        "world": world,
        "replication": replication,
        "signal_strength": signal_strength,
        "horizon_bars": config.horizon_bars,
        "trade_count": config.trade_count,
        "simulations": config.simulations,
        "actual_cumulative_return": actual_cumulative_return,
        "simulated_mean": simulated_mean,
        "simulated_std": simulated_std,
        "p_value": p_value,
        "percentile": percentile,
        "RCSI": rcsi,
        "RCSI_z": rcsi_z,
        "reject_alpha": bool(p_value <= config.alpha),
        "profitable_actual": bool(actual_cumulative_return > 0),
    }


def ks_uniform_test(values: np.ndarray) -> tuple[float, float]:
    """One-sample Kolmogorov-Smirnov test of ``values`` against Uniform[0,1].

    Returns the KS distance D and an asymptotic p-value (Stephens' small-sample
    correction with the Kolmogorov survival series). Self-contained so the
    project does not depend on SciPy.
    """
    x = np.sort(np.clip(np.asarray(values, dtype=float), 0.0, 1.0))
    n = x.size
    if n == 0:
        return float("nan"), float("nan")
    i = np.arange(1, n + 1)
    d = float(max(np.max(i / n - x), np.max(x - (i - 1) / n)))
    t = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d
    if t <= 0:
        return d, 1.0
    series = sum((-1) ** (k - 1) * np.exp(-2.0 * k * k * t * t) for k in range(1, 101))
    return d, float(min(1.0, max(0.0, 2.0 * series)))


def summarize_runs(runs_df: pd.DataFrame, config: SyntheticConfig) -> pd.DataFrame:
    """Aggregate validation metrics by synthetic world."""
    rows: list[dict[str, float | int | str]] = []
    group_columns = ["experiment", "world", "signal_strength"]
    for group_key, world_df in runs_df.groupby(group_columns, sort=True):
        experiment, world, signal_strength = group_key
        p_values = world_df["p_value"].to_numpy(dtype=float)
        n_rep = int(len(world_df))
        r05 = float(np.mean(p_values <= 0.05))
        # Monte Carlo standard error of a rejection-rate estimate (binomial).
        se05 = float(np.sqrt(r05 * (1.0 - r05) / n_rep)) if n_rep > 0 else float("nan")
        ks_d, ks_p = ks_uniform_test(p_values)
        rows.append(
            {
                "experiment": str(experiment),
                "world": world,
                "signal_strength": float(signal_strength),
                "replications": n_rep,
                "simulations_per_replication": config.simulations,
                "mean_actual_cumulative_return": float(world_df["actual_cumulative_return"].mean()),
                "mean_p_value": float(np.mean(p_values)),
                "median_p_value": float(np.median(p_values)),
                "reject_rate_alpha": float(np.mean(p_values <= config.alpha)),
                "reject_rate_0_10": float(np.mean(p_values <= 0.10)),
                "reject_rate_0_05": r05,
                "mc_se_reject_0_05": se05,
                "reject_rate_0_01": float(np.mean(p_values <= 0.01)),
                "ks_stat_uniform": ks_d,
                "ks_pvalue_uniform": ks_p,
                "mean_percentile": float(world_df["percentile"].mean()),
                "mean_RCSI_z": float(world_df["RCSI_z"].mean()),
                "profitable_actual_rate": float(world_df["profitable_actual"].mean()),
            }
        )
    return pd.DataFrame(rows)


def create_type1_calibration_plot(runs_df: pd.DataFrame, output_path: Path) -> None:
    """Save a p-value calibration chart for no-entry-skill worlds."""
    no_skill_worlds = [
        "iid_no_skill",
        "vol_cluster_no_skill",
        "drift_no_skill",
        "structural_skill_absorbed",
    ]
    plot_df = runs_df.loc[
        (runs_df["experiment"] == "world_validation")
        & (runs_df["world"].isin(no_skill_worlds))
    ].copy()
    if plot_df.empty:
        raise ValueError("No no-skill rows available for calibration plotting.")

    labels = {
        "iid_no_skill": "IID",
        "vol_cluster_no_skill": "Vol. clustering",
        "drift_no_skill": "Drift only",
        "structural_skill_absorbed": "Structural exposure",
    }
    colors = {
        "iid_no_skill": "#355C7D",
        "vol_cluster_no_skill": "#2A9D8F",
        "drift_no_skill": "#B76E21",
        "structural_skill_absorbed": "#6D597A",
    }

    fig, (ax_cdf, ax_bar) = plt.subplots(1, 2, figsize=(11.2, 4.8))
    grid = np.linspace(0.0, 1.0, 101)
    ax_cdf.plot(grid, grid, color="#4F4F4F", linewidth=1.2, linestyle="--", label="Uniform")
    for world in no_skill_worlds:
        world_p = np.sort(plot_df.loc[plot_df["world"] == world, "p_value"].to_numpy(dtype=float))
        if world_p.size == 0:
            continue
        empirical = np.arange(1, world_p.size + 1) / world_p.size
        ax_cdf.step(
            world_p,
            empirical,
            where="post",
            linewidth=1.8,
            color=colors[world],
            label=labels[world],
        )
    ax_cdf.set_title("Null p-value calibration")
    ax_cdf.set_xlabel("p-value")
    ax_cdf.set_ylabel("Empirical CDF")
    ax_cdf.set_xlim(0, 1)
    ax_cdf.set_ylim(0, 1)
    ax_cdf.grid(axis="both", color="#D6DCE3", linewidth=0.7, alpha=0.8)
    ax_cdf.legend(frameon=False, fontsize=8)

    alpha = 0.05
    rejection_rates = [
        float(np.mean(plot_df.loc[plot_df["world"] == world, "p_value"].to_numpy(dtype=float) <= alpha))
        for world in no_skill_worlds
    ]
    rejection_se = [
        float(
            np.sqrt(
                rate * (1.0 - rate)
                / max(1, int((plot_df["world"] == world).sum()))
            )
        )
        for world, rate in zip(no_skill_worlds, rejection_rates)
    ]
    x_positions = np.arange(len(no_skill_worlds))
    ax_bar.bar(
        x_positions,
        rejection_rates,
        yerr=rejection_se,
        capsize=4,
        error_kw={"elinewidth": 1.1, "ecolor": "#2E2E2E"},
        color=[colors[world] for world in no_skill_worlds],
        edgecolor="#2E2E2E",
        linewidth=0.6,
    )
    ax_bar.axhline(alpha, color="#D62728", linewidth=1.2, linestyle="--", label="Nominal 5%")
    ax_bar.set_title("False positive rate at 5%")
    ax_bar.set_ylabel("Rejection rate")
    ax_bar.set_xticks(x_positions)
    ax_bar.set_xticklabels([labels[world] for world in no_skill_worlds], rotation=20, ha="right")
    ax_bar.set_ylim(0, max(0.16, max(rejection_rates) * 1.25))
    ax_bar.grid(axis="y", color="#D6DCE3", linewidth=0.7, alpha=0.8)
    ax_bar.legend(frameon=False, fontsize=8)

    fig.suptitle("Synthetic no-entry-skill calibration", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_power_curve_plot(power_summary_df: pd.DataFrame, output_path: Path) -> None:
    """Save the entry-signal power curve."""
    plot_df = power_summary_df.sort_values("signal_strength").copy()
    if plot_df.empty:
        raise ValueError("No power rows available for power plotting.")

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    x_values = plot_df["signal_strength"].to_numpy(dtype=float) * 10000
    curves = [
        ("reject_rate_0_10", "alpha = 0.10", "#355C7D"),
        ("reject_rate_0_05", "alpha = 0.05", "#D62728"),
        ("reject_rate_0_01", "alpha = 0.01", "#2A9D8F"),
    ]
    for column, label, color in curves:
        ax.plot(
            x_values,
            plot_df[column].to_numpy(dtype=float),
            marker="o",
            linewidth=2.0,
            color=color,
            label=label,
        )
    ax.set_title("Power against deterministic entry signal")
    ax.set_xlabel("Injected signal per event bar (basis points)")
    ax.set_ylabel("Rejection rate")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="both", color="#D6DCE3", linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run all synthetic validation worlds and save run-level plus summary outputs."""
    config = SyntheticConfig()
    monte_carlo.SIMULATE_EXECUTION_COSTS = False
    monte_carlo.TRANSACTION_COST = 0.0

    validation_worlds = [
        "iid_no_skill",
        "drift_no_skill",
        "vol_cluster_no_skill",
        "structural_skill_absorbed",
        "entry_skill",
    ]
    rows: list[dict[str, float | int | str | bool]] = []
    for world in validation_worlds:
        signal_strength = 0.0035 if world == "entry_skill" else 0.0
        for replication in range(1, config.replications + 1):
            rows.append(
                run_one(
                    world=world,
                    config=config,
                    replication=replication,
                    signal_strength=signal_strength,
                    experiment="world_validation",
                )
            )

    for signal_strength in parse_signal_strengths():
        for replication in range(1, config.power_replications + 1):
            rows.append(
                run_one(
                    world="entry_skill",
                    config=config,
                    replication=replication,
                    signal_strength=signal_strength,
                    experiment="power_curve",
                )
            )

    runs_df = pd.DataFrame(rows)
    summary_df = summarize_runs(runs_df, config)
    power_summary_df = summary_df.loc[
        (summary_df["experiment"] == "power_curve")
        & (summary_df["world"] == "entry_skill")
    ].copy()

    data_clean_dir = PROJECT_ROOT / "Data_Clean"
    charts_dir = PROJECT_ROOT / "Charts"
    data_clean_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    runs_path = data_clean_dir / "synthetic_null_validation_runs.csv"
    summary_path = data_clean_dir / "synthetic_null_validation_summary.csv"
    power_summary_path = data_clean_dir / "synthetic_power_curve_summary.csv"
    calibration_chart_path = charts_dir / "synthetic_type1_calibration.png"
    power_chart_path = charts_dir / "synthetic_power_curve.png"

    runs_write = write_dataframe_artifact(
        runs_df,
        runs_path,
        producer="synthetic_timing_experiments.main",
        research_grade=(
            config.replications >= 200
            and config.power_replications >= 120
            and config.simulations >= 2000
        ),
        canonical_policy="auto",
        parameters={**config.__dict__, "power_signal_strengths": parse_signal_strengths()},
    )
    summary_write = write_dataframe_artifact(
        summary_df,
        summary_path,
        producer="synthetic_timing_experiments.main",
        dependencies=[runs_path],
        research_grade=(
            config.replications >= 200
            and config.power_replications >= 120
            and config.simulations >= 2000
        ),
        canonical_policy="auto",
        parameters={**config.__dict__, "power_signal_strengths": parse_signal_strengths()},
    )
    power_write = write_dataframe_artifact(
        power_summary_df,
        power_summary_path,
        producer="synthetic_timing_experiments.main",
        dependencies=[summary_path],
        research_grade=(
            config.replications >= 200
            and config.power_replications >= 120
            and config.simulations >= 2000
        ),
        canonical_policy="auto",
        parameters={**config.__dict__, "power_signal_strengths": parse_signal_strengths()},
    )

    create_type1_calibration_plot(runs_df, calibration_chart_path)
    create_power_curve_plot(power_summary_df, power_chart_path)

    print(summary_df.to_string(index=False))
    print(f"\nSaved synthetic validation runs to {runs_write['versioned_path']}")
    print(f"Saved synthetic validation summary to {summary_write['versioned_path']}")
    print(f"Saved synthetic power summary to {power_write['versioned_path']}")
    print(f"Saved calibration chart to {calibration_chart_path}")
    print(f"Saved power curve to {power_chart_path}")
    if (
        runs_write["canonical_updated"]
        and summary_write["canonical_updated"]
        and power_write["canonical_updated"]
    ):
        print(f"Updated canonical runs at {runs_path}")
        print(f"Updated canonical summary at {summary_path}")
        print(f"Updated canonical power summary at {power_summary_path}")


if __name__ == "__main__":
    main()
