"""Run a timing-based Monte Carlo null model for all active strategies."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from artifact_provenance import write_dataframe_artifact
    from execution_model import (
        EXPECTED_COMMISSION_RATE,
        EXPECTED_ROUND_TRIP_EXECUTION_COST,
        HALF_SPREAD_BPS,
        MAX_SLIPPAGE_BPS,
        MIN_SLIPPAGE_BPS,
        TRADE_RETURNS_ALREADY_NET,
    )
    from momentum_relative_strength_agent import load_aligned_universe_data
    from research_metrics import calculate_p_value_prominence
    from regimes import build_regime_dataframe_for_ticker
    from strategy_artifact_utils import ensure_trade_file_exists
    from strategy_config import AGENT_ORDER
    from timeframe_config import RESEARCH_INTERVAL, normalize_timestamp_series, scale_daily_bars, timeframe_output_suffix
except ModuleNotFoundError:
    from Code.artifact_provenance import write_dataframe_artifact
    from Code.execution_model import (
        EXPECTED_COMMISSION_RATE,
        EXPECTED_ROUND_TRIP_EXECUTION_COST,
        HALF_SPREAD_BPS,
        MAX_SLIPPAGE_BPS,
        MIN_SLIPPAGE_BPS,
        TRADE_RETURNS_ALREADY_NET,
    )
    from Code.momentum_relative_strength_agent import load_aligned_universe_data
    from Code.research_metrics import calculate_p_value_prominence
    from Code.regimes import build_regime_dataframe_for_ticker
    from Code.strategy_artifact_utils import ensure_trade_file_exists
    from Code.strategy_config import AGENT_ORDER
    from Code.timeframe_config import RESEARCH_INTERVAL, normalize_timestamp_series, scale_daily_bars, timeframe_output_suffix


# Read the active ticker from the environment, or fall back to SPY.
ticker = os.environ.get("TICKER", "SPY")

# These settings are easy to change at the top of the script.
NUMBER_OF_SIMULATIONS = int(os.environ.get("MONTE_CARLO_SIMULATIONS", "5000"))
TRANSACTION_COST = float(
    os.environ.get(
        "SIMULATED_TRANSACTION_COST",
        str(EXPECTED_ROUND_TRIP_EXECUTION_COST),
    )
)
REPRODUCIBLE = os.environ.get("MONTE_CARLO_REPRODUCIBLE", "1") == "1"
SEED = int(os.environ.get("MONTE_CARLO_SEED", "42"))
SIMULATION_BATCH_SIZE = int(os.environ.get("MONTE_CARLO_BATCH_SIZE", "512"))
NULL_MODEL_NAME = "random_timing_matched_duration"
RELATIVE_STRENGTH_NULL_MODEL_NAME = "random_peer_rotation_matched_duration"
LEGACY_NULL_MODEL_NAME = NULL_MODEL_NAME
LEGACY_RELATIVE_STRENGTH_NULL_MODEL_NAME = RELATIVE_STRENGTH_NULL_MODEL_NAME
_context_lookback_explicit = os.environ.get("MONTE_CARLO_ENTRY_CONTEXT_LOOKBACK_BARS")
ENTRY_CONTEXT_LOOKBACK_BARS = (
    int(_context_lookback_explicit)
    if _context_lookback_explicit is not None
    else scale_daily_bars(20)
)
CONTEXT_MATCHING_ENABLED = os.environ.get("MONTE_CARLO_CONTEXT_MATCHING", "0") == "1"
CONTEXT_BUCKET_COUNT = max(2, int(os.environ.get("MONTE_CARLO_CONTEXT_BUCKET_COUNT", "5")))
SIMULATE_EXECUTION_COSTS = os.environ.get("MONTE_CARLO_SIMULATE_EXECUTION_COSTS", "1") == "1"
# Optional feasibility floor on the leading gap: the first simulated entry cannot
# be placed before this calendar index (e.g. a universal signal warm-up region a
# real strategy could not have traded in). Default 0 reproduces the original
# gap-permutation null exactly; used only for the leading-gap robustness check.
MIN_LEADING_INDEX = max(0, int(os.environ.get("MONTE_CARLO_MIN_LEADING_INDEX", "0")))
MIN_RESEARCH_GRADE_SIMULATIONS = int(os.environ.get("MIN_RESEARCH_GRADE_SIMULATIONS", "5000"))
STRICT_HOLDING_BARS_VALIDATION = (
    os.environ.get("STRICT_HOLDING_BARS_VALIDATION", "0") == "1"
)
TRADE_DIRECTION_COLUMN_CANDIDATES = (
    "direction",
    "position_direction",
    "signal_direction",
    "trade_direction",
    "side",
)

NULL_MODEL_NAME = (
    "structure_preserving_random_timing_context_matched"
    if CONTEXT_MATCHING_ENABLED
    else "structure_preserving_random_timing"
)
RELATIVE_STRENGTH_NULL_MODEL_NAME = (
    "structure_preserving_random_timing_shared_calendar_context_matched"
    if CONTEXT_MATCHING_ENABLED
    else "structure_preserving_random_timing_shared_calendar"
)


@dataclass(frozen=True)
class TradeStructure:
    """Realized trade structure that must be preserved while timing is randomized."""

    entry_indices: np.ndarray
    exit_indices: np.ndarray
    durations: np.ndarray
    position_value_fractions: np.ndarray
    direction_signs: np.ndarray
    asset_indices: np.ndarray
    transition_gap_floors: np.ndarray
    internal_gap_sizes: np.ndarray
    external_slack: int


@dataclass(frozen=True)
class NullModelInputs:
    """All inputs required to simulate one strategy against the timing null."""

    open_price_matrix: np.ndarray
    trade_structure: TradeStructure
    null_model_name: str
    calendar_dates: tuple[pd.Timestamp, ...]
    context_entry_candidate_pools: tuple[np.ndarray, ...]


def resolve_data_clean_dir(project_root: Path) -> Path:
    """Return the clean-data folder, supporting either naming style."""
    suffix = timeframe_output_suffix()
    uppercase_dir = project_root / f"Data_Clean{suffix}"
    lowercase_dir = project_root / f"data_clean{suffix}"

    if uppercase_dir.exists():
        return uppercase_dir
    if lowercase_dir.exists():
        return lowercase_dir

    uppercase_dir.mkdir(parents=True, exist_ok=True)
    return uppercase_dir


def load_csv_checked(
    input_path: Path,
    required_columns: list[str],
    allow_empty: bool = False,
) -> pd.DataFrame:
    """Load a CSV file and confirm that it has the columns we need."""
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    df = pd.read_csv(input_path)

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(
            f"Missing required columns in {input_path}: {', '.join(missing_columns)}"
        )

    if df.empty and not allow_empty:
        raise ValueError(f"The input file is empty: {input_path}")

    return df


def load_trade_data(input_path: Path, allow_empty: bool = False) -> pd.DataFrame:
    """Load one trade file and validate its dates and returns."""
    df = load_csv_checked(
        input_path,
        required_columns=["entry_date", "exit_date", "return"],
        allow_empty=allow_empty,
    )

    df["entry_date"] = normalize_timestamp_series(df["entry_date"])
    df["exit_date"] = normalize_timestamp_series(df["exit_date"])
    df["return"] = pd.to_numeric(df["return"], errors="coerce")
    if "position_value_fraction" in df.columns:
        df["position_value_fraction"] = pd.to_numeric(
            df["position_value_fraction"],
            errors="coerce",
        )
    else:
        df["position_value_fraction"] = 1.0
    if "holding_bars" in df.columns:
        df["holding_bars"] = pd.to_numeric(df["holding_bars"], errors="coerce")

    if df.empty:
        return df.sort_values("exit_date", ascending=True).reset_index(drop=True)

    invalid_rows = df[
        df["entry_date"].isna() | df["exit_date"].isna() | df["return"].isna()
    ]
    if not invalid_rows.empty:
        raise ValueError(
            f"{input_path} contains {len(invalid_rows)} row(s) with invalid dates or returns."
        )

    df = df.sort_values("exit_date", ascending=True).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No usable trade data was found in: {input_path}")

    return df


def build_no_trade_simulated_returns(simulation_count: int) -> pd.Series:
    """Create a degenerate zero-return simulation series for no-trade strategies."""
    return pd.Series(
        np.zeros(simulation_count, dtype=float),
        name="simulated_cumulative_return",
    )


def extract_position_value_fractions(trade_df: pd.DataFrame) -> np.ndarray:
    """Return one portfolio-at-risk fraction for each realized trade."""
    fractions = trade_df["position_value_fraction"].to_numpy(dtype=float)

    if np.any(np.isnan(fractions)):
        raise ValueError("Trade data contains invalid position_value_fraction values.")
    if np.any(fractions <= 0):
        raise ValueError("Trade data contains non-positive position_value_fraction values.")
    if np.any(fractions > 1.0):
        raise ValueError("Trade data contains position_value_fraction values above 1.0.")

    return fractions


def load_market_data(input_path: Path) -> pd.DataFrame:
    """Load the ticker's open-price history used for random-timing simulations."""
    if input_path.exists():
        existing_df = pd.read_csv(input_path)
        interval_is_compatible = (
            existing_df.get("data_interval", pd.Series(dtype="object")).astype(str).str.lower().eq(RESEARCH_INTERVAL).any()
            if "data_interval" in existing_df.columns
            else False
        )
        if not interval_is_compatible:
            build_regime_dataframe_for_ticker(ticker, save_output=True)

    df = load_csv_checked(
        input_path,
        required_columns=["Date", "Open"],
    )

    df["Date"] = normalize_timestamp_series(df["Date"])
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df = df.dropna(subset=["Date", "Open"]).sort_values("Date").reset_index(drop=True)

    if len(df) < 2:
        raise ValueError(
            f"Need at least two usable open-price rows to simulate trades: {input_path}"
        )

    if (df["Open"] <= 0).any():
        raise ValueError(f"Open prices must stay positive for simulation: {input_path}")

    return df


def adjust_trade_returns(
    raw_returns: np.ndarray,
    transaction_cost: float,
    input_path: Path,
) -> np.ndarray:
    """Subtract a fixed transaction cost from each trade return."""
    if TRADE_RETURNS_ALREADY_NET:
        adjusted_returns = raw_returns.astype(float, copy=True)
    else:
        adjusted_returns = raw_returns - transaction_cost

    # Log returns require 1 + adjusted_return to stay positive.
    if np.any(adjusted_returns <= -1):
        smallest_value = float(adjusted_returns.min())
        raise ValueError(
            f"{input_path} contains an adjusted return of {smallest_value:.6f}, "
            "which is too low for log-return conversion. "
            "Reduce the transaction cost or inspect the trade data."
        )

    return adjusted_returns


def convert_to_log_returns(adjusted_returns: np.ndarray) -> np.ndarray:
    """Convert normal trade returns into log returns."""
    return np.log1p(adjusted_returns)


def calculate_cumulative_return_from_log_returns(log_returns: np.ndarray) -> float:
    """Convert a sequence of log returns into a final cumulative return."""
    return float(np.expm1(log_returns.sum()))


def build_random_generator(reproducible: bool, seed: int) -> np.random.Generator:
    """Create either a reproducible or stochastic random number generator."""
    if reproducible:
        return np.random.default_rng(seed)

    # No seed means a fresh stochastic sequence on each run.
    return np.random.default_rng()


def calculate_trade_durations(
    trade_df: pd.DataFrame,
    market_df: pd.DataFrame,
    input_path: Path,
) -> np.ndarray:
    """Convert realized trades into holding periods measured in open-to-open bars."""
    _, _, durations = calculate_trade_indices_and_durations(
        trade_df=trade_df,
        calendar_dates=market_df["Date"],
        input_path=input_path,
    )
    return durations


def calculate_trade_indices_and_durations(
    trade_df: pd.DataFrame,
    calendar_dates: pd.Series | pd.Index | list[pd.Timestamp] | tuple[pd.Timestamp, ...],
    input_path: Path | str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map realized entry and exit timestamps onto one trading calendar."""
    normalized_calendar = normalize_timestamp_series(pd.Series(calendar_dates))
    date_to_index = pd.Series(
        normalized_calendar.index.to_numpy(dtype=np.int64),
        index=normalized_calendar.to_numpy(),
    )

    entry_indices = trade_df["entry_date"].map(date_to_index)
    exit_indices = trade_df["exit_date"].map(date_to_index)
    if entry_indices.isna().any() or exit_indices.isna().any():
        raise ValueError(
            f"{input_path} contains trade dates that do not align with the simulation calendar."
        )

    entry_indices_array = entry_indices.to_numpy(dtype=np.int64)
    exit_indices_array = exit_indices.to_numpy(dtype=np.int64)
    durations = exit_indices_array - entry_indices_array
    if np.any(durations <= 0):
        raise ValueError(
            f"{input_path} contains at least one non-positive holding period, "
            "which is inconsistent with next-open execution."
        )

    if "holding_bars" in trade_df.columns:
        recorded_holding_bars = pd.to_numeric(trade_df["holding_bars"], errors="coerce")
        if recorded_holding_bars.notna().all():
            recorded_durations = recorded_holding_bars.to_numpy(dtype=np.int64)
            if np.any(recorded_durations <= 0):
                raise ValueError(
                    f"{input_path} contains at least one non-positive holding_bars value."
                )
            mismatch_mask = recorded_durations != durations
            if np.any(mismatch_mask):
                mismatch_count = int(np.count_nonzero(mismatch_mask))
                max_abs_difference = int(
                    np.max(np.abs(recorded_durations[mismatch_mask] - durations[mismatch_mask]))
                )
                sample_rows = np.flatnonzero(mismatch_mask)[:5]
                sample_details: list[str] = []
                for sample_row in sample_rows:
                    sample_details.append(
                        "row="
                        f"{int(sample_row)} "
                        f"entry={trade_df.iloc[sample_row]['entry_date']} "
                        f"exit={trade_df.iloc[sample_row]['exit_date']} "
                        f"holding_bars={int(recorded_durations[sample_row])} "
                        f"realized_bars={int(durations[sample_row])}"
                    )

                mismatch_message = (
                    f"{input_path} contains {mismatch_count} holding_bars mismatch(es) "
                    "versus entry/exit timestamps "
                    f"(max absolute difference: {max_abs_difference}). "
                    f"Samples: {'; '.join(sample_details)}"
                )
                if STRICT_HOLDING_BARS_VALIDATION:
                    raise ValueError(mismatch_message)

                print(
                    "Warning: "
                    + mismatch_message
                    + " Monte Carlo will use timestamp-derived durations."
                )

    return entry_indices_array, exit_indices_array, durations.astype(np.int64, copy=False)


def extract_trade_directions(trade_df: pd.DataFrame) -> np.ndarray:
    """Return one direction sign per realized trade when the data provides it."""
    for column_name in TRADE_DIRECTION_COLUMN_CANDIDATES:
        if column_name not in trade_df.columns:
            continue
        raw_values = trade_df[column_name].astype(str).str.strip().str.lower()
        mapped_signs = raw_values.map(
            {
                "1": 1,
                "+1": 1,
                "long": 1,
                "buy": 1,
                "bull": 1,
                "-1": -1,
                "short": -1,
                "sell": -1,
                "bear": -1,
            }
        )
        if mapped_signs.notna().all():
            direction_signs = mapped_signs.to_numpy(dtype=np.int8)
            if np.any(direction_signs == 0):
                raise ValueError("Trade direction values cannot be zero.")
            return direction_signs

    return np.ones(len(trade_df), dtype=np.int8)


def calculate_transition_gap_floors(
    entry_indices: np.ndarray,
    exit_indices: np.ndarray,
) -> np.ndarray:
    """Preserve whether the strategy can rotate on the same bar or needs a wait bar."""
    if len(entry_indices) <= 1:
        return np.array([], dtype=np.int64)

    realized_gaps = entry_indices[1:] - exit_indices[:-1]
    if np.any(realized_gaps < 0):
        raise ValueError(
            "Trade logs contain overlapping realized positions, which this null model "
            "cannot randomize fairly."
        )

    # Zero-gap rotations are preserved only when the realized strategy used them.
    return np.where(realized_gaps == 0, 0, 1).astype(np.int64, copy=False)


def calculate_gap_structure(
    entry_indices: np.ndarray,
    exit_indices: np.ndarray,
    max_open_index: int,
) -> tuple[np.ndarray, int]:
    """Extract the realized internal gap sizes and total external slack."""
    if len(entry_indices) == 0:
        raise ValueError("Gap structure requires at least one realized trade.")

    internal_gap_sizes = (
        entry_indices[1:] - exit_indices[:-1]
        if len(entry_indices) > 1
        else np.array([], dtype=np.int64)
    )
    if np.any(internal_gap_sizes < 0):
        raise ValueError("Realized trades overlap and cannot define a valid gap structure.")

    leading_gap = int(entry_indices[0])
    trailing_gap = int(max_open_index - exit_indices[-1])
    if leading_gap < 0 or trailing_gap < 0:
        raise ValueError("Realized trade structure falls outside the available calendar bounds.")

    return internal_gap_sizes.astype(np.int64, copy=False), int(leading_gap + trailing_gap)


def build_trade_structure(
    trade_df: pd.DataFrame,
    calendar_dates: pd.Series | pd.Index | list[pd.Timestamp] | tuple[pd.Timestamp, ...],
    input_path: Path | str,
    max_open_index: int,
    asset_indices: np.ndarray | None = None,
) -> TradeStructure:
    """Extract the realized structure that the timing null must preserve."""
    entry_indices, exit_indices, durations = calculate_trade_indices_and_durations(
        trade_df=trade_df,
        calendar_dates=calendar_dates,
        input_path=input_path,
    )
    position_value_fractions = extract_position_value_fractions(trade_df)
    direction_signs = extract_trade_directions(trade_df)
    if asset_indices is None:
        asset_indices = np.zeros(len(trade_df), dtype=np.int64)
    else:
        asset_indices = np.asarray(asset_indices, dtype=np.int64)
        if len(asset_indices) != len(trade_df):
            raise ValueError("Asset-index sequence must match the number of realized trades.")

    internal_gap_sizes, external_slack = calculate_gap_structure(
        entry_indices=entry_indices,
        exit_indices=exit_indices,
        max_open_index=max_open_index,
    )

    return TradeStructure(
        entry_indices=entry_indices,
        exit_indices=exit_indices,
        durations=durations,
        position_value_fractions=position_value_fractions.astype(float, copy=False),
        direction_signs=direction_signs.astype(np.int8, copy=False),
        asset_indices=asset_indices,
        transition_gap_floors=calculate_transition_gap_floors(
            entry_indices=entry_indices,
            exit_indices=exit_indices,
        ),
        internal_gap_sizes=internal_gap_sizes,
        external_slack=external_slack,
    )


def calculate_weighted_exposure_share(
    durations: np.ndarray,
    position_value_fractions: np.ndarray,
    max_open_index: int,
) -> float:
    """Calculate the bar-weighted exposure share of the calendar."""
    if max_open_index <= 0:
        return 0.0
    return float(
        np.sum(
            durations.astype(float, copy=False)
            * position_value_fractions.astype(float, copy=False)
        )
        / float(max_open_index)
    )


def calculate_same_bar_turnover_share(
    entry_indices: np.ndarray,
    exit_indices: np.ndarray,
) -> float:
    """Measure how often the next trade begins on the same bar as the prior exit."""
    if len(entry_indices) <= 1:
        return 0.0
    return float(np.mean(entry_indices[1:] == exit_indices[:-1]))


def build_entry_context_matrices(
    open_price_matrix: np.ndarray,
    lookback_bars: int = ENTRY_CONTEXT_LOOKBACK_BARS,
) -> tuple[np.ndarray, np.ndarray]:
    """Build simple entry-context features for fairness diagnostics."""
    asset_count, calendar_length = open_price_matrix.shape
    trailing_returns = np.full((asset_count, calendar_length), np.nan, dtype=float)
    trailing_volatility = np.full((asset_count, calendar_length), np.nan, dtype=float)
    if lookback_bars <= 0 or calendar_length <= lookback_bars:
        return trailing_returns, trailing_volatility

    trailing_returns[:, lookback_bars:] = (
        open_price_matrix[:, lookback_bars:] / open_price_matrix[:, :-lookback_bars]
    ) - 1.0

    open_to_open_returns = (open_price_matrix[:, 1:] / open_price_matrix[:, :-1]) - 1.0
    for end_index in range(lookback_bars, calendar_length):
        trailing_volatility[:, end_index] = np.std(
            open_to_open_returns[:, end_index - lookback_bars : end_index],
            axis=1,
            ddof=0,
        )

    return trailing_returns, trailing_volatility


def build_quantile_bucket_matrix(
    values: np.ndarray,
    bucket_count: int,
) -> np.ndarray:
    """Convert a 2D feature matrix into per-asset quantile buckets."""
    asset_count, calendar_length = values.shape
    bucket_matrix = np.full((asset_count, calendar_length), -1, dtype=np.int16)
    if bucket_count <= 1:
        bucket_matrix[np.isfinite(values)] = 0
        return bucket_matrix

    for asset_index in range(asset_count):
        asset_values = values[asset_index]
        valid_mask = np.isfinite(asset_values)
        if not np.any(valid_mask):
            continue

        valid_values = asset_values[valid_mask]
        quantile_edges = np.nanquantile(
            valid_values,
            np.linspace(0.0, 1.0, bucket_count + 1),
        )
        quantile_edges = np.unique(np.asarray(quantile_edges, dtype=float))
        if len(quantile_edges) <= 1:
            bucket_matrix[asset_index, valid_mask] = 0
            continue

        bucket_matrix[asset_index, valid_mask] = np.digitize(
            valid_values,
            quantile_edges[1:-1],
            right=False,
        ).astype(np.int16, copy=False)

    return bucket_matrix


def build_context_entry_candidate_pools(
    trade_structure: TradeStructure,
    open_price_matrix: np.ndarray,
    calendar_dates: tuple[pd.Timestamp, ...] | list[pd.Timestamp] | pd.Series,
) -> tuple[np.ndarray, ...]:
    """Build feasible entry pools that preserve intraday signature and rough context."""
    calendar_series = pd.to_datetime(pd.Series(calendar_dates), errors="coerce")
    if calendar_series.isna().any():
        raise ValueError("Calendar dates contain invalid timestamps for context matching.")

    trailing_returns, trailing_volatility = build_entry_context_matrices(open_price_matrix)
    trailing_return_buckets = build_quantile_bucket_matrix(
        trailing_returns,
        CONTEXT_BUCKET_COUNT,
    )
    trailing_volatility_buckets = build_quantile_bucket_matrix(
        trailing_volatility,
        CONTEXT_BUCKET_COUNT,
    )
    signature_codes = (
        (calendar_series.dt.dayofweek.astype(np.int16) * 24)
        + calendar_series.dt.hour.astype(np.int16)
    ).to_numpy(dtype=np.int16, copy=False)
    candidate_index_base = np.arange(open_price_matrix.shape[1], dtype=np.int64)
    max_open_index = open_price_matrix.shape[1] - 1
    candidate_pools: list[np.ndarray] = []

    for trade_index, duration in enumerate(trade_structure.durations.astype(np.int64, copy=False)):
        asset_index = int(trade_structure.asset_indices[trade_index])
        actual_entry_index = int(trade_structure.entry_indices[trade_index])
        feasible_mask = candidate_index_base <= (max_open_index - int(duration))
        same_signature_mask = signature_codes == signature_codes[actual_entry_index]

        return_bucket = int(trailing_return_buckets[asset_index, actual_entry_index])
        volatility_bucket = int(trailing_volatility_buckets[asset_index, actual_entry_index])

        fully_matched_mask = feasible_mask & same_signature_mask
        if return_bucket >= 0:
            fully_matched_mask &= trailing_return_buckets[asset_index] == return_bucket
        if volatility_bucket >= 0:
            fully_matched_mask &= trailing_volatility_buckets[asset_index] == volatility_bucket

        candidate_pool = candidate_index_base[fully_matched_mask]
        if len(candidate_pool) == 0:
            candidate_pool = candidate_index_base[feasible_mask & same_signature_mask]
        if len(candidate_pool) == 0 and return_bucket >= 0:
            candidate_pool = candidate_index_base[
                feasible_mask
                & (trailing_return_buckets[asset_index] == return_bucket)
            ]
        if len(candidate_pool) == 0 and volatility_bucket >= 0:
            candidate_pool = candidate_index_base[
                feasible_mask
                & (trailing_volatility_buckets[asset_index] == volatility_bucket)
            ]
        if len(candidate_pool) == 0:
            candidate_pool = candidate_index_base[feasible_mask]

        candidate_pools.append(candidate_pool.astype(np.int64, copy=False))

    return tuple(candidate_pools)


def calculate_tail_minimum_span(
    durations: np.ndarray,
    transition_gap_floors: np.ndarray,
) -> np.ndarray:
    """Return the minimum remaining occupied span from each trade onward."""
    number_of_trades = len(durations)
    tail_minimum_span = np.empty(number_of_trades, dtype=np.int64)
    tail_minimum_span[-1] = int(durations[-1])

    for trade_index in range(number_of_trades - 2, -1, -1):
        tail_minimum_span[trade_index] = int(
            durations[trade_index]
            + transition_gap_floors[trade_index]
            + tail_minimum_span[trade_index + 1]
        )

    return tail_minimum_span


def load_relative_strength_universe_symbols(current_ticker: str) -> list[str]:
    """Load the exact peer universe used by the saved relative-strength run when available."""
    project_root = Path(__file__).resolve().parents[1]
    data_clean_directory = resolve_data_clean_dir(project_root)
    metadata_path = data_clean_directory / f"{current_ticker.upper()}_momentum_relative_strength_universe.csv"

    if metadata_path.exists():
        metadata_df = pd.read_csv(metadata_path)
        required_columns = {"universe_position", "universe_ticker", "included_in_aligned_universe"}
        if required_columns.issubset(metadata_df.columns):
            metadata_df["universe_position"] = pd.to_numeric(
                metadata_df["universe_position"],
                errors="coerce",
            )
            metadata_df["included_in_aligned_universe"] = (
                metadata_df["included_in_aligned_universe"].astype(str).str.lower().map(
                    {"true": True, "false": False}
                )
            )
            metadata_df = metadata_df.dropna(subset=["universe_position", "universe_ticker"])
            metadata_df = metadata_df.sort_values("universe_position")
            aligned_universe = [
                str(symbol).strip().upper()
                for symbol, included in zip(
                    metadata_df["universe_ticker"],
                    metadata_df["included_in_aligned_universe"].fillna(False),
                )
                if bool(included)
            ]
            if len(aligned_universe) >= 2:
                return aligned_universe

    raise FileNotFoundError(
        "The relative-strength universe metadata file is missing or incomplete. "
        "Regenerate the momentum_relative_strength trades before Monte Carlo analysis."
    )


@lru_cache(maxsize=None)
def load_relative_strength_calendar(
    current_ticker: str,
) -> tuple[tuple[pd.Timestamp, ...], tuple[str, ...], np.ndarray]:
    """Load the shared aligned calendar used by the relative-strength strategy."""
    universe = load_relative_strength_universe_symbols(current_ticker)
    common_dates, asset_frames = load_aligned_universe_data(universe)
    aligned_universe = tuple(symbol for symbol in universe if symbol in asset_frames)
    if len(aligned_universe) < 2:
        raise ValueError(
            "Relative-strength Monte Carlo needs at least two aligned peer assets."
        )

    open_price_matrix = np.vstack(
        [
            asset_frames[symbol]["Open"].to_numpy(dtype=float)
            for symbol in aligned_universe
        ]
    )
    if np.any(open_price_matrix <= 0):
        raise ValueError(
            "Relative-strength Monte Carlo requires strictly positive peer open prices."
        )

    return tuple(pd.to_datetime(common_dates)), aligned_universe, open_price_matrix


def build_relative_strength_open_price_matrix(current_ticker: str) -> np.ndarray:
    """Load the shared peer-calendar open prices used by relative-strength rotation."""
    _, _, open_price_matrix = load_relative_strength_calendar(current_ticker)
    return open_price_matrix


def prepare_agent_null_model_inputs(
    *,
    agent_name: str,
    current_ticker: str,
    trade_df: pd.DataFrame,
    market_df: pd.DataFrame,
    input_path: Path | str,
) -> NullModelInputs:
    """Assemble the exact realized trade structure needed for fair timing randomization."""
    if trade_df.empty:
        raise ValueError("Cannot prepare null-model inputs for an empty trade log.")

    if agent_name == "momentum_relative_strength":
        common_dates, aligned_universe, open_price_matrix = load_relative_strength_calendar(
            current_ticker
        )
        if "asset_ticker" not in trade_df.columns:
            raise KeyError(
                "Relative-strength trade logs must include asset_ticker so the null can "
                "preserve the realized asset sequence."
            )
        symbol_to_index = {symbol: index for index, symbol in enumerate(aligned_universe)}
        asset_indices = (
            trade_df["asset_ticker"]
            .astype(str)
            .str.strip()
            .str.upper()
            .map(symbol_to_index)
        )
        if asset_indices.isna().any():
            missing_assets = (
                trade_df.loc[asset_indices.isna(), "asset_ticker"]
                .astype(str)
                .str.strip()
                .str.upper()
                .unique()
                .tolist()
            )
            raise ValueError(
                "Relative-strength trades reference assets that are not available on the "
                f"aligned peer calendar: {missing_assets}"
            )

        trade_structure = build_trade_structure(
            trade_df=trade_df,
            calendar_dates=common_dates,
            input_path=input_path,
            max_open_index=open_price_matrix.shape[1] - 1,
            asset_indices=asset_indices.to_numpy(dtype=np.int64),
        )
        candidate_pools = (
            build_context_entry_candidate_pools(
                trade_structure=trade_structure,
                open_price_matrix=open_price_matrix,
                calendar_dates=common_dates,
            )
            if CONTEXT_MATCHING_ENABLED
            else tuple()
        )
        return NullModelInputs(
            open_price_matrix=open_price_matrix,
            trade_structure=trade_structure,
            null_model_name=RELATIVE_STRENGTH_NULL_MODEL_NAME,
            calendar_dates=tuple(pd.to_datetime(common_dates)),
            context_entry_candidate_pools=candidate_pools,
        )

    trade_structure = build_trade_structure(
        trade_df=trade_df,
        calendar_dates=market_df["Date"],
        input_path=input_path,
        max_open_index=len(market_df["Date"]) - 1,
        asset_indices=np.zeros(len(trade_df), dtype=np.int64),
    )
    single_asset_open_prices = market_df["Open"].to_numpy(dtype=float)
    calendar_dates = tuple(pd.to_datetime(market_df["Date"], errors="coerce"))
    candidate_pools = (
        build_context_entry_candidate_pools(
            trade_structure=trade_structure,
            open_price_matrix=single_asset_open_prices.reshape(1, -1),
            calendar_dates=calendar_dates,
        )
        if CONTEXT_MATCHING_ENABLED
        else tuple()
    )
    return NullModelInputs(
        open_price_matrix=single_asset_open_prices.reshape(1, -1),
        trade_structure=trade_structure,
        null_model_name=NULL_MODEL_NAME,
        calendar_dates=calendar_dates,
        context_entry_candidate_pools=candidate_pools,
    )


def draw_gap_allocation(
    slack_bars: int,
    slot_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomly distribute slack bars across schedule gaps."""
    if slack_bars < 0:
        raise ValueError("Slack bars must be non-negative.")
    if slot_count <= 0:
        raise ValueError("Slot count must be positive.")

    if slack_bars == 0:
        return np.zeros(slot_count, dtype=int)

    probabilities = np.full(slot_count, 1.0 / slot_count, dtype=float)
    return rng.multinomial(slack_bars, probabilities)


def build_random_trade_schedule(
    durations: np.ndarray,
    max_open_index: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a non-overlapping random trade schedule with the same duration profile."""
    if len(durations) == 0:
        raise ValueError("Cannot build a schedule without at least one duration.")

    shuffled_durations = rng.permutation(durations.astype(int, copy=False))
    mandatory_internal_gaps = len(shuffled_durations) - 1
    slack_bars = int(max_open_index - shuffled_durations.sum() - mandatory_internal_gaps)

    if slack_bars < 0:
        raise ValueError(
            "Trade durations cannot fit inside the available price history "
            "without overlapping."
        )

    extra_gaps = draw_gap_allocation(
        slack_bars=slack_bars,
        slot_count=len(shuffled_durations) + 1,
        rng=rng,
    )

    pre_gap = int(extra_gaps[0])
    internal_gaps = extra_gaps[1:-1].astype(int) + 1

    entry_indices = np.empty(len(shuffled_durations), dtype=int)
    exit_indices = np.empty(len(shuffled_durations), dtype=int)

    current_entry = pre_gap
    for index, duration in enumerate(shuffled_durations):
        current_exit = current_entry + int(duration)
        entry_indices[index] = current_entry
        exit_indices[index] = current_exit

        if index < len(shuffled_durations) - 1:
            current_entry = current_exit + int(internal_gaps[index])

    if exit_indices[-1] > max_open_index:
        raise ValueError("Random schedule construction exceeded the available history.")

    return entry_indices, exit_indices


def draw_uniform_slack_allocations(
    slack_bars: int,
    slot_count: int,
    sample_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample slack allocations uniformly over all feasible stars-and-bars schedules."""
    if slack_bars < 0:
        raise ValueError("Slack bars must be non-negative.")
    if slot_count <= 0:
        raise ValueError("Slot count must be positive.")
    if sample_count <= 0:
        raise ValueError("Sample count must be positive.")

    if slot_count == 1:
        return np.full((sample_count, 1), slack_bars, dtype=np.int64)
    if slack_bars == 0:
        return np.zeros((sample_count, slot_count), dtype=np.int64)

    separator_count = slot_count - 1
    total_positions = slack_bars + slot_count - 1
    random_keys = rng.random((sample_count, total_positions))
    separator_positions = np.argpartition(
        random_keys,
        kth=separator_count - 1,
        axis=1,
    )[:, :separator_count]
    separator_positions = np.sort(separator_positions.astype(np.int64, copy=False), axis=1)

    left_boundary = -np.ones((sample_count, 1), dtype=np.int64)
    right_boundary = np.full((sample_count, 1), total_positions, dtype=np.int64)
    return np.diff(
        np.concatenate([left_boundary, separator_positions, right_boundary], axis=1),
        axis=1,
    ) - 1


def build_legacy_trade_schedule_batch(
    durations: np.ndarray,
    max_open_index: int,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the old shuffled-duration schedules for comparison and validation."""
    number_of_trades = len(durations)
    permutation_keys = rng.random((batch_size, number_of_trades))
    permutation_indices = np.argsort(permutation_keys, axis=1)
    shuffled_durations = durations[permutation_indices]

    mandatory_internal_gaps = number_of_trades - 1
    slack_bars = int(max_open_index - durations.sum() - mandatory_internal_gaps)
    if slack_bars < 0:
        raise ValueError(
            "Trade durations cannot fit inside the available price history "
            "without overlapping."
        )

    if slack_bars == 0:
        extra_gaps = np.zeros((batch_size, number_of_trades + 1), dtype=np.int64)
    else:
        probabilities = np.full(number_of_trades + 1, 1.0 / (number_of_trades + 1))
        extra_gaps = rng.multinomial(
            slack_bars,
            probabilities,
            size=batch_size,
        ).astype(np.int64, copy=False)

    leading_gap = extra_gaps[:, [0]]
    if number_of_trades == 1:
        entry_indices = leading_gap.astype(np.int64, copy=False)
    else:
        internal_gaps = extra_gaps[:, 1:-1] + 1
        step_sizes = shuffled_durations[:, :-1] + internal_gaps
        cumulative_steps = np.cumsum(step_sizes, axis=1, dtype=np.int64)
        entry_indices = np.concatenate(
            [
                leading_gap.astype(np.int64, copy=False),
                leading_gap.astype(np.int64, copy=False) + cumulative_steps,
            ],
            axis=1,
        )

    exit_indices = entry_indices + shuffled_durations.astype(np.int64, copy=False)
    if np.any(exit_indices[:, -1] > max_open_index):
        raise ValueError("Legacy random schedule construction exceeded the available history.")

    return entry_indices, exit_indices, permutation_indices.astype(np.int64, copy=False)


def build_structure_preserving_schedule_batch(
    trade_structure: TradeStructure,
    max_open_index: int,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomize timing while preserving durations and the realized gap-size structure."""
    number_of_trades = len(trade_structure.durations)
    if number_of_trades == 0:
        raise ValueError("Cannot build schedules without at least one realized trade.")

    if trade_structure.external_slack < 0:
        raise ValueError("External slack must be non-negative.")
    if trade_structure.internal_gap_sizes.size > 0 and np.any(trade_structure.internal_gap_sizes < 0):
        raise ValueError("Internal gap sizes must be non-negative.")

    leading_low = min(MIN_LEADING_INDEX, trade_structure.external_slack)
    leading_gap = rng.integers(
        leading_low,
        trade_structure.external_slack + 1,
        size=(batch_size, 1),
        dtype=np.int64,
    )
    if number_of_trades == 1:
        entry_indices = leading_gap
    else:
        permutation_keys = rng.random((batch_size, number_of_trades - 1))
        permutation_indices = np.argsort(permutation_keys, axis=1)
        permuted_internal_gaps = trade_structure.internal_gap_sizes[permutation_indices]
        step_sizes = trade_structure.durations[np.newaxis, :-1] + permuted_internal_gaps
        cumulative_steps = np.cumsum(step_sizes, axis=1, dtype=np.int64)
        entry_indices = np.concatenate([leading_gap, leading_gap + cumulative_steps], axis=1)

    exit_indices = entry_indices + trade_structure.durations[np.newaxis, :]
    if np.any(exit_indices[:, -1] > max_open_index):
        raise ValueError(
            "Structure-preserving schedule construction exceeded the available history."
        )

    return entry_indices, exit_indices


def build_context_preserving_schedule_batch(
    trade_structure: TradeStructure,
    max_open_index: int,
    batch_size: int,
    rng: np.random.Generator,
    candidate_pools: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Randomize timing while keeping trades in comparable entry contexts."""
    number_of_trades = len(trade_structure.durations)
    if number_of_trades == 0:
        raise ValueError("Cannot build schedules without at least one realized trade.")

    tail_minimum_span = calculate_tail_minimum_span(
        trade_structure.durations.astype(np.int64, copy=False),
        trade_structure.transition_gap_floors.astype(np.int64, copy=False),
    )
    entry_indices = np.empty((batch_size, number_of_trades), dtype=np.int64)
    exit_indices = np.empty((batch_size, number_of_trades), dtype=np.int64)

    for batch_row in range(batch_size):
        earliest_entry = 0
        for trade_index in range(number_of_trades):
            latest_entry = int(max_open_index - tail_minimum_span[trade_index])
            if latest_entry < earliest_entry:
                raise ValueError(
                    "Context-preserving schedule construction ran out of feasible space."
                )

            candidate_pool = (
                candidate_pools[trade_index]
                if trade_index < len(candidate_pools)
                else np.array([], dtype=np.int64)
            )
            chosen_entry: int
            if len(candidate_pool) > 0:
                left_index = int(np.searchsorted(candidate_pool, earliest_entry, side="left"))
                right_index = int(np.searchsorted(candidate_pool, latest_entry, side="right"))
                if right_index > left_index:
                    chosen_entry = int(
                        candidate_pool[rng.integers(left_index, right_index)]
                    )
                else:
                    chosen_entry = int(rng.integers(earliest_entry, latest_entry + 1))
            else:
                chosen_entry = int(rng.integers(earliest_entry, latest_entry + 1))

            current_exit = chosen_entry + int(trade_structure.durations[trade_index])
            entry_indices[batch_row, trade_index] = chosen_entry
            exit_indices[batch_row, trade_index] = current_exit
            if trade_index < number_of_trades - 1:
                earliest_entry = int(
                    current_exit + trade_structure.transition_gap_floors[trade_index]
                )

    return entry_indices, exit_indices


def calculate_interval_returns_from_open_prices(
    open_prices: np.ndarray,
    entry_indices: np.ndarray,
    exit_indices: np.ndarray,
) -> np.ndarray:
    """Calculate simple returns for a batch of open-to-open trade intervals."""
    entry_prices = open_prices[entry_indices]
    exit_prices = open_prices[exit_indices]
    return (exit_prices / entry_prices) - 1.0


def calculate_directional_returns_from_open_prices(
    open_price_matrix: np.ndarray,
    asset_indices: np.ndarray,
    entry_indices: np.ndarray,
    exit_indices: np.ndarray,
    direction_signs: np.ndarray,
) -> np.ndarray:
    """Calculate long or short open-to-open returns for one batch of schedules."""
    entry_prices = open_price_matrix[asset_indices, entry_indices]
    exit_prices = open_price_matrix[asset_indices, exit_indices]
    long_returns = (exit_prices / entry_prices) - 1.0
    short_returns = (entry_prices / exit_prices) - 1.0
    return np.where(direction_signs > 0, long_returns, short_returns)


def calculate_directional_net_returns_with_execution_costs(
    *,
    open_price_matrix: np.ndarray,
    asset_indices: np.ndarray,
    entry_indices: np.ndarray,
    exit_indices: np.ndarray,
    direction_signs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply the same style of adverse fills used by the live execution model."""
    entry_reference_prices = open_price_matrix[asset_indices, entry_indices]
    exit_reference_prices = open_price_matrix[asset_indices, exit_indices]
    entry_total_bps = HALF_SPREAD_BPS + rng.uniform(
        MIN_SLIPPAGE_BPS,
        MAX_SLIPPAGE_BPS,
        size=entry_indices.shape,
    )
    exit_total_bps = HALF_SPREAD_BPS + rng.uniform(
        MIN_SLIPPAGE_BPS,
        MAX_SLIPPAGE_BPS,
        size=exit_indices.shape,
    )
    entry_multiplier = 1.0 + (entry_total_bps / 10000.0)
    exit_multiplier = 1.0 + (exit_total_bps / 10000.0)

    long_entry_fill = entry_reference_prices * entry_multiplier
    long_exit_fill = exit_reference_prices / exit_multiplier
    short_entry_fill = entry_reference_prices / entry_multiplier
    short_exit_fill = exit_reference_prices * exit_multiplier

    long_returns = (long_exit_fill / long_entry_fill) - 1.0
    short_returns = (short_entry_fill / short_exit_fill) - 1.0
    position_returns = np.where(direction_signs > 0, long_returns, short_returns)
    return position_returns - EXPECTED_COMMISSION_RATE


def benjamini_hochberg_adjusted_p_values(p_values: pd.Series) -> pd.Series:
    """Return Benjamini-Hochberg adjusted p-values in the original row order."""
    numeric_p_values = pd.to_numeric(p_values, errors="coerce")
    adjusted = pd.Series(np.nan, index=numeric_p_values.index, dtype=float)
    valid_mask = numeric_p_values.notna()
    if not valid_mask.any():
        return adjusted

    ordered = numeric_p_values[valid_mask].sort_values()
    ordered_values = ordered.to_numpy(dtype=float)
    test_count = len(ordered_values)
    scaled = ordered_values * float(test_count) / np.arange(1, test_count + 1, dtype=float)
    adjusted_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted.loc[ordered.index] = adjusted_sorted
    return adjusted


def simulate_random_timing_cumulative_returns(
    open_prices: np.ndarray,
    durations: np.ndarray,
    position_value_fractions: np.ndarray,
    simulation_count: int,
    rng: np.random.Generator,
) -> pd.Series:
    """Legacy timing null: shuffle duration ordering and force one idle bar between trades."""
    if len(durations) == 0:
        raise ValueError("Cannot run simulations without at least one realized trade.")
    if len(position_value_fractions) != len(durations):
        raise ValueError("Each duration must have a matching position_value_fraction.")

    durations = durations.astype(np.int32, copy=False)
    position_value_fractions = position_value_fractions.astype(float, copy=False)
    max_open_index = len(open_prices) - 1
    number_of_trades = len(durations)
    mandatory_internal_gaps = number_of_trades - 1
    slack_bars = int(max_open_index - durations.sum() - mandatory_internal_gaps)

    if slack_bars < 0:
        raise ValueError(
            "Trade durations cannot fit inside the available price history "
            "without overlapping."
        )

    gap_probabilities = np.full(number_of_trades + 1, 1.0 / (number_of_trades + 1))
    simulated_cumulative_returns = np.empty(simulation_count, dtype=float)

    for batch_start in range(0, simulation_count, SIMULATION_BATCH_SIZE):
        batch_size = min(SIMULATION_BATCH_SIZE, simulation_count - batch_start)

        # Build one random permutation of the realized duration profile per simulation.
        permutation_keys = rng.random((batch_size, number_of_trades))
        permutation_indices = np.argsort(permutation_keys, axis=1)
        shuffled_durations = durations[permutation_indices]
        shuffled_position_value_fractions = position_value_fractions[permutation_indices]

        if slack_bars == 0:
            extra_gaps = np.zeros((batch_size, number_of_trades + 1), dtype=np.int32)
        else:
            extra_gaps = rng.multinomial(
                slack_bars,
                gap_probabilities,
                size=batch_size,
            ).astype(np.int32, copy=False)

        leading_gap = extra_gaps[:, [0]]
        if number_of_trades == 1:
            entry_indices = leading_gap.astype(np.int64, copy=False)
        else:
            internal_gaps = extra_gaps[:, 1:-1] + 1
            step_sizes = shuffled_durations[:, :-1] + internal_gaps
            cumulative_steps = np.cumsum(step_sizes, axis=1, dtype=np.int64)
            entry_indices = np.concatenate(
                [
                    leading_gap.astype(np.int64, copy=False),
                    leading_gap.astype(np.int64, copy=False) + cumulative_steps,
                ],
                axis=1,
            )

        exit_indices = entry_indices + shuffled_durations.astype(np.int64, copy=False)
        if np.any(exit_indices[:, -1] > max_open_index):
            raise ValueError(
                "Random schedule construction exceeded the available history."
            )

        if SIMULATE_EXECUTION_COSTS:
            simulated_position_returns = calculate_directional_net_returns_with_execution_costs(
                open_price_matrix=open_prices.reshape(1, -1),
                asset_indices=np.zeros_like(entry_indices, dtype=np.int64),
                entry_indices=entry_indices,
                exit_indices=exit_indices,
                direction_signs=np.ones_like(entry_indices, dtype=np.int8),
                rng=rng,
            )
        else:
            entry_prices = open_prices[entry_indices]
            exit_prices = open_prices[exit_indices]
            simulated_position_returns = (exit_prices / entry_prices) - 1.0 - TRANSACTION_COST
        simulated_adjusted_returns = (
            shuffled_position_value_fractions * simulated_position_returns
        )

        if np.any(simulated_adjusted_returns <= -1):
            raise ValueError(
                "A simulated trade produced an adjusted return below -100%, "
                "which cannot be converted to log-return space."
            )

        simulated_log_returns = np.log1p(simulated_adjusted_returns)
        simulated_cumulative_returns[batch_start : batch_start + batch_size] = np.expm1(
            simulated_log_returns.sum(axis=1)
        )

    return pd.Series(
        simulated_cumulative_returns,
        name="simulated_cumulative_return",
    )


def simulate_random_peer_rotation_cumulative_returns(
    open_price_matrix: np.ndarray,
    durations: np.ndarray,
    position_value_fractions: np.ndarray,
    simulation_count: int,
    rng: np.random.Generator,
) -> pd.Series:
    """Legacy relative-strength null kept for direct fairness comparisons."""
    if open_price_matrix.ndim != 2:
        raise ValueError("open_price_matrix must be a 2D array of peer open prices.")

    asset_count, calendar_length = open_price_matrix.shape
    if asset_count < 2 or calendar_length < 2:
        raise ValueError("Relative-strength null simulation needs at least two assets and two dates.")

    durations = durations.astype(np.int32, copy=False)
    position_value_fractions = position_value_fractions.astype(float, copy=False)
    number_of_trades = len(durations)
    if number_of_trades == 0:
        raise ValueError("Cannot run simulations without at least one realized trade.")
    if len(position_value_fractions) != number_of_trades:
        raise ValueError("Each duration must have a matching position_value_fraction.")

    max_open_index = calendar_length - 1
    mandatory_internal_gaps = number_of_trades - 1
    slack_bars = int(max_open_index - durations.sum() - mandatory_internal_gaps)
    if slack_bars < 0:
        raise ValueError(
            "Trade durations cannot fit inside the shared peer calendar without overlapping."
        )

    gap_probabilities = np.full(number_of_trades + 1, 1.0 / (number_of_trades + 1))
    simulated_cumulative_returns = np.empty(simulation_count, dtype=float)

    for batch_start in range(0, simulation_count, SIMULATION_BATCH_SIZE):
        batch_size = min(SIMULATION_BATCH_SIZE, simulation_count - batch_start)

        permutation_keys = rng.random((batch_size, number_of_trades))
        permutation_indices = np.argsort(permutation_keys, axis=1)
        shuffled_durations = durations[permutation_indices]
        shuffled_position_value_fractions = position_value_fractions[permutation_indices]

        if slack_bars == 0:
            extra_gaps = np.zeros((batch_size, number_of_trades + 1), dtype=np.int32)
        else:
            extra_gaps = rng.multinomial(
                slack_bars,
                gap_probabilities,
                size=batch_size,
            ).astype(np.int32, copy=False)

        leading_gap = extra_gaps[:, [0]]
        if number_of_trades == 1:
            entry_indices = leading_gap.astype(np.int64, copy=False)
        else:
            internal_gaps = extra_gaps[:, 1:-1] + 1
            step_sizes = shuffled_durations[:, :-1] + internal_gaps
            cumulative_steps = np.cumsum(step_sizes, axis=1, dtype=np.int64)
            entry_indices = np.concatenate(
                [
                    leading_gap.astype(np.int64, copy=False),
                    leading_gap.astype(np.int64, copy=False) + cumulative_steps,
                ],
                axis=1,
            )

        exit_indices = entry_indices + shuffled_durations.astype(np.int64, copy=False)
        if np.any(exit_indices[:, -1] > max_open_index):
            raise ValueError(
                "Random peer-rotation schedule construction exceeded the available calendar."
            )

        asset_indices = rng.integers(
            low=0,
            high=asset_count,
            size=(batch_size, number_of_trades),
            dtype=np.int64,
        )
        if SIMULATE_EXECUTION_COSTS:
            simulated_position_returns = calculate_directional_net_returns_with_execution_costs(
                open_price_matrix=open_price_matrix,
                asset_indices=asset_indices,
                entry_indices=entry_indices,
                exit_indices=exit_indices,
                direction_signs=np.ones_like(entry_indices, dtype=np.int8),
                rng=rng,
            )
        else:
            entry_prices = open_price_matrix[asset_indices, entry_indices]
            exit_prices = open_price_matrix[asset_indices, exit_indices]
            simulated_position_returns = (exit_prices / entry_prices) - 1.0 - TRANSACTION_COST
        simulated_adjusted_returns = (
            shuffled_position_value_fractions * simulated_position_returns
        )

        if np.any(simulated_adjusted_returns <= -1):
            raise ValueError(
                "A simulated peer-rotation trade produced an adjusted return below -100%."
            )

        simulated_log_returns = np.log1p(simulated_adjusted_returns)
        simulated_cumulative_returns[batch_start : batch_start + batch_size] = np.expm1(
            simulated_log_returns.sum(axis=1)
        )

    return pd.Series(
        simulated_cumulative_returns,
        name="simulated_cumulative_return",
    )


def simulate_structure_preserving_cumulative_returns(
    null_model_inputs: NullModelInputs,
    simulation_count: int,
    rng: np.random.Generator,
) -> pd.Series:
    """Simulate the new timing-only null that preserves realized trade structure exactly."""
    trade_structure = null_model_inputs.trade_structure
    number_of_trades = len(trade_structure.durations)
    if number_of_trades == 0:
        raise ValueError("Cannot run simulations without at least one realized trade.")

    open_price_matrix = null_model_inputs.open_price_matrix
    if open_price_matrix.ndim != 2:
        raise ValueError("open_price_matrix must be a 2D array.")

    max_open_index = open_price_matrix.shape[1] - 1
    simulated_cumulative_returns = np.empty(simulation_count, dtype=float)

    for batch_start in range(0, simulation_count, SIMULATION_BATCH_SIZE):
        batch_size = min(SIMULATION_BATCH_SIZE, simulation_count - batch_start)
        if CONTEXT_MATCHING_ENABLED and null_model_inputs.context_entry_candidate_pools:
            entry_indices, exit_indices = build_context_preserving_schedule_batch(
                trade_structure=trade_structure,
                max_open_index=max_open_index,
                batch_size=batch_size,
                rng=rng,
                candidate_pools=null_model_inputs.context_entry_candidate_pools,
            )
        else:
            entry_indices, exit_indices = build_structure_preserving_schedule_batch(
                trade_structure=trade_structure,
                max_open_index=max_open_index,
                batch_size=batch_size,
                rng=rng,
            )
        asset_indices = np.broadcast_to(
            trade_structure.asset_indices[np.newaxis, :],
            entry_indices.shape,
        )
        direction_signs = np.broadcast_to(
            trade_structure.direction_signs[np.newaxis, :],
            entry_indices.shape,
        )
        if SIMULATE_EXECUTION_COSTS:
            simulated_position_returns = calculate_directional_net_returns_with_execution_costs(
                open_price_matrix=open_price_matrix,
                asset_indices=asset_indices,
                entry_indices=entry_indices,
                exit_indices=exit_indices,
                direction_signs=direction_signs,
                rng=rng,
            )
        else:
            gross_position_returns = calculate_directional_returns_from_open_prices(
                open_price_matrix=open_price_matrix,
                asset_indices=asset_indices,
                entry_indices=entry_indices,
                exit_indices=exit_indices,
                direction_signs=direction_signs,
            )
            simulated_position_returns = gross_position_returns - TRANSACTION_COST
        simulated_adjusted_returns = (
            trade_structure.position_value_fractions[np.newaxis, :]
            * simulated_position_returns
        )

        if np.any(simulated_adjusted_returns <= -1):
            raise ValueError(
                "A structure-preserving simulated trade produced an adjusted return "
                "below -100%, which cannot be converted to log-return space."
            )

        simulated_log_returns = np.log1p(simulated_adjusted_returns)
        simulated_cumulative_returns[batch_start : batch_start + batch_size] = np.expm1(
            simulated_log_returns.sum(axis=1)
        )

    return pd.Series(
        simulated_cumulative_returns,
        name="simulated_cumulative_return",
    )


def simulate_legacy_agent_null_cumulative_returns(
    *,
    agent_name: str,
    null_model_inputs: NullModelInputs,
    simulation_count: int,
    rng: np.random.Generator,
) -> tuple[pd.Series, str]:
    """Run the pre-upgrade null model for fairness comparisons and validation."""
    trade_structure = null_model_inputs.trade_structure
    if agent_name == "momentum_relative_strength":
        return (
            simulate_random_peer_rotation_cumulative_returns(
                open_price_matrix=null_model_inputs.open_price_matrix,
                durations=trade_structure.durations,
                position_value_fractions=trade_structure.position_value_fractions,
                simulation_count=simulation_count,
                rng=rng,
            ),
            LEGACY_RELATIVE_STRENGTH_NULL_MODEL_NAME,
        )

    return (
        simulate_random_timing_cumulative_returns(
            open_prices=null_model_inputs.open_price_matrix[0],
            durations=trade_structure.durations,
            position_value_fractions=trade_structure.position_value_fractions,
            simulation_count=simulation_count,
            rng=rng,
        ),
        LEGACY_NULL_MODEL_NAME,
    )


def simulate_agent_null_cumulative_returns(
    *,
    agent_name: str,
    current_ticker: str,
    trade_df: pd.DataFrame,
    market_df: pd.DataFrame,
    input_path: Path | str,
    simulation_count: int,
    rng: np.random.Generator,
) -> tuple[pd.Series, str]:
    """Dispatch the structure-preserving timing null for one strategy."""
    null_model_inputs = prepare_agent_null_model_inputs(
        agent_name=agent_name,
        current_ticker=current_ticker,
        trade_df=trade_df,
        market_df=market_df,
        input_path=input_path,
    )
    return (
        simulate_structure_preserving_cumulative_returns(
            null_model_inputs=null_model_inputs,
            simulation_count=simulation_count,
            rng=rng,
        ),
        null_model_inputs.null_model_name,
    )


def calculate_p_value(simulated_returns: np.ndarray, actual_cumulative_return: float) -> float:
    """Calculate the one-sided p-value against the simulated baseline."""
    simulation_count = len(simulated_returns)
    if simulation_count <= 0:
        return 1.0
    exceedance_count = int(np.sum(simulated_returns >= actual_cumulative_return))
    return float((exceedance_count + 1.0) / (simulation_count + 1.0))


def calculate_actual_percentile(
    simulated_returns: np.ndarray,
    actual_cumulative_return: float,
) -> float:
    """Calculate a smoothed percentile rank inside the simulated null."""
    simulation_count = len(simulated_returns)
    if simulation_count <= 0:
        return 100.0
    less_equal_count = int(np.sum(simulated_returns <= actual_cumulative_return))
    return float(((less_equal_count + 1.0) / (simulation_count + 1.0)) * 100.0)


def interpret_p_value(p_value: float) -> str:
    """Translate a p-value into a simple evidence label."""
    if p_value < 0.05:
        return "strong evidence"
    if p_value < 0.10:
        return "weak evidence"
    return "no evidence"


def save_simulation_results(
    output_path: Path,
    simulated_returns: pd.Series,
    *,
    current_ticker: str,
    agent_name: str,
    dependencies: list[Path] | tuple[Path, ...] | None = None,
) -> None:
    """Save the full simulation results for one strategy."""
    results_df = pd.DataFrame(
        {
            "simulation_id": range(1, len(simulated_returns) + 1),
            "simulated_cumulative_return": simulated_returns.to_numpy(dtype=float),
        }
    )
    write_dataframe_artifact(
        results_df,
        output_path,
        producer="monte_carlo.save_simulation_results",
        current_ticker=current_ticker,
        dependencies=dependencies,
        research_grade=NUMBER_OF_SIMULATIONS >= MIN_RESEARCH_GRADE_SIMULATIONS,
        canonical_policy="always",
        parameters={
            "agent_name": agent_name,
            "simulation_count": NUMBER_OF_SIMULATIONS,
            "null_model_type": "timing_null_results",
        },
    )


def build_agent_summary(
    agent_name: str,
    actual_cumulative_return: float,
    simulated_returns: pd.Series,
    number_of_trades: int,
    null_model_name: str,
) -> dict[str, float | int | str | bool]:
    """Build one summary row for the Monte Carlo output table."""
    simulated_array = simulated_returns.to_numpy(dtype=float)
    p_value = calculate_p_value(simulated_array, actual_cumulative_return)

    return {
        "agent": agent_name,
        "actual_cumulative_return": actual_cumulative_return,
        "median_simulated_return": float(np.median(simulated_array)),
        "mean_simulated_return": float(np.mean(simulated_array)),
        "std_simulated_return": float(np.std(simulated_array, ddof=0)),
        "actual_percentile": calculate_actual_percentile(simulated_array, actual_cumulative_return),
        "p_value": p_value,
        "p_value_prominence": calculate_p_value_prominence(p_value),
        "p_value_interpretation": interpret_p_value(p_value),
        "lower_5pct": float(np.percentile(simulated_array, 5)),
        "upper_95pct": float(np.percentile(simulated_array, 95)),
        "number_of_trades": number_of_trades,
        "transaction_cost": TRANSACTION_COST,
        "execution_cost_model": (
            "stochastic_spread_slippage_plus_expected_commission"
            if SIMULATE_EXECUTION_COSTS
            else "fixed_transaction_cost"
        ),
        "simulation_count": NUMBER_OF_SIMULATIONS,
        "research_grade": NUMBER_OF_SIMULATIONS >= MIN_RESEARCH_GRADE_SIMULATIONS,
        "reproducible": REPRODUCIBLE,
        "seed_used": SEED if REPRODUCIBLE else "",
        "null_model": null_model_name,
    }


def build_no_trade_summary(agent_name: str, null_model_name: str) -> dict[str, float | int | str | bool]:
    """Build a placeholder summary row for a strategy that produced no trades."""
    return {
        "agent": agent_name,
        "actual_cumulative_return": 0.0,
        "median_simulated_return": 0.0,
        "mean_simulated_return": 0.0,
        "std_simulated_return": 0.0,
        "actual_percentile": 100.0,
        "p_value": 1.0,
        "p_value_prominence": calculate_p_value_prominence(1.0),
        "p_value_interpretation": "no trades",
        "lower_5pct": 0.0,
        "upper_95pct": 0.0,
        "number_of_trades": 0,
        "transaction_cost": TRANSACTION_COST,
        "execution_cost_model": (
            "stochastic_spread_slippage_plus_expected_commission"
            if SIMULATE_EXECUTION_COSTS
            else "fixed_transaction_cost"
        ),
        "simulation_count": NUMBER_OF_SIMULATIONS,
        "research_grade": NUMBER_OF_SIMULATIONS >= MIN_RESEARCH_GRADE_SIMULATIONS,
        "reproducible": REPRODUCIBLE,
        "seed_used": SEED if REPRODUCIBLE else "",
        "null_model": null_model_name,
    }


def print_agent_summary(summary_row: dict[str, float | int | str | bool]) -> None:
    """Print a clean terminal summary for one strategy."""
    print(f"\nAgent: {summary_row['agent']}")
    if int(summary_row["number_of_trades"]) == 0:
        print("No completed trades were generated for this strategy.")
        print("Stored a degenerate zero-return baseline for pipeline compatibility.")
        print(f"Simulation count: {summary_row['simulation_count']}")
        print(f"Null model: {summary_row['null_model']}")
        return

    print(f"Actual cumulative return: {summary_row['actual_cumulative_return']:.6f}")
    print(f"Median simulated return: {summary_row['median_simulated_return']:.6f}")
    print(f"Mean simulated return: {summary_row['mean_simulated_return']:.6f}")
    print(f"Std simulated return: {summary_row['std_simulated_return']:.6f}")
    print(f"Actual percentile: {summary_row['actual_percentile']:.2f}")
    print(f"p-value: {summary_row['p_value']:.6f}")
    print(f"p-value prominence: {summary_row['p_value_prominence']:.6f}")
    print(f"Evidence label: {summary_row['p_value_interpretation']}")
    print(f"5th percentile: {summary_row['lower_5pct']:.6f}")
    print(f"95th percentile: {summary_row['upper_95pct']:.6f}")
    print(f"Number of trades: {summary_row['number_of_trades']}")
    print(f"Transaction cost: {summary_row['transaction_cost']:.6f}")
    print(f"Execution cost model: {summary_row['execution_cost_model']}")
    print(f"Simulation count: {summary_row['simulation_count']}")
    print(f"Research-grade depth: {summary_row['research_grade']}")
    print(f"Reproducible mode: {summary_row['reproducible']}")
    print(f"Seed used: {summary_row['seed_used']}")
    print(f"Null model: {summary_row['null_model']}")


def main() -> None:
    """Run the Monte Carlo engine for all active strategies."""
    project_root = Path(__file__).resolve().parents[1]
    data_clean_dir = resolve_data_clean_dir(project_root)
    market_path = data_clean_dir / f"{ticker}_regimes.csv"
    market_df = load_market_data(market_path)
    seed_sequence = np.random.SeedSequence(SEED)
    child_sequences = seed_sequence.spawn(len(AGENT_ORDER))

    summary_rows = []

    for agent_name, child_sequence in zip(AGENT_ORDER, child_sequences):
        input_path = ensure_trade_file_exists(ticker, agent_name)
        results_output_path = data_clean_dir / f"{ticker}_{agent_name}_monte_carlo_results.csv"
        rng = build_random_generator(
            reproducible=REPRODUCIBLE,
            seed=int(child_sequence.generate_state(1, dtype=np.uint64)[0]),
        )
        null_model_name = (
            RELATIVE_STRENGTH_NULL_MODEL_NAME
            if agent_name == "momentum_relative_strength"
            else NULL_MODEL_NAME
        )

        trade_df = load_trade_data(input_path, allow_empty=True)
        if trade_df.empty:
            simulated_returns = build_no_trade_simulated_returns(NUMBER_OF_SIMULATIONS)
            save_simulation_results(
                results_output_path,
                simulated_returns,
                current_ticker=ticker,
                agent_name=agent_name,
                dependencies=[input_path, market_path],
            )
            summary_row = build_no_trade_summary(agent_name, null_model_name)
            summary_rows.append(summary_row)
            print_agent_summary(summary_row)
            continue

        raw_returns = trade_df["return"].to_numpy(dtype=float)
        adjusted_returns = adjust_trade_returns(
            raw_returns=raw_returns,
            transaction_cost=TRANSACTION_COST,
            input_path=input_path,
        )
        log_returns = convert_to_log_returns(adjusted_returns)

        actual_cumulative_return = calculate_cumulative_return_from_log_returns(log_returns)
        simulated_returns, null_model_name = simulate_agent_null_cumulative_returns(
            agent_name=agent_name,
            current_ticker=ticker,
            trade_df=trade_df,
            market_df=market_df,
            input_path=input_path,
            simulation_count=NUMBER_OF_SIMULATIONS,
            rng=rng,
        )

        save_simulation_results(
            results_output_path,
            simulated_returns,
            current_ticker=ticker,
            agent_name=agent_name,
            dependencies=[input_path, market_path],
        )

        summary_row = build_agent_summary(
            agent_name=agent_name,
            actual_cumulative_return=actual_cumulative_return,
            simulated_returns=simulated_returns,
            number_of_trades=len(trade_df),
            null_model_name=null_model_name,
        )
        summary_rows.append(summary_row)
        print_agent_summary(summary_row)

    summary_df = pd.DataFrame(
        summary_rows,
        columns=[
            "agent",
            "actual_cumulative_return",
            "median_simulated_return",
            "mean_simulated_return",
            "std_simulated_return",
            "actual_percentile",
            "p_value",
            "p_value_prominence",
            "p_value_interpretation",
            "lower_5pct",
            "upper_95pct",
            "number_of_trades",
            "transaction_cost",
            "execution_cost_model",
            "simulation_count",
            "research_grade",
            "reproducible",
            "seed_used",
            "null_model",
        ],
    )
    summary_df["bh_adjusted_p_value"] = benjamini_hochberg_adjusted_p_values(
        summary_df["p_value"]
    )

    summary_output_path = data_clean_dir / f"{ticker}_monte_carlo_summary.csv"
    write_dataframe_artifact(
        summary_df,
        summary_output_path,
        producer="monte_carlo.main",
        current_ticker=ticker,
        dependencies=[
            market_path,
            *[data_clean_dir / f"{ticker}_{agent_name}_monte_carlo_results.csv" for agent_name in AGENT_ORDER],
        ],
        research_grade=NUMBER_OF_SIMULATIONS >= MIN_RESEARCH_GRADE_SIMULATIONS,
        canonical_policy="always",
        parameters={
            "simulation_count": NUMBER_OF_SIMULATIONS,
            "context_matching_enabled": CONTEXT_MATCHING_ENABLED,
            "context_bucket_count": CONTEXT_BUCKET_COUNT,
            "simulate_execution_costs": SIMULATE_EXECUTION_COSTS,
        },
    )

    print("\nMonte Carlo summary table:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
