# ml_drift_detection/cli.py
"""
Command-line interface for the *ml_drift_detection* package.

Typical use (PowerShell example, split for readability):

streamlit run "path\\ml_drift_detection\\cli.py" --
  --prod-data="prod_df.csv"
  --new-data="new_df.csv"
  --numeric-cols="age,income"
  --categorical-cols="gender,plan"
  --target-variable="churn_flag"
  --target-type="categorical"
  --prod-metrics="r2=0.82,mae=127.6"
  --new-metrics="r2=0.74,mae=154.2"
  --background-color="#0E1117"
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# ────────────────────────────────────────────────────────────────
# Robust import so the CLI works whether it’s executed *inside* or
# *outside* the package context.
# ────────────────────────────────────────────────────────────────
from app.dashboard import main as dashboard_main  # type: ignore


# ----------------------------------------------------------------
# Logging helpers
# ----------------------------------------------------------------
def _configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


log = logging.getLogger(__name__)


# ----------------------------------------------------------------
# Parsing helpers
# ----------------------------------------------------------------
def _load_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(
        f"Unsupported file type for '{path.name}'. "
        "Only .csv, .xlsx, .xls are accepted."
    )


def _parse_metrics(raw: str) -> Dict[str, float]:
    """
    Parse 'r2=0.82,mae=127.6' into {'r2': 0.82, 'mae': 127.6}.
    """
    metrics: Dict[str, float] = {}
    if not raw:
        return metrics

    for pair in raw.split(","):
        if "=" not in pair:
            raise ValueError(f"Invalid metric string '{pair}' (use key=value).")

        key, value = (x.strip() for x in pair.split("=", 1))
        try:
            metrics[key] = float(value)
        except ValueError as exc:
            raise ValueError(f"Metric '{key}' value '{value}' is not numeric.") from exc
    return metrics


def _parse_cols(raw: str) -> List[str]:
    return [c.strip() for c in raw.split(",") if c.strip()]


def _normalise_bg(bg: str) -> str:
    white = {"white", "#fff", "#ffffff"}
    if bg.lower() in white:
        return "white"
    if bg.lower() == "#0e1117":
        return "#0E1117"

    raise argparse.ArgumentTypeError(
        "background-color must be 'white' or '#0E1117'."
    )


def _parse_thresholds(raw: str) -> List[dict] | None:
    """
    Accept JSON or Python-literal list of dicts, or an empty string.

    Examples
    --------
    --metric-one-threshold-steps="[]"
    --metric-one-threshold-steps="[{'range':[0,1],'color':'green'}]"
    --metric-one-threshold-steps='[{"range":[0,1],"color":"green"}]'
    """
    if raw is None or raw == "":
        return None
    # Allow "[]" (still truthy) to mean “empty list”
    if raw.strip() in {"[]", "[ ]"}:
        return []

    text = raw.replace('\\"', '"')  # unescape if shell added quotes
    try:
        steps = json.loads(text)
    except json.JSONDecodeError:
        steps = ast.literal_eval(text)

    if not isinstance(steps, list) or not all(isinstance(i, dict) for i in steps):
        raise argparse.ArgumentTypeError(
            "Threshold steps must be a list of dicts like "
            '[{"range":[-1,0],"color":"red"}]'
        )
    return steps


# ----------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------
def main() -> None:  # noqa: C901 (argparse generates long function)
    parser = argparse.ArgumentParser(
        prog="ml-drift-detection",
        description="Launch the Streamlit dashboard for ML drift monitoring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required positional-ish arguments
    parser.add_argument("--prod-data", required=True, help="Baseline CSV/XLSX file")
    parser.add_argument("--new-data", required=True, help="New batch CSV/XLSX file")
    parser.add_argument("--numeric-cols", required=True, help="Comma-separated list")
    parser.add_argument("--categorical-cols", required=True, help="Comma-separated list")
    parser.add_argument("--target-variable", required=True, help="Target column")
    parser.add_argument(
        "--target-type",
        required=True,
        choices={"categorical", "continuous"},
        help="Type of the target variable",
    )
    parser.add_argument("--prod-metrics", required=True)
    parser.add_argument("--new-metrics", required=True)

    # Optional UI customisation
    parser.add_argument(
        "--background-color",
        default="white",
        type=_normalise_bg,
        help="Dashboard background ('white' or '#0E1117')",
    )

    # Per-metric threshold overrides
    parser.add_argument("--metric-one-threshold-steps", type=_parse_thresholds)
    parser.add_argument("--metric-two-threshold-steps", type=_parse_thresholds)
    parser.add_argument("--metric-three-threshold-steps", type=_parse_thresholds)
    parser.add_argument("--metric-four-threshold-steps", type=_parse_thresholds)

    # Diagnostics
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices={"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"},
        help="Logger verbosity",
    )

    args = parser.parse_args()
    _configure_logging(args.log_level)

    # ----------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------
    try:
        prod_df = _load_dataframe(args.prod_data)
        new_df = _load_dataframe(args.new_data)
        log.info("Loaded dataframes: prod=%s rows, new=%s rows",
                 len(prod_df), len(new_df))
    except Exception as exc:
        log.error("Failed loading data: %s", exc)
        sys.exit(1)

    # ----------------------------------------------------------------
    # Parse lists / metrics
    # ----------------------------------------------------------------
    try:
        numeric_cols = _parse_cols(args.numeric_cols)
        categorical_cols = _parse_cols(args.categorical_cols)
        prod_metrics = _parse_metrics(args.prod_metrics)
        new_metrics = _parse_metrics(args.new_metrics)
    except ValueError as exc:
        log.error("Input parsing error: %s", exc)
        sys.exit(1)

    # ----------------------------------------------------------------
    # Fire up Streamlit dashboard (runs inside same Python process)
    # ----------------------------------------------------------------
    try:
        dashboard_main(
            prod_df=prod_df,
            new_df=new_df,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            target_variable=args.target_variable,
            target_type=args.target_type,
            prod_metrics=prod_metrics,
            new_metrics=new_metrics,
            background_color=args.background_color,
            metric_one_threshold_steps=args.metric_one_threshold_steps,
            metric_two_threshold_steps=args.metric_two_threshold_steps,
            metric_three_threshold_steps=args.metric_three_threshold_steps,
            metric_four_threshold_steps=args.metric_four_threshold_steps,
        )
    except Exception as exc:
        log.exception("Dashboard crashed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
