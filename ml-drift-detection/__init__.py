"""
ml_drift_detection package for monitoring ML drift and metrics.

Main public API
---------------
cli_main            -- CLI entrypoint (launches the Streamlit dashboard)
dashboard_main      -- Programmatic dashboard entrypoint
get_plotly_dist     -- KDE distribution figure for numeric features
get_plotly_boxplot  -- Side-by-side box-plot figure for numeric drift
get_plotly_barplot  -- Categorical bar-plot figure for proportion drift
"""

from .cli import main as cli_main
from .app.dashboard import main as dashboard_main

# Plot helpers
from .utils.numeric_drift import (
    get_plotly_dist,
    get_plotly_boxplot,
)
from .utils.categorical_drift import get_plotly_barplot

__all__: list[str] = [
    # Dashboard & CLI
    "cli_main",
    "dashboard_main",
    # Plotting helpers
    "get_plotly_dist",
    "get_plotly_boxplot",
    "get_plotly_barplot",
]
