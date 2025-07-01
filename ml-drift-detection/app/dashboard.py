# ml_drift_detection/app/dashboard.py
from __future__ import annotations

import streamlit as st
import pandas as pd
from typing import Dict, List, Tuple, Optional

from utils.numeric_drift import (
    check_numeric_stats,
    check_numeric_correlation,
    get_plotly_boxplot,
    get_plotly_dist,
)
from utils.categorical_drift import (
    check_categorical_stats,
    check_cat_contingency,
    get_plotly_barplot,
)
from utils.plotting import (
    drift_gauge_ban,
    get_plotly_upset,
)

# ────────────────────────────────────────────────────────────────
# Default colour-coded thresholds for the drift gauge
# (tweak to your own metric semantics if needed)
# ────────────────────────────────────────────────────────────────
THRESHOLD_STEPS: List[Dict] = [
    {"range": [-1.00, -0.10], "color": "firebrick"},
    {"range": [-0.10, -0.05], "color": "orange"},
    {"range": [-0.05,  0.05], "color": "green"},
    {"range": [ 0.05,  1.00], "color": "#2ca02c"},
]

# ────────────────────────────────────────────────────────────────
# Helper utilities
# ────────────────────────────────────────────────────────────────
def _pretty_metric_name(name: str) -> str:
    """Human-friendly capitalisation for metric titles."""
    return name.replace("_", " ").title()


def _get_synced_metrics(
    prod: Dict[str, float],
    new: Dict[str, float],
    limit: int = 4,
) -> List[Tuple[str, Optional[float], Optional[float]]]:
    """
    Align prod/new metric dicts **by key** (not by position).
    Missing metrics are padded with Nones so the UI can render
    a neutral placeholder instead of an incorrect comparison.
    """
    common = [k for k in prod if k in new][:limit]
    rows: List[Tuple[str, Optional[float], Optional[float]]] = [
        (k, prod[k], new[k]) for k in common
    ]

    # Pad up to <limit> metrics
    while len(rows) < limit:
        rows.append(("—", None, None))
    return rows


def _paginated_table(
    df: pd.DataFrame,
    rows_per_page: int = 5,
    key: str = "",
) -> None:
    """Render a paginated DataFrame or a friendly ‘no data’ message."""
    if df.empty:
        st.info("No rows to display.")
        return

    total_rows = len(df)
    total_pages = (total_rows - 1) // rows_per_page + 1

    page = st.selectbox(
        "Page",
        options=list(range(1, total_pages + 1)),
        key=f"page_{key}",
        format_func=str,
    )

    start, end = (page - 1) * rows_per_page, page * rows_per_page
    subset = df.iloc[start:end].copy()
    subset.index = range(start + 1, start + 1 + len(subset))

    # Pad short last page to keep table height constant
    if len(subset) < rows_per_page:
        blanks = pd.DataFrame(
            [[""] * len(subset.columns)]
            * (rows_per_page - len(subset)),
            columns=subset.columns,
            index=range(subset.index[-1] + 1, start + rows_per_page + 1),
        )
        subset = pd.concat([subset, blanks])

    st.write(
        f"Rows {start + 1}-{min(end, total_rows)} "
        f"of {total_rows} (page {page}/{total_pages})"
    )
    st.dataframe(subset, use_container_width=True)


# ────────────────────────────────────────────────────────────────
# Main dashboard entry point
# ────────────────────────────────────────────────────────────────
def main(
    prod_df: pd.DataFrame,
    new_df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    target_variable: str,
    target_type: str,
    prod_metrics: Dict[str, float],
    new_metrics: Dict[str, float],
    *,
    background_color: str = "#0E1117",
    metric_one_threshold_steps: Optional[List[Dict]] = None,
    metric_two_threshold_steps: Optional[List[Dict]] = None,
    metric_three_threshold_steps: Optional[List[Dict]] = None,
    metric_four_threshold_steps: Optional[List[Dict]] = None,
) -> None:
    """Launch the Streamlit ML-drift dashboard."""

    # Decide on text/plot theme based on background
    dark_bg = background_color.lower() not in ("white", "#fff", "#ffffff")
    metric_color = "white" if dark_bg else "black"
    plotly_mode = "white" if dark_bg else "black"

    st.set_page_config(layout="wide")
    st.markdown(
        f"<style>.stApp {{ background-color: {background_color}; }}</style>",
        unsafe_allow_html=True,
    )

    # Banner
    st.markdown(
        """
        <div style='background-color:#f7f7f8;
                    padding:25px 0 10px 35px;
                    border-radius:0 0 25px 25px;
                    margin-bottom:30px;'>
            <h1 style='color:#243354;font-weight:600;'>
                ML Drift Detection Dashboard
            </h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ────────────────────────────────────────────────────────────
    # 1 ▸ Model-level metrics
    # ────────────────────────────────────────────────────────────
    st.markdown("### Model Performance Metrics")

    synced_metrics = _get_synced_metrics(prod_metrics, new_metrics)
    cols = st.columns(4)
    step_lists = [
        metric_one_threshold_steps or THRESHOLD_STEPS,
        metric_two_threshold_steps or THRESHOLD_STEPS,
        metric_three_threshold_steps or THRESHOLD_STEPS,
        metric_four_threshold_steps or THRESHOLD_STEPS,
    ]

    for (name, old_val, new_val), steps, col in zip(
        synced_metrics, step_lists, cols
    ):
        with col:
            if old_val is None or new_val is None:
                st.markdown("**N/A**")
                continue  # skip padded placeholders

            st.plotly_chart(
                drift_gauge_ban(
                    metric_name=_pretty_metric_name(name),
                    old=old_val,
                    new=new_val,
                    steps=steps,
                    background_color=background_color,
                    metric_color=metric_color,
                ),
                use_container_width=True,
            )

    st.markdown("---")

    # ────────────────────────────────────────────────────────────
    # 2 ▸ Feature drift
    # ────────────────────────────────────────────────────────────
    st.markdown("### Feature Drift Analysis")
    left, right = st.columns([1.25, 1.25])

    with left:
        st.markdown("#### Numeric Features")
        num_summary = check_numeric_stats(
            prod_df=prod_df,
            new_df=new_df,
            num_cols=numeric_cols,
            p_threshold=0.05,
        )
        _paginated_table(num_summary, key="num")
        st.plotly_chart(
            get_plotly_dist(
                prod_df=prod_df,
                new_df=new_df,
                numeric_cols=numeric_cols,
                mode=plotly_mode,
            ),
            use_container_width=True,
        )

    with right:
        st.markdown("#### Categorical Features")
        cat_summary = check_categorical_stats(
            prod_df=prod_df,
            new_df=new_df,
            cat_cols=categorical_cols,
            p_threshold=0.05,
        )
        _paginated_table(cat_summary, key="cat")
        # st.plotly_chart(
        #     cat_summary.get_plotly_barplot(mode=plotly_mode),
        #     use_container_width=True,
        # )
        st.plotly_chart(
            get_plotly_barplot(          # ← call the helper directly
                prod_df=prod_df,
                new_df=new_df,
                cat_cols=categorical_cols,
                mode=plotly_mode,
            ),
            use_container_width=True,
        )

    st.markdown("---")

    # ────────────────────────────────────────────────────────────
    # 3 ▸ Label drift
    # ────────────────────────────────────────────────────────────
    st.markdown("### Label Drift Analysis")
    ld_left, ld_right = st.columns([1.25, 1.25])

    with ld_left:
        st.markdown("#### Numeric Features vs Target")
        st.plotly_chart(
            get_plotly_boxplot(      # ← call helper, not num_summary.method
                prod_df=prod_df,
                new_df=new_df,
                num_summary=num_summary,   # summary returned earlier
                mode=plotly_mode,
            ),
            use_container_width=True,
        )

    with ld_right:
        st.markdown("#### Categorical Features vs Target")
        st.plotly_chart(
            get_plotly_upset(
                prod_df=prod_df,
                new_df=new_df,
                categorical_cols=categorical_cols,
                mode=plotly_mode,
            ),
            use_container_width=True,
        )

    st.markdown("---")

    # ────────────────────────────────────────────────────────────
    # 4 ▸ Concept drift
    # ────────────────────────────────────────────────────────────
    st.markdown("### Concept Drift Analysis")
    cd_left, cd_right = st.columns([1.25, 1.25])

    with cd_left:
        st.markdown("#### Numeric Features ↔ Target")
        num_concept = check_numeric_correlation(
            prod_df=prod_df,
            new_df=new_df,
            num_cols=numeric_cols,
            target=target_variable,
            target_type=target_type,
        )
        _paginated_table(num_concept, key="concept_num")

    with cd_right:
        st.markdown("#### Categorical Features ↔ Target")
        cat_concept = check_cat_contingency(
            prod_df=prod_df,
            new_df=new_df,
            cat_cols=categorical_cols,
            target=target_variable,
            target_type=target_type,
        )
        _paginated_table(cat_concept, key="concept_cat")

    # ────────────────────────────────────────────────────────────
    # Footer
    # ────────────────────────────────────────────────────────────
    st.markdown(
        """
        <span style='font-size:11px;color:#b0b0b0;'>
        ml-drift-detection © 2025 — <a href="https://github.com/knowusuboaky/ml-drift-detection"
        style="color:#b0b0b0;text-decoration:none;">Open Source</a>
        </span>
        """,
        unsafe_allow_html=True,
    )
