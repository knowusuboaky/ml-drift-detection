# ml-drift-detection/categorical_drift.py

import pandas as pd
import numpy as np
from scipy import stats
import warnings
import plotly.graph_objs as go
from typing import List

########################## CATEGORICAL #############################
# ----------------------------------------------------------------------
# Data-frame subclass that adds .interpret()
# ----------------------------------------------------------------------
class CatStatsSummary(pd.DataFrame):
    _metadata = ["_p_threshold"]

    @property
    def _constructor(self):
        return CatStatsSummary

    def interpret(self) -> str:
        """
        Return a single formal sentence listing only features flagged as strong drift (🔴),
        with a formal recommendation, matching NumStatsSummary.interpret().
        """
        α = getattr(self, "_p_threshold", 0.05)

        # Only features with a "🔴" detection (strong drift)
        red_features = [
            row["Feature"] for _, row in self.iterrows()
            if row["Detection"] == "🔴"
        ]

        if not red_features:
            return (
                "All categorical features examined exhibit distributions that are "
                f"statistically indistinguishable from the production benchmark at a "
                f"chi-square p-value threshold of {α}."
            )

        plural = "feature" if len(red_features) == 1 else "feature(s)"
        feature_list = ", ".join(red_features)

        return (
            f"The {plural} {feature_list} display categorical distributions that "
            f"differ substantially from those observed in the production data, as "
            f"determined by at least two drift criteria—including the chi-square "
            f"test at a p-value threshold of {α}, a change in the modal category, or "
            "a change in the number of unique category levels. This indicates "
            "meaningful changes in category frequencies or the emergence of new or "
            "missing levels. Prior to promoting this data to production, it is "
            "recommended that the model be retrained on the updated distribution and "
            "that the underlying process be investigated for potential root causes "
            "of the observed categorical drift."
        )

def get_plotly_barplot(
    *,
    prod_df: pd.DataFrame,
    new_df: pd.DataFrame,
    cat_cols: List[str],
    mode: str = "dark",
    height: int = 300,
):
    """
    Faceted bar-chart widget that lets you flip through every categorical
    feature and compare class proportions between the production baseline
    and the new batch.

    Parameters
    ----------
    prod_df, new_df : pd.DataFrame
        DataFrames that contain (at least) the columns in `cat_cols`.
    cat_cols : list[str]
        Categorical columns to visualise.
    mode : {"dark", "light"}
        Plotly colour template. Anything other than "dark" defaults to light.
    height : int
        Pixel height of the figure.

    Returns
    -------
    plotly.graph_objs.Figure
    """
    
    if mode == "dark":
        bgcolor = "#000"
        textcolor = "#fff"
        prod_color = "#b0b0b0"
        new_color = "#d32f2f"
        gridcolor = "#444"
    else:
        bgcolor = "#fff"
        textcolor = "#000"
        prod_color = "#b0b0b0"
        new_color = "#d32f2f"
        gridcolor = "#eee"

    data = []
    for i, col in enumerate(cat_cols):
        prod_counts = prod_df[col].value_counts(normalize=True).sort_index()
        new_counts = new_df[col].value_counts(normalize=True).sort_index()
        all_cats = sorted(set(prod_counts.index).union(new_counts.index))

        prod_counts = prod_counts.reindex(all_cats, fill_value=0)
        new_counts = new_counts.reindex(all_cats, fill_value=0)

        data.append(go.Bar(
            x=all_cats, y=prod_counts,
            name="Production Data",
            marker_color=prod_color,
            visible=(i == 0)
        ))
        data.append(go.Bar(
            x=all_cats, y=new_counts,
            name="Incoming Data",
            marker_color=new_color,
            visible=(i == 0)
        ))

    steps = []
    for i, col in enumerate(cat_cols):
        vis = [False]*2*len(cat_cols)
        vis[2*i] = True
        vis[2*i+1] = True
        steps.append(
            dict(
                label=str(col).capitalize(),
                method="update",
                args=[
                    {"visible": vis},
                    {"title": {
                        "text": f"Categorical Drift: {str(col).capitalize()}",
                        "x": 0.5,
                        "xanchor": "center"
                    },
                    "xaxis.title": str(col).capitalize()}
                ]
            )
        )

    layout = go.Layout(
        title={
            "text": f"Categorical Drift: {str(cat_cols[0]).capitalize()}",
            "x": 0.5,
            "xanchor": "center"
        },
        xaxis=dict(
            title=str(cat_cols[0]).capitalize(),
            gridcolor=gridcolor,
            color="black",  # <-- force axis label/tick color black
            title_font=dict(color="black"),  # <-- force axis title black
            tickfont=dict(color="black")  # <-- force tick numbers black
        ),
        yaxis=dict(
            title="Proportion",
            gridcolor=gridcolor,
            color="black",  # <-- force axis label/tick color black
            title_font=dict(color="black"),  # <-- force axis title black
            tickfont=dict(color="black")  # <-- force tick numbers black
        ),
        plot_bgcolor=bgcolor, paper_bgcolor=bgcolor,
        font=dict(color=textcolor),
        legend=dict(font=dict(color=textcolor)),
        updatemenus=[
            dict(
                buttons=steps,
                direction="down",
                x=0.01, xanchor="left",
                y=1.15, yanchor="top",
                showactive=True,
                bgcolor=bgcolor,
                bordercolor=textcolor,
                font=dict(color=textcolor)
            ),
        ],
        margin=dict(l=40, r=20, t=60, b=40),
        barmode='group',
        height=height
    )
    fig = go.Figure(data=data, layout=layout)
    return fig

# ----------------------------------------------------------------------
# Main routine (unchanged except for return type and storing α)
# ----------------------------------------------------------------------
def check_categorical_stats(
    prod_df: pd.DataFrame,
    new_df: pd.DataFrame,
    cat_cols: list[str],
    p_threshold: float = 0.05,
) -> CatStatsSummary:
    """Compare new vs. reference categorical columns and add drift flags."""
    def pct_change(a, b):
        if pd.isna(a) or a == 0 or pd.isna(b):
            return "—"
        return f"{(b - a) / abs(a):+0.2%}"

    rows = []
    for col in cat_cols:
        ref, new = prod_df[col], new_df[col]

        stats_ref = {
            "mode":   ref.mode(dropna=True).iat[0] if not ref.mode().empty else pd.NA,
            "unique": ref.nunique(dropna=True),
            "nulls":  ref.isna().mean(),
        }
        stats_new = {
            "mode":   new.mode(dropna=True).iat[0] if not new.mode().empty else pd.NA,
            "unique": new.nunique(dropna=True),
            "nulls":  new.isna().mean(),
        }

        deltas = {
            "mode":   "same" if stats_ref["mode"] == stats_new["mode"]
                               else f'{stats_ref["mode"]} → {stats_new["mode"]}',
            "unique": f'{stats_new["unique"] - stats_ref["unique"]:+d}',
            "nulls":  pct_change(stats_ref["nulls"], stats_new["nulls"]),
        }

        cats = sorted(set(ref.dropna().unique()) | set(new.dropna().unique()))
        exp  = ref.value_counts().reindex(cats, fill_value=0).to_numpy()
        obs  = new.value_counts().reindex(cats, fill_value=0).to_numpy()
        exp  = np.where(exp == 0, 1e-6, exp)

        _, chi_p = stats.chisquare(obs, f_exp=exp)

       # ------ New flag logic ------
        drift_criteria = [
            chi_p <= p_threshold,
            deltas["mode"] != "same",
            int(deltas["unique"]) != 0
        ]
        drift_count = sum(drift_criteria)

        if drift_count >= 2:
            flag = "🔴"
        elif drift_count == 1:
            flag = "🟡"
        else:
            flag = "🟢"
        # ---------------------------

        rows.append({
            "Feature":  col,
            "Mode Δ":   deltas["mode"],
            "Unique Δ": deltas["unique"],
            "Nulls Δ":  deltas["nulls"],
            "Chi² p":   f"{chi_p:.3g}",
            "Detection": flag,
        })

    df = CatStatsSummary(rows)
    df._p_threshold = p_threshold
    return df


######################### CONCEPT DRIFT ############################
def check_cat_contingency(
    prod_df: pd.DataFrame,
    new_df: pd.DataFrame,
    cat_cols: list[str],
    target: str,
    target_type: str = "auto",
    low_threshold: float = 10.0,   # as percent
    high_threshold: float = 20.0,  # as percent
    cat_max_uniques: int = 10
) -> pd.DataFrame:
    """
    Checks drift in the relationship between categorical features and the target variable,
    handling both categorical and continuous targets.
    Returns most changed category, target value, and percentage change.
    """
    results = []
    # Auto-detect type if needed
    if target_type == "auto":
        nunique = prod_df[target].nunique()
        if prod_df[target].dtype == 'O' or nunique <= cat_max_uniques:
            use_mode = "categorical"
        else:
            use_mode = "continuous"
    else:
        use_mode = target_type.lower()

    for col in cat_cols:
        if col not in prod_df.columns or col not in new_df.columns:
            continue

        if use_mode == "categorical":
            ref_table = pd.crosstab(prod_df[col], prod_df[target], normalize='index')
            new_table = pd.crosstab(new_df[col], new_df[target], normalize='index')
            # Align
            ref_table, new_table = ref_table.align(new_table, join='outer', fill_value=0)
            diff = (new_table - ref_table)
            idx = np.unravel_index(np.abs(diff.values).argmax(), diff.values.shape)
            most_changed_cat = ref_table.index[idx[0]]
            target_val = ref_table.columns[idx[1]]
            ref_val = ref_table.values[idx]
            new_val = new_table.values[idx]
            if ref_val == 0:
                if new_val == 0:
                    pct_change = 0.0
                else:
                    pct_change = 100.0 * np.sign(new_val)
            else:
                pct_change = ((new_val - ref_val) / abs(ref_val)) * 100
        elif use_mode == "continuous":
            ref_means = prod_df.groupby(col)[target].mean()
            new_means = new_df.groupby(col)[target].mean()
            all_cats = ref_means.index.union(new_means.index)
            ref_means = ref_means.reindex(all_cats, fill_value=0)
            new_means = new_means.reindex(all_cats, fill_value=0)
            biggest_cat = None
            ref_val = None
            new_val = None
            pct_change = None
            for cat in all_cats:
                rv = ref_means[cat]
                nv = new_means[cat]
                if rv == 0:
                    if nv == 0:
                        change = 0.0
                    else:
                        change = 100.0 * np.sign(nv)
                else:
                    change = ((nv - rv) / abs(rv)) * 100
                if (pct_change is None) or (abs(change) > abs(pct_change)):
                    pct_change = change
                    biggest_cat = cat
                    ref_val = rv
                    new_val = nv
            most_changed_cat = biggest_cat
            target_val = None
        else:
            raise ValueError("target_type must be 'categorical', 'continuous', or 'auto'.")

        # Flag (absolute for flagging)
        if pct_change is None or pd.isna(pct_change):
            flag = "N/A"
        elif abs(pct_change) >= high_threshold:
            flag = "🔴"
        elif abs(pct_change) >= low_threshold:
            flag = "🟡"
        else:
            flag = "🟢"

        results.append({
            'Feature': col,
            'Most Changed Category': most_changed_cat,
            'Target': target_val,
            #'Reference Value': f"{ref_val:.3f}" if ref_val is not None else "",
            #'New Value': f"{new_val:.3f}" if new_val is not None else "",
            'Change (%)': f"{pct_change:+.1f}%" if pct_change is not None and not pd.isna(pct_change) else "NaN",
            'Detection': flag
        })
    return pd.DataFrame(results)

