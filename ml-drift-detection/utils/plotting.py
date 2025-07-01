# ml-drift-detection/plotting.py

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.graph_objs as go
from typing import List, Dict, Optional
from plotly.subplots import make_subplots
import warnings

################# MODEL PERFORMANCE GAUGE #######################
# Function to create a half-donut gauge for visualizing percentage change
def _is_dark(col: str) -> bool:
    """Rudimentary luminance check (hex or 'black' / 'white')."""
    col = col.lower().lstrip("#")
    if col in {"black", "000", "000000"}:
        return True
    if col in {"white", "fff", "ffffff"}:
        return False
    if len(col) in {3, 6} and all(c in "0123456789abcdef" for c in col):
        if len(col) == 3:
            col = "".join(ch*2 for ch in col)
        r, g, b = int(col[0:2], 16), int(col[2:4], 16), int(col[4:6], 16)
        return 0.2126*r + 0.7152*g + 0.0722*b < 128  # WCAG-like weight
    # fallback – treat anything else as light
    return False

def drift_gauge_ban(
    metric_name: str,
    old: float,
    new: float,
    steps: List[Dict[str, object]],
    *,
    bar_color: str = "#00BFAE",
    bar_thickness: float = 0.4,
    background_color: str = "black",
    metric_color: str = "white",
    axis_color: Optional[str] = None,
    height: int = 200,
    number_y: float = 0.37,
    number_size: Optional[int] = 32,
    label_y: float = 0.00,
    label_size: Optional[int] = 18
) -> go.Figure:
    """
    Draw a customizable half-donut gauge for visualizing the percentage change between two values.

    This function creates a half-donut (or "ban") gauge visualization using Plotly to illustrate 
    the relative change (as a percentage) from an old (baseline) value to a new (current) value.
    The percentage change is capped between -100% and +100% (i.e., -1.0 to +1.0 in decimal form)
    to prevent the gauge from displaying out-of-bounds values. Color bands, a central percentage
    number, and a label make the gauge clear and visually appealing for dashboards and reports.

    Parameters
    ----------
    metric_name : str
        The label for the metric, displayed beneath the gauge.
    old : float
        The baseline or previous value used to compute percentage change.
    new : float
        The new or current value used to compute percentage change.
    steps : List[Dict[str, object]]
        A list of dictionaries, each specifying a color band for a range on the gauge.
        Each dictionary must have keys: 'range' (a tuple of two floats, lower and upper bounds, in [-1, 1]),
        and 'color' (a valid color string).
    bar_color : str, optional
        Color of the gauge's indicator bar (default "#00BFAE").
    bar_thickness : float, optional
        Thickness of the indicator bar (default 0.4).
    background_color : str, optional
        Background color of the chart (default "black").
    metric_color : str, optional
        Color for the metric name/label text (default "white").
    axis_color : str or None, optional
        Color for the gauge's axis, ticks, and labels. If None, will auto-detect for contrast based on background.
    height : int, optional
        Height of the chart in pixels (default 200). Controls auto-scaling of fonts.
    number_y : float, optional
        Vertical position of the central percentage number (0 = bottom, 1 = top; default 0.37).
    number_size : int or None, optional
        Font size of the central percentage number. If None, scales with height.
    label_y : float, optional
        Vertical position of the metric name label (default 0.00).
    label_size : int or None, optional
        Font size of the metric name label. If None, scales with height.

    Returns
    -------
    go.Figure
        A Plotly Figure object containing the formatted half-donut gauge visualization.
        The gauge displays:
            - The percentage change (centered, colored according to bands).
            - A colored arc representing the percentage change.
            - Custom color bands for specified ranges.
            - The metric label beneath the gauge.
            - The axis, ticks, and overall chart styled according to customization parameters.

    Example
    -------
    >>> steps = [
    ...     {"range": (-1, -0.5), "color": "#FF4D4F"},
    ...     {"range": (-0.5, 0), "color": "#FFDD57"},
    ...     {"range": (0, 0.5), "color": "#00BFAE"},
    ...     {"range": (0.5, 1), "color": "#1890FF"},
    ... ]
    >>> fig = drift_gauge_ban("Accuracy", 0.8, 0.9, steps)
    >>> fig.show()
    """
    # Calculate % change and cap to [-1, 1]
    if old == 0:
        pct_change = 0  # To avoid division by zero
    else:
        pct_change = (new - old) / old
        pct_change = max(-1, min(1, pct_change))  # Cap to [-1, 1]

    # Find the color for the central number based on the step bands
    number_colour = next(
        (band["color"] for band in steps if band["range"][0] <= pct_change < band["range"][1]),
        "#000000",
    )

    if axis_color is None:
        axis_color = "white" if _is_dark(background_color) else "black"

    pct_font_size = number_size if number_size is not None else int(height * 0.24)
    metric_font_size = label_size if label_size is not None else int(height * 0.14)
    tick_font_size = int(height * 0.08)

    # Custom tickvals and ticktext to ensure axis displays as -100%, -50%, 0%, 50%, 100%
    tickvals = [-1, -0.5, 0, 0.5, 1]
    ticktext = ["-1", "-0.5", "0", "+0.5", "+1"]

    fig = go.Figure(
        go.Indicator(
            mode="gauge",
            value=pct_change,
            gauge=dict(
                axis=dict(
                    range=[-1, 1],
                    tickvals=tickvals,
                    ticktext=ticktext,
                    tickcolor=axis_color,
                    tickfont=dict(color=axis_color, size=tick_font_size),
                ),
                bar=dict(color=bar_color, thickness=bar_thickness),
                steps=steps,
                threshold=dict(
                    line=dict(color=axis_color, width=4),
                    thickness=0.75,
                    value=pct_change,
                ),
                bgcolor=background_color,
            ),
        )
    )

    fig.add_annotation(
        x=0.5, y=number_y, xref="paper", yref="paper",
        text=f"{pct_change:+.2%}",
        showarrow=False,
        font=dict(size=pct_font_size, family="Roboto", color=number_colour),
    )
    fig.add_annotation(
        x=0.5, y=label_y, xref="paper", yref="paper",
        text=metric_name,
        showarrow=False,
        font=dict(size=metric_font_size, family="Roboto", color=metric_color),
    )

    fig.update_layout(
        margin=dict(l=int(height * 0.1), r=int(height * 0.1), t=int(height * 0.15), b=int(height * 0.25)),
        paper_bgcolor=background_color,
        plot_bgcolor=background_color,
        font=dict(color=axis_color),
        height=height,
    )
    return fig




###################### PLOTLY UPSET ################################

def get_plotly_upset(prod_df: pd.DataFrame, new_df: pd.DataFrame, categorical_cols: list[str], mode="dark"):
    if mode == "dark":
        bgcolor = "#000"
        textcolor = "#fff"
        bar_colors = dict(new="forestgreen", old="firebrick", both="gray")
        stripe_colors = ("#222", "#111")
        gridcolor = "#444"
        table_header_color = "#222"
        table_cell_color = "#111"
    else:
        bgcolor = "#fff"
        textcolor = "#000"
        bar_colors = dict(new="forestgreen", old="firebrick", both="gray")
        stripe_colors = ("#ebf0f8", "#ced2d9")
        gridcolor = "#eee"
        table_header_color = "#e2e2e2"
        table_cell_color = "#fff"

    def get_fixed_intersections(prod_set, new_set):
        new_only = sorted(list(new_set - prod_set))
        old_only = sorted(list(prod_set - new_set))
        both = sorted(list(prod_set & new_set))
        return [
            {"samples": ("New",), "n": len(new_only), "elements": new_only},
            {"samples": ("Old",), "n": len(old_only), "elements": old_only},
            {"samples": ("New", "Old"), "n": len(both), "elements": both}
        ]

    filtered_features = []
    for col in categorical_cols:
        prod_cats = set(prod_df[col].dropna().unique())
        new_cats = set(new_df[col].dropna().unique())
        if (new_cats - prod_cats) or (prod_cats - new_cats):
            filtered_features.append(col)
    if not filtered_features:
        filtered_features = ["None"]

    if filtered_features == ["None"]:
        fig = go.Figure()
        fig.update_layout(
            plot_bgcolor=bgcolor,
            paper_bgcolor=bgcolor,
            font=dict(color=textcolor),
            showlegend=False,
            margin=dict(l=40, r=20, t=60, b=40),
            xaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False, visible=False
            ),
            yaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False, visible=False
            ),
            title=None
        )
        fig.add_annotation(
            text="<b>No Significant Drift Detected<br>in Categorical Features</b>",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=28, color=textcolor),
            align="center"
        )
        return fig

    up_fig = make_subplots(
        rows=2, cols=2,
        shared_xaxes=True,
        vertical_spacing=0.01,
        horizontal_spacing=0.07,
        row_heights=[0.60, 0.15],
        column_widths=[0.7, 0.3],
        specs=[[{"type": "xy"}, {"type": "table"}],
               [{"type": "xy"}, None]]
    )

    traces_per_feature = []
    for i, col in enumerate(filtered_features):
        ref_set = set(prod_df[col].dropna().unique())
        new_set = set(new_df[col].dropna().unique())
        intersections = get_fixed_intersections(ref_set, new_set)
        y_bars = [x['n'] for x in intersections]
        intersection_names = ['New', 'Old', 'New & Old']
        bar_colors_list = [bar_colors["new"], bar_colors["old"], bar_colors["both"]]
        x_labels = [0, 1, 2]
        show_this = (i==0)

        hover_text = [f"{label}, {count}" for label, count in zip(intersection_names, y_bars)]
        up_fig.add_trace(go.Bar(
            x=x_labels,
            y=y_bars,
            marker_color=bar_colors_list,
            width=0.7,
            showlegend=False,
            hovertext=hover_text,
            hoverinfo="text",
            visible=show_this
        ), row=1, col=1)

        marker_size = 28
        dot_traces = 0
        for col_idx, inter in enumerate(intersections):
            for row_idx, setname in enumerate(['New', 'Old']):
                is_on = setname in inter["samples"]
                up_fig.add_trace(
                    go.Scatter(
                        x=[col_idx],
                        y=[-row_idx],
                        mode="markers",
                        marker=dict(
                            size=marker_size,
                            color="black" if is_on else "lightgray",
                            line=dict(width=0)
                        ),
                        showlegend=False,
                        hoverinfo='skip',
                        visible=show_this
                    ),
                    row=2, col=1
                )
                dot_traces += 1
            if inter["samples"] == ("New", "Old") or inter["samples"] == ("Old", "New"):
                up_fig.add_shape(
                    type="line",
                    xref="x1", yref="y2",
                    x0=col_idx, x1=col_idx,
                    y0=-1, y1=0,
                    line=dict(color="black", width=3),
                    row=2, col=1
                )

        for k in range(2):
            up_fig.add_shape(
                type="rect",
                xref="x1", yref="y2",
                x0=-0.5, x1=2.5,
                y0=-k+0.5-1, y1=-k-0.5-1,
                fillcolor=stripe_colors[k%2],
                line_width=0,
                layer="below",
                row=2, col=1
            )

        unchanged = sorted(list(ref_set & new_set))
        removed = sorted(list(ref_set - new_set))
        added = sorted(list(new_set - ref_set))
        up_fig.add_trace(
            go.Table(
                header=dict(values=["Type", "Subcategories"], font=dict(color=textcolor), fill_color=table_header_color),
                cells=dict(
                    values=[
                        ["Unchanged", "Removed", "New"],
                        [", ".join(unchanged), ", ".join(removed), ", ".join(added)]
                    ],
                    fill_color=[[table_header_color]*3, [table_cell_color]*3],
                    align="left",
                    font=dict(color=textcolor)
                ),
                visible=show_this
            ),
            row=1, col=2
        )
        traces_per_feature.append(1 + dot_traces + 1)  # bar + dots + table

    total_traces = sum(traces_per_feature)
    steps = []
    idx = 0
    for i, col in enumerate(filtered_features):
        vis = [False] * total_traces
        for j in range(traces_per_feature[i]):
            vis[idx + j] = True
        idx += traces_per_feature[i]
        steps.append(dict(
            label=str(col).capitalize(),
            method="update",
            args=[
                {"visible": vis},
                {"title.text": f"Subcategory Comparison: {col.capitalize()}"}
            ]
        ))

    # Bar chart (row 1, col 1)
    up_fig.update_xaxes(visible=False, row=1, col=1)
    up_fig.update_yaxes(
        title="Intersection size",
        title_font=dict(color="black"),
        showgrid=True,
        gridcolor=gridcolor,
        zeroline=False,
        color="black",  # axis line and label
        tickfont=dict(color="black", size=12),  # force y-axis tick labels to black
        row=1, col=1
    )

    # Dot-matrix grid (row 2, col 1): show black numbers and colored annotation labels
    up_fig.update_xaxes(
        title="Intersection",
        title_font=dict(color="black"),
        tickvals=[0,1,2],
        ticktext=["", "", ""],  # Hide tick labels for x
        showticklabels=True,
        color="black",
        row=2, col=1
    )
    up_fig.update_yaxes(
        title="Set",
        title_font=dict(color="black"),
        tickvals=[-0, -1],
        ticktext=['0', '1'],  # Black numbers for y
        range=[-1.5, 0.5],
        showgrid=False,
        zeroline=False,
        color="black",
        row=2, col=1
    )
    up_fig.update_xaxes(visible=False, row=2, col=2)
    up_fig.update_yaxes(visible=False, row=2, col=2)
    up_fig.update_xaxes(visible=False, row=1, col=2)
    up_fig.update_yaxes(visible=False, row=1, col=2)

    label_colors = ["forestgreen", "firebrick", "gray"]
    label_texts = ["New", "Old", "New & Old"]
    for i, (lab, color) in enumerate(zip(label_texts, label_colors)):
        up_fig.add_annotation(
            text=f"<b>{lab}</b>",
            x=i,
            y=-1.50,
            xref="x1",
            yref="y2",
            showarrow=False,
            font=dict(color=color, size=12),
            align="center",
            row=2, col=1
        )

    for i, (lab, color) in enumerate(zip(["New", "Old"], ["forestgreen", "firebrick"])):
        up_fig.add_annotation(
            text=f"<b>{lab}</b>",
            x=-0.7,
            y=-i,
            xref="x1",
            yref="y2",
            showarrow=False,
            font=dict(color=color, size=12),
            align="right",
            row=2, col=1
        )

    up_fig.update_layout(
        title={"text": f"Subcategory Comparison: {filtered_features[0].capitalize()}", "x": 0.5, "xanchor": "center"},
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
        plot_bgcolor=bgcolor,
        paper_bgcolor=bgcolor,
        font=dict(color=textcolor),
        showlegend=False,
        margin=dict(l=40, r=20, t=60, b=40)
    )

    return up_fig

