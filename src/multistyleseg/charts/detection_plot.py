from multistyleseg.data.fundus.consts import MAPPING_STR, TEST_DATASET_SIZE
from multistyleseg.analysis.utils import DATASET_COLORS, LesionsUtils
from matplotlib.patches import Patch

import math
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl

mpl.rcParams["hatch.linewidth"] = 0.25  # previous pdf hatch linewidth


LOW_YLIM = (0, 45)
HIGH_YLIM = (45, 160)
HEIGHT_RATIO = [1, 2.5]


def comparative_detection_plot(
    df: pd.DataFrame,
    n_cols=2,
    hatch="//",
    model_choices: list = None,
    legend_yloc=0.925,
    ax_width=6,
    ax_height=4,
    tp_width_factor=0.5,
    include_titles=True,
):
    if model_choices is not None:
        df = df[df["Model"].isin(model_choices)]

    lesion_order = LesionsUtils.reorder(df["Lesion"].unique())
    x_ticks = pd.Series(lesion_order).map(MAPPING_STR).values
    x_values = np.arange(len(x_ticks)) * 2

    n_models = df["Model"].nunique()
    n_rows = math.ceil(n_models / n_cols)

    fig = plt.figure(figsize=(ax_width * n_cols, ax_height * n_rows))
    outer_grid = GridSpec(n_rows, n_cols, figure=fig, hspace=0.08, wspace=0.0)
    n_cols = 2

    # Track first ax_bot per column for sharex
    col_ref_axes = {}
    col_ref_axes = {}
    row_ref_top = {}
    row_ref_bot = {}
    all_datasets = df["Dataset"].unique()
    for idx, model in enumerate(df["Model"].unique()):
        df_model = df[df["Model"] == model]
        row, col = divmod(idx, n_cols)
        is_last_row = (row == n_rows - 1) or (idx + n_cols >= n_models)
        inner_grid = GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=outer_grid[idx],
            height_ratios=HEIGHT_RATIO,
            hspace=0.04,
            wspace=0,
        )

        share_x = col_ref_axes.get(col)
        share_y_top = row_ref_top.get(row)
        share_y_bot = row_ref_bot.get(row)

        ax_top = fig.add_subplot(inner_grid[0], sharey=share_y_top)
        ax_bot = fig.add_subplot(
            inner_grid[1],
            sharex=share_x if share_x else ax_top,
            sharey=share_y_bot,
        )

        if share_x is None:
            col_ref_axes[col] = ax_bot
        if share_y_top is None:
            row_ref_top[row] = ax_top
        if share_y_bot is None:
            row_ref_bot[row] = ax_bot

        # Hide y-tick labels on non-first columns
        if col > 0:
            ax_top.tick_params(labelleft=False)
            ax_bot.tick_params(labelleft=False)
            ax_bot.set_ylabel("")
        interval = 1.85
        step = interval / (len(all_datasets) * 2)
        separator = step * 0.25
        for i, (dataset_test, color) in enumerate(zip(all_datasets, DATASET_COLORS)):
            dataset_size = TEST_DATASET_SIZE[dataset_test]
            df_filtered = (
                df_model[df_model["Dataset"] == dataset_test]
                .set_index("Lesion")
                .loc[lesion_order]
                .reset_index()
            )

            for ax in (ax_top, ax_bot):
                ax.bar(
                    x_values - interval / 2 + 2 * i * step + separator,
                    df_filtered["n_gt"] / dataset_size,
                    width=step - separator,
                    hatch=hatch,
                    color=color,
                    # edgecolor="k",
                    linewidth=0.5,
                )
                ax.bar(
                    x_values - interval / 2 + (2 * i + 1) * step,
                    df_filtered["n_pred"] / dataset_size,
                    width=step - separator,
                    edgecolor="k",
                    color=color,
                    linewidth=0.5,
                )
                mid = (x_values - interval / 2 + (2 * i + 0.5) * step) + separator / 2
                ax.bar(
                    mid,
                    df_filtered["TP"] / dataset_size,
                    alpha=0.5,
                    width=(step - separator) * tp_width_factor,
                    color="#40a02b",
                    edgecolor="white",
                    linewidth=0.5,
                )

        ax_top.set_ylim(*HIGH_YLIM)
        ax_bot.set_ylim(*LOW_YLIM)

        ax_top.spines["bottom"].set_visible(False)
        ax_bot.spines["top"].set_visible(False)
        ax_top.tick_params(bottom=False, labelbottom=False)
        d = 0.012
        d_top = d * sum(HEIGHT_RATIO) / HEIGHT_RATIO[0]
        d_bot = d * sum(HEIGHT_RATIO) / HEIGHT_RATIO[1]

        kwargs = dict(color="k", clip_on=False, linewidth=1)
        ax_top.plot((-d, +d), (-d_top, +d_top), transform=ax_top.transAxes, **kwargs)
        ax_top.plot(
            (1 - d, 1 + d), (-d_top, +d_top), transform=ax_top.transAxes, **kwargs
        )
        ax_bot.plot(
            (-d, +d), (1 - d_bot, 1 + d_bot), transform=ax_bot.transAxes, **kwargs
        )
        ax_bot.plot(
            (1 - d, 1 + d), (1 - d_bot, 1 + d_bot), transform=ax_bot.transAxes, **kwargs
        )
        ax_bot.set_xticks(x_values)
        if is_last_row:
            ax_bot.set_xticklabels(x_ticks, fontsize=12)
        else:
            ax_bot.tick_params(labelbottom=False)

        for x in x_values[:-1] + interval / 2:
            for ax in (ax_top, ax_bot):
                ax.axvline(x=x, color="k", linestyle="--", alpha=0.5, linewidth=0.5)
        for j, lesion in enumerate(lesion_order):
            mean_f1 = df_model[df_model["Lesion"] == lesion]["F1"].mean()
            ax_bot.text(
                x_values[j] + interval / 4,
                25,
                f"F1={mean_f1:.2f}",
                ha="center",
                va="top",
                fontsize=7,
                fontstyle="italic",
            )

        for ax in (ax_top, ax_bot):
            ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
            ax.set_axisbelow(True)

        if col == 0:
            ax_bot.set_ylabel("Count / dataset size", fontsize=8)

        average_f1 = df_model.groupby("Lesion")["F1"].mean().mean()
        ax_bot.set_title(f"Avg. F1={average_f1:.2f}", fontsize=8, fontweight="bold")
        model_name = model
        if include_titles:
            ax_top.set_title(model_name, fontsize=10, fontweight="bold")
        min_x = x_values[0] - interval / 2 - step + separator
        max_x = x_values[-1] + interval / 2
        for ax in (ax_top, ax_bot):
            for j, lesion in enumerate(lesion_order):
                ax.axvspan(
                    x_values[j] - interval / 2 - step + separator,
                    x_values[j] + interval / 2,
                    color=LesionsUtils.get_color(lesion),
                    alpha=0.2,
                    zorder=-5,
                )
        ax.set_xlim(min_x, max_x)
        for tick, lesion in zip(ax.get_xticklabels(), lesion_order):
            tick.set_color(LesionsUtils.get_color(lesion))
            # Set it to bold
            tick.set_fontweight("bold")
        if idx == 0:
            dataset_handles = [
                Patch(facecolor=color, edgecolor="k", label=dataset, linewidth=0.25)
                for dataset, color in zip(all_datasets, DATASET_COLORS)
            ]
            type_handles = [
                Patch(facecolor="white", hatch=hatch, label="GT count"),
                Patch(facecolor="white", edgecolor="k", label="Pred. count"),
                Patch(
                    facecolor="#40a02b",
                    edgecolor="white",
                    alpha=0.5,
                    label="True positives",
                ),
            ]
            handles = dataset_handles + type_handles
            n_legend_cols = 5
            n_legend_rows = math.ceil(len(handles) / n_legend_cols)

            reordered = []
            for col in range(n_legend_cols):
                for row in range(n_legend_rows):
                    idx = row * n_legend_cols + col
                    if idx < len(handles):
                        reordered.append(handles[idx])
            # Reorder the handles to have them grouped by dataset first, then by type
            fig.legend(
                handles=reordered,
                ncol=n_legend_cols,
                loc="upper center",
                bbox_to_anchor=(0.5, legend_yloc),
                fontsize="small",
                frameon=False,
            )

    return fig


def comparative_detection_radar_plot(
    df: pd.DataFrame,
    metric: str = "F1",
    per: str = "lesion",  # "lesion" or "dataset"
    n_cols: int = 2,
    model_choices: list[str] | None = None,
    ax_size: float = 4,
    include_titles: bool = True,
    ylim: tuple[float, float] = (0, 1),
    metric_name: str | None = None,
):
    """
    Radar chart for detection performance.

    Parameters
    ----------
    df : pd.DataFrame
        Flat DataFrame with columns: Model, Lesion, Dataset, F1, TP, n_gt, n_pred, ...
    per : str
        "lesion"  → one spoke per lesion, value = mean across datasets.
        "dataset" → one spoke per dataset, value = mean across lesions.
    """
    if per not in ("lesion", "dataset"):
        raise ValueError(f"per must be 'lesion' or 'dataset', got '{per}'")
    lesion_order = LesionsUtils.reorder(df["Lesion"].unique())

    if model_choices is not None:
        df = df[df["Model"].isin(model_choices)]
        # Sort the models in the order of model_choices
        df["Model"] = pd.Categorical(
            df["Model"], categories=model_choices, ordered=True
        )
        df = df.sort_values("Model")

    all_datasets = df["Dataset"].unique()
    models = df["Model"].unique()
    n_models = len(models)
    n_rows = math.ceil(n_models / n_cols)

    fig = plt.figure(figsize=(ax_size * n_cols, ax_size * n_rows))
    grid = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)

    for idx, model in enumerate(models):
        df_model = df[df["Model"] == model]
        ax = fig.add_subplot(grid[idx], polar=True)

        if per == "lesion":
            categories = [MAPPING_STR.get(l, l) for l in lesion_order]
            colors = [LesionsUtils.get_color(l) for l in lesion_order]
            values = [
                df_model[df_model["Lesion"] == lesion][metric].mean()
                for lesion in lesion_order
            ]
        else:
            categories = [str(ds) for ds in all_datasets]
            colors = list(DATASET_COLORS[: len(all_datasets)])
            values = [
                df_model[df_model["Dataset"] == dataset][metric].mean()
                for dataset in all_datasets
            ]

        n_cats = len(categories)
        angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
        angles_closed = angles + [angles[0]]
        values_closed = values + [values[0]]

        ax.set_ylim(*ylim)
        ax.set_xticks(angles)
        ax.set_xticklabels(categories, fontsize=8, fontweight="bold")
        for tick, color in zip(ax.get_xticklabels(), colors):
            tick.set_color(color)

        ax.yaxis.grid(True, linewidth=0.5, alpha=0.5)

        for i in range(n_cats):
            j = (i + 1) % n_cats
            seg_angles = [angles[i], angles[j], 0]
            seg_values = [values[i], values[j], 0]
            ax.fill(seg_angles, seg_values, color=colors[i], alpha=0.5)

        ax.plot(angles_closed, values_closed, color="k", linewidth=1.5)

        for a, v, c in zip(angles, values, colors):
            ax.plot(a, v, "o", color=c, markersize=5, zorder=5)

        ax.set_yticklabels([])

        for a, v, c in zip(angles, values, colors):
            cos_a = np.cos(a)
            ha = "center"
            va = "bottom" if cos_a > 0.1 else "top" if cos_a < -0.1 else "center"
            ax.text(
                a,
                v + 0.2,
                f"{v:.2f}",
                ha=ha,
                va=va,
                fontsize=10,
                fontweight="bold",
                color=c,
            )

        avg = np.mean(values)
        ax.text(
            0.5,
            0.5,
            f"{metric_name or metric}\n{avg:.3f}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            transform=ax.transAxes,
        )

        if include_titles:
            letter = chr(ord("a") + idx)
            ax.set_title(
                f"{letter}) {model.upper()}", fontsize=10, fontweight="bold", pad=16
            )

    return fig
