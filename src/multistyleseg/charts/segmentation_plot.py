import pandas as pd
from multistyleseg.data.fundus.consts import MAPPING_STR
from multistyleseg.analysis.utils import DATASET_COLORS, LesionsUtils
from matplotlib.patches import Patch
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def comparative_segmentation_plot(
    df: pd.DataFrame,
    metrics: tuple[str] = ("IoU",),
    metrics_hatches: tuple[str] = ("",),
    n_cols=2,
    model_choices: tuple[str] = None,
    legend_yloc=0.925,
    ax_width=6,
    ax_height=4,
    include_titles=True,
):
    lesions = [
        c
        for c in df.columns.get_level_values(0).unique()
        if (c not in ["MEAN", "Model"])
    ]
    if model_choices is not None:
        df = df[df["Model"].isin(model_choices)]
    lesion_order = LesionsUtils.reorder(lesions)
    x_ticks = pd.Series(lesion_order).map(MAPPING_STR).values
    x_values = np.arange(len(x_ticks)) * 2

    n_models = df["Model"].nunique()
    n_rows = math.ceil(n_models / n_cols)

    fig = plt.figure(figsize=(ax_width * n_cols, ax_height * n_rows))
    grid = GridSpec(n_rows, n_cols, figure=fig, hspace=0.1, wspace=0.0)
    row_ref_ax = {}

    for idx, model in enumerate(df["Model"].unique()):
        df_model = df[df["Model"] == model]
        row, col = divmod(idx, n_cols)
        all_datasets = df_model.index.tolist()
        share_y = row_ref_ax.get(row)
        ax = fig.add_subplot(grid[idx], sharey=share_y)
        if share_y is None:
            row_ref_ax[row] = ax

        if col > 0:
            ax.tick_params(labelleft=False)

        interval = 1.85
        n_bars = len(all_datasets) * len(metrics)
        step = interval / n_bars
        separator = step * 0.25
        for i, (dataset, color) in enumerate(zip(all_datasets, DATASET_COLORS)):
            for m, metric in enumerate(metrics):
                bar_idx = i * len(metrics) + m
                values = [
                    df_model.loc[dataset, (lesion, metric)] for lesion in lesion_order
                ]
                ax.bar(
                    x_values - interval / 2 + bar_idx * step + separator,
                    values,
                    width=step - separator,
                    hatch=metrics_hatches[m],
                    color=color,
                    edgecolor="k",
                    linewidth=0.5,
                )

        ax.set_xticks(x_values)
        is_last_row = (row == n_rows - 1) or (idx + n_cols >= n_models)
        if is_last_row:
            ax.set_xticklabels(x_ticks, fontsize=12)
        else:
            ax.tick_params(labelbottom=False)

        ax.set_ylim(0, 1)
        ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)

        for x in x_values[:-1] + interval / 2:
            ax.axvline(x=x, color="k", linestyle="--", alpha=0.5, linewidth=0.5)

        # Annotate mean IoU per lesion
        for j, lesion in enumerate(lesion_order):
            mean_iou = df_model[(lesion, "IoU")].mean()
            ax.text(
                x_values[j],
                0.85,
                f"Average IoU={mean_iou:.2f}",
                ha="center",
                va="top",
                fontsize=7,
                fontstyle="italic",
            )
        # The width of an individual bar in your loop
        n_bars = len(all_datasets) * len(metrics)
        bar_width = step - separator

        min_x = x_values[0] - interval / 2 - step + separator + bar_width / 2
        max_x = x_values[-1] + interval / 2
        for j, lesion in enumerate(lesion_order):
            ax.axvspan(
                x_values[j] - interval / 2 - step + separator + bar_width / 2,
                x_values[j] + interval / 2,
                color=LesionsUtils.get_color(lesion),
                alpha=0.2,
                zorder=-5,
            )
        ax.set_xlim(min_x, max_x)
        if col == 0:
            ax.set_ylabel("Score", fontsize=8)
        if include_titles:
            ax.set_title(model, fontsize=10, fontweight="bold")
        mean_iou = df_model[("MEAN", "IoU")].mean()
        ax.text(
            0.5,
            ax.get_ylim()[1] - 0.05,
            f"Avg. mIoU={mean_iou:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            transform=ax.transAxes,
        )
        # Change the color of the xlabels to match the lesion color
        for tick, lesion in zip(ax.get_xticklabels(), lesion_order):
            tick.set_color(LesionsUtils.get_color(lesion))
            # Set it to bold
            tick.set_fontweight("bold")
        if idx == 0:
            dataset_handles = [
                Patch(
                    facecolor=color,
                    edgecolor="k",
                    label=dataset.replace("FundusDataset.", ""),
                    linewidth=0.5,
                )
                for dataset, color in zip(all_datasets, DATASET_COLORS)
            ]
            # metric_handles = [
            #     Patch(
            #         facecolor="white",
            #         edgecolor="k",
            #         hatch=metrics_hatches[m],
            #         label=metrics[m],
            #     )
            #     for m in range(len(metrics))
            # ]
            fig.legend(
                handles=dataset_handles,
                ncol=len(all_datasets) + len(metrics),
                loc="upper center",
                bbox_to_anchor=(0.5, legend_yloc),
                fontsize="small",
                frameon=False,
            )
    return fig


def comparative_segmentation_radar_plot(
    df: pd.DataFrame,
    metric: str = "IoU",
    per: str = "lesion",  # "lesion" or "dataset"
    n_cols: int = 2,
    model_choices: list[str] | None = None,
    ax_size: float = 4,
    include_titles: bool = True,
    ylim: tuple[float, float] = (0, 1),
    metric_name: str | None = None,
):
    """
    Radar chart comparing models on a given metric.

    Parameters
    ----------
    per : str
        "lesion"  → one spoke per lesion, value = mean across datasets. 4 points.
        "dataset" → one spoke per dataset, value = mean across lesions. 5 points.
    """
    if per not in ("lesion", "dataset"):
        raise ValueError(f"per must be 'lesion' or 'dataset', got '{per}'")

    lesions = [
        c for c in df.columns.get_level_values(0).unique() if c not in ["MEAN", "Model"]
    ]
    lesion_order = LesionsUtils.reorder(lesions)

    if model_choices is not None:
        df = df[df["Model"].isin(model_choices)]
        # Order the models as in model_choices
        df["Model"] = pd.Categorical(
            df["Model"], categories=model_choices, ordered=True
        )
        df = df.sort_values("Model")

    models = df["Model"].unique()
    n_models = len(models)
    n_rows = math.ceil(n_models / n_cols)

    fig = plt.figure(figsize=(ax_size * n_cols, ax_size * n_rows))
    grid = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)

    for idx, model in enumerate(models):
        df_model = df[df["Model"] == model]
        all_datasets = df_model.index.tolist()

        ax = fig.add_subplot(grid[idx], polar=True)

        if per == "lesion":
            categories = [MAPPING_STR.get(l, l) for l in lesion_order]
            colors = [LesionsUtils.get_color(l) for l in lesion_order]
            values = [df_model[(lesion, metric)].mean() for lesion in lesion_order]
        else:
            categories = [ds.replace("FundusDataset.", "") for ds in all_datasets]
            colors = list(DATASET_COLORS[: len(all_datasets)])
            values = [
                np.mean(
                    [df_model.loc[dataset, (lesion, metric)] for lesion in lesion_order]
                )
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

        # Draw segments with per-spoke colors
        for i in range(n_cats):
            j = (i + 1) % n_cats
            seg_angles = [angles[i], angles[j], 0]
            seg_values = [values[i], values[j], 0]
            ax.fill(seg_angles, seg_values, color=colors[i], alpha=0.5)

        ax.plot(angles_closed, values_closed, color="k", linewidth=1.5)
        # Dots at each spoke
        for a, v, c in zip(angles, values, colors):
            ax.plot(a, v, "o", color=c, markersize=5, zorder=5)

        # Remove ring labels
        ax.set_yticklabels([])

        # Value labels at each spoke
        for a, v, c in zip(angles, values, colors):
            # Alignment based on spoke direction
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
        # Center text
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
