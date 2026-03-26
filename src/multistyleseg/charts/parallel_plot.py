import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import plasma


def bezier_curve(p0, p1, num_points=60):
    """Cubic Bezier S-curve between two points."""
    x0, y0 = p0
    x1, y1 = p1
    cx0 = x0 + (x1 - x0) * 0.4
    cx1 = x0 + (x1 - x0) * 0.6
    t = np.linspace(0, 1, num_points)
    x = (
        (1 - t) ** 3 * x0
        + 3 * (1 - t) ** 2 * t * cx0
        + 3 * (1 - t) * t**2 * cx1
        + t**3 * x1
    )
    y = (
        (1 - t) ** 3 * y0
        + 3 * (1 - t) ** 2 * t * y0
        + 3 * (1 - t) * t**2 * y1
        + t**3 * y1
    )
    return x, y


# ── Color palettes ──────────────────────────────────────────────────────
_HIGHLIGHT_CMAP = LinearSegmentedColormap.from_list(
    "warm_highlight",
    ["#4361ee", "#7209b7", "#f72585", "#ff6d00", "#ffbe0b"],
)

_BOTTOM_CMAP = LinearSegmentedColormap.from_list(
    "cool_bottom",
    ["#064e3b", "#047857", "#34d399", "#6ee7b7"],
)

_DARK = {
    "bg": "#0f1117",
    "axis": "#3a3d4a",
    "tick": "#8b8fa3",
    "label": "#c9ccd5",
    "title": "#eceef4",
    "grid": "#22252f",
    "bg_line": "#4a4e5c",
}

_LIGHT = {
    "bg": "#ffffff",
    "axis": "#c0c3cc",
    "tick": "#5c5f6e",
    "label": "#33363f",
    "title": "#111318",
    "grid": "#f0f0f3",
    "bg_line": "#b0b3be",
}


def parallel_coordinates_highlighted(
    df,
    dimensions,
    color_col,
    top_n=5,
    bottom_n=0,
    figsize=(16, 7),
    linewidth=1.0,
    background_alpha=0.08,
    highlight_cmap=None,
    bottom_cmap=None,
    jitter_strength=0.02,
    legend=True,
    seed=42,
    dark=True,
    gradient_lines=True,
):
    """
    Parallel-coordinates plot with top-N and bottom-N highlighting.

    Parameters
    ----------
    top_n : int
        Number of best runs to highlight (warm palette).
    bottom_n : int
        Number of worst runs to highlight (cool palette). 0 to disable.
    dark : bool
        Use dark background theme.
    gradient_lines : bool
        Draw highlighted lines with glow effect.
    """

    np.random.seed(seed)
    pal = _DARK if dark else _LIGHT
    if highlight_cmap is None:
        highlight_cmap = _HIGHLIGHT_CMAP
    if bottom_cmap is None:
        bottom_cmap = _BOTTOM_CMAP

    # ── Normalize dimensions ────────────────────────────────────────────
    normalized_dims = []
    for dim in dimensions:
        vals = np.array(dim["values"], dtype=float)
        vmin, vmax = vals.min(), vals.max()
        if vmax - vmin < 1e-9:
            normalized = np.zeros_like(vals, dtype=float)
        else:
            normalized = (vals - vmin) / (vmax - vmin)
        normalized_dims.append(normalized)

    normalized_data = np.array(normalized_dims).T
    n_dims = len(dimensions)
    n_runs = len(df)
    x_positions = np.arange(n_dims)

    # ── Jitter categorical axes ─────────────────────────────────────────
    jittered_data = normalized_data.copy()
    for i, dim in enumerate(dimensions):
        if "tickvals" in dim and len(dim["tickvals"]) <= 10:
            jitter = np.random.uniform(-jitter_strength, jitter_strength, n_runs)
            jittered_data[:, i] = np.clip(normalized_data[:, i] + jitter, 0, 1)

    # ── Top-N indices ───────────────────────────────────────────────────
    top_indices = df[color_col].nlargest(top_n).index.tolist()
    top_values = df.loc[top_indices, color_col]
    norm_top = Normalize(vmin=top_values.min(), vmax=top_values.max())

    # ── Bottom-N indices ────────────────────────────────────────────────
    if bottom_n > 0:
        bottom_indices = df[color_col].nsmallest(bottom_n).index.tolist()
        bottom_values = df.loc[bottom_indices, color_col]
        bmin, bmax = bottom_values.min(), bottom_values.max()
        if bmax - bmin < 1e-9:
            # All bottom values identical → spread colors evenly
            norm_bottom = Normalize(vmin=bmin - 0.01, vmax=bmax + 0.01)
        else:
            norm_bottom = Normalize(vmin=bmin, vmax=bmax)
    else:
        bottom_indices = []

    all_highlighted = set(top_indices) | set(bottom_indices)

    # ── Figure setup ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize, facecolor=pal["bg"])
    ax.set_facecolor(pal["bg"])

    # ── Helper: build full polyline points ──────────────────────────────
    def _build_polyline(run_idx, npts=60):
        xs, ys = [], []
        for i in range(n_dims - 1):
            bx, by = bezier_curve(
                (x_positions[i], jittered_data[run_idx, i]),
                (x_positions[i + 1], jittered_data[run_idx, i + 1]),
                num_points=npts,
            )
            if i == 0:
                xs.extend(bx)
                ys.extend(by)
            else:
                xs.extend(bx[1:])
                ys.extend(by[1:])
        return np.array(xs), np.array(ys)

    # ── Draw background lines ──────────────────────────────────────────
    for run_idx in range(n_runs):
        if run_idx not in all_highlighted:
            xs, ys = _build_polyline(run_idx, npts=40)
            ax.plot(
                xs,
                ys,
                color=pal["bg_line"],
                alpha=background_alpha,
                linewidth=linewidth * 0.8,
                solid_capstyle="round",
                zorder=1,
            )

    # ── Helper: draw a highlighted line with optional glow ─────────────
    def _draw_highlighted(run_idx, color, zorder, dashed=False, glow_alpha=0.20):
        xs, ys = _build_polyline(run_idx, npts=60)
        ls = (0, (6, 3)) if dashed else "-"
        if gradient_lines:
            points = np.column_stack([xs, ys])
            segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
            lc_glow = LineCollection(
                segments,
                colors=[(color[0], color[1], color[2], glow_alpha)],
                linewidths=linewidth * 6,
                capstyle="round",
                zorder=zorder,
            )
            ax.add_collection(lc_glow)
            ax.plot(
                xs,
                ys,
                color=color,
                alpha=0.95,
                linewidth=linewidth * 2.5,
                solid_capstyle="round",
                linestyle=ls,
                zorder=zorder + 1,
            )
        else:
            ax.plot(
                xs,
                ys,
                color=color,
                alpha=0.9,
                linewidth=linewidth * 2.5,
                solid_capstyle="round",
                linestyle=ls,
                zorder=zorder,
            )

    # ── Draw bottom-N lines ────────────────────────────────────────────
    for rank, run_idx in enumerate(reversed(bottom_indices)):
        c = bottom_cmap(norm_bottom(df.loc[run_idx, color_col]))
        _draw_highlighted(run_idx, c, zorder=5 + rank, dashed=True, glow_alpha=0.06)

    # ── Draw top-N lines (on top of bottom) ────────────────────────────
    for rank, run_idx in enumerate(reversed(top_indices)):
        c = highlight_cmap(norm_top(df.loc[run_idx, color_col]))
        _draw_highlighted(run_idx, c, zorder=10 + rank)

    # ── Draw axes and tick labels ──────────────────────────────────────
    for i, dim in enumerate(dimensions):
        # Axis line
        ax.plot(
            [i, i],
            [-0.02, 1.02],
            color=pal["axis"],
            linewidth=1.2,
            zorder=20,
            solid_capstyle="round",
        )

        vals = np.array(dim["values"], dtype=float)
        vmin, vmax = vals.min(), vals.max()

        for tv, tt in zip(dim["tickvals"], dim["ticktext"]):
            y_pos = 0.5 if vmax - vmin < 1e-9 else (tv - vmin) / (vmax - vmin)
            # Small tick mark
            ax.plot(
                [i - 0.03, i + 0.03],
                [y_pos, y_pos],
                color=pal["axis"],
                linewidth=1.0,
                zorder=25,
            )
            # Label on the left for all but last axis, right for last
            if i < n_dims - 1:
                ha, offset = "right", -0.06
            else:
                ha, offset = "left", 0.06
            ax.text(
                i + offset,
                y_pos,
                str(tt).upper(),
                ha=ha,
                va="center",
                fontsize=12,
                color=pal["tick"],
                fontfamily="monospace",
                zorder=40,
            )

        # Axis title
        ax.text(
            i,
            1.10,
            dim["name"].upper(),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="600",
            color=pal["title"],
            zorder=40,
        )

    # ── Limits and cleanup ─────────────────────────────────────────────
    ax.set_xlim(-0.5, n_dims - 0.5)
    ax.set_ylim(-0.08, 1.20)
    ax.axis("off")

    # ── Colorbar (top-N only) ─────────────────────────────────────────
    sm_top = plt.cm.ScalarMappable(cmap=highlight_cmap, norm=norm_top)
    sm_top.set_array([])
    cbar = plt.colorbar(sm_top, ax=ax, shrink=0.45, pad=0.02, aspect=20)
    cbar.set_label(
        f"{color_col} (Top {top_n})",
        fontsize=10,
        color=pal["label"],
        labelpad=8,
    )
    cbar.ax.tick_params(colors=pal["tick"], labelsize=8)
    cbar.outline.set_edgecolor(pal["axis"])
    cbar.outline.set_linewidth(0.5)

    # ── Legend ─────────────────────────────────────────────────────────
    if legend:
        handles = []
        # Top runs
        for rank, run_idx in enumerate(top_indices):
            row = df.loc[run_idx]
            c = highlight_cmap(norm_top(row[color_col]))
            (h,) = ax.plot(
                [],
                [],
                color=c,
                linewidth=3,
                solid_capstyle="round",
                label=f"Best #{rank + 1}: {row[color_col]:.3f}",
            )
            handles.append(h)

        # Bottom runs (dashed)
        if bottom_n > 0:
            for rank, run_idx in enumerate(bottom_indices):
                row = df.loc[run_idx]
                c = bottom_cmap(norm_bottom(row[color_col]))
                (h,) = ax.plot(
                    [],
                    [],
                    color=c,
                    linewidth=3,
                    solid_capstyle="round",
                    linestyle=(0, (6, 3)),
                    label=f"Worst #{rank + 1}: {row[color_col]:.3f}",
                )
                handles.append(h)

        leg = ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1, 1.0),
            title=f"Top {top_n}" + (f" / Bottom {bottom_n}" if bottom_n else ""),
            fontsize=9,
            title_fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=False,
            edgecolor=pal["axis"],
            facecolor=pal["bg"],
            labelcolor=pal["label"],
        )
        leg.get_title().set_color(pal["title"])

    fig.tight_layout()
    return fig, ax
