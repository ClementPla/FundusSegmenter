import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from pathlib import Path
from fundus_data_toolkit.utils.composer import get_generic_composer
import re
from multistyleseg.analysis.utils import DATASET_COLORS, LesionsUtils
from collections import defaultdict


REGION_STYLES = {
    "OD region": {"color": "lime", "linestyle": "-"},
    "macula region": {"color": "cyan", "linestyle": "-"},
    "1OD-2OD from macula": {"color": "cyan", "linestyle": "--"},
}


def _parse_coord(val):
    if isinstance(val, np.ndarray):
        return val.astype(float)
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(val))]
    return np.array(nums[:2])


def _parse_centroid(val):
    if isinstance(val, tuple):
        return np.array(val, dtype=float)
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(val))]
    return np.array(nums[:2])


def _parse_contours(val):
    """Parse the contours column back into a list of (N,2) arrays."""
    if isinstance(val, list) and len(val) > 0 and isinstance(val[0], np.ndarray):
        return val
    s = str(val)
    # Extract individual array blocks
    blocks = re.findall(r"array\(\[(.*?)\]\)", s, re.DOTALL)
    contours = []
    for block in blocks:
        rows = re.findall(r"\[([\d\s\.\-\+eE,]+)\]", block)
        pts = []
        for row in rows:
            nums = [
                float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", row)
            ]
            if len(nums) == 2:
                pts.append(nums)
        if pts:
            contours.append(np.array(pts))
    return contours


def plot_image_lesions(
    root_imgs,
    image_id,
    lesion_df,
    od_mac_df,
    img_size=(1024, 1024),
    od_diameter=150,
    show_region_counts=False,
    show_regions=True,
    ax=None,
    include_title=False,
    save_file=None,
    fig_size=5,
):
    """Plot an image with its lesions and OD/Macula landmarks.

    Parameters
    ----------
    root_imgs : str or Path
        Directory containing the fundus images.
    image_id : str
        Image identifier.
    lesion_df : DataFrame
        Lesion dataframe (image_id, lesion_id, centroid, contours, coordinates …).
    od_mac_df : DataFrame
        Landmarks dataframe (image_id, od, od_valid, macula, macula_valid).
    img_size : tuple
        Target image size used by the composer.
    od_diameter : float
        Optic-disc diameter in pixels (after resize to img_size).
    show_region_counts : bool
        If True, display a text box with lesion counts per region.
    show_regions : bool
        If True, draw the OD / macula / perimacular region circles.
    ax : matplotlib Axes or None
        If provided, plot on this axes; otherwise create a new figure.
    """
    # --- load and preprocess image -------------------------------------
    if "laterality" in lesion_df.columns:
        probably_hmr = True
        img_path = next(
            (
                Path(root_imgs)
                / image_id
                / lesion_df[lesion_df["image_id"] == image_id]["laterality"].iloc[0]
            ).glob("*.*"),
            None,
        )
    else:
        probably_hmr = False
        img_path = next(Path(root_imgs).glob(f"{image_id}.*"), None)

    if img_path is None:
        print(f"Image {image_id} not found in {root_imgs}")
        return

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR_RGB)
    composer = get_generic_composer(img_size, precise=not probably_hmr)
    composer.ops = composer.ops[:-2]
    img = composer(image=img)["image"]

    # --- set up axes ---------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
    ax.imshow(img)
    if include_title:
        ax.set_title(f"Image ID: {image_id}")
    ax.axis("off")

    # --- landmarks -----------------------------------------------------
    lm_row = od_mac_df[od_mac_df["image_id"] == image_id]
    od_xy, mac_xy = None, None
    od_valid = mac_valid = False

    if len(lm_row):
        lm_row = lm_row.iloc[0]
        od_xy = _parse_coord(lm_row["od"])
        mac_xy = _parse_coord(lm_row["macula"])
        od_valid = bool(
            lm_row["od_valid"]
            if not isinstance(lm_row["od_valid"], str)
            else "True" in str(lm_row["od_valid"])
        )
        mac_valid = bool(
            lm_row["macula_valid"]
            if not isinstance(lm_row["macula_valid"], str)
            else "True" in str(lm_row["macula_valid"])
        )

    od_radius = od_diameter / 2.0

    if show_regions:
        if od_valid and od_xy is not None:
            ax.add_patch(
                Circle(
                    (od_xy[0], od_xy[1]),
                    od_radius,
                    fill=False,
                    edgecolor="lime",
                    linewidth=1.5,
                    linestyle="-",
                    label="OD region",
                )
            )
            ax.plot(*od_xy, "x", color="lime", markersize=8, markeredgewidth=2)

        if mac_valid and mac_xy is not None:
            ax.add_patch(
                Circle(
                    (mac_xy[0], mac_xy[1]),
                    od_diameter,
                    fill=False,
                    edgecolor="cyan",
                    linewidth=1.5,
                    linestyle="-",
                    label="Macula region (1 OD)",
                )
            )
            ax.add_patch(
                Circle(
                    (mac_xy[0], mac_xy[1]),
                    2 * od_diameter,
                    fill=False,
                    edgecolor="cyan",
                    linewidth=1.2,
                    linestyle="--",
                    label="Perimacular (2 OD)",
                )
            )
            ax.plot(*mac_xy, "+", color="cyan", markersize=10, markeredgewidth=2)

    # --- draw lesions --------------------------------------------------
    img_lesions = lesion_df[lesion_df["image_id"] == image_id]
    img_lesions.sort_values(
        "lesion_id", key=LesionsUtils.sort_key_from_series, inplace=True
    )  # Inplace sort by lesion type

    # Make sure the lesions are always in the same order in the plots, regardless of the order they appear in the input data.
    legend_handles = {}

    for _, row in img_lesions.iterrows():
        ltype = row["lesion_id"]
        color = LesionsUtils.get_color(ltype)

        # draw contour outline
        contours = _parse_contours(row.get("contours", "[]"))
        for cnt in contours:
            if len(cnt) >= 3:
                poly = plt.Polygon(
                    cnt[:, ::-1],
                    closed=True,
                    fill=True,
                    edgecolor=color,
                    facecolor=color + "80",  # Add alpha to color
                    linewidth=2.0,
                )
                ax.add_patch(poly)

        if ltype not in legend_handles:
            legend_handles[ltype] = mpatches.Patch(color=color, label=ltype, alpha=0.6)

    # --- region counts text box ----------------------------------------
    if show_region_counts and len(img_lesions):
        # Group counts by region → lesion type
        region_counts = defaultdict(lambda: defaultdict(int))
        for _, row in img_lesions.iterrows():
            ltype = row["lesion_id"]
            centroid = _parse_centroid(row["centroid"])
            dist_od = np.linalg.norm(centroid - od_xy[::-1]) if od_valid else np.inf
            dist_mac = np.linalg.norm(centroid - mac_xy[::-1]) if mac_valid else np.inf

            if dist_od <= od_radius:
                region = "OD region"
            elif dist_mac <= od_diameter:
                region = "Macula"
            elif dist_mac <= 2 * od_diameter:
                region = "Perimacular (1-2 OD)"
            else:
                region = "Elsewhere"
            region_counts[region][ltype] += 1

        # Format: one line per region, listing lesion counts inline
        lines = []
        for region in ["OD region", "Macula", "Perimacular (1-2 OD)", "Elsewhere"]:
            if region not in region_counts:
                continue
            items = region_counts[region]
            parts = [
                f"{n} {lt.lower().replace('_', ' ')}"
                for lt, n in sorted(
                    items.items(), key=lambda x: LesionsUtils.sort_key(x[0])
                )
            ]
            lines.append(f"{region}: {', '.join(parts)}")

        textstr = "\n".join(lines)
        props = dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.7)
        ax.text(
            0.02,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            color="white",
            fontfamily="monospace",
            bbox=props,
        )

    # --- legend --------------------------------------------------------
    handles = list(legend_handles.values())
    if show_regions:
        if od_valid:
            handles.append(
                mpatches.Patch(
                    edgecolor="lime", facecolor="none", linewidth=1.5, label="OD region"
                )
            )
        if mac_valid:
            handles.append(
                mpatches.Patch(
                    edgecolor="cyan",
                    facecolor="none",
                    linewidth=1.5,
                    label="Macula (1 OD)",
                )
            )
            handles.append(
                mpatches.Patch(
                    edgecolor="cyan",
                    facecolor="none",
                    linewidth=1.2,
                    linestyle="--",
                    label="Perimacular (2 OD)",
                )
            )
    if handles:
        ax.legend(
            handles=handles,
            loc="lower right",
            fontsize=8,
            framealpha=0.7,
            facecolor="black",
            labelcolor="white",
        )

    plt.gca().set_axis_off()

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if save_file is not None:
        save_file = Path(save_file)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_file, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.show()
