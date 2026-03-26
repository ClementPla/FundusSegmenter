import pandas as pd
import numpy as np
import re


REGION_ORDER = ["OD region", "macula region", "1OD-2OD from macula", "elsewhere"]


def categorize_lesions_by_region(
    lesions_df: pd.DataFrame,
    landmarks_df: pd.DataFrame,
    od_diameter: float = 150.0,
    lesion_order: list[str] | None = None,
) -> pd.DataFrame:
    """Categorize lesions into spatial regions relative to the optic disc and macula.

    Regions (distances measured from lesion centroid):
      - OD region: within 0.5 * od_diameter of the OD centre
      - Macula region: within 1 * od_diameter of the macula centre
      - Perimacular (1-2 OD): between 1 and 2 * od_diameter from the macula centre
      - Elsewhere: everything else

    Priority: OD region > Macula region > Perimacular > Elsewhere

    Parameters
    ----------
    lesions_df : DataFrame
        Columns: image_id, lesion_id (label), centroid, area, …
    landmarks_df : DataFrame
        Columns: image_id, od, od_valid, macula, macula_valid
    od_diameter : float
        Estimated optic-disc diameter in pixels.
    lesion_order : list[str] or None
        Canonical ordering of lesion types for columns.
        If None, sorted alphabetically.

    Returns
    -------
    DataFrame with one row per image and a MultiIndex on columns:
        level 0 – metric:  "count", "total_area", "mean_area"
        level 1 – region:  "OD region", "macula region", "1OD-2OD from macula", "elsewhere"
        level 2 – lesion:  each lesion type
    """

    od_radius = od_diameter / 2.0

    # --- helpers -------------------------------------------------------
    def _parse_coord(val):
        if isinstance(val, np.ndarray):
            return val.astype(float)
        nums = [
            float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(val))
        ]
        return np.array(nums[:2])

    def _parse_centroid(val):
        """regionprops (row, col) → (x, y) to match landmarks."""
        if isinstance(val, tuple):
            return np.array([val[1], val[0]], dtype=float)
        nums = [
            float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(val))
        ]
        return np.array([nums[1], nums[0]], dtype=float)

    def _parse_valid(val):
        if isinstance(val, str):
            return "True" in val
        return bool(val)

    # --- landmark lookup -----------------------------------------------
    landmark_map = {}
    for _, row in landmarks_df.iterrows():
        landmark_map[row["image_id"]] = (
            _parse_coord(row["od"]),
            _parse_coord(row["macula"]),
            _parse_valid(row["od_valid"]),
            _parse_valid(row["macula_valid"]),
        )

    # --- classify each lesion ------------------------------------------
    records = []
    for _, row in lesions_df.iterrows():
        img_id = row["image_id"]
        lesion_type = row["lesion_id"]
        centroid = _parse_centroid(row["centroid"])
        area = float(row["area"]) if "area" in row.index else 0.0

        if img_id not in landmark_map:
            records.append((img_id, lesion_type, "elsewhere", area))
            continue

        od_xy, mac_xy, od_valid, mac_valid = landmark_map[img_id]
        dist_od = np.linalg.norm(centroid - od_xy) if od_valid else np.inf
        dist_mac = np.linalg.norm(centroid - mac_xy) if mac_valid else np.inf

        if dist_od <= od_radius:
            region = "OD region"
        elif dist_mac <= od_diameter:
            region = "macula region"
        elif dist_mac <= 2 * od_diameter:
            region = "1OD-2OD from macula"
        else:
            region = "elsewhere"

        records.append((img_id, lesion_type, region, area))

    df = pd.DataFrame(records, columns=["image_id", "lesion_type", "region", "area"])
    # --- aggregate per (image, lesion_type, region) --------------------
    agg = (
        df.groupby(["image_id", "lesion_type", "region"])
        .agg(
            count=("area", "count"),
            total_area=("area", "sum"),
            mean_area=("area", "mean"),
        )
        .reset_index()
    )

    # --- melt metrics into a single column for clean pivoting ----------
    agg_melted = agg.melt(
        id_vars=["image_id", "lesion_type", "region"],
        value_vars=["count", "total_area", "mean_area"],
        var_name="metric",
        value_name="value",
    )

    # --- pivot to MultiIndex columns -----------------------------------
    result = agg_melted.pivot_table(
        index="image_id",
        columns=["metric", "region", "lesion_type"],
        values="value",
        fill_value=0,
    )

    # --- reindex to guarantee all images, regions, lesions present -----
    if lesion_order is None:
        lesion_types = sorted(lesions_df["lesion_id"].unique())
    else:
        lesion_types = [l for l in lesion_order if l in lesions_df["lesion_id"].values]

    metrics = ["count", "total_area", "mean_area"]

    all_images = sorted(
        set(lesions_df["image_id"].unique()) | set(landmarks_df["image_id"].unique())
    )

    multi_cols = pd.MultiIndex.from_product(
        [metrics, REGION_ORDER, lesion_types],
        names=["metric", "region", "lesion"],
    )

    result = result.reindex(index=all_images, columns=multi_cols, fill_value=0)
    result.index.name = "image_id"
    result["count"] = result["count"].astype(int)

    return result


def attach_gt(result_df: pd.DataFrame, gt_df: pd.DataFrame) -> pd.DataFrame:
    """Add ground-truth DR grade to the lesion-region result DataFrame.

    Parameters
    ----------
    result_df : DataFrame
        Output of categorize_lesions_by_region (indexed by image_id).
    gt_df : DataFrame
        Columns: image_id, label

    Returns
    -------
    DataFrame with a ("label", "", "") column added,
    restricted to images present in gt_df and ordered to match it.
    """
    gt = gt_df.set_index("image_id")
    # Keep only images present in both, ordered as in gt
    common = gt.index.intersection(result_df.index)
    out = result_df.loc[common].copy()
    out[("label", "", "")] = gt.loc[common, "label"]
    return out
