from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from tqdm.auto import tqdm

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    F1Score,
    JaccardIndex,
    Recall,
    Precision,
    Accuracy,
    Specificity,
    AUROC,
    PrecisionRecallCurve,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES = 5
IGNORE_INDEX = 0


# ===================================================================
# PART 1 — Pixel-level evaluation (refactored)
# ===================================================================


def build_pixel_metrics(
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
    prc_thresholds: int = 33,
) -> MetricCollection:
    """Instantiate the MetricCollection used for pixel-level evaluation."""
    common = dict(
        num_classes=num_classes,
        task="multiclass",
        average="none",
        ignore_index=ignore_index,
    )
    return MetricCollection(
        {
            "Dice": F1Score(**common),
            "Recall": Recall(**common),
            "Precision": Precision(**common),
            "Accuracy": Accuracy(**common),
            "Specificity": Specificity(**common),
            "AUROC": AUROC(**common),
            "IoU": JaccardIndex(**common),
            "prc": PrecisionRecallCurve(
                num_classes=num_classes,
                task="multiclass",
                average=None,
                thresholds=prc_thresholds,
            ),
        }
    )


def _compute_pr_auc(prc_result, num_classes: int = NUM_CLASSES) -> list[float]:
    """Compute per-class PR-AUC from a PrecisionRecallCurve result."""
    p, r, _ = prc_result
    pr_aucs = []
    for i in range(1, num_classes):
        prec, rec = p[i], r[i]
        valid = ~(torch.isnan(prec) | torch.isnan(rec))
        pr_aucs.append(torch.trapz(prec[valid], rec[valid]).abs().item())
    return pr_aucs


@torch.inference_mode()
def evaluate_model_pixel(
    model: torch.nn.Module,
    test_dataloaders: Sequence,
    all_classes: Sequence,
    num_classes: int = NUM_CLASSES,
    device: str = "cuda",
) -> pd.DataFrame:
    """
    Run pixel-level evaluation of *model* over a list of dataloaders.

    Returns a MultiIndex DataFrame:
        rows    = datasets
        columns = (Lesion, Metric)  with a MEAN lesion column appended.
    """
    model = model.eval().to(device)
    metrics = build_pixel_metrics(num_classes=num_classes).to(device)
    metric_names = [k for k in metrics.keys() if k != "prc"]

    rows: dict[str, list] = {
        "PR_AUC": [],
        "Lesion": [],
        "Dataset": [],
        **{m: [] for m in metric_names},
    }

    for dataloader in test_dataloaders:
        tag = dataloader.dataset.tag
        for batch in tqdm(dataloader, desc=f"Eval on {tag.name}"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device).long().clamp(0, num_classes - 1)
            logits = model(images).softmax(dim=1)
            metrics.update(logits, masks)

        result = metrics.compute()
        metrics.reset()

        pr_aucs = _compute_pr_auc(result["prc"], num_classes)

        for i, lesion in enumerate(all_classes):
            rows["PR_AUC"].append(pr_aucs[i])
            rows["Lesion"].append(lesion.name)
            rows["Dataset"].append(tag)
            for m in metric_names:
                rows[m].append(result[m][i + 1].item())

    df = pd.DataFrame(rows)
    df = df.pivot(
        index="Dataset",
        columns="Lesion",
        values=metric_names + ["PR_AUC"],
    )
    df = df.swaplevel(axis=1).sort_index(axis=1)

    # Append MEAN across lesions
    mean_df = df.groupby(level=1, axis=1).mean()
    mean_df.columns = pd.MultiIndex.from_tuples(
        [("MEAN", m) for m in mean_df.columns],
        names=df.columns.names,
    )
    df = pd.concat([df, mean_df], axis=1)
    return df


# ===================================================================
# PART 2 — Detection-level analysis (new)
# ===================================================================


@dataclass
class BlobMatch:
    """Result of matching one GT blob to predicted blobs."""

    gt_label: int
    gt_area: int
    matched: bool
    pred_label: int | None = None
    pred_area: int | None = None
    iou: float = 0.0


@dataclass
class DetectionResult:
    """Per-image, per-class detection bookkeeping."""

    class_id: int
    dataset: str
    image_idx: int
    n_gt: int = 0
    n_pred: int = 0
    tp: int = 0  # GT blobs that were detected
    fn: int = 0  # GT blobs missed
    fp: int = 0  # predicted blobs with no GT match
    gt_areas: list[int] = field(default_factory=list)
    pred_areas_tp: list[int] = field(default_factory=list)  # matched pred
    gt_areas_tp: list[int] = field(default_factory=list)  # matched GT
    pred_areas_fp: list[int] = field(default_factory=list)  # spurious pred
    gt_areas_fn: list[int] = field(default_factory=list)  # missed GT
    ious: list[float] = field(default_factory=list)


def match_blobs(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    iou_threshold: float = 0.1,
    min_blob_area: int = 0,
) -> DetectionResult:
    """
    Match connected components between a binary GT mask and a binary
    predicted mask for **one class** in **one image**.

    Matching strategy (greedy, GT-centric):
      1. Label connected components in both masks.
      2. For each GT blob, find all overlapping pred blobs.
      3. Pick the pred blob with highest IoU; if IoU >= threshold → TP.
      4. A pred blob can only be matched once; leftover preds → FP.

    Parameters
    ----------
    gt_mask, pred_mask : 2-D bool / uint8 arrays (H, W).
    iou_threshold : minimum IoU to accept a match.
    min_blob_area : ignore blobs smaller than this (pixels).
    """
    result = DetectionResult(class_id=0, dataset="", image_idx=0)

    gt_labels, n_gt = ndimage.label(gt_mask)
    pred_labels, n_pred = ndimage.label(pred_mask)

    gt_areas = ndimage.sum(gt_mask, gt_labels, range(1, n_gt + 1)).astype(int)
    pred_areas = ndimage.sum(pred_mask, pred_labels, range(1, n_pred + 1)).astype(int)

    # Filter by min area
    valid_gt = {i + 1 for i, a in enumerate(gt_areas) if a >= min_blob_area}
    valid_pred = {i + 1 for i, a in enumerate(pred_areas) if a >= min_blob_area}

    result.n_gt = len(valid_gt)
    result.n_pred = len(valid_pred)
    result.gt_areas = [int(gt_areas[i - 1]) for i in valid_gt]

    matched_pred: set[int] = set()

    # For each GT blob, find best overlapping pred blob
    for g in sorted(valid_gt):
        g_mask = gt_labels == g
        # Which pred labels overlap?
        overlapping = set(pred_labels[g_mask].ravel()) - {0}
        overlapping &= valid_pred

        best_iou, best_p = 0.0, None
        for p in overlapping:
            p_mask = pred_labels == p
            inter = np.count_nonzero(g_mask & p_mask)
            union = np.count_nonzero(g_mask | p_mask)
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_p = iou, p

        if best_iou >= iou_threshold and best_p is not None:
            result.tp += 1
            result.ious.append(best_iou)
            result.gt_areas_tp.append(int(gt_areas[g - 1]))
            result.pred_areas_tp.append(int(pred_areas[best_p - 1]))
            matched_pred.add(best_p)
        else:
            result.fn += 1
            result.gt_areas_fn.append(int(gt_areas[g - 1]))

    # Unmatched predictions → FP
    fp_set = valid_pred - matched_pred
    result.fp = len(fp_set)
    result.pred_areas_fp = [int(pred_areas[p - 1]) for p in fp_set]

    return result


@torch.inference_mode()
def evaluate_model_detection(
    model: torch.nn.Module,
    test_dataloaders: Sequence,
    all_classes: Sequence,
    num_classes: int = NUM_CLASSES,
    iou_threshold: float = 0.1,
    min_blob_area: int = 0,
    device: str = "cuda",
) -> list[DetectionResult]:
    """
    Run detection-level evaluation.

    For each image × class, produces a DetectionResult with:
      - TP / FP / FN counts
      - areas of matched, missed, and spurious blobs
      - per-match IoU

    Returns a flat list of DetectionResult (one per image × class × dataset).
    """
    model = model.eval().to(device)
    results: list[DetectionResult] = []

    for dataloader in test_dataloaders:
        tag = dataloader.dataset.tag
        img_idx = 0
        for batch in tqdm(dataloader, desc=f"Detection eval on {tag.name}"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device).long().clamp(0, num_classes - 1)
            preds = model(images).softmax(dim=1).argmax(dim=1)  # (B, H, W)

            masks_np = masks.cpu().numpy()
            preds_np = preds.cpu().numpy()

            for b in range(masks_np.shape[0]):
                for cls_idx, cls in enumerate(all_classes):
                    class_id = cls_idx + 1  # skip background=0
                    gt_bin = masks_np[b] == class_id
                    pred_bin = preds_np[b] == class_id

                    det = match_blobs(
                        gt_bin,
                        pred_bin,
                        iou_threshold=iou_threshold,
                        min_blob_area=min_blob_area,
                    )
                    det.class_id = class_id
                    det.dataset = tag.name if hasattr(tag, "name") else str(tag)
                    det.image_idx = img_idx + b
                    results.append(det)

            img_idx += masks_np.shape[0]

    return results


# ===================================================================
# PART 3 — Aggregation helpers
# ===================================================================


def aggregate_detection_results(
    results: list[DetectionResult],
    all_classes: Sequence,
) -> pd.DataFrame:
    """
    Aggregate DetectionResult list into a summary DataFrame.

    For each (Dataset, Lesion) pair, reports:
      - total GT / Pred blob counts
      - TP, FP, FN
      - Sensitivity (TP / (TP+FN))  — "how many GT blobs did we find?"
      - Precision   (TP / (TP+FP))  — "how many detections are real?"
      - F1
      - detection_ratio = n_pred / n_gt  — >1 means over-detection
      - mean_area_ratio = mean(pred_area_tp / gt_area_tp) per matched pair
      - median GT area for TP vs FN (are we missing the small ones?)
    """
    rows = []
    for r in results:
        cls_name = all_classes[r.class_id - 1].name
        rows.append(
            {
                "Dataset": r.dataset,
                "Lesion": cls_name,
                "n_gt": r.n_gt,
                "n_pred": r.n_pred,
                "TP": r.tp,
                "FP": r.fp,
                "FN": r.fn,
                # keep raw areas for later distribution plots
                "gt_areas_tp": r.gt_areas_tp,
                "pred_areas_tp": r.pred_areas_tp,
                "gt_areas_fn": r.gt_areas_fn,
                "pred_areas_fp": r.pred_areas_fp,
                "ious": r.ious,
            }
        )

    raw = pd.DataFrame(rows)

    # --- per (Dataset, Lesion) aggregation ---
    def _agg(group: pd.DataFrame) -> pd.Series:
        tp = group["TP"].sum()
        fp = group["FP"].sum()
        fn = group["FN"].sum()
        n_gt = group["n_gt"].sum()
        n_pred = group["n_pred"].sum()

        sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        f1 = 2 * sens * prec / (sens + prec) if (sens + prec) > 0 else float("nan")

        det_ratio = n_pred / n_gt if n_gt > 0 else float("nan")

        # Matched blob size ratio
        gt_tp = (
            np.concatenate(group["gt_areas_tp"].values)
            if any(len(x) > 0 for x in group["gt_areas_tp"])
            else np.array([])
        )
        pred_tp = (
            np.concatenate(group["pred_areas_tp"].values)
            if any(len(x) > 0 for x in group["pred_areas_tp"])
            else np.array([])
        )
        gt_fn = (
            np.concatenate(group["gt_areas_fn"].values)
            if any(len(x) > 0 for x in group["gt_areas_fn"])
            else np.array([])
        )
        pred_fp = (
            np.concatenate(group["pred_areas_fp"].values)
            if any(len(x) > 0 for x in group["pred_areas_fp"])
            else np.array([])
        )

        if len(gt_tp) > 0 and len(pred_tp) > 0:
            area_ratios = pred_tp / np.maximum(gt_tp, 1)
            mean_area_ratio = float(np.mean(area_ratios))
            median_area_ratio = float(np.median(area_ratios))
        else:
            mean_area_ratio = median_area_ratio = float("nan")

        median_gt_area_tp = float(np.median(gt_tp)) if len(gt_tp) > 0 else float("nan")
        median_gt_area_fn = float(np.median(gt_fn)) if len(gt_fn) > 0 else float("nan")

        median_pred_area_fp = (
            float(np.median(pred_fp)) if len(pred_fp) > 0 else float("nan")
        )

        all_ious = (
            np.concatenate(group["ious"].values)
            if any(len(x) > 0 for x in group["ious"])
            else np.array([])
        )
        mean_iou = float(np.mean(all_ious)) if len(all_ious) > 0 else float("nan")

        return pd.Series(
            {
                "n_gt": n_gt,
                "n_pred": n_pred,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "Sensitivity": sens,
                "Precision": prec,
                "F1": f1,
                "detection_ratio": det_ratio,
                "mean_area_ratio": mean_area_ratio,
                "median_area_ratio": median_area_ratio,
                "median_gt_area_TP": median_gt_area_tp,
                "median_gt_area_FN": median_gt_area_fn,
                "median_pred_area_FP": median_pred_area_fp,
                "mean_IoU_matched": mean_iou,
            }
        )

    summary = raw.groupby(["Dataset", "Lesion"]).apply(_agg).reset_index()
    return summary


def detection_size_distributions(
    results: list[DetectionResult],
    all_classes: Sequence,
) -> pd.DataFrame:
    """
    Flatten all blob areas into a long-form DataFrame suitable for
    violin / box / histogram plots.

    Columns: Dataset, Lesion, category (TP_gt, TP_pred, FN, FP), area
    """
    rows = []
    for r in results:
        cls_name = all_classes[r.class_id - 1].name
        ds = r.dataset
        for a in r.gt_areas_tp:
            rows.append(
                {"Dataset": ds, "Lesion": cls_name, "category": "TP_gt", "area": a}
            )
        for a in r.pred_areas_tp:
            rows.append(
                {"Dataset": ds, "Lesion": cls_name, "category": "TP_pred", "area": a}
            )
        for a in r.gt_areas_fn:
            rows.append(
                {"Dataset": ds, "Lesion": cls_name, "category": "FN_gt", "area": a}
            )
        for a in r.pred_areas_fp:
            rows.append(
                {"Dataset": ds, "Lesion": cls_name, "category": "FP_pred", "area": a}
            )

    return pd.DataFrame(rows)
