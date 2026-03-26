from multistyleseg.data.synthetic.generator import SynthTriangle
import torch
import streamlit as st
from multistyleseg.data.synthetic import (
    AnnotationType,
    Task,
    swap_annotators_tensor,
)
from torch.autograd.functional import jvp


def get_dataset(
    n_shapes: int = 1,
    return_all_styles: bool = False,
    annotation_type=[AnnotationType.FINE, AnnotationType.COARSE],
):
    return SynthTriangle(
        resolution=64,
        n_shapes=n_shapes,
        return_all_styles=return_all_styles,
        annotation_types=annotation_type,
        task=st.session_state.task,
    )


def compute_IoU(preds, masks, apply_sigmoid: bool = False) -> float:
    """Compute the Intersection over Union (IoU) between predictions and masks tensors (B, H, W) or (B, 1, H, W).

    Args:
        preds (torch.Tensor): _predicted masks (B, H, W) or (B, 1, H, W).
        masks (torch.Tensor): _ground truth masks (B, H, W) or (B, 1, H, W).
    """
    if preds.dim() == 4:
        preds = preds.squeeze(1)
    if masks.dim() == 4:
        masks = masks.squeeze(1)

    if apply_sigmoid:
        preds = torch.sigmoid(preds)

    # If float, threshold at 0.5
    if preds.dtype.is_floating_point:
        preds = (preds > 0.5).float()

    intersection = (preds * masks).sum(dim=(1, 2))
    union = ((preds + masks) > 0).sum(dim=(1, 2))
    iou = intersection / (union + 1e-6)
    return iou.mean().item()


@st.cache_data
def get_empirical_style_steering(_model, _dataloader, findex, n_iterations, task):
    direction = None
    total_samples = 0
    progress_bar = st.progress(0, text="Computing Empirical Style Steering")

    for i in range(n_iterations):
        data = next(iter(_dataloader))
        images = data["image"]
        coarse_images = swap_annotators_tensor(
            data, AnnotationType.FINE, AnnotationType.COARSE
        ).cuda()
        fine_images = swap_annotators_tensor(
            data, AnnotationType.COARSE, AnnotationType.FINE
        ).cuda()
        total_samples += images.shape[0]
        with torch.inference_mode():
            fine_features = _model.encoder(fine_images)[findex]
            coarse_features = _model.encoder(coarse_images)[findex]
            diff = fine_features.mean((2, 3)) - coarse_features.mean((2, 3))
            if direction is None:
                direction = diff
            else:
                direction = direction + diff
        progress_bar.progress((i + 1) / n_iterations)

    direction = direction.sum(0) / total_samples

    progress_bar.empty()
    return direction / direction.norm()


def compute_Gw(encoder, x, probe, target: int, findex):
    """
    Compute G @ w = J_E @ J_E^T @ w efficiently.

    Uses backward-mode AD for J^T @ w, then forward-mode AD for J @ (J^T @ w).
    """

    target = torch.full(
        (x.shape[0],),
        target,
        device=x.device,
    )

    def forward_probe(inp):
        features = encoder(inp)[findex]
        predicted = probe(features).squeeze(1)
        # Return loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            predicted, target.float()
        )
        return loss

    def encoded_pooled(inp):
        features = encoder(inp)[findex]
        pooled = features
        return pooled

    # Step 1: Compute v = J_E^T @ w via standard backward pass
    x_grad = x.clone().requires_grad_(True)
    f = forward_probe(x_grad)

    # grad of (w · f) w.r.t. x gives J_E^T @ w
    v = torch.autograd.grad(f, x_grad)[0]  # [B, C_in, H, W]

    # Step 2: Compute Gw = J_E @ v via forward-mode AD (jvp)
    x_detached = x.clone().detach()
    _, Gw = jvp(encoded_pooled, (x_detached,), (v.detach(),))

    return Gw
