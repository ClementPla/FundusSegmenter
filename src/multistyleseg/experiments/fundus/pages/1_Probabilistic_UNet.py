import streamlit as st
from multistyleseg.models.factory import ModelType
import torch
from torchvision.utils import make_grid, draw_segmentation_masks
from multistyleseg.experiments.fundus.ui.cache import get_dataset, get_cache_model
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="Fundus Probabilistic U-Net Segmentation",
    layout="wide",
)
with st.sidebar:
    model_type = st.selectbox(
        "Select Model Type",
        options=[
            ModelType.PROBABILISTIC_UNET,
            ModelType.HIERARCHICAL_PROBABILISTIC_UNET,
        ],
        format_func=lambda x: x.value,
        index=1,
    )
    cmap = st.selectbox(
        "Select a colormap",
        options=plt.colormaps(),
        index=0,
        help="Select a colormap for the uncertainty map.",
    )
model = get_cache_model(model_type)

datamodule = get_dataset(batch_size=4)

test_dataloaders = datamodule.test_dataloader()
st.title("Fundus Probabilistic U-Net Segmentation")

cols = st.columns(len(test_dataloaders))
for c, dataloader in zip(cols, test_dataloaders):
    c.subheader(f"{dataloader.dataset.id}")
    batch = next(iter(dataloader))
    images = batch["image"].cuda()
    grid_images = make_grid(images, nrow=2, normalize=True)
    with c:
        st.image(grid_images.permute(1, 2, 0).cpu().numpy(), caption="Input Images")

    with st.spinner("Running inference..."):
        model.eval()
        with torch.inference_mode():
            outputs = model.predict_multiple_hypotheses(images, num_samples=10).softmax(
                2
            )
    stds = outputs.std(0).mean(1)

    out = outputs.mean(0)[:, 1:] > 1 / 5
    grid_pred = make_grid(out.float(), nrow=2, normalize=False).bool()
    overlay = draw_segmentation_masks(
        grid_images.cpu(),
        masks=grid_pred.squeeze(1).cpu(),
        alpha=0.5,
        colors=["red", "green", "blue", "yellow"],
    )
    with c:
        st.image(
            overlay.permute(1, 2, 0).cpu().numpy(),
            caption="Predicted Segmentations",
        )
    std_grid = make_grid(
        stds.unsqueeze(1).cpu(), nrow=2, normalize=True, scale_each=True
    )
    with c:
        uncertainty = std_grid.permute(1, 2, 0).cpu().numpy().mean(axis=2)
        uncertainty_cm = (plt.get_cmap(cmap)(uncertainty) * 255).astype(np.uint8)

        st.image(
            uncertainty_cm,
            caption="Prediction Uncertainty (std over samples)",
        )
