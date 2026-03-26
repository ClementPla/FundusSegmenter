import streamlit as st
from torch.utils.data import DataLoader
from multistyleseg.data.synthetic import (
    AnnotationType,
    swap_annotators_tensor,
)
from multistyleseg.experiments.synthetic.ui_utils import global_sidebar

from multistyleseg.experiments.synthetic.utils import get_dataset, compute_IoU
from multistyleseg.experiments.synthetic.train import get_joint_model
from torchvision.utils import make_grid
import torch
import pandas as pd

st.set_page_config(page_title="Causal Pathways Experiment", layout="wide")
global_sidebar()
with st.sidebar:
    n_shapes: int = st.slider(
        "Number of shapes", min_value=1, max_value=5, value=1, step=1, key="n_shapes"
    )


dataloader = DataLoader(
    get_dataset(
        n_shapes=st.session_state.n_shapes,
        return_all_styles=True,
    ),
    batch_size=32,
    num_workers=8,
    pin_memory=True,
)
model = get_joint_model(task=st.session_state.task)
model.eval()
data = next(iter(dataloader))
annotation_types = data["expected_style"].view(-1, 1, 1)
actual_masks = (
    data["fine_mask"] * (1 - annotation_types) + data["coarse_mask"] * annotation_types
)

coarse_images = swap_annotators_tensor(data, AnnotationType.FINE, AnnotationType.COARSE)
fine_images = swap_annotators_tensor(data, AnnotationType.COARSE, AnnotationType.FINE)

images = data["image"].cuda()
fine_masks = data["fine_mask"].cuda()
coarse_masks = data["coarse_mask"].cuda()
coarse_images = coarse_images.cuda()
fine_images = fine_images.cuda()

with torch.no_grad():
    fine_preds = model(fine_images).squeeze(1).sigmoid()
    coarse_preds = model(coarse_images).squeeze(1).sigmoid()
tabInference, tabCausal = st.tabs(["Inference Results", "Causal Pathways"])

with tabInference:
    cols = st.columns(3)
    with cols[0]:
        st.header("Image/GT")
        st.image(make_grid(images).permute(1, 2, 0).cpu().numpy())
        st.image(
            make_grid(actual_masks.unsqueeze(1)).permute(1, 2, 0).cpu().numpy() * 255,
        )

        with cols[1]:
            st.header("Converted to Fine")
            st.image(
                make_grid(fine_images).permute(1, 2, 0).cpu().numpy(),
            )
            st.image(
                make_grid(fine_preds.unsqueeze(1)).permute(1, 2, 0).cpu().numpy(),
            )
        with cols[2]:
            st.header("Converted to Coarse")
            st.image(
                make_grid(coarse_images).permute(1, 2, 0).cpu().numpy(),
            )
            st.image(
                make_grid(coarse_preds.unsqueeze(1)).permute(1, 2, 0).cpu().numpy(),
            )
with tabCausal:
    st.header("Causal Pathway Results moving from fine to coarse")

    findices = st.slider(
        "Choose a range of feature indices, features within this range will be swapped",
        min_value=0,
        max_value=len(model.encoder.out_channels) - 1,
        value=(0, len(model.encoder.out_channels) - 1),
    )
    cols = st.columns(2)

    with torch.inference_mode():
        fine_features = model.encoder(fine_images)
        coarse_features = model.encoder(coarse_images)

    fpreds_fine_to_coarse = []
    fpreds_coarse_to_fine = []
    with torch.inference_mode():
        mixed_features_1_fine_2_coarse = []
        mixed_features_2_fine_1_coarse = []
        for i in range(len(fine_features)):
            if findices[0] <= i <= findices[1]:
                mixed_features_1_fine_2_coarse.append(coarse_features[i])
                mixed_features_2_fine_1_coarse.append(fine_features[i])
            else:
                mixed_features_1_fine_2_coarse.append(fine_features[i])
                mixed_features_2_fine_1_coarse.append(coarse_features[i])

        mixed_output = model.decoder(mixed_features_1_fine_2_coarse)
        pred = model.segmentation_head(mixed_output).sigmoid()
        fpreds_fine_to_coarse.append(pred)
        mixed_output = model.decoder(mixed_features_2_fine_1_coarse)
        pred = model.segmentation_head(mixed_output).sigmoid()
        fpreds_coarse_to_fine.append(pred)
    with cols[0]:
        st.image(
            make_grid(coarse_preds.unsqueeze(1)).cpu().permute(1, 2, 0).numpy(),
            clamp=True,
            use_container_width=True,
            caption="Original Coarse Predictions",
        )
        st.image(
            make_grid(fine_preds.unsqueeze(1)).cpu().permute(1, 2, 0).numpy(),
            clamp=True,
            use_container_width=True,
            caption="Original Fine Predictions",
        )

    with cols[1]:
        st.image(
            make_grid(fpreds_coarse_to_fine[0]).cpu().permute(1, 2, 0).numpy(),
            clamp=True,
            use_container_width=True,
            caption="Causal Pathway Coarse to Fine Predictions",
        )
        st.image(
            make_grid(fpreds_fine_to_coarse[0]).cpu().permute(1, 2, 0).numpy(),
            clamp=True,
            use_container_width=True,
            caption="Causal Pathway Fine to Coarse Predictions",
        )
