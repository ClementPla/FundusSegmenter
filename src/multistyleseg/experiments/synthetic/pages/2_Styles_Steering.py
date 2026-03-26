import streamlit as st
from torch.utils.data import DataLoader
from multistyleseg.data.synthetic import (
    AnnotationType,
    swap_annotators_tensor,
)
from multistyleseg.experiments.synthetic.utils import (
    get_dataset,
    compute_IoU,
    get_empirical_style_steering,
)
from multistyleseg.experiments.synthetic.ui_utils import global_sidebar

from multistyleseg.experiments.synthetic.misc import jacobian_verification
from multistyleseg.experiments.synthetic.train import get_joint_model
from torchvision.utils import make_grid
import torch
import pandas as pd

st.set_page_config(page_title="Styles Steering Experiment", layout="wide")

global_sidebar()

dataloader = DataLoader(
    get_dataset(
        n_shapes=1,
        return_all_styles=True,
    ),
    batch_size=128,
    num_workers=8,
    pin_memory=True,
)
model = get_joint_model(task=st.session_state.task)
model.eval()

findex = st.slider(
    "Feature index to analyze",
    min_value=0,
    max_value=len(model.encoder.out_channels) - 1,
    value=3,
    step=1,
    key="feature_index",
)
empirical_steering = get_empirical_style_steering(
    model, dataloader, findex=findex, n_iterations=10, task=st.session_state.task
)

# Random direction

random_direction = torch.randn_like(empirical_steering)
random_direction /= random_direction.norm()
data = next(iter(dataloader))
images = data["image"]
fine_masks = data["fine_mask"]
coarse_masks = data["coarse_mask"]
annotation_types = data["expected_style"].view(-1, 1, 1)

coarse_images = swap_annotators_tensor(data, AnnotationType.FINE, AnnotationType.COARSE)
fine_images = swap_annotators_tensor(data, AnnotationType.COARSE, AnnotationType.FINE)
coarse_images = coarse_images.cuda()
fine_images = fine_images.cuda()
alpha = st.slider(
    "Steering magnitude",
    min_value=0.0,
    max_value=50.0,
    value=5.0,
    step=0.1,
    key="steering_magnitude",
)
with torch.no_grad():
    coarse_preds = model(coarse_images).sigmoid()

    coarse_features = model.encoder(coarse_images)
    fine_features = model.encoder(fine_images)

    mixed_empirical_steered = []
    mixed_real_steered = []
    mixed_random_steered = []
    for layer in range(len(coarse_features)):
        if layer != findex:
            mixed_empirical_steered.append(coarse_features[layer])
            mixed_random_steered.append(coarse_features[layer])
            mixed_real_steered.append(coarse_features[layer])
        elif layer == findex:
            real_difference = fine_features[layer].mean((2, 3)) - coarse_features[
                layer
            ].mean((2, 3))
            real_difference = real_difference / real_difference.norm(
                dim=1, keepdim=True
            )
            B = coarse_features[layer].shape[0]
            alignment = torch.cosine_similarity(
                real_difference, empirical_steering.view(1, -1), dim=1
            ).mean()
            mixed_empirical_steered.append(
                coarse_features[layer] + alpha * empirical_steering.view(1, -1, 1, 1)
            )
            mixed_random_steered.append(
                coarse_features[layer] + alpha * random_direction.view(1, -1, 1, 1)
            )
            mixed_real_steered.append(
                coarse_features[layer] + alpha * real_difference.view(B, -1, 1, 1)
            )

    preds_steered = model.decoder(mixed_empirical_steered)
    preds_random_steered = model.decoder(mixed_random_steered)
    preds_real_steered = model.decoder(mixed_real_steered)
    preds_steered = model.segmentation_head(preds_steered).sigmoid()
    preds_random_steered = model.segmentation_head(preds_random_steered).sigmoid()
    preds_real_steered = model.segmentation_head(preds_real_steered).sigmoid()

tabs = st.tabs(
    [
        "Original vs Steered",
        "Jacobian Analysis",
    ]
)
with tabs[0]:
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(
            make_grid(coarse_preds).permute(1, 2, 0).cpu().numpy(),
            caption="Original Coarse Predictions",
        )
    with col2:
        st.image(
            make_grid(preds_real_steered).permute(1, 2, 0).cpu().numpy(),
            caption="Real Direction Steering",
        )
    with col3:
        st.image(
            make_grid(preds_steered).permute(1, 2, 0).cpu().numpy(),
            caption="Empirical Direction Steering",
        )
    with col4:
        st.image(
            make_grid(preds_random_steered).permute(1, 2, 0).cpu().numpy(),
            caption="Random Direction Steering",
        )
    with col5:
        st.write(f"Alignment with empirical direction: {alignment:.4f}")
        st.bar_chart(
            pd.DataFrame(
                {
                    "IoU Coarse": [compute_IoU(coarse_preds, fine_masks.cuda())],
                    "IoU Real Steered": [
                        compute_IoU(preds_real_steered, fine_masks.cuda())
                    ],
                    "IoU Empirical Steered": [
                        compute_IoU(preds_steered, fine_masks.cuda())
                    ],
                    "IoU Random Steered": [
                        compute_IoU(preds_random_steered, fine_masks.cuda())
                    ],
                }
            ),
            stack=False,
            y_label="IoU",
            x_label="Setup",
        )
with tabs[1]:
    jacobian_verification(
        model,
        fine_images,
        coarse_images,
        findex,
        empirical_steering,
    )
