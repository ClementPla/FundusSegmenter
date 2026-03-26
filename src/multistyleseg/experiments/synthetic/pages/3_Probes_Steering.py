import streamlit as st
from torch.utils.data import DataLoader
import pandas as pd
from multistyleseg.data.synthetic import (
    AnnotationType,
    swap_annotators_tensor,
)
from multistyleseg.experiments.synthetic.utils import (
    compute_Gw,
    get_dataset,
    get_empirical_style_steering,
)
from multistyleseg.experiments.synthetic.train import (
    get_joint_model,
    get_probe_linear_model,
    adversarial_steer,
)
from multistyleseg.experiments.synthetic.ui_utils import global_sidebar

from torchvision.utils import make_grid
import torch

st.set_page_config(page_title="Causal Pathways Experiment", layout="wide")

global_sidebar()
model = get_joint_model(task=st.session_state.task)
model.eval()

dataloader = DataLoader(
    get_dataset(
        n_shapes=1,
        return_all_styles=True,
    ),
    batch_size=128,
    num_workers=8,
    pin_memory=True,
)

f_index = st.slider(
    "Feature index to analyze",
    min_value=0,
    max_value=len(model.encoder.out_channels) - 1,
    value=3,
    step=1,
    key="feature_index",
)
style_steering = get_empirical_style_steering(
    model,
    dataloader,
    findex=f_index,
    n_iterations=10,
    task=st.session_state.task,
)
probe = get_probe_linear_model(f_index, 1000, task=st.session_state.task)
probe.eval()

data = next(iter(dataloader))
images = data["image"]
fine_masks = data["fine_mask"]
coarse_masks = data["coarse_mask"]
annotation_types = data["expected_style"].view(-1, 1, 1)
actual_masks = fine_masks * (1 - annotation_types) + coarse_masks * annotation_types

coarse_images = swap_annotators_tensor(data, AnnotationType.FINE, AnnotationType.COARSE)
fine_images = swap_annotators_tensor(data, AnnotationType.COARSE, AnnotationType.FINE)
coarse_images = coarse_images.cuda()
fine_images = fine_images.cuda()
with st.sidebar:
    epsilon = st.slider(
        "Adversarial epsilon",
        min_value=0.0,
        max_value=0.2,
        value=0.05,
        step=0.005,
        key="adversarial_epsilon",
    )
    alpha = st.slider(
        "Adversarial alpha",
        min_value=0.0,
        max_value=0.005,
        value=0.001,
        step=0.00001,
        key="adversarial_alpha",
    )
    n_iters = st.slider(
        "Adversarial iterations",
        min_value=1,
        max_value=100,
        value=30,
        step=1,
        key="adversarial_n_iters",
    )
coarse_to_fine = adversarial_steer(
    model,
    probe,
    f_index,
    coarse_images,
    target_style=AnnotationType.FINE.value,
    epsilon=epsilon,
    alpha=alpha,
    n_iters=n_iters,
)
fine_to_coarse = adversarial_steer(
    model,
    probe,
    f_index,
    fine_images,
    target_style=AnnotationType.COARSE.value,
    epsilon=epsilon,
    alpha=alpha,
    n_iters=n_iters,
)

with torch.no_grad():
    features_coarse = model.encoder(coarse_images)
    features_fine = model.encoder(fine_images)
    features_coarse_to_fine = model.encoder(coarse_to_fine)
    features_fine_to_coarse = model.encoder(fine_to_coarse)

    pred_coarse_to_fine = model.decoder(features_coarse_to_fine)
    pred_fine_to_coarse = model.decoder(features_fine_to_coarse)

    pred_coarse_to_fine = model.segmentation_head(pred_coarse_to_fine).sigmoid()
    pred_fine_to_coarse = model.segmentation_head(pred_fine_to_coarse).sigmoid()

tabInference, tabGramSteering, tabDiff = st.tabs(
    ["Adversarial Style Steering", "Gram Matrix Steering", "Diff"]
)
with tabInference:
    col1, col2 = st.columns(2)
    with col1:
        st.header("Coarse to Fine Adversarial Steering")
        st.image(
            make_grid(pred_coarse_to_fine.cpu()).permute(1, 2, 0).numpy(), clamp=True
        )

        diff = features_coarse_to_fine[f_index].mean((2, 3)) - features_coarse[
            f_index
        ].mean((2, 3))
        alignment = (
            torch.cosine_similarity(diff, style_steering.view(1, -1), dim=1)
            .mean()
            .item()
        )
        st.write(f"Alignment with style steering direction: {alignment:.4f}")

    with col2:
        st.header("Fine to Coarse Adversarial Steering")
        st.image(
            make_grid(pred_fine_to_coarse.cpu()).permute(1, 2, 0).numpy(), clamp=True
        )
        diff = features_fine[f_index] - features_fine_to_coarse[f_index]
        alignment = (
            torch.cosine_similarity(
                diff.mean((2, 3)), style_steering.view(1, -1), dim=1
            )
            .mean()
            .item()
        )
        st.write(f"Alignment with style steering direction: {alignment:.4f}")
with tabGramSteering:
    # w is the gradient direction of the probe computed from coarse to fine features
    coarse_features_findex = features_coarse[f_index]
    coarse_features_findex.requires_grad_()
    probe_outputs = probe(coarse_features_findex).squeeze()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        probe_outputs,
        torch.full(
            (probe_outputs.shape[0],),
            AnnotationType.FINE.value,
            device=probe_outputs.device,
        ).float(),
    )
    w = torch.autograd.grad(loss, coarse_features_findex)[0]
    Gw = compute_Gw(
        model.encoder, coarse_images.cuda(), probe, AnnotationType.FINE.value, f_index
    )
    diff = features_coarse_to_fine[f_index].mean((2, 3)) - features_coarse[
        f_index
    ].mean((2, 3))

    cos_Gw_deltaF = torch.cosine_similarity(
        Gw.mean((0, 2, 3)), -diff.mean(0, keepdim=True), dim=1
    )
    cos_Gw_directions = torch.cosine_similarity(
        Gw.mean((0, 2, 3)), -style_steering.unsqueeze(0).cuda(), dim=1
    )

    st.bar_chart(
        pd.DataFrame(
            {
                "COSSIM Gw and Delta F": cos_Gw_deltaF.cpu().detach().numpy(),
                "COSSIM Gw and Style Steering": cos_Gw_directions.cpu()
                .detach()
                .numpy(),
            }
        ),
        stack=False,
    )

    st.header("Steering using Gram Matrix Direction (coarse -> fine)")
    # Gws = []
    # for i, image in enumerate(coarse_images):
    #     Gw = compute_Gw(
    #         model.encoder,
    #         image.unsqueeze(0),
    #         probe,
    #         AnnotationType.FINE.value,
    #         f_index,
    #     )
    #     Gws.append(Gw.unsqueeze(0))
    # Gw = torch.cat(Gws, dim=0)
    gram_direction = Gw  # / Gw.norm(dim=1, keepdim=True)
    B = coarse_images.shape[0]
    features_coarse_gram_steered = []
    alpha_gram = st.slider(
        "Gram Steering Alpha",
        min_value=0.0,
        max_value=200.0,
        value=10.0,
        step=0.1,
        key="gram_steering_alpha",
    )
    for layer in range(len(features_coarse)):
        if layer != f_index:
            features_coarse_gram_steered.append(features_coarse[layer])
        elif layer == f_index:
            # features_coarse_gram_steered.append(
            #     features_coarse[layer] + 10 * style_steering.view(1, -1, 1, 1)
            # )
            features_coarse_gram_steered.append(
                features_coarse[layer] - alpha_gram * gram_direction
            )
    preds_coarse_gram_steered = model.decoder(features_coarse_gram_steered)
    preds_coarse_gram_steered = model.segmentation_head(
        preds_coarse_gram_steered
    ).sigmoid()
    st.image(
        make_grid(preds_coarse_gram_steered.cpu()).permute(1, 2, 0).numpy(), clamp=True
    )


with tabDiff:
    diff_image = coarse_images - coarse_to_fine
    diff_image -= diff_image.min()
    diff_image /= diff_image.max()

    st.header("Diff Images (Coarse - Coarse to Fine Adversarially Steered)")

    st.image(make_grid(diff_image.cpu()).permute(1, 2, 0).numpy(), clamp=True)
