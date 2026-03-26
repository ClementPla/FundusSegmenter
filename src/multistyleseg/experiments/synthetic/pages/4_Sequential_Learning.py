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
import numpy as np
import pandas as pd
from multistyleseg.experiments.synthetic.train import get_sequential_model
from multistyleseg.experiments.synthetic.utils import compute_IoU
import copy


st.set_page_config(page_title="Sequential Learning Experiment", layout="wide")
global_sidebar()
model1, model = get_sequential_model(task=st.session_state.task)

dataloader = DataLoader(
    get_dataset(
        n_shapes=1,
        return_all_styles=True,
    ),
    batch_size=32,
    num_workers=8,
    pin_memory=True,
)

data = next(iter(dataloader))
images = data["image"]
fine_masks = data["fine_mask"]
coarse_masks = data["coarse_mask"]
annotation_types = data["expected_style"]
annotation_types = annotation_types.view(-1, 1, 1)
actual_masks = fine_masks * (1 - annotation_types) + coarse_masks * annotation_types

coarse_images = swap_annotators_tensor(data, AnnotationType.FINE, AnnotationType.COARSE)
fine_images = swap_annotators_tensor(data, AnnotationType.COARSE, AnnotationType.FINE)

tabInference, tabSequential, tabDiff = st.tabs(
    ["Inference Results", "Sequential Learning", "Difference"]
)
with tabInference:
    cols = st.columns(2)

    with cols[0]:
        st.header("Model First Half Predictions (Fine)")
        images = images.cuda()
        fine_images = fine_images.cuda()
        with torch.no_grad():
            fine_pred_features_model1 = model1.encoder(fine_images)
            fine_preds = model1.decoder(fine_pred_features_model1)
            fine_preds_model1 = (
                model1.segmentation_head(fine_preds).squeeze(1).sigmoid()
            )
        st.image(
            make_grid(fine_images).permute(1, 2, 0).cpu().numpy(),
            caption="Input Fine Images",
        )

        st.image(
            make_grid(fine_preds_model1.unsqueeze(1)).permute(1, 2, 0).cpu().numpy(),
            caption=f"IoU: {compute_IoU(fine_preds_model1, fine_masks.cuda()):.4f}",
        )
        st.header("Model First Half Predictions (Coarse)")
        images = images.cuda()
        coarse_images = coarse_images.cuda()
        with torch.no_grad():
            coarse_pred_features_model1 = model1.encoder(coarse_images)
            coarse_preds = model1.decoder(coarse_pred_features_model1)
            coarse_preds_model1 = (
                model1.segmentation_head(coarse_preds).squeeze(1).sigmoid()
            )
        st.image(
            make_grid(coarse_images).permute(1, 2, 0).cpu().numpy(),
            caption="Input Coarse Images",
        )
        st.image(
            make_grid(coarse_preds_model1.unsqueeze(1)).permute(1, 2, 0).cpu().numpy(),
            caption=f"IoU: {compute_IoU(coarse_preds_model1, coarse_masks.cuda()):.4f}",
        )
    with cols[1]:
        st.header("Model Full Predictions (Fine)")
        images = images.cuda()
        fine_images = fine_images.cuda()
        with torch.no_grad():
            fine_pred_features_full_model = model.encoder(fine_images)
            fine_preds_full_model = model.decoder(fine_pred_features_full_model)
            fine_preds_full_model = (
                model.segmentation_head(fine_preds_full_model).squeeze(1).sigmoid()
            )
        st.image(
            make_grid(fine_images).permute(1, 2, 0).cpu().numpy(),
            caption="Input Fine Images",
        )

        st.image(
            make_grid(fine_preds_full_model.unsqueeze(1))
            .permute(1, 2, 0)
            .cpu()
            .numpy(),
            caption=f"IoU: {compute_IoU(fine_preds_full_model, fine_masks.cuda()):.4f}",
        )
        st.header("Model Full Predictions (Coarse Annotations)")
        images = images.cuda()
        coarse_images = coarse_images.cuda()
        with torch.no_grad():
            coarse_pred_features_full_model = model.encoder(coarse_images)
            coarse_preds_full_model = model.decoder(coarse_pred_features_full_model)
            coarse_preds_full_model = (
                model.segmentation_head(coarse_preds_full_model).squeeze(1).sigmoid()
            )

        st.image(
            make_grid(coarse_images).permute(1, 2, 0).cpu().numpy(),
            caption="Input Coarse Images",
        )
        st.image(
            make_grid(coarse_preds_full_model.unsqueeze(1))
            .permute(1, 2, 0)
            .cpu()
            .numpy(),
            caption=f"IoU: {compute_IoU(coarse_preds_full_model, coarse_masks.cuda()):.4f}",
        )

with tabSequential:
    # Find the alignment between the two models
    st.header("Sequential Learning Experiment")
    alignment = []
    names = []
    model_mixed = get_joint_model(task=st.session_state.task)
    alignment_mixed_one = []
    alignment_mixed_full = []
    for (name1, param1), (name2, param2), (name_mixed, param_mixed) in zip(
        list(model1.named_parameters()),
        list(model.named_parameters()),
        list(model_mixed.named_parameters()),
    ):
        alignment.append(
            torch.cosine_similarity(param1.view(-1), param2.view(-1), dim=0).item()
        )
        alignment_mixed_one.append(
            torch.cosine_similarity(param1.view(-1), param_mixed.view(-1), dim=0).item()
        )
        alignment_mixed_full.append(
            torch.cosine_similarity(param2.view(-1), param_mixed.view(-1), dim=0).item()
        )
        names.append(name1)
    st.write(f"Alignment between Model 1 and Model 2: {np.mean(alignment):.4f}")
    st.line_chart(
        pd.DataFrame(
            {
                "Alignment (T1 ↔ Sequential) ": alignment,
                "Alignment (T1 ↔ Mixed)": alignment_mixed_one,
                "Alignment (Sequential ↔ Mixed)": alignment_mixed_full,
            },
            index=names,
        )
    )

    st.write(
        f"Minimum Alignment: {min(alignment):.4f} for layer {names[alignment.index(min(alignment))]}"
    )

    # Compute alignments for decoder layers only
    decoder_alignments = {
        "fine_vs_seq": [],
        "fine_vs_joint": [],
        "seq_vs_joint": [],
        "names": [],
    }

    for (name1, param1), (name2, param2), (name_mixed, param_mixed) in zip(
        list(model1.named_parameters()),
        list(model.named_parameters()),
        list(model_mixed.named_parameters()),
    ):
        if "decoder" in name1 and "weight" in name1:
            wf = param1.flatten()
            ws = param2.flatten()
            wj = param_mixed.flatten()

            cos_fs = torch.cosine_similarity(wf.unsqueeze(0), ws.unsqueeze(0)).item()
            cos_fj = torch.cosine_similarity(wf.unsqueeze(0), wj.unsqueeze(0)).item()
            cos_sj = torch.cosine_similarity(ws.unsqueeze(0), wj.unsqueeze(0)).item()

            decoder_alignments["fine_vs_seq"].append(cos_fs)
            decoder_alignments["fine_vs_joint"].append(cos_fj)
            decoder_alignments["seq_vs_joint"].append(cos_sj)
            decoder_alignments["names"].append(name1.replace("decoder.", ""))

    st.write("Decoder layer alignments:")
    st.write(f"{'Layer':<40} Fine↔Seq  Fine↔Joint  Seq↔Joint")
    st.write("-" * 75)
    for i, name in enumerate(decoder_alignments["names"]):
        st.write(
            f"{name:<40} {decoder_alignments['fine_vs_seq'][i]:.3f}     {decoder_alignments['fine_vs_joint'][i]:.3f}       {decoder_alignments['seq_vs_joint'][i]:.3f}"
        )

    st.write("Mean alignment:")
    st.write(f"  Fine ↔ Sequential: {np.mean(decoder_alignments['fine_vs_seq']):.3f}")
    st.write(f"  Fine ↔ Joint:      {np.mean(decoder_alignments['fine_vs_joint']):.3f}")
    st.write(f"  Sequential ↔ Joint: {np.mean(decoder_alignments['seq_vs_joint']):.3f}")
