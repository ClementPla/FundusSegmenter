import streamlit as st
from multistyleseg.models.factory import ModelType
import torch
from torchvision.utils import make_grid, draw_segmentation_masks
from multistyleseg.experiments.fundus.ui.cache import get_dataset, get_cache_model
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from fundus_data_toolkit.datasets.generic import get_generic_composer

st.set_page_config(
    page_title="Fundus Segmentation",
    layout="wide",
)

with st.sidebar:
    model_type = st.selectbox(
        "Select Model Type",
        options=[
            ModelType.UNET,
            ModelType.SERESNET_UNET,
            ModelType.CONVNEXT_UNET,
            ModelType.DEEPLABV3_PLUS,
            ModelType.SWIN_UNETR,
            ModelType.UNETR,
            ModelType.SEGFORMER,
        ],
        format_func=lambda x: x.value,
        index=0,
    )

    file = st.file_uploader("Upload a fundus image", type=["png", "jpg", "jpeg"])
composer = get_generic_composer((1024, 1024), True)
if file is not None:
    image = np.asarray(Image.open(file).convert("RGB"))
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Input Image", use_container_width=True)
    input_tensor = composer(image=image)["image"].cuda()

    with st.spinner("Running inference..."):
        model = get_cache_model(model_type).cuda()
        model.eval()
        with torch.inference_mode():
            output = model(input_tensor.unsqueeze(0)).softmax(1)
    pred = output.argmax(1)
    one_hot_encoded = (
        torch.nn.functional.one_hot(pred, num_classes=5).squeeze(0).permute(2, 0, 1)
    ).bool()
    colors = ["green", "blue", "yellow", "purple"]
    input_tensor = (input_tensor - input_tensor.min()) / (
        input_tensor.max() - input_tensor.min()
    )
    overlay = draw_segmentation_masks(
        input_tensor,
        masks=one_hot_encoded[1:],
        alpha=0.5,
        colors=colors,
    )
    with col2:
        st.image(
            overlay.permute(1, 2, 0).cpu().numpy(),
            caption="Segmentation Overlay",
            use_container_width=True,
        )
