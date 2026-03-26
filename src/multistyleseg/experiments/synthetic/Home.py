import streamlit as st

from multistyleseg.experiments.synthetic.ui_utils import global_sidebar
from multistyleseg.experiments.synthetic.train import get_joint_model
from multistyleseg.experiments.synthetic.utils import get_dataset
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

st.set_page_config(page_title="Synthetic Data Generator", layout="wide")

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
data = next(iter(dataloader))
image = data["image"]
fine_mask = data["fine_mask"]
coarse_mask = data["coarse_mask"]


train = st.button("Load/Train Joint Model")
if train:
    model = get_joint_model(task=st.session_state.task)

cols = st.columns(3)
with cols[0]:
    st.header("Synthetic Image")
    st.image(make_grid(image).permute(1, 2, 0).numpy())
with cols[1]:
    st.header("Fine Annotation Mask")
    st.image(
        make_grid(fine_mask.unsqueeze(1)).permute(1, 2, 0).numpy() * 255,
    )
with cols[2]:
    st.header("Coarse Annotation Mask")
    st.image(
        make_grid(coarse_mask.unsqueeze(1)).permute(1, 2, 0).numpy() * 255,
    )
