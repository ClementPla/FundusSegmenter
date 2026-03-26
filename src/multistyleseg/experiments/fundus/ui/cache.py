import streamlit as st
from multistyleseg.data.fundus.factory import ALL_DATASETS, get_datamodule_from_config
from nntools.utils import Config
from pathlib import Path
from multistyleseg.models.factory import ModelType, get_model
from multistyleseg.trainers.fundus import (
    FundusProbabilisticUnet,
    FundusConditionalProbabilisticUnet,
    FundusSegmentationTrainer,
    FundusHierarchicalProbabilisticUnet,
)


@st.cache_resource
def get_dataset(batch_size=4):
    config = Config("configs/fundus.yaml")
    config["data"]["batch_size"] = batch_size
    config["data"]["num_workers"] = 0
    config["data"]["use_cache"] = False
    datamodule = get_datamodule_from_config(
        config["datasets"], ALL_DATASETS, config["data"]
    )
    datamodule.setup_all()
    return datamodule


@st.cache_resource
def get_cache_model(model_type: ModelType):
    ckpt_path = Path("checkpoints") / f"fundus_{model_type.value}"
    model_path = next(ckpt_path.glob("*.ckpt"))
    model = get_model(
        model_type=model_type, in_channels=3, out_channels=5, img_size=1024
    )
    match model_type:
        case ModelType.PROBABILISTIC_UNET:
            pl_module = FundusProbabilisticUnet.load_from_checkpoint(
                model_path, model=model, learning_rate=1e-4
            )
        case ModelType.HIERARCHICAL_PROBABILISTIC_UNET:
            pl_module = FundusHierarchicalProbabilisticUnet.load_from_checkpoint(
                model_path, model=model, learning_rate=1e-4
            )
        case ModelType.CONDITIONAL_PROBABILISTIC_UNET:
            pl_module = FundusConditionalProbabilisticUnet.load_from_checkpoint(
                model_path, model=model, learning_rate=1e-4
            )
        case _:
            pl_module = FundusSegmentationTrainer.load_from_checkpoint(
                model_path, model=model, learning_rate=1e-4
            )

    model = pl_module.model.cuda()
    return model
