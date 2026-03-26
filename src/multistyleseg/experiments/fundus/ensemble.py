import torch
import torch.nn as nn
from multistyleseg.models.factory import ModelType, get_model
from multistyleseg.trainers.fundus import FundusSegmentationTrainer
from pathlib import Path
from monai.inferers import SlidingWindowInferer
from multistyleseg.utils import CKPT_DIR
from fundus_odmac_toolkit.models.hf_hub import HuggingFaceModel


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        # Assuming the outputs are logits, we average them
        avg_output = torch.mean(torch.stack(outputs), dim=0)
        return avg_output


class EnsembleModelForOnnx(nn.Module):
    def __init__(self, models):
        super(EnsembleModelForOnnx, self).__init__()
        self.models = nn.ModuleList(models)
        self.normalization_mean = nn.Parameter(
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), requires_grad=False
        )
        self.normalization_std = nn.Parameter(
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), requires_grad=False
        )
        self.od_mac_model = HuggingFaceModel.from_pretrained(
            "ClementP/fundus-odmac-segmentation-unet-maxvit_small_tf_512",
            force_download=False,
            local_files_only=True,
        ).model

    def forward(self, x):
        x = (x - self.normalization_mean) / self.normalization_std
        outputs = [model(x) for model in self.models]
        # Assuming the outputs are logits, we average them
        avg_output = torch.mean(torch.stack(outputs), dim=0).argmax(dim=1)
        od_mac = self.od_mac_model(x).argmax(dim=1)
        return avg_output, od_mac


def get_ensemble_model(
    img_size: int = 1024,
    model_choices: list = None,
    to_onnx=False,
) -> EnsembleModel:
    # List of model checkpoint paths
    ckpt_folder = CKPT_DIR.resolve()
    models_paths = ckpt_folder.rglob("*.ckpt")
    models = []
    for model_path in models_paths:
        # Parent folder is assumed to be the model type
        model_type = model_path.parent.name.split("fundus_")[1]
        if model_choices and model_type not in model_choices:
            continue
        arch = get_model(
            model_type=ModelType(model_type),
            in_channels=3,
            out_channels=5,
            img_size=img_size,
        )
        trainer = FundusSegmentationTrainer.load_from_checkpoint(
            checkpoint_path=model_path,
            model=arch,
        )
        models.append(trainer.model)
    if to_onnx:
        ensemble_model = EnsembleModelForOnnx(models)
    else:
        ensemble_model = EnsembleModel(models)
    return ensemble_model
