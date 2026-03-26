from enum import Enum
import segmentation_models_pytorch as smp
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.networks.nets.unetr import UNETR
from multistyleseg.models.probabilistic_unet.probabilistic_unet import ProbabilisticUnet
from multistyleseg.models.probabilistic_unet.conditional_probabilistic_unet import (
    ConditionalProbabilisticUnet,
)
from multistyleseg.models.probabilistic_unet.hierarchical_probabilistic_unet import (
    HierarchicalProbabilisticUNet,
)


class ModelType(Enum):
    UNET = "UNET"
    UNET_BASELINE = "UNET_BASELINE"
    SERESNET_UNET = "SERESNET_UNET"
    CONVNEXT_UNET = "CONVNEXT_UNET"
    DEEPLABV3_PLUS = "DEEPLABV3_PLUS"
    SWIN_UNETR = "SWIN_UNETR"
    UNETR = "UNETR"
    SEGFORMER = "SEGFORMER"
    PROBABILISTIC_UNET = "PROBABILISTIC_UNET_SERESNET"
    CONDITIONAL_PROBABILISTIC_UNET = "CONDITIONAL_PROBABILISTIC_UNET_SERESNET"
    CONDITIONAL_PROBABILISTIC_UNET_INPUT = (
        "CONDITIONAL_PROBABILISTIC_UNET_SERESNET_INPUT"
    )
    CONDITIONAL_PROBABILISTIC_UNET_OUTPUT = (
        "CONDITIONAL_PROBABILISTIC_UNET_SERESNET_OUTPUT"
    )
    HIERARCHICAL_PROBABILISTIC_UNET = "HIERARCHICAL_PROBABILISTIC_UNET_SERESNET"


def get_model(
    model_type: ModelType,
    in_channels: int = 3,
    out_channels: int = 5,
    img_size: int = 1024,
):
    match model_type:
        case ModelType.UNET:
            return smp.create_model(
                arch="unet",
                encoder_name="resnet50",
                encoder_weights="imagenet",
                in_channels=in_channels,
                classes=out_channels,
            )
        case ModelType.SERESNET_UNET:
            return smp.create_model(
                arch="unet",
                encoder_name="se_resnet50",
                encoder_weights="imagenet",
                in_channels=in_channels,
                classes=out_channels,
            )
        case ModelType.CONVNEXT_UNET:
            return smp.create_model(
                arch="unet",
                encoder_name="tu-convnext_small",
                encoder_weights="imagenet",
                in_channels=in_channels,
                classes=out_channels,
            )
        case ModelType.UNET_BASELINE:
            return smp.Unet(
                encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=in_channels,
                classes=out_channels,
            )
        case ModelType.SWIN_UNETR:
            return SwinUNETR(
                in_channels=in_channels,
                out_channels=out_channels,
                use_checkpoint=True,
                spatial_dims=2,
            )
        case ModelType.DEEPLABV3_PLUS:
            return smp.create_model(
                arch="deeplabv3plus",
                encoder_name="resnet50",
                encoder_weights="imagenet",
                in_channels=in_channels,
                classes=out_channels,
            )
        case ModelType.UNETR:
            return UNETR(
                in_channels=in_channels,
                out_channels=out_channels,
                img_size=(img_size, img_size),
                spatial_dims=2,
                feature_size=16,
                hidden_size=768,
                mlp_dim=3072,
                num_heads=12,
            )
        case ModelType.SEGFORMER:
            return smp.create_model(
                arch="segformer",
                encoder_name="mit_b3",
                encoder_weights="imagenet",
                in_channels=in_channels,
                classes=out_channels,
                img_size=img_size,
            )
            
        case ModelType.PROBABILISTIC_UNET:
            return ProbabilisticUnet(
                input_channels=in_channels,
                num_classes=out_channels,
                latent_dim=6,
                no_convs_fcomb=4,
                beta=10.0,
            )
        case ModelType.CONDITIONAL_PROBABILISTIC_UNET:
            return ConditionalProbabilisticUnet(
                input_channels=in_channels,
                num_classes=out_channels,
                latent_dim=6,
                no_convs_fcomb=4,
                beta=10.0,
                embedding_dim=8,
                embedding_classes=5,
                embedding_to_input=False,
            )
        case ModelType.CONDITIONAL_PROBABILISTIC_UNET_INPUT:
            return ConditionalProbabilisticUnet(
                input_channels=in_channels,
                num_classes=out_channels,
                latent_dim=6,
                no_convs_fcomb=4,
                beta=10.0,
                embedding_dim=8,
                embedding_classes=5,
                embedding_to_input=True,
            )
        case ModelType.CONDITIONAL_PROBABILISTIC_UNET_OUTPUT:
            return ConditionalProbabilisticUnet(
                input_channels=in_channels,
                num_classes=out_channels,
                latent_dim=6,
                no_convs_fcomb=4,
                beta=10.0,
                embedding_dim=8,
                embedding_classes=5,
                embedding_to_input=False,
                embedding_to_output=True,
            )
        case ModelType.HIERARCHICAL_PROBABILISTIC_UNET:
            return HierarchicalProbabilisticUNet(
                encoder_name="se_resnet50",
                encoder_weights="imagenet",
                in_channels=in_channels,
                num_classes=out_channels,
                latent_channels=6,
                num_latent_scales=2,
                beta=10.0,
            )

        case _:
            raise ValueError(f"Unsupported model type: {model_type}")

def main():
    import torch
    model = get_model(ModelType.UNETR, in_channels=3, out_channels=5, img_size=1024).cuda()
    foo = torch.randn(2, 3, 512, 512).cuda()
    bar = model(foo)
    print(bar.shape)

if __name__ == "__main__":
    main()