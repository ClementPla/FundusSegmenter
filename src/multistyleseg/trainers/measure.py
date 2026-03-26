from torchmetrics import MetricCollection
from torchmetrics.segmentation import MeanIoU, DiceScore
from monai.losses import DiceCELoss, FocalLoss
from enum import Enum
from torch.nn import CrossEntropyLoss

def get_metrics(num_classes: int) -> MetricCollection:
    metrics = MetricCollection(
        {
            "MeanIoU": MeanIoU(num_classes=num_classes, input_format="mixed"),
            "DiceScore": DiceScore(
                num_classes=num_classes, average="macro", input_format="mixed"
            ),
        }
    )
    return metrics

class LossType(str, Enum):
    DICE_CE = "DICE_CE"
    DICE = "DICE"
    CE = "CE"
    FOCAL = "FOCAL"


def get_loss(loss_type: LossType = LossType.DICE_CE):
    match loss_type:
        case LossType.DICE_CE:
            return DiceCELoss(to_onehot_y=True, softmax=True, label_smoothing=0.3)
        case LossType.DICE:
            return DiceCELoss(to_onehot_y=True, softmax=True, lambda_ce=0.0)
        case LossType.CE:
            return DiceCELoss(to_onehot_y=True, softmax=True, label_smoothing=0.3, lambda_dice=0.0)
        case LossType.FOCAL:
            return FocalLoss(to_onehot_y=True, use_softmax=True)
        case _:
            raise ValueError(f"Unsupported loss type: {loss_type}")