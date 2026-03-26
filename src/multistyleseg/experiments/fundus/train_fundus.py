import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path
from multistyleseg.models.factory import ModelType, get_model
from multistyleseg.trainers.fundus import FundusSegmentationTrainer
from nntools.utils import Config
from multistyleseg.data.fundus.factory import ALL_DATASETS, get_datamodule_from_config


def main(model_type: ModelType):
    config = Config("configs/fundus.yaml")
    # Example usage of FundusSegmentationTrainer

    model = get_model(
        model_type=model_type, in_channels=3, out_channels=5, img_size=1024
    )
    pl_module = FundusSegmentationTrainer(model=model, learning_rate=1e-4)

    datamodule = get_datamodule_from_config(
        config["datasets"], ALL_DATASETS, config["data"]
    )
    ckpt_path = Path("checkpoints") / f"fundus_{model_type.value}"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            dirpath=ckpt_path,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    wandb_logger = WandbLogger(
        project="MultiStyle Fundus Segmentation", name=model_type.value
    )
    trainer = Trainer(callbacks=callbacks, logger=wandb_logger, **config["trainer"])

    trainer.fit(
        pl_module,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
    )
    for test_dataloader in datamodule.test_dataloader():
        pl_module.test_metrics.postfix = f" {test_dataloader.dataset.id}"
        trainer.test(
            dataloaders=test_dataloader,
            ckpt_path="best",
            verbose=True,
        )

    wandb.finish()


if __name__ == "__main__":
    for model_type in [
        # ModelType.UNET,
        # ModelType.SERESNET_UNET,
        # ModelType.CONVNEXT_UNET,
        # ModelType.SWIN_UNETR,
        ModelType.UNETR,
        # ModelType.DEEPLABV3_PLUS,
        # ModelType.SEGFORMER,
    ]:
        main(model_type)
