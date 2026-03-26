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
from fundus_data_toolkit.data_aug import DAType
from multistyleseg.trainers.measure import LossType
from multistyleseg.experiments.fundus.baseline_setup import iterable_configurations
def main(model_type: ModelType, image_resolution: int, data_augmentation: DAType, loss_type: LossType):
    config = Config("configs/fundus.yaml")
    # Example usage of FundusSegmentationTrainer
    if image_resolution <= 256:
        config['data']['random_crop'] = None
    config['data']['data_augmentation_type'] = data_augmentation.value
    config['data']['img_size'] = (image_resolution, image_resolution)
    model = get_model(
        model_type=model_type, in_channels=3, out_channels=5, img_size=image_resolution
    )
    pl_module = FundusSegmentationTrainer(model=model, learning_rate=1e-3, loss_type=loss_type)

    datamodule = get_datamodule_from_config(
        config["datasets"], ALL_DATASETS, config["data"]
    )
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    wandb_logger = WandbLogger(
        project="Baseline Fundus Segmentation",
        config={**config['data'], 'model_type': model_type.value, 'loss_type': loss_type.value}
        
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
    
    for i, img_res, da_type, loss_type in enumerate(iterable_configurations()):
        main(
            model_type=ModelType.UNET_BASELINE,
            image_resolution=img_res,
            data_augmentation=da_type,
            loss_type=loss_type,
        )
        print(f"Completed experiment {i+1}/64")

    
    
