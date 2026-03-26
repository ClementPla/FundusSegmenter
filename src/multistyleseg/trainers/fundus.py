import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from multistyleseg.models.probabilistic_unet.utils import l2_regularisation
from multistyleseg.trainers.measure import get_loss, get_metrics, LossType
from multistyleseg.data.fundus.utils import convert_list_datasets_to_tensor


class FundusSegmentationTrainer(pl.LightningModule):
    def __init__(self, model: nn.Module, learning_rate: float = 1e-4, loss_type: LossType = LossType.DICE_CE):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = get_loss(loss_type)
        self.metrics = get_metrics(num_classes=5)
        self.test_metrics = get_metrics(num_classes=5)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["mask"].long().clamp(0, 4)
        outputs = self(images)
        loss = self.criterion(outputs, masks.unsqueeze(1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["mask"].long().clamp(0, 4)
        outputs = self(images)
        loss = self.criterion(outputs, masks.unsqueeze(1))
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.metrics.update(outputs, masks)

    def on_validation_epoch_end(self):
        results = self.metrics.compute()
        self.log_dict(results, prog_bar=True, sync_dist=True)
        self.metrics.reset()

    def test_step(self, batch, batch_idx):
        images = batch["image"]
        masks = batch["mask"].long().clamp(0, 4)
        outputs = self(images)
        loss = self.criterion(outputs, masks.unsqueeze(1))
        self.log("test_loss", loss)
        self.test_metrics.update(outputs, masks)

    def on_test_epoch_end(self):
        results = self.test_metrics.compute()
        self.log_dict(
            results,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.test_metrics.reset()


class FundusProbabilisticUnet(pl.LightningModule):
    def __init__(self, model: nn.Module, learning_rate: float = 1e-4):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = get_loss()
        self.metrics = get_metrics(num_classes=5)
        self.test_metrics = get_metrics(num_classes=5)
        self.first_call = True

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-5
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def forward(self, batch, is_training=True):
        images = batch["image"]
        masks = batch["mask"].long().clamp(0, 4)
        self.model(images, segm=masks, training=is_training)
        if is_training:
            return self.model.elbo(masks)
        else:
            outputs = self.model.predict_segmentation(images, num_samples=4)
            return outputs, masks

    def training_step(self, batch, batch_idx):
        elbo = self.forward(batch, is_training=True)
        reg_loss = (
            l2_regularisation(self.model.posterior)
            + l2_regularisation(self.model.prior)
            + l2_regularisation(self.model.fcomb.layers)
        )
        loss = -elbo + 1e-5 * reg_loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0 and self.first_call:
            self.forward(batch, is_training=True)
            self.first_call = False
        outputs, masks = self.forward(batch, is_training=False)
        self.metrics.update(outputs, masks)

    def on_validation_epoch_end(self):
        results = self.metrics.compute()
        self.log_dict(results, prog_bar=True, sync_dist=True)
        self.metrics.reset()

    def test_step(self, batch, batch_idx):
        outputs, masks = self.forward(batch, is_training=False)
        self.test_metrics.update(outputs, masks)

    def on_test_epoch_end(self):
        results = self.test_metrics.compute()
        self.log_dict(
            results,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.test_metrics.reset()


class FundusHierarchicalProbabilisticUnet(FundusProbabilisticUnet):
    def training_step(self, batch, batch_idx):
        elbo = self.forward(batch, is_training=True)
        loss = elbo
        self.log("train_loss", loss)
        return loss


class FundusConditionalProbabilisticUnet(FundusProbabilisticUnet):
    def forward(self, batch, is_training=True):
        images = batch["image"]
        masks = batch["mask"].long().clamp(0, 4)
        class_labels = batch["tag"]
        class_labels = convert_list_datasets_to_tensor(
            class_labels, device=images.device
        )
        self.model(
            patch=images, segm=masks, class_label=class_labels, training=is_training
        )
        if is_training:
            return self.model.elbo(masks)
        else:
            outputs = self.model.predict_segmentation(
                images, class_labels, num_samples=4
            )
            return outputs, masks
