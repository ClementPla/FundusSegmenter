import torch
from .measure import get_loss, get_metrics


class TestMeasure:
    def test_get_metrics(self):
        metrics = get_metrics(num_classes=5)
        assert "MeanIoU" in metrics
        assert "DiceScore" in metrics

    def test_get_loss(self):
        loss_fn = get_loss()
        assert loss_fn is not None
        assert callable(loss_fn)

    def test_loss_forward(self):
        loss_fn = get_loss()
        predictions = torch.randn(2, 5, 256, 256)  # Example prediction tensor
        targets = torch.randint(0, 5, (2, 1, 256, 256))  # Example target tensor
        loss = loss_fn(predictions, targets)
        assert loss.item() >= 0
        assert isinstance(loss, torch.Tensor)

    def test_metrics_forward(self):
        metrics = get_metrics(num_classes=5)
        predictions = torch.randn(2, 5, 256, 256)  # Example prediction tensor
        targets = torch.randint(0, 5, (2, 256, 256))  # Example target tensor

        results = metrics(predictions, targets)
        assert "MeanIoU" in results
        assert "DiceScore" in results
        assert results["MeanIoU"].item() >= 0
        assert results["DiceScore"].item() >= 0
