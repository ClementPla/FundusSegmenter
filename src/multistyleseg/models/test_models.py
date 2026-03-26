from .factory import ModelType, get_model
import torch


class TestModelCreation:
    def test_unet_creation(self):
        get_model(ModelType.UNET, in_channels=3, out_channels=5)

    def test_seresnet_unet_creation(self):
        get_model(ModelType.SERESNET_UNET, in_channels=3, out_channels=5)

    def test_convnext_unet_creation(self):
        get_model(ModelType.CONVNEXT_UNET, in_channels=3, out_channels=5)

    def test_swin_unetr_creation(self):
        get_model(ModelType.SWIN_UNETR, in_channels=3, out_channels=5)


class TestForwardPass:
    def test_unet_forward(self):
        model = get_model(ModelType.UNET, in_channels=3, out_channels=5).cuda()
        x = torch.randn(1, 3, 1024, 1024).cuda()
        y = model(x)
        assert y.shape == (1, 5, 1024, 1024)

    def test_seresnet_unet_forward(self):
        model = get_model(
            ModelType.SERESNET_UNET, in_channels=3, out_channels=5, img_size=1024
        ).cuda()
        x = torch.randn(1, 3, 1024, 1024).cuda()
        y = model(x)
        assert y.shape == (1, 5, 1024, 1024)

    def test_convnext_unet_forward(self):
        model = get_model(ModelType.CONVNEXT_UNET, in_channels=3, out_channels=5).cuda()
        x = torch.randn(1, 3, 1024, 1024).cuda()
        y = model(x)
        assert y.shape == (1, 5, 1024, 1024)

    def test_swin_unetr_forward(self):
        model = get_model(ModelType.SWIN_UNETR, in_channels=3, out_channels=5).cuda()
        x = torch.randn(1, 3, 1024, 1024).cuda()
        y = model(x)
        assert y.shape == (1, 5, 1024, 1024)
