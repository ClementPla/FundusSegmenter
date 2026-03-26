"""
Hierarchical Probabilistic U-Net (HPU-Net) - PyTorch Implementation
Using Segmentation Models PyTorch (smp) pretrained encoders

Based on: "A Hierarchical Probabilistic U-Net for Modeling Multi-Scale Ambiguities"
Kohl et al., 2019 (arXiv:1905.13077)

Architecture:
- Pretrained encoder from smp (e.g., ResNet, EfficientNet, etc.)
- Custom decoder with latent injection points
- Hierarchical prior/posterior networks
- 1x1 convolutions to match latent dimensions to decoder channels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl_divergence
from typing import List, Tuple, Optional, Dict, Union
from monai.losses import DiceCELoss
from multistyleseg.models.probabilistic_unet.hierarchical_probabilistic_unet import (
    HierarchicalPrior,
    UNetDecoderWithLatents,
    HierarchicalPosterior,
)

try:
    import segmentation_models_pytorch as smp
    from segmentation_models_pytorch.encoders import get_encoder

    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print(
        "Warning: segmentation_models_pytorch not installed. "
        "Install with: pip install segmentation-models-pytorch"
    )

    """Predicts mean and log-variance for a Gaussian at a single scale."""

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        hidden_channels: Optional[int] = None,
    ):
        super().__init__()
        hidden_channels = hidden_channels or max(in_channels // 2, latent_channels * 2)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv_mean = nn.Conv2d(hidden_channels, latent_channels, 1)
        self.conv_logvar = nn.Conv2d(hidden_channels, latent_channels, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.conv_mean(h), self.conv_logvar(h)


class ConditionalHierarchicalProbabilisticUNet(nn.Module):
    """
    Hierarchical Probabilistic U-Net using pretrained smp encoders.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        num_classes: int = 2,
        decoder_channels: Tuple[int, ...] = (256, 128, 64, 32, 16),
        latent_channels: int = 6,
        num_latent_scales: int = 4,
        beta: float = 1.0,
        embedding_dim=8,
        embedding_classes=5,
        embedding_to_input=False,
        embedding_to_output=False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_latent_scales = num_latent_scales
        self.latent_channels = latent_channels
        self.beta = beta
        self.conditional_embedding = nn.Embedding(embedding_classes, embedding_dim)
        self.embedding_to_input = embedding_to_input
        self.embedding_to_output = embedding_to_output

        # Main encoder
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels + (embedding_dim if embedding_to_input else 0),
            weights=encoder_weights,
        )
        encoder_channels = list(self.encoder.out_channels)

        # Decoder
        self.decoder = UNetDecoderWithLatents(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            latent_channels=latent_channels,
            num_latent_scales=num_latent_scales,
            num_classes=num_classes,
        )

        # Prior
        self.prior = HierarchicalPrior(
            encoder_channels=encoder_channels,
            latent_channels=latent_channels,
            num_latent_scales=num_latent_scales,
        )

        # Posterior (uses lightweight encoder for segmentation)
        self.posterior = HierarchicalPosterior(
            encoder_channels=encoder_channels,
            num_classes=num_classes,
            in_channels=in_channels + (embedding_dim if embedding_to_input else 0),
            latent_channels=latent_channels,
            num_latent_scales=num_latent_scales,
        )

        self.encoder_channels = encoder_channels

        # Storage for intermediate results
        self.encoder_features = None
        self.prior_dists = None
        self.posterior_dists = None
        self.criterion = DiceCELoss(to_onehot_y=True, softmax=True, label_smoothing=0.3)

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        return list(self.encoder(x))

    def forward(
        self,
        patch: torch.Tensor,
        segm: torch.Tensor,
        class_label: torch.Tensor,
        training: bool = True,
        temperature: float = 1.0,
    ):
        """
        Forward pass - encodes image and computes distributions.

        Args:
            patch: Input image (B, C, H, W)
            segm: Segmentation mask (B, H, W) or (B, 1, H, W) or one-hot (B, num_classes, H, W)
            training: Whether to compute posterior (requires segm)
            temperature: Sampling temperature
        """

        self.class_emb = (
            self.conditional_embedding(class_label)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, patch.size(2), patch.size(3))
        )
        self.encoder_features = self.encode(patch)
        self.prior_dists = self.prior(self.encoder_features, temperature=temperature)

        if training and segm is not None:
            self.posterior_dists = self.posterior(
                self.encoder_features, segmentation=segm, temperature=temperature
            )

    def sample(self, testing: bool = False) -> torch.Tensor:
        """Sample from prior and decode."""
        if testing:
            z_samples = [dist.sample() for dist in self.prior_dists]
        else:
            z_samples = [dist.rsample() for dist in self.prior_dists]
        return self.decoder(self.encoder_features, latents=z_samples)

    def reconstruct(self, use_posterior_mean: bool = False) -> torch.Tensor:
        """Reconstruct from posterior."""
        if use_posterior_mean:
            z_samples = [dist.mean for dist in self.posterior_dists]
        else:
            z_samples = [dist.rsample() for dist in self.posterior_dists]
        return self.decoder(self.encoder_features, latents=z_samples)

    def elbo(
        self,
        segm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute negative ELBO loss.

        Args:
            segm: Target segmentation (B, H, W) or (B, 1, H, W)
            criterion: Reconstruction loss function

        Returns:
            Negative ELBO (to minimize)
        """
        # Sample from posterior
        z_samples = [dist.rsample() for dist in self.posterior_dists]

        # Reconstruction
        reconstruction = self.decoder(self.encoder_features, latents=z_samples)

        # Handle segmentation shape for loss
        if segm.dim() == 3:
            target = segm
        elif segm.dim() == 4 and segm.shape[1] == 1:
            target = segm.squeeze(1)
        elif segm.dim() == 4 and segm.shape[1] == self.num_classes:
            target = segm.argmax(dim=1)
        else:
            target = segm

        reconstruction_loss = self.criterion(reconstruction, target.long().unsqueeze(1))

        # KL divergence
        kl_loss = sum(
            kl_divergence(post, prior).mean()
            for post, prior in zip(self.posterior_dists, self.prior_dists)
        )

        # Return negative ELBO (reconstruction_loss is already positive)
        return reconstruction_loss + self.beta * kl_loss

    def predict_multiple_hypotheses(
        self, image: torch.Tensor, num_samples: int = 4
    ) -> torch.Tensor:
        """
        Generate multiple segmentation hypotheses.

        Returns:
            Logits (num_samples, B, num_classes, H, W)
        """
        self.encoder_features = self.encode(image)
        self.prior_dists = self.prior(self.encoder_features)

        samples = []
        for _ in range(num_samples):
            z_samples = [dist.rsample() for dist in self.prior_dists]
            logits = self.decoder(self.encoder_features, latents=z_samples)
            samples.append(logits)

        return torch.stack(samples, dim=0)

    def predict_segmentation(
        self, image: torch.Tensor, num_samples: int = 4
    ) -> torch.Tensor:
        """
        Predict segmentation by averaging multiple hypotheses.

        Returns:
            Averaged probabilities (B, num_classes, H, W)
        """
        logits_samples = self.predict_multiple_hypotheses(
            image, num_samples=num_samples
        )
        probs_samples = F.softmax(logits_samples, dim=2)
        mean_probs = probs_samples.mean(dim=0)
        return mean_probs

    def freeze_encoder(self, freeze: bool = True):
        """Freeze/unfreeze the pretrained encoder."""
        for param in self.encoder.parameters():
            param.requires_grad = not freeze
