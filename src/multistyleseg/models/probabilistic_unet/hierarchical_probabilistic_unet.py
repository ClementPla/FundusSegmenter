"""
Hierarchical Probabilistic U-Net (HPU-Net) - PyTorch Implementation
Using Segmentation Models PyTorch (smp) pretrained encoders

Based on: "A Hierarchical Probabilistic U-Net for Modeling Multi-Scale Ambiguities"
Kohl et al., 2019 (arXiv:1905.13077)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl_divergence
from typing import List, Tuple, Optional
from monai.losses import DiceCELoss

try:
    import segmentation_models_pytorch as smp
    from segmentation_models_pytorch.encoders import get_encoder

    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("Warning: segmentation_models_pytorch not installed.")


# ==============================================================================
# Decoder Blocks
# ==============================================================================


class DecoderBlock(nn.Module):
    """Decoder block with optional latent injection."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        latent_channels: int = 0,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        self.latent_channels = latent_channels
        if latent_channels > 0:
            self.latent_proj = nn.Conv2d(latent_channels, out_channels, 1)

        total_in = in_channels + skip_channels

        self.conv1 = nn.Conv2d(
            total_in, out_channels, 3, padding=1, bias=not use_batchnorm
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1, bias=not use_batchnorm
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)

        x = F.relu(self.bn1(self.conv1(x)))

        if latent is not None and self.latent_channels > 0:
            if latent.shape[2:] != x.shape[2:]:
                latent = F.interpolate(
                    latent, size=x.shape[2:], mode="bilinear", align_corners=False
                )
            x = x + self.latent_proj(latent)

        x = F.relu(self.bn2(self.conv2(x)))
        return x


class UNetDecoderWithLatents(nn.Module):
    """U-Net decoder with latent injections, designed for smp encoders."""

    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        latent_channels: int = 6,
        num_latent_scales: int = 4,
        num_classes: int = 2,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        self.num_latent_scales = num_latent_scales
        self.latent_channels = latent_channels
        self.encoder_channels_raw = list(encoder_channels)

        enc_ch = encoder_channels[1:][::-1]
        n_blocks = len(enc_ch) - 1

        decoder_channels = list(decoder_channels)
        if len(decoder_channels) < n_blocks:
            decoder_channels += [decoder_channels[-1]] * (
                n_blocks - len(decoder_channels)
            )
        decoder_channels = decoder_channels[:n_blocks]

        self.decoder_blocks = nn.ModuleList()
        in_ch = enc_ch[0]

        for i in range(n_blocks):
            skip_ch = enc_ch[i + 1]
            out_ch = decoder_channels[i]
            lat_ch = latent_channels if i < num_latent_scales else 0
            self.decoder_blocks.append(
                DecoderBlock(in_ch, skip_ch, out_ch, lat_ch, use_batchnorm)
            )
            in_ch = out_ch

        self.segmentation_head = nn.Conv2d(decoder_channels[-1], num_classes, 1)
        self.encoder_channels = enc_ch
        self.decoder_channels = decoder_channels
        self.n_blocks = n_blocks

    def forward(
        self,
        encoder_features: List[torch.Tensor],
        latents: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        output_size = encoder_features[0].shape[2:]
        features = encoder_features[1:][::-1]
        x = features[0]

        for i, block in enumerate(self.decoder_blocks):
            skip = features[i + 1] if i + 1 < len(features) else None
            latent = latents[i] if latents is not None and i < len(latents) else None
            x = block(x, skip, latent)

        x = self.segmentation_head(x)
        if x.shape[2:] != output_size:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x


# ==============================================================================
# Latent Distribution Networks
# ==============================================================================


class LatentDistributionBlock(nn.Module):
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


class HierarchicalPrior(nn.Module):
    def __init__(
        self,
        encoder_channels: List[int],
        latent_channels: int = 6,
        num_latent_scales: int = 4,
        condition_on_previous: bool = True,
    ):
        super().__init__()

        self.num_latent_scales = num_latent_scales
        self.latent_channels = latent_channels
        self.condition_on_previous = condition_on_previous
        self.feature_channels = list(reversed(encoder_channels[1:]))

        self.dist_blocks = nn.ModuleList()
        for i in range(num_latent_scales):
            feat_idx = min(i, len(self.feature_channels) - 1)
            in_ch = self.feature_channels[feat_idx]
            # Add previous feature + previous latent channels for conditioning
            if condition_on_previous and i > 0:
                prev_feat_idx = min(i - 1, len(self.feature_channels) - 1)
                in_ch += self.feature_channels[prev_feat_idx] + latent_channels
            self.dist_blocks.append(LatentDistributionBlock(in_ch, latent_channels))

    def _get_features_for_scale(
        self, encoder_features: List[torch.Tensor], scale_idx: int
    ) -> torch.Tensor:
        reversed_features = encoder_features[1:][::-1]
        feat_idx = min(scale_idx, len(reversed_features) - 1)
        return reversed_features[feat_idx]

    def _sample_latent(
        self, mean: torch.Tensor, logvar: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        if temperature > 0:
            std = torch.exp(0.5 * logvar)
            return mean + temperature * torch.randn_like(std) * std
        return mean

    def forward(
        self, encoder_features: List[torch.Tensor], temperature: float = 1.0, **kwargs
    ) -> List:
        latents = []
        distributions = []

        for i, dist_block in enumerate(self.dist_blocks):
            feat = self._get_features_for_scale(encoder_features, i)

            # Condition on previous feature AND previous latent
            if self.condition_on_previous and i > 0:
                # Previous feature
                prev_feat = self._get_features_for_scale(encoder_features, i - 1)
                if prev_feat.shape[2:] != feat.shape[2:]:
                    prev_feat = F.interpolate(
                        prev_feat,
                        size=feat.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                # Previous latent
                prev_latent = latents[-1]
                if prev_latent.shape[2:] != feat.shape[2:]:
                    prev_latent = F.interpolate(
                        prev_latent,
                        size=feat.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                feat = torch.cat([feat, prev_feat, prev_latent], dim=1)

            mean, logvar = dist_block(feat)
            scale = F.softplus(logvar) + 1e-6
            dist = Independent(Normal(loc=mean, scale=scale), 3)
            distributions.append(dist)
            latents.append(self._sample_latent(mean, logvar, temperature))

        return distributions


class SegmentationEncoder(nn.Module):
    """Simple encoder for image+segmentation concatenation."""

    def __init__(
        self, in_channels: int, out_channels_list: List[int], num_stages: int = 5
    ):
        super().__init__()

        self.stages = nn.ModuleList()
        self.out_channels = [in_channels]

        ch = in_channels
        for i in range(num_stages):
            out_ch = (
                out_channels_list[i + 1]
                if i + 1 < len(out_channels_list)
                else out_channels_list[-1]
            )
            self.stages.append(
                nn.Sequential(
                    nn.Conv2d(ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
            )
            self.out_channels.append(out_ch)
            ch = out_ch

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = [x]
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class HierarchicalPosterior(nn.Module):
    """Posterior network conditioned on image AND segmentation."""

    def __init__(
        self,
        encoder_channels: List[int],
        in_channels: int = 3,
        latent_channels: int = 6,
        num_latent_scales: int = 4,
        num_classes: int = 2,
    ):
        super().__init__()

        self.num_latent_scales = num_latent_scales
        self.latent_channels = latent_channels
        self.num_classes = num_classes

        # Encoder for image + one-hot segmentation
        self.seg_encoder = SegmentationEncoder(
            in_channels=in_channels + num_classes,
            out_channels_list=encoder_channels,
            num_stages=len(encoder_channels) - 1,
        )

        # Get reversed channel list (bottleneck first)
        seg_channels_rev = list(reversed(self.seg_encoder.out_channels[1:]))

        # Build distribution blocks
        # Each block takes: current_feat + previous_feat + previous_latent (if not first scale)
        self.dist_blocks = nn.ModuleList()
        for i in range(num_latent_scales):
            seg_idx = min(i, len(seg_channels_rev) - 1)
            in_ch = seg_channels_rev[seg_idx]

            # Condition on previous feature AND previous latent
            if i > 0:
                prev_feat_idx = min(i - 1, len(seg_channels_rev) - 1)
                in_ch += seg_channels_rev[prev_feat_idx] + latent_channels

            self.dist_blocks.append(LatentDistributionBlock(in_ch, latent_channels))

        self.seg_channels_rev = seg_channels_rev

    def forward(
        self, images: torch.Tensor, segmentation: torch.Tensor, temperature: float = 1.0
    ) -> List:
        """
        Args:
            images: Input images (B, C, H, W)
            segmentation: Segmentation mask - can be (B, H, W), (B, 1, H, W), or (B, num_classes, H, W)
            temperature: Sampling temperature
        """
        # Convert segmentation to one-hot if needed
        if segmentation.dim() == 3:
            segmentation = (
                F.one_hot(segmentation.long(), self.num_classes)
                .permute(0, 3, 1, 2)
                .float()
            )
        elif segmentation.dim() == 4 and segmentation.shape[1] == 1:
            segmentation = (
                F.one_hot(segmentation.squeeze(1).long(), self.num_classes)
                .permute(0, 3, 1, 2)
                .float()
            )
        elif segmentation.dim() == 4 and segmentation.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} channels for one-hot, got {segmentation.shape[1]}"
            )

        # Concatenate image and segmentation, then encode
        posterior_input = torch.cat([images, segmentation], dim=1)
        seg_features = self.seg_encoder(posterior_input)

        # Reverse to go bottleneck -> shallow
        seg_features_rev = seg_features[1:][::-1]

        latents = []
        distributions = []

        for i, dist_block in enumerate(self.dist_blocks):
            feat_idx = min(i, len(seg_features_rev) - 1)
            feat = seg_features_rev[feat_idx]

            # Condition on previous feature AND previous latent
            if i > 0:
                # Previous feature
                prev_feat_idx = min(i - 1, len(seg_features_rev) - 1)
                prev_feat = seg_features_rev[prev_feat_idx]
                if prev_feat.shape[2:] != feat.shape[2:]:
                    prev_feat = F.interpolate(
                        prev_feat,
                        size=feat.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                # Previous latent
                prev_latent = latents[-1]
                if prev_latent.shape[2:] != feat.shape[2:]:
                    prev_latent = F.interpolate(
                        prev_latent,
                        size=feat.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                feat = torch.cat([feat, prev_feat, prev_latent], dim=1)

            mean, logvar = dist_block(feat)
            scale = F.softplus(logvar) + 1e-6
            dist = Independent(Normal(loc=mean, scale=scale), 3)
            distributions.append(dist)

            # Sample for next iteration conditioning
            latent = (
                mean + temperature * torch.randn_like(scale) * scale
                if temperature > 0
                else mean
            )
            latents.append(latent)

        return distributions


class HierarchicalProbabilisticUNet(nn.Module):
    """Hierarchical Probabilistic U-Net using pretrained smp encoders."""

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
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_latent_scales = num_latent_scales
        self.latent_channels = latent_channels
        self.beta = beta
        self.in_channels = in_channels

        # Main encoder
        self.encoder = get_encoder(
            encoder_name, in_channels=in_channels, weights=encoder_weights
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

        # Posterior
        self.posterior = HierarchicalPosterior(
            encoder_channels=encoder_channels,
            in_channels=in_channels,
            num_classes=num_classes,
            latent_channels=latent_channels,
            num_latent_scales=num_latent_scales,
        )

        self.encoder_channels = encoder_channels
        self.encoder_features = None
        self.prior_dists = None
        self.posterior_dists = None
        self.criterion = DiceCELoss(to_onehot_y=True, softmax=True, label_smoothing=0.3)

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        return list(self.encoder(x))

    def forward(
        self,
        patch: torch.Tensor,
        segm: Optional[torch.Tensor] = None,
        training: bool = True,
        temperature: float = 1.0,
    ):
        self.encoder_features = self.encode(patch)
        self.prior_dists = self.prior(self.encoder_features, temperature=temperature)

        if training and segm is not None:
            self.posterior_dists = self.posterior(
                patch, segmentation=segm, temperature=temperature
            )

    def sample(self, testing: bool = False) -> torch.Tensor:
        if testing:
            z_samples = [dist.sample() for dist in self.prior_dists]
        else:
            z_samples = [dist.rsample() for dist in self.prior_dists]
        return self.decoder(self.encoder_features, latents=z_samples)

    def reconstruct(self, use_posterior_mean: bool = False) -> torch.Tensor:
        if use_posterior_mean:
            z_samples = [dist.mean for dist in self.posterior_dists]
        else:
            z_samples = [dist.rsample() for dist in self.posterior_dists]
        return self.decoder(self.encoder_features, latents=z_samples)

    def elbo(self, segm: torch.Tensor) -> torch.Tensor:
        z_samples = [dist.rsample() for dist in self.posterior_dists]
        reconstruction = self.decoder(self.encoder_features, latents=z_samples)

        if segm.dim() == 3:
            target = segm
        elif segm.dim() == 4 and segm.shape[1] == 1:
            target = segm.squeeze(1)
        elif segm.dim() == 4 and segm.shape[1] == self.num_classes:
            target = segm.argmax(dim=1)
        else:
            target = segm

        reconstruction_loss = self.criterion(reconstruction, target.long().unsqueeze(1))

        kl_loss = sum(
            kl_divergence(post, prior).mean()
            for post, prior in zip(self.posterior_dists, self.prior_dists)
        )

        return reconstruction_loss + self.beta * kl_loss

    def predict_multiple_hypotheses(
        self, image: torch.Tensor, num_samples: int = 4
    ) -> torch.Tensor:
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
        logits_samples = self.predict_multiple_hypotheses(
            image, num_samples=num_samples
        )
        probs_samples = F.softmax(logits_samples, dim=2)
        return probs_samples.mean(dim=0)

    def freeze_encoder(self, freeze: bool = True):
        for param in self.encoder.parameters():
            param.requires_grad = not freeze


# ==============================================================================
# Test
# ==============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HierarchicalProbabilisticUNet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=5,
        latent_channels=6,
        num_latent_scales=4,
        beta=1.0,
    ).to(device)

    print(f"Encoder channels: {model.encoder_channels}")
    print(
        f"Posterior seg_encoder out_channels: {model.posterior.seg_encoder.out_channels}"
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    batch_size = 2
    image = torch.randn(batch_size, 3, 256, 256).to(device)
    segm = torch.randint(0, 5, (batch_size, 256, 256)).to(device)

    model.train()
    model(image, segm=segm, training=True)

    loss = model.elbo(segm)
    print(f"ELBO loss: {loss.item():.4f}")

    # Backward pass test
    loss.backward()
    print("Backward pass OK")

    model.eval()
    with torch.no_grad():
        samples = model.predict_multiple_hypotheses(image, num_samples=8)
        print(f"Samples shape: {samples.shape}")

        avg_pred = model.predict_segmentation(image, num_samples=4)
        print(f"Avg prediction shape: {avg_pred.shape}")

    print("✓ All tests passed!")
