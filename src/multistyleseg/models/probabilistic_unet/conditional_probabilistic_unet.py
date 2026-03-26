# This code is based on: https://github.com/SimonKohl/probabilistic_unet

from torch.distributions import Normal, Independent, kl
import torch
import torch.nn as nn
from monai.losses import DiceCELoss
import segmentation_models_pytorch as smp
from multistyleseg.models.probabilistic_unet.probabilistic_unet import (
    Fcomb,
    AxisAlignedConvGaussian,
)


class ConditionalProbabilisticUnet(nn.Module):
    """
    A probabilistic UNet (https://arxiv.org/abs/1806.05034) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: is a list consisint of the amount of filters layer
    latent_dim: dimension of the latent space
    no_cons_per_block: no convs per block in the (convolutional) encoder of prior and posterior
    """

    def __init__(
        self,
        input_channels=1,
        num_classes=1,
        latent_dim=6,
        no_convs_fcomb=4,
        beta=10.0,
        embedding_dim=8,
        embedding_classes=5,
        embedding_to_input=False,
        embedding_to_output=False,
    ):
        super(ConditionalProbabilisticUnet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.initializers = {"w": "he_normal", "b": "normal"}
        self.beta = beta
        self.z_prior_sample = 0
        self.conditional_embedding = nn.Embedding(embedding_classes, embedding_dim)
        self.embedding_to_input = embedding_to_input
        self.embedding_to_output = embedding_to_output

        if self.embedding_to_input and self.embedding_to_output:
            raise ValueError(
                "You can only use one of embedding_to_input or embedding_to_output"
            )
        num_filters = [32, 64, 128, 192]

        self.unet = smp.create_model(
            arch="unet",
            encoder_name="se_resnet50",
            encoder_weights="imagenet",
            in_channels=self.input_channels
            + (embedding_dim if embedding_to_input else 0),
            classes=self.num_classes,
        )
        self.unet.segmentation_head = nn.Identity()
        self.prior = AxisAlignedConvGaussian(
            self.input_channels + embedding_dim,
            num_filters,
            self.no_convs_per_block,
            self.latent_dim,
            self.initializers,
        )
        self.posterior = AxisAlignedConvGaussian(
            self.input_channels + self.num_classes - 1 + embedding_dim,
            num_filters,
            self.no_convs_per_block,
            self.latent_dim,
            self.initializers,
            posterior=True,
        )
        last_unet_chan = self.unet.decoder.blocks[-1].conv2[0].out_channels
        self.fcomb = Fcomb(
            self.latent_dim,
            last_unet_chan + (embedding_dim if embedding_to_output else 0),
            self.num_classes,
            self.no_convs_fcomb,
            {"w": "orthogonal", "b": "normal"},
        )

        self.criterion = DiceCELoss(to_onehot_y=True, softmax=True, label_smoothing=0.3)

    def forward(self, patch, segm, class_label, training=True):
        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        self.class_emb = (
            self.conditional_embedding(class_label)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, patch.size(2), patch.size(3))
        )
        if self.embedding_to_input:
            patch = torch.cat([patch, self.class_emb], dim=1)
        self.unet_features = self.unet(patch)
        if not self.embedding_to_input:
            patch = torch.cat([patch, self.class_emb], dim=1)
        if training:
            self.posterior_latent_space = self.posterior.forward(patch, segm)
        self.prior_latent_space = self.prior.forward(patch)

    def sample(self, testing=False):
        """
        Sample a segmentation by reconstructing from a prior sample
        and combining this with UNet features
        """
        if not testing:
            z_prior = self.prior_latent_space.rsample()
            self.z_prior_sample = z_prior
        else:
            # You can choose whether you mean a sample or the mean here. For the GED it is important to take a sample.
            # z_prior = self.prior_latent_space.base_dist.loc
            z_prior = self.prior_latent_space.sample()
            self.z_prior_sample = z_prior
        unet_features = self.unet_features
        if self.embedding_to_output:
            unet_features = torch.cat([self.unet_features, self.class_emb], dim=1)
        return self.fcomb.forward(unet_features, z_prior)

    def predict_segmentation(self, image, class_label, num_samples=4):
        """
        Predict segmentation by sampling from the prior latent space multiple times
        and averaging the resulting segmentations
        num_samples: number of samples to draw from the prior latent space
        """
        self.forward(image, class_label=class_label, segm=None, training=False)
        segm_samples = []
        for _ in range(num_samples):
            segm_sample = self.sample(testing=True)
            segm_samples.append(segm_sample.unsqueeze(0))
        segm_samples = torch.cat(segm_samples, dim=0)
        return torch.mean(segm_samples, dim=0)

    def predict_multiple_hypotheses(self, image, class_label, num_samples=4):
        """
        Generate multiple segmentation hypotheses by sampling from the prior latent space multiple times
        num_samples: number of samples to draw from the prior latent space
        """
        self.forward(image, class_label=class_label, segm=None, training=False)
        segm_samples = []
        for _ in range(num_samples):
            segm_sample = self.sample(testing=True)
            segm_samples.append(segm_sample.unsqueeze(0))
        segm_samples = torch.cat(segm_samples, dim=0)
        return segm_samples  # Return all samples without averaging

    def reconstruct(
        self, use_posterior_mean=False, calculate_posterior=False, z_posterior=None
    ):
        """
        Reconstruct a segmentation from a posterior sample (decoding a posterior sample) and UNet feature map
        use_posterior_mean: use posterior_mean instead of sampling z_q
        calculate_posterior: use a provided sample or sample from posterior latent space
        """
        if use_posterior_mean:
            z_posterior = self.posterior_latent_space.loc
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
        unet_features = self.unet_features
        if self.embedding_to_output:
            unet_features = torch.cat([self.unet_features, self.class_emb], dim=1)
        return self.fcomb.forward(unet_features, z_posterior)

    def kl_divergence(self, analytic=True, calculate_posterior=False, z_posterior=None):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
        """
        if analytic:
            kl_div = kl.kl_divergence(
                self.posterior_latent_space, self.prior_latent_space
            )
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        return kl_div

    def elbo(self, segm, analytic_kl=True, reconstruct_posterior_mean=False):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """

        z_posterior = self.posterior_latent_space.rsample()

        self.kl = torch.mean(
            self.kl_divergence(
                analytic=analytic_kl, calculate_posterior=False, z_posterior=z_posterior
            )
        )

        # Here we use the posterior sample sampled above
        self.reconstruction = self.reconstruct(
            use_posterior_mean=reconstruct_posterior_mean,
            calculate_posterior=False,
            z_posterior=z_posterior,
        )

        reconstruction_loss = self.criterion(
            input=self.reconstruction, target=segm.unsqueeze(1)
        )
        self.reconstruction_loss = torch.sum(reconstruction_loss)
        self.mean_reconstruction_loss = torch.mean(reconstruction_loss)

        return -(self.reconstruction_loss + self.beta * self.kl)
