# This code is based on: https://github.com/SimonKohl/probabilistic_unet
from multistyleseg.models.probabilistic_unet.unet import Unet
from multistyleseg.models.probabilistic_unet.utils import (
    init_weights,
    init_weights_orthogonal_normal,
)
from torch.distributions import Normal, Independent, kl
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from monai.losses import DiceCELoss
import segmentation_models_pytorch as smp


class Encoder(nn.Module):
    """
    A convolutional neural network, consisting of len(num_filters) times a block of no_convs_per_block convolutional layers,
    after each block a pooling operation is performed. And after each convolutional layer a non-linear (ReLU) activation function is applied.
    """

    def __init__(
        self,
        input_channels,
        num_filters,
        no_convs_per_block,
        initializers,
        padding=True,
        posterior=False,
    ):
        super(Encoder, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters

        if posterior:
            # To accomodate for the mask that is concatenated at the channel axis, we increase the input_channels.
            self.input_channels += 1

        layers = []
        output_dim = None
        for i in range(len(self.num_filters)):
            """
            Determine input_dim and output_dim of conv layers in this block. The first layer is input x output,
            All the subsequent layers are output x output.
            """
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]

            if i != 0:
                layers.append(
                    nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
                )

            layers.append(
                nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding))
            )
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block - 1):
                layers.append(
                    nn.Conv2d(
                        output_dim, output_dim, kernel_size=3, padding=int(padding)
                    )
                )
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, input):
        output = self.layers(input)
        return output


class AxisAlignedConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """

    def __init__(
        self,
        input_channels,
        num_filters,
        no_convs_per_block,
        latent_dim,
        initializers,
        posterior=False,
    ):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        if self.posterior:
            self.name = "Posterior"
        else:
            self.name = "Prior"
        self.encoder = Encoder(
            self.input_channels,
            self.num_filters,
            self.no_convs_per_block,
            initializers,
            posterior=self.posterior,
        )
        self.conv_layer = nn.Conv2d(
            num_filters[-1], 2 * self.latent_dim, (1, 1), stride=1
        )
        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

        nn.init.kaiming_normal_(
            self.conv_layer.weight, mode="fan_in", nonlinearity="relu"
        )
        nn.init.normal_(self.conv_layer.bias)

    def forward(self, input, segm=None):
        # If segmentation is not none, concatenate the mask to the channel axis of the input
        if segm is not None:
            self.show_img = input
            self.show_seg = segm
            # Convert segm to one hot encoding
            segm = F.one_hot(segm.long().squeeze(1), num_classes=5)
            segm = segm.permute(0, 3, 1, 2).float()
            input = torch.cat((input, segm), dim=1)
            self.show_concat = input
            self.sum_input = torch.sum(input)

        encoding = self.encoder(input)

        # We only want the mean of the resulting hxw image
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)

        # Convert encoding to 2 x latent dim and split up for mu and log_sigma
        mu_log_sigma = self.conv_layer(encoding)

        # We squeeze the second dimension twice, since otherwise it won't work when batch size is equal to 1
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        mu = mu_log_sigma[:, : self.latent_dim]
        log_sigma = mu_log_sigma[:, self.latent_dim :]

        # This is a multivariate normal with diagonal covariance matrix sigma
        # https://github.com/pytorch/pytorch/pull/11178
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)
        return dist


class Fcomb(nn.Module):
    """
    A function composed of no_convs_fcomb times a 1x1 convolution that combines the sample taken from the latent space,
    and output of the UNet (the feature map) by concatenating them along their channel axis.
    """

    def __init__(
        self,
        latent_dim,
        num_output_channels,
        num_classes,
        no_convs_fcomb,
        initializers,
    ):
        super(Fcomb, self).__init__()
        self.num_channels = num_output_channels  # output channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.no_convs_fcomb = no_convs_fcomb
        self.name = "Fcomb"

        layers = []

        # Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the last layer
        layers.append(
            nn.Conv2d(
                self.num_channels + self.latent_dim,
                self.num_channels,
                kernel_size=1,
            )
        )
        layers.append(nn.ReLU(inplace=True))

        for _ in range(no_convs_fcomb - 2):
            layers.append(
                nn.Conv2d(self.num_channels, self.num_channels, kernel_size=1)
            )
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.last_layer = nn.Conv2d(self.num_channels, self.num_classes, kernel_size=1)

        if initializers["w"] == "orthogonal":
            self.layers.apply(init_weights_orthogonal_normal)
            self.last_layer.apply(init_weights_orthogonal_normal)
        else:
            self.layers.apply(init_weights)
            self.last_layer.apply(init_weights)

    def forward(self, feature_map, z):
        """
        Z is batch_sizexlatent_dim and feature_map is batch_sizexno_channelsxHxW.
        So broadcast Z to batch_sizexlatent_dimxHxW. Behavior is exactly the same as tf.tile (verified)
        """
        h, w = feature_map.shape[-2:]
        # Tile z to have same spatial dimensions as feature map
        z = z.unsqueeze(2)
        z = z.unsqueeze(3)
        z = z.repeat(
            1,
            1,
            h,
            w,
        )
        # Concatenate feature map and z along the channel axis
        x = torch.cat((feature_map, z), dim=1)

        x = self.layers(x)
        x = self.last_layer(x)
        return x


class ProbabilisticUnet(nn.Module):
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
    ):
        super(ProbabilisticUnet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.initializers = {"w": "he_normal", "b": "normal"}
        self.beta = beta
        self.z_prior_sample = 0
        num_filters = [32, 64, 128, 192]

        self.unet = smp.create_model(
            arch="unet",
            encoder_name="se_resnet50",
            encoder_weights="imagenet",
            in_channels=self.input_channels,
            classes=self.num_classes,
        )
        self.unet.segmentation_head = nn.Identity()
        self.prior = AxisAlignedConvGaussian(
            self.input_channels,
            num_filters,
            self.no_convs_per_block,
            self.latent_dim,
            self.initializers,
        )
        self.posterior = AxisAlignedConvGaussian(
            self.input_channels + self.num_classes - 1,
            num_filters,
            self.no_convs_per_block,
            self.latent_dim,
            self.initializers,
            posterior=True,
        )
        last_unet_chan = self.unet.decoder.blocks[-1].conv2[0].out_channels
        self.fcomb = Fcomb(
            self.latent_dim,
            last_unet_chan,
            self.num_classes,
            self.no_convs_fcomb,
            {"w": "orthogonal", "b": "normal"},
        )

        self.criterion = DiceCELoss(to_onehot_y=True, softmax=True, label_smoothing=0.3)

    def forward(self, patch, segm, training=True):
        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        if training:
            self.posterior_latent_space = self.posterior.forward(patch, segm)
        self.prior_latent_space = self.prior.forward(patch)
        self.unet_features = self.unet(patch)

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
        return self.fcomb.forward(self.unet_features, z_prior)

    def predict_segmentation(self, image, num_samples=4):
        """
        Predict segmentation by sampling from the prior latent space multiple times
        and averaging the resulting segmentations
        num_samples: number of samples to draw from the prior latent space
        """
        self.forward(image, segm=None, training=False)
        segm_samples = []
        for _ in range(num_samples):
            segm_sample = self.sample(testing=True)
            segm_samples.append(segm_sample.unsqueeze(0))
        segm_samples = torch.cat(segm_samples, dim=0)
        return torch.mean(segm_samples, dim=0)

    def predict_multiple_hypotheses(self, image, num_samples=4):
        """
        Generate multiple segmentation hypotheses by sampling from the prior latent space multiple times
        num_samples: number of samples to draw from the prior latent space
        """
        self.forward(image, segm=None, training=False)
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
        return self.fcomb.forward(self.unet_features, z_posterior)

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
        self.reconstruction_loss = reconstruction_loss

        return -self.beta * self.kl - self.reconstruction_loss
