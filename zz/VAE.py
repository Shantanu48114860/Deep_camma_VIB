import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channel=1, out_channel=64, lin_dims=500,
                 dim_Y=10, dim_M=32, dim_Z=64):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=5,
                               stride=1, padding=2)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=5,
                               stride=1, padding=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=5,
                               stride=1, padding=2)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(in_features=out_channel * 4 * 4, out_features=lin_dims)
        self.fc2 = nn.Linear(in_features=lin_dims, out_features=lin_dims)
        self.fc_mu = nn.Linear(in_features=lin_dims, out_features=dim_Z)

        self.fc_logvar = nn.Linear(in_features=lin_dims, out_features=dim_Z)


class Decoder(nn.Module):
    def __init__(self, capacity=64, latent_dims=2):
        super(Decoder, self).__init__()
        self.c = capacity
        self.fc = nn.Linear(in_features=latent_dims, out_features=self.c * 2 * 7 * 7)
        self.conv2 = nn.ConvTranspose2d(in_channels=self.c * 2, out_channels=self.c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=self.c, out_channels=1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.fc(x)

        # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = x.view(x.size(0), self.c * 2, 7, 7)

        # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv1(x))
        return x


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparametrize(self, mu, logvar):
        # the reparameterization trick
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()

            return mu + eps * std

        else:
            return mu

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent_z = self.reparametrize(latent_mu, latent_logvar)
        x_hat = self.decoder(latent_z)
        # print("x_hat")
        # print(x_hat)
        return x_hat, latent_mu, latent_logvar

enc = Encoder()
dec = Decoder()
print(enc)
print(dec)
