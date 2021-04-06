import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, capacity=64, latent_dims=2):
        super(Encoder, self).__init__()
        c = capacity
        # kernel: c x 1 x4 x 4
        # output: c x 14 x 14
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1)

        # output: c*2 x 7 x 7
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c * 2, kernel_size=4, stride=2, padding=1)

        self.fc_mu = nn.Linear(in_features=c * 2 * 7 * 7, out_features=latent_dims)

        self.fc_logvar = nn.Linear(in_features=c * 2 * 7 * 7, out_features=latent_dims)

    def forward(self, x):
        x = F.relu((self.conv1(x)))
        print("########### Encoder ")
        print(x[0][0])
        x = F.relu((self.conv2(x)))
        print("########### Encoder")
        print(x[0][0])
        x = x.view(x.size(0), -1)
        x_mu = self.fc_mu(x)
        print("########### Mu")
        print(x[0])
        x_logvar = self.fc_logvar(x)
        print("########### fc_logvar")
        print(x[0])
        return x_mu, x_logvar


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
        print("x_hat")
        print(x_hat)
        return x_hat, latent_mu, latent_logvar

enc = Encoder()
dec = Decoder()
print(enc)
print(dec)
