import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_dims=784, shared_nodes=1024, out_dims=256):
        super(Encoder, self).__init__()
        self.shared1 = nn.Linear(in_features=in_dims, out_features=shared_nodes)

        self.shared2 = nn.Linear(in_features=shared_nodes, out_features=shared_nodes)

        self.fc_mu = nn.Linear(in_features=shared_nodes, out_features=out_dims)
        self.fc_log_var = nn.Linear(in_features=shared_nodes, out_features=out_dims)

    def forward(self, x):
        x = F.relu((self.shared1(x)))
        x = F.relu((self.shared2(x)))

        m_mu = self.fc_mu(x)
        m_log_var = self.fc_log_var(x)
        return m_mu, m_log_var


class Decoder(nn.Module):
    def __init__(self, in_dims=256, out_dims=10):
        super(Decoder, self).__init__()
        self.shared1 = nn.Linear(in_features=in_dims, out_features=out_dims)

    def forward(self, x):
        return self.shared1(x)


class VIB(nn.Module):
    def __init__(self):
        super(VIB, self).__init__()
        self.encoder = Encoder(in_dims=784, shared_nodes=1024, out_dims=256)
        self.decoder = Decoder(in_dims=256, out_dims=10)

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()

            return mu + eps * std

        else:
            return mu

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent_z = self.reparametrize(latent_mu, latent_logvar)
        y_hat = self.decoder(latent_z)

        return y_hat, latent_mu, latent_logvar
