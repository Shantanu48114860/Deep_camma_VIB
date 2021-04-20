import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils import Utils


class Encoder_NN_q_M(nn.Module):
    def __init__(self, in_channel=1, out_channel=64, lin_dims=500, dim_M=32):
        super(Encoder_NN_q_M, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1))
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1))
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1))
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(in_features=out_channel * 4 * 4, out_features=lin_dims)
        self.fc_mu = nn.Linear(in_features=lin_dims, out_features=dim_M)

        self.fc_log_var = nn.Linear(in_features=lin_dims, out_features=dim_M)

    def forward(self, x):
        x = self.max_pool1(F.relu((self.conv1(x))))
        x = self.max_pool2(F.relu((self.conv2(x))))
        x = self.max_pool3(F.relu((self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        m_mu = self.fc_mu(x)
        m_logvar = self.fc_log_var(x)
        return m_mu, m_logvar


class Encoder_NN_q_Z(nn.Module):
    def __init__(self, in_channel=1, out_channel=64, lin_dims=500,
                 dim_Y=10, dim_M=32, dim_Z=64):
        super(Encoder_NN_q_Z, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=(5, 5),
                               stride=(1, 1), padding=(2, 2))
        self.max_pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=(5, 5),
                               stride=(1, 1), padding=(2, 2))
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1))

        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=(5, 5),
                               stride=(1, 1), padding=(2, 2))
        self.max_pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.fc1 = nn.Linear(in_features=out_channel * 4 * 4, out_features=lin_dims)
        self.fc2 = nn.Linear(in_features=lin_dims + dim_Y + dim_M, out_features=lin_dims)
        self.fc_mu = nn.Linear(in_features=lin_dims, out_features=dim_Z)

        self.fc_log_var = nn.Linear(in_features=lin_dims, out_features=dim_Z)

    def forward(self, x, y, m):
        x = self.max_pool1(F.relu((self.conv1(x))))
        x = self.max_pool2(F.relu((self.conv2(x))))
        x = self.max_pool3(F.relu((self.conv3(x))))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x_concat = F.relu(self.fc2(torch.cat((x, y, m), dim=1)))
        x_mu = self.fc_mu(x_concat)
        x_logvar = self.fc_log_var(x_concat)
        return x_mu, x_logvar


class Decoder_NN_p_Y(nn.Module):
    def __init__(self, dim_Y=10, lin_dims=500):
        super(Decoder_NN_p_Y, self).__init__()
        self.fc1 = nn.Linear(in_features=dim_Y, out_features=lin_dims)
        self.fc2 = nn.Linear(in_features=lin_dims, out_features=lin_dims)

    def forward(self, y):
        y = F.relu((self.fc1(y)))
        y = F.relu((self.fc2(y)))
        return y


class Decoder_NN_p_Z(nn.Module):
    def __init__(self, dim_Z=64, lin_dims=500):
        super(Decoder_NN_p_Z, self).__init__()
        self.fc1 = nn.Linear(in_features=dim_Z, out_features=lin_dims)
        self.fc2 = nn.Linear(in_features=lin_dims, out_features=lin_dims)

    def forward(self, z):
        z = F.relu((self.fc1(z)))
        z = F.relu((self.fc2(z)))
        return z


class Decoder_NN_p_M(nn.Module):
    def __init__(self, dim_M=32, lin_dims=500):
        super(Decoder_NN_p_M, self).__init__()
        self.fc1 = nn.Linear(in_features=dim_M, out_features=lin_dims)
        self.fc2 = nn.Linear(in_features=lin_dims, out_features=lin_dims)
        self.fc3 = nn.Linear(in_features=lin_dims, out_features=lin_dims)
        self.fc4 = nn.Linear(in_features=lin_dims, out_features=lin_dims)

    def forward(self, m):
        m = F.relu((self.fc1(m)))
        m = F.relu((self.fc2(m)))
        m = F.relu((self.fc3(m)))
        m = F.relu((self.fc4(m)))
        return m


class Decoder_NN_p_merge(nn.Module):
    def __init__(self, dim_hidden=1500, proj_dims=4, channels=64, out_channels=1):
        super(Decoder_NN_p_merge, self).__init__()
        self.channels = channels
        self.proj_dims = proj_dims
        self.fc1 = nn.Linear(in_features=dim_hidden, out_features=channels * proj_dims * proj_dims)
        self.conv1 = nn.ConvTranspose2d(in_channels=channels, out_channels=channels,
                                        kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.ConvTranspose2d(in_channels=channels, out_channels=channels,
                                        kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv3 = nn.ConvTranspose2d(in_channels=channels, out_channels=out_channels,
                                        kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))

    def forward(self, z, y, m):
        x = F.relu((self.fc1(torch.cat((z, y, m), dim=1))))
        x = x.view(x.size(0), self.channels, self.proj_dims, self.proj_dims)
        x = F.relu((self.conv1(x)))
        x = F.relu((self.conv2(x)))
        x = torch.sigmoid((self.conv3(x)))
        return x


class Deep_Camma(nn.Module):
    def __init__(self, dim_M=32):
        super(Deep_Camma, self).__init__()
        self.dim_M = dim_M
        self.encoder_NN_q_M = Encoder_NN_q_M(dim_M=dim_M)
        self.encoder_NN_q_Z = Encoder_NN_q_Z()

        self.decoder_NN_p_Y = Decoder_NN_p_Y()
        self.decoder_NN_p_Z = Decoder_NN_p_Z()
        self.decoder_NN_p_M = Decoder_NN_p_M()
        self.decoder_NN_p_merge = Decoder_NN_p_merge()

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()

            return mu + eps * std

        else:
            return mu

    def forward(self, x, y_one_hot, do_m):
        device = Utils.get_device()
        y_one_hot = y_one_hot.float()

        if do_m == 0:
            m_mu = torch.zeros((x.size(0), self.dim_M))
            m_logvar = torch.zeros((x.size(0), self.dim_M))
            latent_m = torch.zeros((x.size(0), self.dim_M))
        else:
            m_mu, m_logvar = self.encoder_NN_q_M(x)
            latent_m = self.reparametrize(m_mu, m_logvar)

        latent_m = latent_m.to(device)
        z_mu, z_logvar = self.encoder_NN_q_Z(x, y_one_hot, latent_m)
        latent_z = self.reparametrize(z_mu, z_logvar)

        z = self.decoder_NN_p_Z(latent_z)
        y = self.decoder_NN_p_Y(y_one_hot)
        m = self.decoder_NN_p_M(latent_m)
        x_hat = self.decoder_NN_p_merge(z, y, m)

        return x_hat, z_mu, z_logvar, m_mu, m_logvar


class Classifier(nn.Module):
    def __init__(self, input_channel=64, out_channel=10):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_features=input_channel, out_features=out_channel)

    def forward(self, x):
        return self.fc1(x)


