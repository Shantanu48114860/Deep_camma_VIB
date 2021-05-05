import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal
from torch.utils.data import DataLoader
from tqdm import tqdm

from Deep_camma_vib import Deep_Camma
from Utils import Utils


class Deep_Camma_Manager_Predict:
    def __init__(self, n_classes, batch_size, test_parameters, m=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_classes = n_classes
        self.batch_size = batch_size

        model_save_path = test_parameters["model_save_path"]
        print(model_save_path)

        self.deep_camma = Deep_Camma()
        self.deep_camma.load_state_dict(torch.load(model_save_path))
        self.deep_camma = self.deep_camma.to(self.device)
        self.deep_camma.eval()
        self.do_m = m

    def __call__(self, x):
        if len(x.shape) != 4:
            x = x.unsqueeze(0)

        with torch.no_grad():
            x = x.to(self.device)
            activation_tensor = torch.from_numpy(np.array(
                self.get_activations(x, self.deep_camma, self.do_m)))
            probs = F.softmax(activation_tensor, dim=1)

        return probs

    def cuda(self):
        self.deep_camma = self.deep_camma.to("cuda")

    def eval(self):
        self.deep_camma.eval()

    def get_activations(self, x_img, deep_camma, m):
        class_val = torch.empty(x_img.size(0), dtype=torch.float)
        activations = torch.zeros((x_img.size(0), 1))
        for y_c in range(10):
            class_val.fill_(y_c)
            # print(class_val.size())
            y_one_hot = Utils.get_one_hot_labels(class_val.to(torch.int64),
                                                 self.n_classes).to(self.device)
            x_hat, z_mu, z_log_var, latent_z, m_mu, m_log_var, latent_m = deep_camma(x_img, y_one_hot,
                                                                                     do_m=m)

            p_yc = torch.tensor(1000 / 10000)
            z_normal = MultivariateNormal(torch.zeros((z_mu.size(0), z_mu.size(1))),
                                          torch.eye(z_mu.size(1)))
            log_p_z = z_normal.log_prob(z_normal.sample())
            p_z_proba = log_p_z.exp()

            x_hat_flatten = x_hat.view(x_hat.size(0), -1).cpu()
            x_img_flatten = x_img.view(x_img.size(0), -1).cpu()
            # print(x_hat_flatten.size())

            p_theta_normal = MultivariateNormal(x_hat_flatten, torch.eye(x_hat_flatten.size(1)))
            log_p_theta = p_theta_normal.log_prob(x_img_flatten.cpu())
            # print(log_p_theta.size())
            # print(log_p_theta[0])
            # print(log_p_theta.exp())

            # p_theta_normal = Normal(x_hat_flatten, 1.0)
            # log_p_theta = p_theta_normal.log_prob(x_img_flatten.cpu())
            # p_theta_proba = torch.sum(log_p_theta.exp(), dim=1)

            z_mu = z_mu.cpu()
            z_log_var = z_log_var.exp().cpu()
            latent_z = latent_z.cpu()

            # print(z_log_var.size())
            # print(z_mu.size())

            q_phi_normal = Normal(z_mu, z_log_var.exp())
            log_q_phi = q_phi_normal.log_prob(latent_z.cpu())
            q_phi_proba = torch.sum(log_q_phi.exp(), dim=1)

            # print(q_phi_proba.size())
            # print(p_theta_proba.size())
            # print(p_z_proba.size())
            #
            # print(q_phi_proba)
            # print(p_theta_proba)
            # print(p_z_proba)
            # activation_val = torch.log((p_theta_proba * p_yc * p_z_proba) / q_phi_proba)

            kl = Utils.kl_loss_clean_predict(z_mu, z_log_var)
            activation_val = log_p_theta + torch.log(p_yc) + log_p_z - kl
            # q_phi_normal = MultivariateNormal(z_mu, z_log_var.exp())
            # log_q_phi = q_phi_normal.log_prob(latent_z)
            # activation_val = log_p_theta + p_yc + log_p_z - log_q_phi

            # p_theta = recons_loss(x_hat,
            #                       x_img).to(self.device)
            # activation_val = p_theta - Utils.kl_loss_clean(z_mu, z_log_var) + \
            #                  torch.log(torch.tensor(1 / 10))

            activation_val = activation_val.reshape(activation_val.size(0), -1)
            activations = torch.cat((activations, activation_val), dim=1)

        activation_tensor = torch.from_numpy(np.array(activations))
        # print(activation_tensor.size())
        # print(activation_tensor)
        activation_tensor = activation_tensor[:, 1:]
        # print(activation_tensor.size())
        # print(activation_tensor)
        # print(x)
        return activation_tensor