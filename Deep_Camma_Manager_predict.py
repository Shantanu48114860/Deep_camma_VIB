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

        test_dataset = test_parameters["test_dataset"]
        shuffle = test_parameters["shuffle"]
        model_save_path = test_parameters["model_save_path"]
        print(model_save_path)

        self.deep_camma = Deep_Camma()
        self.deep_camma.load_state_dict(torch.load(model_save_path))
        self.deep_camma = self.deep_camma.to(self.device)
        self.deep_camma.eval()
        self.do_m = m
        self.test_data_loader = DataLoader(test_dataset,
                                           batch_size=128,
                                           shuffle=shuffle)

    # def __call__(self, x):
    #     recons_loss = nn.BCELoss(reduction="sum")
    #     running_loss = 0.0
    #     test_size = 0.0
    #     total_correct = 0.0
    #     correct = []
    #     with tqdm(total=len(self.test_data_loader)) as t:
    #         for x_img in self.x:
    #             with torch.no_grad():
    #                 test_size += x_img.size(0)
    #                 x_img = x_img.to(self.device)
    #                 label = label.to(self.device)
    #                 activation_tensor = torch.from_numpy(np.array(
    #                     self.get_activations(x_img,
    #                                          self.deep_camma, recons_loss,
    #                                          label, self.do_m)))
    #                 softmax = nn.Softmax(dim=0)
    #                 preds = softmax(activation_tensor)
    #                 # print(op)
    #                 # print(label)
    #                 # print(op.argmax())
    #                 total_correct += Utils.get_num_correct(preds.cpu(), label.cpu())
    #                 t.set_postfix(total_correct='{:05.3f}'.format(total_correct),
    #                               test_size='{:05.3f}'.format(test_size),
    #                               accuracy='{:0}'.format(total_correct / test_size))
    #                 t.update()
    #
    #     # correct_estimate = np.count_nonzero(np.array(correct))
    #     print("Total correct: {0}".format(total_correct))
    #     print("Accuracy: {0}".format(total_correct / test_size))
    #     # return .....

    def __call__(self, x):
        recons_loss = nn.BCELoss(reduction="sum")
        running_loss = 0.0
        test_size = 0.0
        total_correct = 0.0
        correct = []
        probs = []
        count = 0
        with tqdm(total=len(self.test_data_loader)) as t:
            for x_img, label in self.test_data_loader:
                with torch.no_grad():
                    test_size += x_img.size(0)
                    x_img = x_img.to(self.device)
                    label = label.to(self.device)
                    activation_tensor = torch.from_numpy(np.array(
                        self.get_activations(x_img,
                                             self.deep_camma, recons_loss,
                                             label, self.do_m)))
                    preds = F.softmax(activation_tensor, dim=1)
                    probs.append(preds)
                    # print(op)
                    # print(label)
                    # print(op.argmax())
                    total_correct += Utils.get_num_correct(preds.cpu(), label.cpu())
                    t.set_postfix(total_correct='{:05.3f}'.format(total_correct),
                                  test_size='{:05.3f}'.format(test_size),
                                  accuracy='{:0}'.format(total_correct / test_size))
                    t.update()
                    
        # correct_estimate = np.count_nonzero(np.array(correct))
        print("Total correct: {0}".format(total_correct))
        print("Accuracy: {0}".format(total_correct / test_size))
        probs_output = torch.cat(probs, dim=0)
        return probs_output

    def get_activations(self, x_img, deep_camma, recons_loss, label, m):
        class_val = torch.empty(label.size(0), dtype=torch.float)
        activations = torch.zeros((label.size(0), 1))
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
