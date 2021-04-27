import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision


class Utils:
    @staticmethod
    def convert_df_to_np_arr(data):
        return data.to_numpy()

    @staticmethod
    def convert_to_col_vector(np_arr):
        return np_arr.reshape(np_arr.shape[0], 1)

    @staticmethod
    def get_device():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_one_hot_labels(labels, n_classes):
        return F.one_hot(labels, n_classes)

    @staticmethod
    def kl_loss_clean(mu, log_var):
        return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

    @staticmethod
    def kl_loss_do_m(z_mu_clean, z_log_var_clean,
                     m_mu_clean, m_log_var_clean):
        z_kl = torch.mean(-0.5 * torch.sum(1 + z_log_var_clean - z_mu_clean.pow(2) - z_log_var_clean.exp(),
                                           dim=1), dim=0)
        m_kl = torch.mean(-0.5 * torch.sum(1 + m_log_var_clean - m_mu_clean.pow(2) - m_log_var_clean.exp(),
                                           dim=1), dim=0)

        return z_kl + m_kl

    @staticmethod
    def get_num_correct(preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    @staticmethod
    def reparametrize(mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.empty_like(std).normal_()

        return mu + eps * std

    @staticmethod
    def plot_loss(train_loss_avg, fig_name):
        plt.ion()
        fig = plt.figure()
        plt.plot(train_loss_avg)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        # plt.show()
        plt.draw()
        plt.savefig(fig_name, dpi=220)
        plt.clf()

    @staticmethod
    def to_img(x):
        x = x.clamp(0, 1)
        return x

    @staticmethod
    def save_input_image(img, fig_name):
        img = Utils.to_img(img)
        npimg = img.cpu().numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.draw()
        plt.savefig(fig_name, dpi=220)
        plt.clf()
        # plt.show()

    @staticmethod
    def reconstruct_image(images, labels, n_classes, deep_camma, fig_name, device):
        with torch.no_grad():
            x_img = images.to(device)
            labels = labels.to(device)
            y_one_hot = Utils.get_one_hot_labels(labels, n_classes)
            images, _, _, _, _, _, _ = deep_camma(x_img, y_one_hot, do_m=0)
            images = images.cpu()
            images = Utils.to_img(images)
            np_imagegrid = torchvision.utils.make_grid(images[1:50], 10, 5).numpy()
            plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
            plt.draw()
            plt.savefig(fig_name, dpi=220)
            plt.clf()
            # plt.show()
