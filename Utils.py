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
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    @staticmethod
    def kl_loss_do_m(z_mu_clean, z_log_var_clean,
                     m_mu_clean, m_log_var_clean):
        return -torch.sum(1 + 0.5 * z_log_var_clean
                          - 0.5 * (z_mu_clean.pow(2)
                                   + z_log_var_clean.exp())) + torch.sum(0.5 * m_log_var_clean
                                                                         - 0.5 * (m_mu_clean.pow(2)
                                                                                  + m_log_var_clean.exp()))

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
        npimg = img.numpy()
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
            images, _, _, _, _ = deep_camma(x_img, y_one_hot, "clean")
            images = images.cpu()
            images = Utils.to_img(images)
            np_imagegrid = torchvision.utils.make_grid(images[1:50], 10, 5).numpy()
            plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
            plt.draw()
            plt.savefig(fig_name, dpi=220)
            plt.clf()
            # plt.show()
