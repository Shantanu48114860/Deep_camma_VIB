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
    def show_input_image(img, fig_name):
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

    @staticmethod
    def deep_camma_generator(deep_camma, x_img, labels,
                             batch_size,
                             n_classes,
                             device,
                             fig_name,
                             dim_M=32,
                             dim_Z=64,
                             do_m=0):
        deep_camma.eval()

        with torch.no_grad():
            x_img = x_img.to(device)
            labels = labels.to(device)
            y_one_hot = Utils.get_one_hot_labels(labels, n_classes)
            # sample latent vectors from the normal distribution
            if do_m == 0:
                latent_m = torch.zeros((x_img.size(0), dim_M)).to(device)
            else:
                latent_m = torch.randn(batch_size, dim_M, device=device)

            latent_z = torch.randn(batch_size, dim_Z, device=device)

            if torch.cuda.is_available():
                y_one_hot = y_one_hot.float().cuda()
            else:
                y_one_hot = y_one_hot.float()

            z = deep_camma.decoder_NN_p_Z(latent_z)
            y = deep_camma.decoder_NN_p_Y(y_one_hot)
            m = deep_camma.decoder_NN_p_M(latent_m)
            x_hat = deep_camma.decoder_NN_p_merge(z, y, m)

            # reconstruct images from the latent vectors
            x_hat = x_hat.to(device)

            fig, ax = plt.subplots(figsize=(5, 5))
            Utils.show_input_image(torchvision.utils.make_grid(x_hat.data[:100], 10, 5),
                                   fig_name=fig_name)
            # plt.show()
