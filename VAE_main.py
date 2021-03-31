import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from VAE_Model import VAE


class VAE_Model:
    @staticmethod
    def vae_loss(x_hat, x, mu, logvar, variational_beta):
        recon_loss = F.binary_cross_entropy(x_hat.view(-1, 784), x.view(-1, 784), reduction='sum')

        kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return variational_beta * kldivergence

    def train(self, train_data_loader, device):
        num_epochs = 10
        variational_beta = 1
        learning_rate = 1e-3
        vae = VAE()
        vae = vae.to(device)

        num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
        print('Number of parameters: %d' % num_params)

        optimizer = torch.optim.Adam(params=vae.parameters(),
                                     lr=learning_rate, weight_decay=1e-5)
        vae.train()

        train_loss_avg = []
        loss_VAE_recons_MSE = nn.MSELoss()
        print('Training ...')
        for epoch in range(num_epochs):
            train_loss_avg.append(0)
            num_batches = 0

            for image_batch, _ in train_data_loader:
                 with torch.autograd.detect_anomaly():
                    image_batch = image_batch.to(device)

                    # vae reconstruction
                    image_batch_recon, latent_mu, latent_logvar = vae(image_batch)

                    # reconstruction error
                    loss_recons = F.binary_cross_entropy(image_batch_recon.view(-1, 784),
                                                         image_batch.view(-1, 784), reduction='sum')
                    # loss_recons = loss_VAE_recons_MSE(image_batch_recon.float(),
                    #                                   image_batch.float()).to(device)
                    loss = loss_recons + self.vae_loss(image_batch_recon, image_batch,
                                                       latent_mu, latent_logvar,
                                                       variational_beta)

                    # backpropagation
                    optimizer.zero_grad()
                    loss.backward()

                    # one step of the optmizer (using the gradients from backpropagation)
                    optimizer.step()

                    train_loss_avg[-1] += loss.item()
                    num_batches += 1

            train_loss_avg[-1] /= num_batches
            print('Epoch [%d / %d] average reconstruction error: %f' % (epoch + 1, num_epochs, train_loss_avg[-1]))

        torch.save(vae.state_dict(), "./Model/VAE_model.pth")
        return vae, train_loss_avg

    def test_vae(self, vae, test_data_loader, device):
        vae.eval()

        test_loss_avg, num_batches = 0, 0
        for image_batch, _ in test_data_loader:
            with torch.no_grad():
                image_batch = image_batch.to(device)

                # vae reconstruction
                image_batch_recon, latent_mu, latent_logvar = vae(image_batch)

                # reconstruction error
                loss = self.vae_loss(image_batch_recon, image_batch,
                                     latent_mu, latent_logvar, variational_beta=1)

                test_loss_avg += loss.item()
                num_batches += 1

        test_loss_avg /= num_batches
        print('average reconstruction error: %f' % (test_loss_avg))

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

    def show_image(self, img, fig_name):
        img = self.to_img(img)
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.draw()
        plt.savefig(fig_name, dpi=220)
        plt.clf()
        # plt.show()

    def visualise_output(self, images, model, fig_name):
        with torch.no_grad():
            images = images.to(device)
            images, _, _ = model(images)
            images = images.cpu()
            images = self.to_img(images)
            np_imagegrid = torchvision.utils.make_grid(images[1:50], 10, 5).numpy()
            plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
            plt.draw()
            plt.savefig(fig_name, dpi=220)
            plt.clf()
            # plt.show()

    def vae_generator(self, vae, latent_dims=2):
        vae.eval()

        with torch.no_grad():
            # sample latent vectors from the normal distribution
            latent = torch.randn(128, latent_dims, device=device)

            # reconstruct images from the latent vectors
            img_recon = vae.decoder(latent)
            img_recon = img_recon.cpu()

            fig, ax = plt.subplots(figsize=(5, 5))
            self.show_image(torchvision.utils.make_grid(img_recon.data[:100], 10, 5),
                            fig_name="./Plots/VAE_Generated_image.jpeg")
            # plt.show()


if __name__ == "__main__":
    img_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    batch_size = 128

    train_dataset = MNIST(root='./data/MNIST', download=True, train=True,
                          transform=img_transform)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MNIST(root='./data/MNIST', download=True, train=False, transform=img_transform)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    vae_model = VAE_Model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("### Device: {0} ###".format(device))

    vae, train_loss_avg = vae_model.train(train_data_loader, device)

    # vae = VAE().to(device)
    # vae.load_state_dict(torch.load("./Model/VAE_model_1000epoch.pth",
    #                                    map_location=device))

    vae_model.plot_loss(train_loss_avg, fig_name="./Plots/Loss_plot.jpeg")
    vae_model.test_vae(vae, test_data_loader, device)

    images, labels = iter(test_data_loader).next()

    # First visualise the original images
    print('Original images')
    vae_model.show_image(torchvision.utils.make_grid(images[1:50], 10, 5), fig_name="./Plots/Original_image.jpeg")
    # plt.show()

    # Reconstruct and visualise the images using the vae
    print('VAE reconstruction:')
    vae_model.visualise_output(images, vae, fig_name="./Plots/Reconstructed_image.jpeg")

    print('VAE Generated:')
    vae_model.vae_generator(vae, latent_dims=2)
