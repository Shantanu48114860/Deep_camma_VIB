import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm

from VIB_Model import VIB


class VAE_Model:
    @staticmethod
    def get_num_correct(preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    @staticmethod
    def vae_loss(y_hat, y, mu, logvar):
        classification_loss = F.cross_entropy(y_hat, y)

        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return classification_loss, kl_divergence

    def train(self, train_data_loader, device):
        num_epochs = 100
        variational_beta = 1e-3
        learning_rate = 1e-3
        vib = VIB()
        vib = vib.to(device)

        num_params = sum(p.numel() for p in vib.parameters() if p.requires_grad)
        print('Number of parameters: %d' % num_params)

        optimizer = torch.optim.Adam(params=vib.parameters(),
                                     lr=learning_rate, weight_decay=1e-5)
        vib.train()

        print('Training ...')
        for epoch in range(num_epochs):
            running_loss = 0.0
            running_classification_loss = 0.0
            running_kl_loss = 0.0
            train_size = 0.0
            total_correct = 0.0

            with tqdm(total=len(train_data_loader)) as t:
                for x_img, y in train_data_loader:
                    x_img = x_img.to(device)
                    x_img = x_img.view(x_img.size(0), -1)
                    y = y.to(device)
                    y_hat, latent_mu, latent_log_var = vib(x_img)

                    classification_loss, kl_divergence = self.vae_loss(y_hat, y,
                                                                       latent_mu,
                                                                       latent_log_var)

                    total_loss = classification_loss + variational_beta * kl_divergence

                    # backpropagation
                    optimizer.zero_grad()
                    total_loss.backward()

                    optimizer.step()

                    running_loss += total_loss.item()
                    running_classification_loss += classification_loss
                    running_kl_loss += kl_divergence
                    train_size += x_img.size(0)
                    total_correct += self.get_num_correct(y_hat, y)

                    t.set_postfix(epoch='{:0}'.format(epoch + 1),
                                  classification_loss='{:05.3f}'.format(running_classification_loss / train_size),
                                  kl_loss='{:05.3f}'.format(running_kl_loss / train_size),
                                  deep_camma_loss='{:05.3f}'.format(running_loss / train_size),
                                  total_correct='{:05.3f}'.format(total_correct),
                                  train_accuracy='{:05.3f}'.format(total_correct / train_size),
                                  samples_trained='{:05.3f}'.format(train_size))
                    t.update()
        torch.save(vib.state_dict(), "./Model/VIB/VAE_model.pth")
        return vib

    def test_vae(self, vib, test_data_loader, device):
        vib.eval()

        total_loss = 0
        total_correct = 0
        test_size = 0.0
        for x_img, y in test_data_loader:
            x_img = x_img.to(device)
            x_img = x_img.view(x_img.size(0), -1)
            y = y.to(device)

            # forward propagation
            y_hat, latent_mu, latent_log_var = vib(x_img)
            total_correct += self.get_num_correct(y_hat, y)
            test_size += x_img.size(0)

        print("Test Accuracy: {0}".format(total_correct / test_size))


if __name__ == "__main__":
    img_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    batch_size = 128
    path = "./data/MNIST"
    train_dataset = MNIST(root=path,
                          download=True,
                          train=True,
                          transform=img_transform)

    test_dataset = MNIST(root=path,
                         download=True,
                         train=False,
                         transform=img_transform)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    vae_model = VAE_Model()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("### Device: {0} ###".format(device))

    vib = vae_model.train(train_data_loader, device)

    # vae = VAE().to(device)
    # vae.load_state_dict(torch.load("./Model/VAE_model_1000epoch.pth",
    #                                    map_location=device))

    vae_model.test_vae(vib, test_data_loader, device)
