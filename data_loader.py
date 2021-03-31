import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST


class Data_Loader:
    def load_train_MNIST(self, path, train_set_size, val_set_size):
        img_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        train_dataset = MNIST(root=path, download=True, train=True,
                              transform=img_transform)

        test_dataset = MNIST(root=path, download=True, train=False, transform=img_transform)

        val_set = []
        train_set, val_set = torch.utils.data.random_split(train_dataset,
                                                           [train_set_size, val_set_size])
        return train_set, val_set, test_dataset
