import torch
from torchvision.datasets import MNIST


class Data_Loader:
    def __init__(self, path, train_set_size, val_set_size):
        self.path = path
        self.train_set_size = train_set_size
        self.val_set_size = val_set_size

    def load_train_MNIST(self, img_transform_obj):
        dataset_do_m = None
        dataset_clean = self.get_train_val_test_dataset(
            img_transform=img_transform_obj["img_transform_clean"])

        if img_transform_obj["img_transform_manipulated"] is not None:
            dataset_do_m = self.get_train_val_test_dataset(
                img_transform=img_transform_obj["img_transform_manipulated"])

        return {
            "dataset_clean": dataset_clean,
            "dataset_do_m": dataset_do_m
        }

    def get_train_val_test_dataset(self, img_transform):
        train_dataset = MNIST(root=self.path,
                              download=True,
                              train=True,
                              transform=img_transform)

        test_dataset = MNIST(root=self.path,
                             download=True,
                             train=False,
                             transform=img_transform)

        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset,
            [self.train_set_size,
             self.val_set_size])

        return {
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "test_dataset": test_dataset
        }
