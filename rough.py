from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from Utils import Utils
from data_loader import Data_Loader


def do_rough(img_transform1, img_transform2):
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    print("Current day: ", dt_string)

    device = Utils.get_device()
    print("Device: {0}".format(device))

    hyper_parameters = {
        "num_epochs": 100,
        "variational_beta": 1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "n_classes": 10,
        "train_set_size": 57000,
        "val_set_size": 3000,
        "test_set_size": 10000,
        "batch_size": 128,
        "shuffle": True,
        "do_m": 0,
        "model_save_path": "Model/deep_camma_clean_{0}.pth".format(dt_string),
        "dataset_path": "./data/MNIST",
        "original_file_name": "./Plots/Original_image_{0}.jpeg".format(dt_string),
        "recons_file_name": "./Plots/Reconstructed_image_{0}.jpeg".format(dt_string),
        "deep_camma_generated_img_file_name": "./Plots/VAE_Generated_image_{0}.jpeg".format(dt_string)
    }

    torch.manual_seed(1)

    dL = Data_Loader()
    train_dataset, val_set, test_dataset = dL.load_train_MNIST(
        hyper_parameters["dataset_path"],
        img_transform1,
        train_set_size=hyper_parameters["train_set_size"],
        val_set_size=hyper_parameters["val_set_size"])

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=hyper_parameters["batch_size"],
                                   shuffle=hyper_parameters["shuffle"])

    x_img, label = iter(train_data_loader).next()
    x_img = x_img.to(device)[0]
    label = label.to(device)[0]
    print(x_img.size())
    print(label.item())

    img = Utils.to_img(x_img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap="gray")
    plt.draw()
    plt.savefig("./Plots/Rough_Orig.png", dpi=220)
    plt.clf()

    dL = Data_Loader()
    train_dataset, val_set, test_dataset = dL.load_train_MNIST(
        hyper_parameters["dataset_path"],
        img_transform2,
        train_set_size=hyper_parameters["train_set_size"],
        val_set_size=hyper_parameters["val_set_size"])

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=hyper_parameters["batch_size"],
                                   shuffle=hyper_parameters["shuffle"])

    x_img, label = iter(train_data_loader).next()
    x_img = x_img.to(device)[0]
    label = label.to(device)[0]
    print(x_img.size())
    print(label.item())

    img = Utils.to_img(x_img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap="gray")
    plt.draw()
    plt.savefig("./Plots/Rough_Hori.png", dpi=220)
    plt.clf()


img_transform1 = transforms.Compose([
    transforms.ToTensor()
])

img_transform2 = transforms.Compose([
    transforms.RandomAffine(degrees=0, translate=(0, 0.78)),
    transforms.ToTensor()
])

do_rough(img_transform1, img_transform2)
