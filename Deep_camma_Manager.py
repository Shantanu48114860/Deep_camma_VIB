import torch
import torch.nn as nn
import torchvision.utils
from torch.utils.data import DataLoader
from tqdm import tqdm

from Deep_camma_vib import Deep_Camma
from Utils import Utils


class Deep_Camma_Manager:
    def __init__(self, device, model_save_path, n_classes, batch_size, do_m):
        self.device = device
        self.model_save_path = model_save_path
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.do_m = do_m

    def train(self, train_parameters):
        num_epochs = train_parameters["num_epochs"]
        variational_beta = train_parameters["variational_beta"]
        learning_rate = train_parameters["learning_rate"]
        weight_decay = train_parameters["weight_decay"]
        train_set_size = train_parameters["train_set_size"]
        train_dataset = train_parameters["train_dataset"]
        shuffle = train_parameters["shuffle"]

        deep_camma = Deep_Camma()
        deep_camma = deep_camma.to(self.device)

        train_data_loader = DataLoader(train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=shuffle)
        optimizer = torch.optim.Adam(params=deep_camma.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay)
        deep_camma.train()

        recons_loss = nn.BCELoss(reduction="sum")
        print("Training started..")
        for epoch in range(num_epochs):
            running_loss = 0.0
            train_size = 0.0
            with tqdm(total=len(train_data_loader)) as t:
                for x_img, label in train_data_loader:
                    x_img = x_img.to(self.device)
                    label = label.to(self.device)
                    y_one_hot = Utils.get_one_hot_labels(label, self.n_classes)
                    x_hat, z_mu, z_log_var, m_mu, m_log_var = deep_camma(x_img, y_one_hot, self.do_m)

                    loss_recons = recons_loss(x_hat, x_img)
                    loss = loss_recons + variational_beta * Utils.kl_loss_clean(z_mu,
                                                                                z_log_var)
                    optimizer.zero_grad()
                    loss.backward()

                    optimizer.step()

                    running_loss += loss.item()
                    train_size += x_img.size(0)
                    t.set_postfix(epoch='{:0}'.format(epoch + 1),
                                  deep_camma_loss='{:05.3f}'.format(running_loss / train_size),
                                  samples_trained='{:05.3f}'.format(train_size))
                    t.update()

        torch.save(deep_camma.state_dict(), self.model_save_path)
        print("Training completed..")

    def evaluate(self, test_parameters):
        test_dataset = test_parameters["test_dataset"]
        shuffle = test_parameters["shuffle"]
        original_file_name = test_parameters["original_file_name"]
        recons_file_name = test_parameters["recons_file_name"]
        deep_camma_generated_img_file_name = test_parameters["deep_camma_generated_img_file_name"]

        deep_camma = Deep_Camma()
        deep_camma = deep_camma.to(self.device)
        deep_camma.load_state_dict(torch.load(self.model_save_path))
        deep_camma.eval()

        test_data_loader = DataLoader(test_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=shuffle)

        recons_loss = nn.BCELoss(reduction="sum")
        running_loss = 0.0
        test_size = 0.0

        for x_img, label in test_data_loader:
            with torch.no_grad():
                x_img = x_img.to(self.device)
                label = label.to(self.device)
                y_one_hot = Utils.get_one_hot_labels(label, self.n_classes)
                x_hat, z_mu, z_log_var, m_mu, m_log_var = deep_camma(x_img, y_one_hot, self.do_m)

                loss_recons = recons_loss(x_hat,
                                          x_img).to(self.device)
                loss = loss_recons + Utils.kl_loss_clean(z_mu, z_log_var)

                running_loss += loss.item()
                test_size += x_img.size(0)

        running_loss /= test_size
        print("Avg reconstruction error: {0}".format(running_loss))

        self.visualize(test_data_loader,
                       deep_camma,
                       original_file_name,
                       recons_file_name,
                       deep_camma_generated_img_file_name)

    def visualize(self, test_data_loader,
                  deep_camma,
                  original_file_name,
                  recons_file_name,
                  deep_camma_generated_img_file_name):
        images, labels = iter(test_data_loader).next()

        # First visualise the original images
        Utils.show_input_image(torchvision.utils.make_grid(images[1:50], 10, 5),
                               fig_name=original_file_name)
        print("Original images saved.")

        # Reconstruct and visualise the images using the vae
        Utils.reconstruct_image(images,
                                labels,
                                self.n_classes,
                                deep_camma,
                                fig_name=recons_file_name,
                                device=self.device)
        print("Deep Camma reconstruction has been completed.")

        Utils.deep_camma_generator(deep_camma,
                                   images,
                                   labels,
                                   self.batch_size,
                                   self.n_classes,
                                   self.device,
                                   fig_name=deep_camma_generated_img_file_name,
                                   dim_M=32,
                                   dim_Z=64,
                                   do_m=self.do_m)
        print("Deep Camma generated image has been saved.")
