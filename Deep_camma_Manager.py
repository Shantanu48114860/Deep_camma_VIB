import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader
from tqdm import tqdm

from Deep_camma_vib import Deep_Camma, Classifier
from Utils import Utils


class Deep_Camma_Manager:
    def __init__(self, device, n_classes, batch_size):
        self.device = device
        self.n_classes = n_classes
        self.batch_size = batch_size

    def train_clean(self, train_parameters):
        num_epochs = train_parameters["num_epochs"]
        variational_beta = train_parameters["variational_beta"]
        learning_rate = train_parameters["learning_rate"]
        weight_decay = train_parameters["weight_decay"]
        train_set_size = train_parameters["train_set_size"]
        train_dataset_clean = train_parameters["train_dataset_clean"]
        shuffle = train_parameters["shuffle"]
        model_save_path = train_parameters["model_save_path_clean"]

        deep_camma = Deep_Camma()
        deep_camma = deep_camma.to(self.device)

        optimizer = torch.optim.Adam(params=deep_camma.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay)
        deep_camma.train()
        BCE_Loss = nn.BCELoss(reduction="sum")
        train_data_loader_clean = DataLoader(train_dataset_clean,
                                             batch_size=self.batch_size,
                                             shuffle=shuffle)
        print(len(train_data_loader_clean))
        print("Training started..")
        for epoch in range(num_epochs):
            running_loss = 0.0
            running_recons_loss = 0.0
            running_kl_loss = 0.0
            train_size = 0.0
            with tqdm(total=len(train_data_loader_clean)) as t:
                for x_img, label in train_data_loader_clean:
                    x_img = x_img.to(self.device)
                    label = label.to(self.device)
                    y_one_hot = Utils.get_one_hot_labels(label, self.n_classes)
                    x_hat, z_mu, z_log_var, z, m_mu, m_log_var, m = deep_camma(x_img, y_one_hot, do_m=0)

                    recons_loss = BCE_Loss(x_hat, x_img)
                    kl_loss = variational_beta * Utils.kl_loss_clean(z_mu,
                                                                     z_log_var)
                    loss = recons_loss + kl_loss
                    optimizer.zero_grad()
                    loss.backward()

                    optimizer.step()

                    running_loss += loss.item()
                    running_recons_loss += recons_loss
                    running_kl_loss += kl_loss
                    train_size += x_img.size(0)
                    t.set_postfix(epoch='{:0}'.format(epoch + 1),
                                  recons_loss='{:05.3f}'.format(running_recons_loss / train_size),
                                  kl_loss='{:05.3f}'.format(running_kl_loss / train_size),
                                  deep_camma_loss='{:05.3f}'.format(running_loss / train_size),
                                  samples_trained='{:05.3f}'.format(train_size))
                    t.update()

        torch.save(deep_camma.state_dict(), model_save_path)
        print("Training completed..")

    def train_manipulated(self, train_parameters):
        num_epochs = train_parameters["num_epochs"]
        variational_beta = train_parameters["variational_beta"]
        learning_rate = train_parameters["learning_rate"]
        weight_decay = train_parameters["weight_decay"]
        train_set_size = train_parameters["train_set_size"]
        train_dataset_clean = train_parameters["train_dataset_clean"]
        train_dataset_do_m = train_parameters["train_dataset_do_m"]
        shuffle = train_parameters["shuffle"]
        model_save_path = train_parameters["model_save_path"]

        deep_camma = Deep_Camma()
        deep_camma = deep_camma.to(self.device)

        optimizer = torch.optim.Adam(params=deep_camma.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay)
        deep_camma.train()
        recons_loss = nn.BCELoss(reduction="sum")
        train_data_loader_clean = DataLoader(train_dataset_clean,
                                             batch_size=self.batch_size,
                                             shuffle=shuffle)
        train_data_loader_do_m = DataLoader(train_dataset_do_m,
                                            batch_size=self.batch_size,
                                            shuffle=shuffle)

        print("Training started..")
        for epoch in range(num_epochs):
            running_loss = 0.0
            train_size = 0.0
            train_clean_size = 0.0
            train_m_size = 0.0
            running_ELBO_clean = 0.0
            running_ELBO_do_m = 0.0

            with tqdm(total=len(train_data_loader_clean)) as t:
                for index, data in enumerate(zip(train_data_loader_clean, train_data_loader_do_m)):
                    x_img_clean, y_clean, x_img_do_m, y_do_m = data[0][0], data[0][1], data[1][0], data[1][1]
                    x_img_clean, y_clean, x_img_do_m, y_do_m = x_img_clean.to(self.device), \
                                                               y_clean.to(self.device), \
                                                               x_img_do_m.to(self.device), \
                                                               y_do_m.to(self.device)

                    y_one_hot_clean = Utils.get_one_hot_labels(y_clean, self.n_classes)
                    x_hat_clean, z_mu_clean, z_log_var_clean, z, \
                    m_mu_clean, m_log_var_clean, m = deep_camma(x_img_clean,
                                                                y_one_hot_clean,
                                                                do_m=0)
                    loss_recons_clean = recons_loss(x_hat_clean, x_img_clean)
                    ELBO_clean = 0.5 * (loss_recons_clean + variational_beta * Utils.kl_loss_clean(z_mu_clean,
                                                                                                   z_log_var_clean))

                    y_one_hot_do_m = Utils.get_one_hot_labels(y_do_m, self.n_classes)
                    x_hat_do_m, z_mu_do_m, z_log_var_do_m, z, \
                    m_mu_do_m, m_log_var_do_m, m = deep_camma(x_img_do_m,
                                                              y_one_hot_do_m,
                                                              do_m=1)
                    loss_recons_clean = recons_loss(x_hat_do_m, x_img_do_m)
                    ELBO_do_m = 0.5 * (loss_recons_clean + variational_beta * Utils.kl_loss_do_m(z_mu_do_m,
                                                                                                 z_log_var_do_m,
                                                                                                 m_mu_do_m,
                                                                                                 m_log_var_do_m))
                    loss = ELBO_clean + ELBO_do_m

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    running_ELBO_clean += ELBO_clean.item()
                    running_ELBO_do_m += ELBO_do_m.item()

                    train_clean_size += x_img_clean.size(0)
                    train_m_size += x_img_do_m.size(0)
                    train_size += x_img_clean.size(0) + x_img_do_m.size(0)
                    t.set_postfix(epoch='{:0}'.format(epoch + 1),
                                  ELBO_clean='{:05.3f}'.format(running_ELBO_clean / train_clean_size),
                                  ELBO_do_m='{:05.3f}'.format(running_ELBO_do_m / train_m_size),
                                  deep_camma_loss='{:05.3f}'.format(running_loss / train_size),
                                  samples_trained='{:0}'.format(train_size))
                    t.update()

        torch.save(deep_camma.state_dict(), model_save_path)
        print("Training completed..")

    def evaluate(self, test_parameters, do_m=0):
        print("Evaluate")
        test_dataset = test_parameters["test_dataset"]
        shuffle = test_parameters["shuffle"]
        original_file_name = test_parameters["original_file_name"]
        recons_file_name = test_parameters["recons_file_name"]
        deep_camma_generated_img_file_name = test_parameters["deep_camma_generated_img_file_name"]
        model_save_path = test_parameters["model_save_path"]

        deep_camma = Deep_Camma()
        deep_camma = deep_camma.to(self.device)
        deep_camma.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
        deep_camma.eval()

        test_data_loader = DataLoader(test_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=shuffle)

        BCE_Loss = nn.BCELoss(reduction="sum")
        running_loss = 0.0
        test_size = 0.0

        for x_img, label in test_data_loader:
            with torch.no_grad():
                x_img = x_img.to(self.device)
                label = label.to(self.device)
                y_one_hot = Utils.get_one_hot_labels(label, self.n_classes)
                x_hat, z_mu, z_log_var, z, m_mu, m_log_var, m = deep_camma(x_img, y_one_hot, do_m=do_m)

                recons_loss = BCE_Loss(x_hat, x_img).to(self.device)
                kl_loss = Utils.kl_loss_clean(z_mu,
                                              z_log_var)

                loss = recons_loss + kl_loss

                running_loss += loss.item()
                test_size += x_img.size(0)
        running_loss /= test_size
        print("Avg reconstruction error: {0}".format(running_loss))

        self.visualize(test_data_loader,
                       deep_camma,
                       original_file_name,
                       recons_file_name,
                       deep_camma_generated_img_file_name, do_m=do_m)

    def visualize(self, test_data_loader,
                  deep_camma,
                  original_file_name,
                  recons_file_name,
                  deep_camma_generated_img_file_name, do_m=0):
        images, labels = iter(test_data_loader).next()

        # First visualise the original images
        Utils.save_input_image(torchvision.utils.make_grid(images[1:50], 10, 5),
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

        self.deep_camma_generator(deep_camma,
                                  images,
                                  labels,
                                  self.batch_size,
                                  self.n_classes,
                                  self.device,
                                  fig_name=deep_camma_generated_img_file_name,
                                  dim_M=32,
                                  dim_Z=64,
                                  do_m=do_m)
        print("Deep Camma generated image has been saved.")

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

            print(latent_m)
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

            Utils.save_input_image(torchvision.utils.make_grid(x_hat.data[:100], 10, 5),
                                   fig_name=fig_name)
            # plt.show()

    def train_classifier(self, train_parameters):

        num_epochs = train_parameters["num_epochs"]
        learning_rate = train_parameters["learning_rate"]
        weight_decay = train_parameters["weight_decay"]
        train_set_size = train_parameters["train_set_size"]
        train_dataset_clean = train_parameters["train_dataset_clean"]
        train_dataset_do_m = train_parameters["train_dataset_do_m"]
        shuffle = train_parameters["shuffle"]
        deep_camma_save_path = train_parameters["deep_camma_save_path"]
        classifier_save_path = train_parameters["classifier_save_path"]

        deep_camma = Deep_Camma()
        deep_camma = deep_camma.to(self.device)
        deep_camma.load_state_dict(torch.load(deep_camma_save_path))

        classifier = Classifier()
        classifier = classifier.to(self.device)
        optimizer = torch.optim.Adam(params=classifier.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay)

        merged_train_dataset = torch.utils.data.ConcatDataset([train_dataset_clean, train_dataset_do_m])
        data_loader = DataLoader(merged_train_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=shuffle)

        for epoch in range(num_epochs):
            running_loss = 0.0
            train_size = 0.0
            running_correct = 0.0

            with tqdm(total=len(data_loader)) as t:
                for x_img, y in data_loader:
                    x_img = x_img.to(self.device)
                    y = y.to(self.device)
                    y_one_hot = Utils.get_one_hot_labels(y, self.n_classes)
                    x_hat, z_mu, z_log_var, m_mu, m_log_var = deep_camma(x_img, y_one_hot, do_m=1)
                    latent_z = Utils.reparametrize(z_mu, z_log_var).to(self.device)
                    y_hat = classifier(latent_z)
                    loss = F.cross_entropy(y_hat, y)

                    optimizer.zero_grad()
                    loss.backward()

                    optimizer.step()

                    running_loss += loss.item()
                    running_correct += Utils.get_num_correct(y_hat, y)

                    train_size += x_img.size(0)
                    t.set_postfix(epoch='{:0}'.format(epoch + 1),
                                  classifier_loss='{:05.3f}'.format(running_loss / train_size),
                                  classifier_accuracy='{:05.3f}'.format(running_correct / train_size),
                                  samples_trained='{:0}'.format(train_size))
                    t.update()

        torch.save(classifier.state_dict(), classifier_save_path)
        print("Training completed..")

    def predict(self, test_parameters, m):
        test_dataset = test_parameters["test_dataset"]
        shuffle = test_parameters["shuffle"]
        model_save_path = test_parameters["model_save_path"]

        deep_camma = Deep_Camma()
        deep_camma.load_state_dict(torch.load(model_save_path))
        deep_camma = deep_camma.to(self.device)
        deep_camma.eval()

        test_data_loader = DataLoader(test_dataset,
                                      batch_size=1,
                                      shuffle=shuffle)

        recons_loss = nn.BCELoss(reduction="sum")
        running_loss = 0.0
        test_size = 0.0
        correct = []
        for x_img, label in test_data_loader:
            with torch.no_grad():
                test_size += 1
                x_img = x_img.to(self.device)
                label = label.to(self.device)
                activation_tensor = torch.from_numpy(np.array(
                    self.get_activations(x_img,
                                         deep_camma, recons_loss,
                                         label, m)))
                softmax = nn.Softmax(dim=0)
                op = softmax(activation_tensor)
                # print(op)
                # print(label)
                # print(op.argmax())
                correct.append(op.argmax().eq(label).item())

        correct_estimate = np.count_nonzero(np.array(correct))
        print(correct_estimate / test_size)

    def get_activations(self, x_img, deep_camma, recons_loss, label, m):
        class_val = torch.empty(label.size(0), dtype=torch.float)
        activations = []
        for y_c in range(10):
            class_val.fill_(y_c)
            # print(class_val.size())
            y_one_hot = Utils.get_one_hot_labels(class_val.to(torch.int64), self.n_classes).to(self.device)
            x_hat, z_mu, z_log_var, latent_z, m_mu, m_log_var, latent_m = deep_camma(x_img, y_one_hot,
                                                                                     do_m=m)
            log_p_yc = torch.log(torch.tensor(1000 / 10000))
            z_normal = MultivariateNormal(torch.zeros((z_mu.size(0), z_mu.size(1))),
                                          torch.eye(z_mu.size(1)))
            log_p_z = z_normal.log_prob(z_normal.sample())
            # print(log_p_z)

            x_hat_flatten = x_hat.view(x_hat.size(0), -1)
            x_img_flatten = x_img.view(x_img.size(0), -1)
            # print(x_hat_flatten.size())

            print(x_hat_flatten.get_device())
            print(x_img_flatten.get_device())
            p_theta_normal = MultivariateNormal(x_hat_flatten, torch.eye(x_hat_flatten.size(1)))
            log_p_theta = p_theta_normal.log_prob(x_img_flatten)
            # print(log_p_theta)

            q_phi_normal = MultivariateNormal(z_mu, torch.eye(z_mu.size(1)))
            log_q_phi = q_phi_normal.log_prob(latent_z).to(self.device)
            activation_val = log_p_theta + log_p_yc + log_p_z - log_q_phi

            # p_theta = recons_loss(x_hat,
            #                       x_img).to(self.device)
            # activation_val = p_theta - Utils.kl_loss_clean(z_mu, z_log_var) + \
            #                  torch.log(torch.tensor(1 / 10))
            activations.append(activation_val.item())

        activation_tensor = torch.from_numpy(np.array(activations))
        return activation_tensor
