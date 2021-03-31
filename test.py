def _train(self, train_params, device):
        num_epochs = train_params["num_epochs"]
        VAE_BETA = train_params["variational_beta"]
        learning_rate = train_params["num_epochs"]
        weight_decay = train_params["weight_decay"]
        train_dataset_clean = train_params["train_dataset_clean"]
        batch_size = train_params["batch_size"]
        shuffle = train_params["shuffle"]
        n_classes = train_params["n_classes"]

        # deep_camma = Deep_Camma()
        deep_camma = VAE()
        deep_camma = deep_camma.to(device)

        optimizer = torch.optim.Adam(params=deep_camma.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay)
        # train_data_loader_clean = DataLoader(train_dataset_clean, batch_size=batch_size,
        #                                      shuffle=shuffle)
        deep_camma.train()

        train_loss_avg = []
        loss_VAE_recons_MSE = nn.MSELoss()
        for epoch in range(num_epochs):
            train_loss_avg.append(0)
            num_batches = 0

            for image_batch, _ in train_dataset_clean:
                with torch.autograd.detect_anomaly():
                    image_batch = image_batch.to(device)

                    # vae reconstruction
                    image_batch_recon, latent_mu, latent_logvar = deep_camma(image_batch)

                    # reconstruction error
                    loss_recons = loss_VAE_recons_MSE(image_batch_recon.float(),
                                                      image_batch.float()).to(device)
                    loss = loss_recons + Utils.kl_loss_clean(latent_mu, latent_logvar)

                    # backpropagation
                    optimizer.zero_grad()
                    loss.backward()

                    # one step of the optmizer (using the gradients from backpropagation)
                    optimizer.step()

                    train_loss_avg[-1] += loss.item()
                    num_batches += 1

            train_loss_avg[-1] /= num_batches
            print('Epoch [%d / %d] average reconstruction error: %f' % (epoch + 1, num_epochs, train_loss_avg[-1]))

        for epoch in range(num_epochs):
            train_loss_clean = 0
            with tqdm(total=len(train_data_loader_clean)) as t:
                for image_batch, label in train_data_loader_clean:
                    with torch.autograd.detect_anomaly():
                        image_batch = image_batch.to(device)
                        label = label.to(device)
                        y_one_hot = Utils.get_one_hot_labels(label, n_classes)
                        # x_hat, z_mu, z_logvar = deep_camma(image_batch, y_one_hot, "clean")
                        x_hat, z_mu, z_logvar = deep_camma(image_batch)

                        if torch.cuda.is_available():
                            loss_recons = loss_VAE_recons_MSE(x_hat.float().cuda(),
                                                              image_batch.float().cuda()).to(device)
                        else:
                            loss_recons = loss_VAE_recons_MSE(x_hat.float(),
                                                              image_batch.float()).to(device)

                        kl_loss_clean = Utils.kl_loss_clean(z_mu, z_logvar)
                        loss_deep_camma_clean = loss_recons + VAE_BETA * kl_loss_clean

                        # print(deep_camma.encoder_NN_q_Z.fc1.weight[])
                        optimizer.zero_grad()
                        loss_deep_camma_clean.backward()
                        optimizer.step()
                        print("Grad:")
                        # print(deep_camma.encoder_NN_q_Z.fc1.weight)
                        train_loss_clean += loss_deep_camma_clean.item()
                        # print(image_batch[16].size())
                        # print(x_hat[16].size())
                        # print("Image:")
                        # print(image_batch[16])
                        # print("x_hat:")
                        # print(x_hat[16])
                        # print(loss_recons.item())
                        # print(kl_loss_clean)
                        # print(loss_deep_camma_clean)
                        print("---------------------")
                    break
                    t.set_postfix(epoch='{0}'.format(epoch),
                                  VAE_loss='{0}'.format(loss_recons.item()))
                    t.update()

            #     break
            # break