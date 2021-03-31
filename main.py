# Press the green button in the gutter to run the script.
from datetime import datetime

from Experiments import Experiments

if __name__ == '__main__':
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    print("Current day: ", dt_string)

    hyper_parameters_MNIST = {
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

    experiments = Experiments()
    experiments.run_all_experiments(hyper_parameters_MNIST)
