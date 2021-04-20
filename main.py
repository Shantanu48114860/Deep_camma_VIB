# Press the green button in the gutter to run the script.
from datetime import datetime

from Experiments import Experiments

if __name__ == '__main__':
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    print("Current day: ", dt_string)

    hyper_parameters_MNIST = {
        "num_epochs": 500,
        "num_epochs_classifier": 10,
        "variational_beta": 1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "n_classes": 10,
        "train_set_size": 57000,
        "val_set_size": 3000,
        "test_set_size": 10000,
        "batch_size": 128,
        "shuffle": True,
        "model_save_path_clean": "Model/deep_camma_clean_{0}.pth".format(dt_string),
        "model_save_path_do_m": "Model/deep_camma_do_m_{0}.pth".format(dt_string),
        "classifier_save_path": "Model/classifier_{0}.pth".format(dt_string),
        "dataset_path": "./data/MNIST",

        "original_file_name_clean": "./Plots/Clean_Original_image_{0}.jpeg".format(dt_string),
        "recons_file_name_clean": "./Plots/Clean_Reconstructed_image_{0}.jpeg".format(dt_string),
        "deep_camma_generated_img_file_name_clean": "./Plots/Clean_Generated_image_{0}.jpeg".format(dt_string),

        "original_file_name_do_m": "./Plots/Do_m_Original_image_{0}.jpeg".format(dt_string),
        "recons_file_name_do_m": "./Plots/Do_m_Reconstructed_image_{0}.jpeg".format(dt_string),
        "deep_camma_generated_img_file_name_do_m": "./Plots/Do_m_Generated_image_{0}.jpeg".format(dt_string)
    }

    do_m = {
        "m": 1,
        "degrees": 0,
        "horizontal_shift": 0,
        "vertical_shift": 0.98
    }

    experiments = Experiments()

    # train clean
    # experiments.run_all_experiments_clean(hyper_parameters_MNIST, do_m)

    # train do_m
    experiments.run_all_experiments_do_m(hyper_parameters_MNIST, do_m)

    # clean_model_name = "Model/deep_camma_clean_17_04_2021_01_50_19.pth"
    clean_model_name = "Model/deep_camma_do_m_17_04_2021_01_50_53.pth"
    original_file_name_disentangle = "./Plots/Disentangle_Original_image_{0}.jpeg".format(dt_string)
    recons_file_name_disentangle = "./Plots/Disentangle_Reconstructed_image_{0}.jpeg".format(dt_string)
    deep_camma_generated_img_file_name_disentangle = "./Plots/Disentangle_Generated_image_{0}.jpeg".format(dt_string)

    # run disentangle
    # experiments.run_disentangle_experiments_do_m(hyper_parameters_MNIST, do_m,
    #                                              clean_model_name,
    #                                              original_file_name_disentangle,
    #                                              recons_file_name_disentangle,
    #                                              deep_camma_generated_img_file_name_disentangle)

    # predict
    do_m_model_name = "Model/deep_camma_do_m_17_04_2021_01_50_53.pth"
    # experiments.predict(hyper_parameters_MNIST, do_m_model_name)

    # classify using DNN
    # experiments.train_classifier(hyper_parameters_MNIST, do_m_model_name, do_m)

