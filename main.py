# Press the green button in the gutter to run the script.
from datetime import datetime

from Experiments import Experiments

if __name__ == '__main__':
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    print("Current day: ", dt_string)
    num_epochs = 15

    hyper_parameters_MNIST = {
        # "num_epochs": 75,
        "num_epochs": num_epochs,
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
        "model_save_path_clean": "Model/deep_camma_clean_{0}_epochs_{1}.pth".format(dt_string, num_epochs),
        "model_save_path_do_m": "Model/deep_camma_do_m_{0}_epochs_{1}.pth".format(dt_string, num_epochs),
        "classifier_save_path": "Model/classifier_{0}_epochs_{1}.pth".format(dt_string, num_epochs),
        "dataset_path": "./data/MNIST",

        "original_file_name_clean": "./Plots/Clean_Original_image_{0}_epochs_{1}.jpeg".format(dt_string, num_epochs),
        "recons_file_name_clean": "./Plots/Clean_Reconstructed_image_{0}_epochs_{1}.jpeg".format(dt_string, num_epochs),
        "deep_camma_generated_img_file_name_clean": "./Plots/Clean_Generated_image_{0}_epochs_{1}.jpeg".format(
            dt_string, num_epochs),

        "original_file_name_do_m": "./Plots/Do_m_Original_image_{0}_epochs_{1}.jpeg".format(dt_string, num_epochs),
        "recons_file_name_do_m": "./Plots/Do_m_Reconstructed_image_{0}_epochs_{1}.jpeg".format(dt_string, num_epochs),
        "deep_camma_generated_img_file_name_do_m": "./Plots/Do_m_Generated_image_{0}_epochs_{1}.jpeg".format(dt_string,
                                                                                                             num_epochs)
    }

    do_m = {
        "m": 1,
        "degrees": 0,
        "horizontal_shift": 0,
        "vertical_shift": .25
    }

    experiments = Experiments()

    # train clean
    # experiments.run_all_experiments_clean(hyper_parameters_MNIST)

    # train do_m
    # experiments.run_all_experiments_do_m(hyper_parameters_MNIST, do_m)

    # clean_model_name = "Model/deep_camma_clean_17_04_2021_01_50_19.pth"
    clean_model_name = "Model/clean/deep_camma_clean_24_04_2021_02_30_11_epochs_100.pth"
    original_file_name_disentangle = "./Plots/Disentangle_Original_image_{0}_epochs_20.jpeg".format(dt_string)
    recons_file_name_disentangle = "./Plots/Disentangle_Reconstructed_image_{0}_epochs_20.jpeg".format(dt_string)
    deep_camma_generated_img_file_name_disentangle = "./Plots/Disentangle_Generated_image_{0}_epochs_20.jpeg".format(dt_string)

    # run disentangle
    # experiments.run_disentangle_experiments_do_m(hyper_parameters_MNIST, do_m,
    #                                              clean_model_name,
    #                                              original_file_name_disentangle,
    #                                              recons_file_name_disentangle,
    #                                              deep_camma_generated_img_file_name_disentangle)

    # predict
    do_m_model_name = "Model/do_m_vertical_shift_0.25/deep_camma_do_m_27_04_2021_08_09_54_epochs_35.pth"
    do_m_model_name = "Model/deep_camma_do_m_27_04_2021_22_23_32_epochs_15.pth"
    experiments.predict(hyper_parameters_MNIST, do_m_model_name)

    # classify using DNN
    # experiments.train_classifier(hyper_parameters_MNIST, do_m_model_name, do_m)
