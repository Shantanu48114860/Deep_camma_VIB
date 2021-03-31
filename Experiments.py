from Deep_camma_Manager import Deep_Camma_Manager
from Utils import Utils
from data_loader import Data_Loader


class Experiments:
    def run_all_experiments(self, hyper_parameters):
        device = Utils.get_device()
        print("Device: {0}".format(device))

        dL = Data_Loader()
        train_dataset, val_set, test_dataset = dL.load_train_MNIST(hyper_parameters["dataset_path"],
                                                                   train_set_size=hyper_parameters["train_set_size"],
                                                                   val_set_size=hyper_parameters["val_set_size"])

        train_parameters = {
            "num_epochs": hyper_parameters["num_epochs"],
            "variational_beta": hyper_parameters["variational_beta"],
            "learning_rate": hyper_parameters["learning_rate"],
            "weight_decay": hyper_parameters["weight_decay"],
            "train_set_size": hyper_parameters["train_set_size"],
            "train_dataset": train_dataset,
            "shuffle": hyper_parameters["shuffle"]
        }

        deep_camma_manager = Deep_Camma_Manager(device,
                                                hyper_parameters["model_save_path"],
                                                hyper_parameters["n_classes"],
                                                hyper_parameters["batch_size"],
                                                hyper_parameters["do_m"])
        deep_camma_manager.train(train_parameters)

        test_parameters = {
            "test_dataset": test_dataset,
            "shuffle": hyper_parameters["shuffle"],
            "original_file_name": hyper_parameters["original_file_name"],
            "recons_file_name": hyper_parameters["recons_file_name"],
            "deep_camma_generated_img_file_name": hyper_parameters["deep_camma_generated_img_file_name"]
        }
        deep_camma_manager.evaluate(test_parameters)
