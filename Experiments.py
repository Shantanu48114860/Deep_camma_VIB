import torch
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

from Deep_Camma_Manager_predict import Deep_Camma_Manager_Predict
from Deep_camma_Manager import Deep_Camma_Manager
from Utils import Utils
from data_loader import Data_Loader


class Experiments:
    def run_all_experiments_clean(self, hyper_parameters):
        device = Utils.get_device()
        print("Device: {0}".format(device))
        img_transform_manipulated = None

        img_transform_clean = transforms.Compose([
            transforms.ToTensor()
        ])

        dL = Data_Loader(path=hyper_parameters["dataset_path"],
                         train_set_size=hyper_parameters["train_set_size"],
                         val_set_size=hyper_parameters["val_set_size"])

        dataset = dL.load_train_MNIST(
            {
                "img_transform_clean": img_transform_clean,
                "img_transform_manipulated": img_transform_manipulated
            })

        dataset_clean = dataset["dataset_clean"]
        train_dataset_clean = dataset_clean["train_dataset"]
        test_dataset_clean = dataset_clean["test_dataset"]

        train_parameters = {
            "num_epochs": hyper_parameters["num_epochs"],
            "variational_beta": hyper_parameters["variational_beta"],
            "learning_rate": hyper_parameters["learning_rate"],
            "weight_decay": hyper_parameters["weight_decay"],
            "train_set_size": hyper_parameters["train_set_size"],
            "train_dataset_clean": train_dataset_clean,
            "model_save_path_clean": hyper_parameters["model_save_path_clean"],
            "shuffle": hyper_parameters["shuffle"]
        }

        deep_camma_manager = Deep_Camma_Manager(device,
                                                hyper_parameters["n_classes"],
                                                hyper_parameters["batch_size"])
        deep_camma_manager.train_clean(train_parameters)

        test_parameters = {
            "test_dataset": test_dataset_clean,
            "shuffle": hyper_parameters["shuffle"],
            "original_file_name": hyper_parameters["original_file_name_clean"],
            "recons_file_name": hyper_parameters["recons_file_name_clean"],
            "deep_camma_generated_img_file_name": hyper_parameters["deep_camma_generated_img_file_name_clean"],
            "model_save_path": hyper_parameters["model_save_path_clean"]
        }
        deep_camma_manager.evaluate(test_parameters, do_m=0)

    def run_all_experiments_do_m(self, hyper_parameters, do_m):
        device = Utils.get_device()
        print("Device: {0}".format(device))
        img_transform_manipulated = None

        img_transform_clean = transforms.Compose([
            transforms.ToTensor()
        ])

        if do_m["degrees"] != 0 or do_m["horizontal_shift"] != 0 or do_m["vertical_shift"] != 0:
            img_transform_manipulated = transforms.Compose([
                transforms.RandomAffine(degrees=do_m["degrees"],
                                        translate=(
                                            do_m["horizontal_shift"],
                                            do_m["vertical_shift"])
                                        ),
                transforms.ToTensor()
            ])

        dL = Data_Loader(path=hyper_parameters["dataset_path"],
                         train_set_size=hyper_parameters["train_set_size"],
                         val_set_size=hyper_parameters["val_set_size"])

        dataset = dL.load_train_MNIST(
            {
                "img_transform_clean": img_transform_clean,
                "img_transform_manipulated": img_transform_manipulated
            })

        dataset_clean = dataset["dataset_clean"]
        train_dataset_clean = dataset_clean["train_dataset"]
        test_dataset_clean = dataset_clean["test_dataset"]

        train_dataset_do_m = dataset["dataset_do_m"]["train_dataset"]
        test_dataset_do_m = dataset["dataset_do_m"]["test_dataset"]

        train_parameters = {
            "num_epochs": hyper_parameters["num_epochs"],
            "variational_beta": hyper_parameters["variational_beta"],
            "learning_rate": hyper_parameters["learning_rate"],
            "weight_decay": hyper_parameters["weight_decay"],
            "train_set_size": hyper_parameters["train_set_size"],
            "train_dataset_clean": train_dataset_clean,
            "train_dataset_do_m": train_dataset_do_m,
            "model_save_path": hyper_parameters["model_save_path_do_m"],
            "shuffle": hyper_parameters["shuffle"]
        }

        deep_camma_manager = Deep_Camma_Manager(device,
                                                hyper_parameters["n_classes"],
                                                hyper_parameters["batch_size"])
        deep_camma_manager.train_manipulated(train_parameters)

        test_parameters = {
            "test_dataset": test_dataset_do_m,
            "shuffle": hyper_parameters["shuffle"],
            "original_file_name": hyper_parameters["original_file_name_do_m"],
            "recons_file_name": hyper_parameters["recons_file_name_do_m"],
            "deep_camma_generated_img_file_name": hyper_parameters["deep_camma_generated_img_file_name_do_m"],
            "model_save_path": hyper_parameters["model_save_path_do_m"]
        }
        deep_camma_manager.evaluate(test_parameters, do_m=0)

    def run_disentangle_experiments_do_m(self, hyper_parameters, do_m,
                                         clean_model_name,
                                         original_file_name_disentangle,
                                         recons_file_name_disentangle,
                                         deep_camma_generated_img_file_name_disentangle):
        device = Utils.get_device()
        print("Device: {0}".format(device))
        img_transform_manipulated = None

        img_transform_clean = transforms.Compose([
            transforms.ToTensor()
        ])

        if do_m["degrees"] != 0 or do_m["horizontal_shift"] != 0 or do_m["vertical_shift"] != 0:
            img_transform_manipulated = transforms.Compose([
                transforms.RandomAffine(degrees=do_m["degrees"],
                                        translate=(
                                            do_m["horizontal_shift"],
                                            do_m["vertical_shift"])
                                        ),
                transforms.ToTensor()
            ])

        dL = Data_Loader(path=hyper_parameters["dataset_path"],
                         train_set_size=hyper_parameters["train_set_size"],
                         val_set_size=hyper_parameters["val_set_size"])

        dataset = dL.load_train_MNIST(
            {
                "img_transform_clean": img_transform_clean,
                "img_transform_manipulated": img_transform_manipulated
            })

        test_dataset_do_m = dataset["dataset_do_m"]["test_dataset"]

        test_parameters = {
            "test_dataset": test_dataset_do_m,
            "shuffle": hyper_parameters["shuffle"],
            "original_file_name": original_file_name_disentangle,
            "recons_file_name": recons_file_name_disentangle,
            "deep_camma_generated_img_file_name": deep_camma_generated_img_file_name_disentangle,
            "model_save_path": clean_model_name
        }
        deep_camma_manager = Deep_Camma_Manager(device,
                                                hyper_parameters["n_classes"],
                                                hyper_parameters["batch_size"])
        deep_camma_manager.evaluate(test_parameters, do_m=0)

    def train_classifier(self, hyper_parameters, do_m_model_name, do_m):
        device = Utils.get_device()
        print("Device: {0}".format(device))
        img_transform_manipulated = None

        img_transform_clean = transforms.Compose([
            transforms.ToTensor()
        ])

        if do_m["degrees"] != 0 or do_m["horizontal_shift"] != 0 or do_m["vertical_shift"] != 0:
            img_transform_manipulated = transforms.Compose([
                transforms.RandomAffine(degrees=do_m["degrees"],
                                        translate=(
                                            do_m["horizontal_shift"],
                                            do_m["vertical_shift"])
                                        ),
                transforms.ToTensor()
            ])

        dL = Data_Loader(path=hyper_parameters["dataset_path"],
                         train_set_size=hyper_parameters["train_set_size"],
                         val_set_size=hyper_parameters["val_set_size"])

        dataset = dL.load_train_MNIST(
            {
                "img_transform_clean": img_transform_clean,
                "img_transform_manipulated": img_transform_manipulated
            })

        dataset_clean = dataset["dataset_clean"]
        train_dataset_clean = dataset_clean["train_dataset"]
        test_dataset_clean = dataset_clean["test_dataset"]

        train_dataset_do_m = dataset["dataset_do_m"]["train_dataset"]
        test_dataset_do_m = dataset["dataset_do_m"]["test_dataset"]

        train_parameters = {
            "num_epochs": hyper_parameters["num_epochs_classifier"],
            "variational_beta": hyper_parameters["variational_beta"],
            "learning_rate": hyper_parameters["learning_rate"],
            "weight_decay": hyper_parameters["weight_decay"],
            "train_set_size": hyper_parameters["train_set_size"],
            "train_dataset_clean": train_dataset_clean,
            "train_dataset_do_m": train_dataset_do_m,
            "deep_camma_save_path": do_m_model_name,
            "classifier_save_path": hyper_parameters["classifier_save_path"],
            "shuffle": hyper_parameters["shuffle"]
        }

        deep_camma_manager = Deep_Camma_Manager(device,
                                                hyper_parameters["n_classes"],
                                                hyper_parameters["batch_size"])
        deep_camma_manager.train_classifier(train_parameters)

        # test_parameters = {
        #     "test_dataset": test_dataset_do_m,
        #     "shuffle": hyper_parameters["shuffle"],
        #     "original_file_name": hyper_parameters["original_file_name_do_m"],
        #     "recons_file_name": hyper_parameters["recons_file_name_do_m"],
        #     "deep_camma_generated_img_file_name": hyper_parameters["deep_camma_generated_img_file_name_do_m"],
        #     "model_save_path": hyper_parameters["model_save_path_do_m"]
        # }
        # deep_camma_manager.evaluate(test_parameters, m=1)

    def predict(self, hyper_parameters, do_m_model_name, do_m=1):
        device = Utils.get_device()
        print("Device: {0}".format(device))
        img_transform_clean = transforms.Compose([
            transforms.ToTensor()
        ])

        dL = Data_Loader(path=hyper_parameters["dataset_path"],
                         train_set_size=hyper_parameters["train_set_size"],
                         val_set_size=hyper_parameters["val_set_size"])
        img_transform_manipulated = None

        dataset = dL.load_train_MNIST(
            {
                "img_transform_clean": img_transform_clean,
                "img_transform_manipulated": img_transform_manipulated
            })

        dataset_clean = dataset["dataset_clean"]
        train_dataset_clean = dataset_clean["train_dataset"]
        test_dataset_clean = dataset_clean["test_dataset"]

        test_parameters = {
            "test_dataset": test_dataset_clean,
            "shuffle": hyper_parameters["shuffle"],
            "model_save_path": do_m_model_name
        }

        self.test_data_loader = DataLoader(test_dataset_clean,
                                           batch_size=128,
                                           shuffle=True)

        deep_camma_manager = Deep_Camma_Manager_Predict(n_classes=10,
                                                        batch_size=128,
                                                        test_parameters=test_parameters,
                                                        m=1)
        probs_output = deep_camma_manager(x=self.test_data_loader)
        print(probs_output.size())
