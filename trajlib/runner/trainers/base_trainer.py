import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model.model_factory import create_model
from dataset.data_factory import create_dataset

class BaseTrainer:
    def __init__(self, model_config, dataset_config, training_config):
        self.__init__()
        self.model = create_model(model_config)
        self.dataset_loader = create_dataset(dataset_config)
        self.training_config = training_config

    def train(self):
        return

    def validate(self):
        return
