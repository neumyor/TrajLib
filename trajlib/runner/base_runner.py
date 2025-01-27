import torch
import os
from tqdm import tqdm
from evaluators.base_evaluator import BaseEvaluator
from trainers.trainer_factory import create_trainer
from torch.utils.data import DataLoader


class BaseRunner:
    def __init__(
        self, 
        trainer_config_list,
    ):
        self.trainers = [create_trainer(trainer_config=_cfg) for _cfg in trainer_config_list]

    def run(self):
        return
