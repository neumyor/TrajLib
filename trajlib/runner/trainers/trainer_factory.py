from trainers.base_trainer import BaseTrainer


def create_trainer(trainer_config):
    return BaseTrainer(
        model_config=trainer_config.model_config,
        dataset_config=trainer_config.dataset_config,
        training_config=trainer_config.training_config,
    )
