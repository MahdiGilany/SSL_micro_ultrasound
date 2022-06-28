import os
from typing import List, Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils

log = utils.get_logger(__name__)


def train(config) -> Optional[float]:
    """Contains the training pipeline. Can additionally evaluate model on a testset, using best
    weights achieved during training.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # =========================
    # Pretraining
    # ==========================

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("pretrain_seed"):
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    log.info(
        f"Instantiating self-supervised datamodule <{config.datamodule['self_supervised']['_target_']}>"
    )
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config.datamodule.self_supervised
    )

    # Init lightning model
    log.info(f"Instantiating model <{config.model.pretrain._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model.pretrain)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.pretrain.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating pretraining callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating pretraining trainer <{config.trainer.pretrain._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer.pretrain, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    if config.get("pretrain"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)

    ckpt_path = trainer.checkpoint_callback.best_model_path

    # FINETUNING ====================================

    # Init lightning datamodule
    log.info(
        f"Instantiating supervised datamodule <{config.datamodule.self_supervised._target_}>"
    )
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config.datamodule.supervised
    )

    # Init lightning model
    log.info(f"Instantiating model <{config.model.finetune._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        config.model.finetune, ckpt_path=ckpt_path
    )

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.finetune.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating pretraining callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning trainer
    log.info(f"Instantiating pretraining trainer <{config.trainer.finetune._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer.finetune, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Train the model
    if config.get("finetune"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run") and config.get("train"):
        log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")
