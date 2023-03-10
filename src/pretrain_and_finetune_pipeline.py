import os
from typing import List, Optional

import hydra
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase
import torch
from omegaconf import DictConfig, OmegaConf

from src import utils
from src.callbacks.online_evaluation import OnlineEvaluation

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
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
        seed_everything(config["seed"], workers=True)

    # Init lightning datamodule
    log.info(
        f"Instantiating self-supervised datamodule <{config['datamodule']['self_supervised']['_target_']}>"
    )
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config["datamodule"]["self_supervised"]
    )

    # Init lightning model
    log.info(f"Instantiating model <{config['model']['pretrain']['_target_']}>")
    model: LightningModule = hydra.utils.instantiate(config["model"]["pretrain"])

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"]["pretrain"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating pretraining callback <{cb_conf['_target_']}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf['_target_']}>")
                logger.append(hydra.utils.instantiate(lg_conf))

        if "wandb" in config.logger:
            import wandb

            wandb.config = OmegaConf.to_container(
                config, resolve=True, throw_on_missing=True
            )

    # Init lightning trainer
    log.info(
        f"Instantiating pretraining trainer <{config['trainer']['pretrain']['_target_']}>"
    )
    trainer: Trainer = hydra.utils.instantiate(
        config["trainer"]["pretrain"],
        callbacks=callbacks,
        logger=logger,
        _convert_="partial",
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

    # prefer to resume training from online eval checkpoint
    if hasattr(trainer, "online_eval_callback"):
        online_eval_callback: OnlineEvaluation = trainer.online_eval_callback
        optimized_metric = "val_auroc"
        score = online_eval_callback.best_val_metrics_global["auroc"]
        ckpt_path = online_eval_callback.checkpoint_paths["val_auroc"]
    else:
        assert trainer.checkpoint_callback is not None

        ckpt_path = trainer.checkpoint_callback.best_model_path
        optimized_metric = trainer.checkpoint_callback.monitor
        score = trainer.checkpoint_callback.best_model_score

    log.info(
        f"""
        PRETRAINING RESULTS ==== 
        Best model score -- {optimized_metric}: {score}.
        Best model checkpoint to saved locally at {ckpt_path}."""
    )

    if config.get("save_feature_extractor_to_wandb"):

        log.info("Saving feature extractor as artifact to wandb.")
        import wandb

        run = wandb.run
        assert run is not None

        artifact = wandb.Artifact(
            f"feature_extractor.{run.name}",
            type="model",
            description="""
            This artifact consists of the preprocessing function and pretrained 
            feature extractor which can be used in combination to directly extract
            features from patches of RF data, for use in downstream tasks.
            """,
        )

        from exactvu.data import ExactSSLDataModule
        from src.models.self_supervised.exact_ssl_module import ExactSSLModule

        assert isinstance(model, ExactSSLModule)
        assert isinstance(datamodule, ExactSSLDataModule)

        feature_extractor = model.backbone
        transform = datamodule.eval_transform
        transform.create_pairs = False
        import pickle

        sample_batch = next(iter(datamodule.val_dataloader()[0]))
        X1, y = sample_batch
        sample_input = X1

        feature_extractor.eval()
        traced_model = torch.jit.trace(feature_extractor, (sample_input))
        torch.jit.save(traced_model, "feature_extractor.pt")

        with open("transform.pkl", "wb") as f:
            pickle.dump(transform, f)

        artifact.add_file("feature_extractor.pt")
        artifact.add_file("transform.pkl")

        run.log_artifact(artifact)

    # == FINETUNING ====================================

    # Init lightning datamodule
    log.info(
        f"Instantiating supervised datamodule <{config['datamodule']['self_supervised']['_target_']}>"
    )
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config["datamodule"]["supervised"]
    )

    # Init lightning model
    log.info(f"Instantiating model <{config['model']['finetune']['_target_']}>")
    model: LightningModule = hydra.utils.instantiate(
        config["model"]["finetune"], ckpt_path=ckpt_path
    )

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"]["finetune"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating pretraining callback <{cb_conf['_target_']}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning trainer
    log.info(
        f"Instantiating pretraining trainer <{config['trainer']['finetune']['_target_']}>"
    )
    trainer: Trainer = hydra.utils.instantiate(
        config["trainer"]["finetune"],
        callbacks=callbacks,
        logger=logger,
        _convert_="partial",
    )

    # Train the model
    if config.get("finetune"):
        log.info("Starting Finetuning!")
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
    if not config["trainer"].get("fast_dev_run") and config.get("finetune"):

        assert trainer.checkpoint_callback is not None

        ckpt_path = trainer.checkpoint_callback.best_model_path
        optimized_metric = trainer.checkpoint_callback.monitor
        score = trainer.checkpoint_callback.best_model_score
        log.info(
            f"""
            FINETUNING RESULTS ==== 
            Best model score -- {optimized_metric}: {score}.
            Best model checkpoint to saved locally at {ckpt_path}."""
        )
