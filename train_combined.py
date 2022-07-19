import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=False)


@hydra.main(config_path="configs/", config_name="combined_config.yaml")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.pretrain_and_finetune_pipeline import train

    # Applies optional utilities
    utils.extras(config)

    import wandb

    wandb.init(
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        name=config.name,
        project=config.project,
    )
    config = wandb.config

    # Train model
    return train(config)


if __name__ == "__main__":
    main()
