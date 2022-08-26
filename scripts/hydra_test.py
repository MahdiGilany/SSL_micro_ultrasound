import argparse
from ast import parse
import sys
import hydra


@hydra.main(config_path="../configs", config_name="combined_config")
def main(cfg):

    from rich import print as pprint
    from omegaconf import OmegaConf

    pprint(OmegaConf.to_object(cfg))


if __name__ == "__main__":
    main()
