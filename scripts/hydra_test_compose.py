import argparse
from ast import parse
import sys
import hydra


def main(args):

    from hydra import initialize, compose

    initialize("../configs", version_base="1.2")
    cfg = compose("combined_config", args)

    from rich import print as pprint
    from omegaconf import OmegaConf

    pprint(OmegaConf.to_object(cfg))


if __name__ == "__main__":
    args = sys.argv[1:]
    print(args)

    main(args)
