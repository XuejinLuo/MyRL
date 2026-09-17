"""Inspect the same configured data used by offline training."""
from hydra import compose, initialize_config_dir
from pathlib import Path
from data.demonstrations import load_demonstrations


def main():
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[2]/'configs')):
        cfg = compose(config_name='train_offline')
    for index, ep in enumerate(load_demonstrations(cfg)):
        print(index, {key: value.shape for key, value in ep.items()})


if __name__ == '__main__':
    main()
