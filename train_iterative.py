"""Run only the iterative stage; settings are in configs/config.yaml."""
import hydra


@hydra.main(version_base=None, config_path="configs", config_name="train_iterative")
def main(cfg):
    from workflows.iterative import run
    run(cfg)


if __name__ == "__main__":
    main()
