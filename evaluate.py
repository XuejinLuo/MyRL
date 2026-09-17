"""Compare checkpoints independently, using configs/config.yaml comparison settings."""
import hydra


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    from evaluation.compare import run
    run(cfg)


if __name__ == "__main__":
    main()
