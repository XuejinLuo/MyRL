"""Run only the online stage; settings are in configs/config.yaml."""
import hydra


@hydra.main(version_base=None, config_path="configs", config_name="train_online")
def main(cfg):
    from workflows.online import run
    run(cfg)


if __name__ == "__main__":
    main()
