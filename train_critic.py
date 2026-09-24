"""Train only Q/V on existing primitive data; never collect or update Actor."""
import hydra


@hydra.main(version_base=None, config_path='configs', config_name='train_critic')
def main(cfg):
    from workflows.critic import run
    run(cfg)


if __name__ == '__main__':
    main()
