"""Matched single-sample versus best-of-K Q-selection evaluation."""
import hydra


@hydra.main(version_base=None, config_path='configs', config_name='evaluate_q_selection')
def main(cfg):
    from evaluation.q_selection import run
    run(cfg)


if __name__ == '__main__':
    main()
