import logging
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="quest_config")
def my_conf(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    print(cfg.quest_config.data_path)


if __name__ == "__main__":
    my_conf()
