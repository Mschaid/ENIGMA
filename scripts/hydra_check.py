import hydra
from omegaconf import DictConfig, OmegaConf
@hydra.main(version_base=None,
            config_path="/projects/p31961/ENIGMA/modeling/experiments/endpoint_experiments/conf",
            config_name="configs_normalzied_tune"
            )
def main(cfg: DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    print(cfg.experiment_query)
    print(type(cfg.experiment_query))
if __name__ == "__main__":
    main()