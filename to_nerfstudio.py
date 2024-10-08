import os
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="to_nerfstudio")
def main(cfg: DictConfig):
    dataset = hydra.utils.instantiate(cfg.dataset)
    os.makedirs(cfg.dir, exist_ok=True)
    dataset.to_nerfstudio_config(cfg.dir)


if __name__ == "__main__":
    main()
