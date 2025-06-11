from pathlib import Path
from sys import path

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pytorch_lightning import seed_everything

path.append(str(Path(__file__).parent.parent.parent))

@hydra.main(
    config_path="../../configs/classification", config_name="config", version_base="1.3"
)
def main(cfg: DictConfig):
    seed_everything(cfg.get("seed", 42), workers=True)

    # get dataset
    datamodule = instantiate(cfg.datamodule)
    
    # NOTE: consider moving this directly into the training and testing
    datamodule.extract_features(inplace=True)
    datamodule.train_test_split(inplace=True)
    
    # NOTE: this needs to be an SKLearn model
    engine = instantiate(cfg.engine)
    
    # NOTE: inside the fit, we can consider implementing 
    engine.fit(datamodule)
    engine.test(datamodule)

if __name__ == "__main__":
    main()
