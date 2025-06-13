from pathlib import Path
from sys import path

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pytorch_lightning import seed_everything
import wandb
import os

# Set wandb to offline mode
os.environ["WANDB_MODE"] = "offline"

# path.append(str(Path(__file__)))
# print the current working directory
# path.append(str(Path(__file__).parent.parent))

@hydra.main(
    config_path="./configs/classification", 
    # config_name="config",
    version_base="1.3"
)
def main(cfg: DictConfig):
    wandb.init(project="classification_experiment", config=OmegaConf.to_container(cfg, resolve=True))
    wandb.log({"info": "Starting classification experiment..."})
    wandb.log({"config": OmegaConf.to_yaml(cfg)})
    seed_everything(cfg.get("seed", 42), workers=True)
    wandb.log({"info": f"Seed set to: {cfg.get('seed', 42)}"})

    # get dataset
    wandb.log({"info": "Instantiating datamodule..."})
    datamodule = instantiate(cfg.datamodule)
    wandb.log({"info": f"Datamodule instantiated: {datamodule}"})
    
    # NOTE: consider moving this directly into the training and testing
    wandb.log({"info": "Extracting features..."})
    datamodule.extract_features(inplace=True)
    wandb.log({"info": "Features extracted."})
    wandb.log({"info": "Performing train/test split..."})
    datamodule.train_test_split(inplace=True)
    wandb.log({"info": "Train/test split done."})
    
    # NOTE: this needs to be an SKLearn model
    wandb.log({"info": "Instantiating engine..."})
    engine = instantiate(cfg.engine)
    wandb.log({"info": f"Engine instantiated: {engine}"})
    
    # NOTE: inside the fit, we can consider implementing 
    wandb.log({"info": "Fitting engine..."})
    engine.fit(datamodule)
    wandb.log({"info": "Engine fit complete."})
    wandb.log({"info": "Testing engine..."})
    engine.test(datamodule)
    wandb.log({"info": "Engine test complete."})
    wandb.finish()

if __name__ == "__main__":
    main()
