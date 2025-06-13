from pathlib import Path
from sys import path

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pytorch_lightning import seed_everything
# set seeds for sklearn
from sklearn.utils import check_random_state
import wandb
import os

# Set wandb to offline mode
# os.environ["WANDB_MODE"] = "offline"

# path.append(str(Path(__file__)))
# print the current working directory
# path.append(str(Path(__file__).parent.parent))

@hydra.main(
    config_path="./configs/classification", 
    # config_name="config",
    version_base="1.3"
)
def main(cfg: DictConfig):
    run = wandb.init(project="pretrained_foundation_models_physiological_data", 
                     config=OmegaConf.to_container(cfg, resolve=True))
    
    run.log({"info": "Starting classification experiment..."})
    run.log({"config": OmegaConf.to_yaml(cfg)})
    
    seed_everything(cfg['seed'], workers=True)
    check_random_state(cfg['seed'])
    
    run.log({"info": f"Seed set to: {cfg['seed']}"})

    # get dataset
    run.log({"info": "Instantiating datamodule..."})
    datamodule = instantiate(cfg.datamodule)
    run.log({"info": f"Datamodule instantiated: {datamodule}"})
    
    # NOTE: consider moving this directly into the training and testing
    run.log({"info": "Extracting features..."})
    datamodule.extract_features(inplace=True)
    run.log({"info": "Features extracted."})
    run.log({"info": "Performing train/test split..."})
    datamodule.train_test_split(inplace=True)
    run.log({"info": "Train/test split done."})
    
    # NOTE: this needs to be an SKLearn model
    run.log({"info": "Instantiating engine..."})
    engine = instantiate(cfg.engine)
    run.log({"info": f"Engine instantiated: {engine}"})
    
    # NOTE: inside the fit, we can consider implementing 
    run.log({"info": "Fitting engine..."})
    engine.fit(datamodule)
    run.log({"info": "Engine fit complete."})
    run.log({"info": "Testing engine..."})
    engine.test(datamodule)
    run.log({"info": "Engine test complete."})
    run.finish()

if __name__ == "__main__":
    main()
