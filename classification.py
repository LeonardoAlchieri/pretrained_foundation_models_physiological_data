from pathlib import Path
from sys import path

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pytorch_lightning import seed_everything
# set seeds for sklearn
from sklearn.utils import check_random_state
# import wandb
# import os

# Set wandb to offline mode
# os.environ["WANDB_MODE"] = "offline"

# path.append(str(Path(__file__)))
# print the current working directory
# path.append(str(Path(__file__).parent.parent))


# def check_conf_validity(cfg: DictConfig):
#     """
#     Check if the configuration is valid.
#     """
#     # whenever we have handcrafted features, only "none" aggregator is allowed
#     if cfg.datamodule.feature_extractor == "handcrafted" and cfg.datamodule.aggregator != "none":
#         return False
#     return True

@hydra.main(
    config_path="./configs/classification", 
    # config_name="config",
    version_base="1.3"
)
def main(cfg: DictConfig):
    # run = wandb.init(project="pretrained_foundation_models_physiological_data", 
    #                  config=OmegaConf.to_container(cfg, resolve=True))
    
    print({"info": "Starting classification experiment..."})
    print({"config": OmegaConf.to_yaml(cfg)})
    
    seed_everything(cfg['seed'], workers=True)
    check_random_state(cfg['seed'])
    # cfg = clean_config(cfg) # to clean the configs, e.g. for None values
    
    print({"info": f"Seed set to: {cfg['seed']}"})

    # if not check_conf_validity(cfg):
    #     print({"error": "Configuration is not valid. Exiting..."})
    #     return None
    # get dataset
    print({"info": "Instantiating datamodule..."})
    datamodule = instantiate(cfg.datamodule)
    # cfg = update_config_from_data(datamodule, cfg) # you do this if you need information about the dataset, e.g. the dataset size
    print({"info": f"Datamodule instantiated: {datamodule}"})
    
    # NOTE: consider moving this directly into the training and testing
    print({"info": "Extracting features..."})
    datamodule.extract_features(inplace=True)
    print({"info": "Features extracted."})
    print({"info": "Performing train/test split..."})
    datamodule.train_test_split(inplace=True)
    print({"info": "Train/test split done."})
    
    # NOTE: this needs to be an SKLearn model
    print({"info": "Instantiating engine..."})
    engine = instantiate(cfg.engine)
    print({"info": f"Engine instantiated: {engine}"})
    
    # NOTE: inside the fit, we can consider implementing 
    print({"info": "Fitting engine..."})
    engine.fit(datamodule)
    print({"info": "Engine fit complete."})
    print({"info": "Testing engine..."})
    engine.test(datamodule)
    print({"info": "Engine test complete."})
    # run.finish()

if __name__ == "__main__":
    main()
