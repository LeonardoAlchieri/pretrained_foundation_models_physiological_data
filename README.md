# Pretrained Foundation Models for EDA Data

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)
[![Hydra](https://img.shields.io/badge/Hydra-1.3-green.svg)](https://hydra.cc)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the official implementation of research comparing pretrained foundation models for physiological data classification tasks. The codebase focuses on **electrodermal activity (EDA) signal analysis** using state-of-the-art foundation models including MOMENT, Chronos, PatchTSMixer, and Mantis.

## 🔬 Abstract

Recent advances in foundation models have demonstrated remarkable capabilities in natural language processing and computer vision. This work investigates their effectiveness in physiological signal analysis, specifically focusing on electrodermal activity (EDA) classification tasks. We evaluate multiple pretrained foundation models across four diverse datasets (USILaughs, SEED, BiHeartS, APSYNC) and compare their performance against traditional handcrafted features, trainable feature extractors (e.g., MiniRocket), and machine learning approaches.

## 📊 Key Features

- **Multi-model evaluation**: Comparison of MOMENT, Chronos, PatchTSMixer, Mantis, and handcrafted features
- **Trainable feature extractors**: Support for self-supervised feature learning methods like MiniRocket
- **Comprehensive datasets**: Four diverse EDA datasets for robust evaluation
- **Feature extraction pipeline**: Unified framework for both fixed and trainable feature extractors
- **Cross-validation**: Support for Leave-One-Person-Out (LOPO) and Time-Aware Cross-Validation (TACV)
- **Modular design**: Easy extension for new models, datasets, and feature extractors
- **Reproducible experiments**: Hydra configuration management for systematic experiments
- **Fair evaluation**: Trainable extractors are fitted per-fold to prevent data leakage

## 🏗️ Architecture

The codebase is organized into several key components:

```
edamame_downstream/
├── data/               # Dataset loading and preprocessing
├── feature_extraction/ # Foundation model feature extractors
│   ├── chronos.py     # Amazon Chronos models
│   ├── moment.py      # MOMENT models
│   ├── mantis.py      # Mantis models
│   ├── timemixer.py   # PatchTSMixer models
│   ├── handcrafted.py # Traditional handcrafted features
│   └── none.py        # Pass-through for raw data
├── model/             # Model implementations
│   ├── customKernelSVC.py           # Custom kernel SVC classifier
│   └── trainable_feature_extraction/ # Trainable feature extractors
│       └── minirocket.py            # MiniRocket implementation
├── engine/            # Training and evaluation engine
├── validation/        # Cross-validation strategies
└── utils/             # Utility functions and configurations
```

## 🚀 Quick Start

### Prerequisites

```bash
# Clone the repository
git clone <repository-url>
cd pretrained-foundation-models-physiological-data

# Install dependencies
pip install -e .
```

### Basic Usage

1. **Single experiment**:
```bash
python classification.py --config-name=default dataset=usilaughs
```

2. **Hyperparameter sweep**:
```bash
python classification.py --config-name=usilaughs --multirun
```

3. **Custom configuration with overrides**:
```bash
python classification.py --config-name=default feature_extractor=moment_large model=logistic_regression validation_method=lopo dataset=seed
```

## 📈 Datasets

The framework supports four EDA datasets:

- **USILaughs**: congitive load/relaxation classification
- **SEED**: low/high engagement
- **BiHeartS**: sleep/wake
- **APSYNC**: low/high engagement; this dataset can be used for enjoyment and immersion

These datasets can be shared, either in raw format or in the pre-processed format used in this work, upon signing a data sharing agreement.

### Data Format

Each dataset should be stored as `.npz` files following this naming convention:
```
data_{side}_{label_name}_{segment_length}s.npz
```

Examples:
- `data_right_engagement_10s.npz`
- `data_left_enjoyment_5s.npz`
- `data_unknown_performance_30s.npz`

Each `.npz` file must contain exactly 4 keys with numpy arrays:
- `values`: EDA signal data (shape: [N, T, A] where N=samples, T=time points, A=channels)
- `labels`: Classification labels (shape: [N] - one label per sample)
- `groups`: Subject/session identifiers for cross-validation (shape: [N] - one group ID per sample)
- `name`: Dataset name (numpy array containing the dataset name as string)

## 🤖 Supported Models

### Foundation Models (Fixed Feature Extractors)
- **MOMENT** (`AutonLab/MOMENT-1-large`): Time series foundation model
- **Chronos** (`amazon/chronos-t5-large/small`): Amazon's time series forecasting model
- **PatchTSMixer** (`ibm-granite/granite-timeseries-patchtsmixer`): IBM's patch-based model
- **Mantis** (`paris-noah/Mantis-8M`): Multi-modal foundation model

### Trainable Feature Extractors
In addition to fixed pretrained models, the framework supports **trainable feature extractors** that learn representations in a self-supervised fashion during training:

- **MiniRocket**: Mini Random Convolutional Kernel Transform for time series classification
- Custom extractors can be easily added by implementing `fit` and `transform` methods

These extractors are trained independently for each fold during cross-validation, enabling fair comparison with fixed foundation models.

### Baseline Models
- **Handcrafted Features**: Traditional signal processing features (min, max, mean, std, slopes, peaks, spectral features)
- **Machine Learning Classifiers**: 
  - Logistic Regression
  - XGBoost
  - K-Nearest Neighbors (KNN)
  - Custom Kernel SVC with RBF kernel and adaptive gamma

## ⚙️ Configuration

The framework uses Hydra for configuration management with a modular, composable structure. Configuration files are organized as follows:

```
configs/classification/
├── default/
│   └── default.yaml        # Base configuration with all default settings
├── sweeps/
│   └── basic.yaml          # Sweep configuration for hyperparameter searches
│   └── giofeatures.yaml    # Advanced sweep with trainable feature extractors
├── model/                  # Model configurations
│   ├── logistic_regression.yaml
│   ├── xgboost.yaml
│   ├── knn.yaml           # K-Nearest Neighbors
│   └── cksvc.yaml         # Custom Kernel SVC
├── feature_extractor/      # Feature extractor configurations (fixed)
├── trainable_feature_extractor/  # Trainable feature extractor configs
│   ├── minirocket.yaml    # MiniRocket configuration
│   └── none.yaml          # No trainable extractor
├── dataset/               # Dataset-specific configurations
├── validation_method/     # Cross-validation strategies
├── aggregator/           # Feature aggregation methods
├── label_processor/      # Label processing methods
├── feature_scaling_method/       # Data scaling methods
├── resampling/           # Resampling strategies
├── usilaughs.yaml        # Dataset-specific experiment configs
├── seed.yaml
├── bihearts.yaml
├── apsync.yaml
└── workplace.yaml
```

### Configuration Composition

Each experiment configuration composes multiple components:

```yaml
defaults:
  - _self_
  - default      # Import base configuration
  - sweeps@_here_: basic         # Import sweep configuration (for multirun)
  - override dataset: usilaughs  # Override dataset selection
  - override label_processor: binarizer  # Override label processing

# Experiment-specific parameters
label_name: engagement
segment_length: 10
device_map: "cpu"

# Additional sweep parameters (merged with sweeps/basic.yaml)
hydra:
  sweeper:
    grid_params:
      label_name: engagement, enjoyment, motivation
```

### Available Configurations

- **Models**: `logistic_regression`, `xgboost`, `knn`, `cksvc`, `random_baseline`
- **Feature Extractors** (Fixed): `moment_large`, `chronos_large`, `chronos_small`, `mantis`, `timemixer`, `handcrafted`, `none`
- **Trainable Feature Extractors**: `minirocket`, `none`
- **Validation Methods**: `lopo` (Leave-One-Person-Out), `tacv` (Time-Aware Cross-Validation), `lnpo` (Leave-N-Persons-Out)
- **Aggregators**: `mean_chan`, `mean_time`, `concat`, `none`
- **Label Processors**: `binarizer`, `none`, `extreme_only`, `inside`
- **Scaling Methods**: `standard_scaler`, `min_max_scaler`, `robust_scaler`, `none`

### Trainable vs Fixed Feature Extraction

The framework distinguishes between two types of feature extraction:

1. **Fixed Feature Extractors** (`feature_extractor`): Pretrained foundation models or handcrafted features that don't require training. These are applied identically to all folds.

2. **Trainable Feature Extractors** (`trainable_feature_extractor`): Self-supervised methods that learn representations from the training data. These are:
   - Fitted independently on each fold's training set
   - Used to transform both training and test data for that fold
   - Ensuring fair cross-validation without data leakage

Example configuration combining both:
```yaml
feature_extractor: none              # No fixed features
trainable_feature_extractor: minirocket  # Use MiniRocket (trained per fold)
model: cksvc                         # Custom kernel SVC classifier
```

## 📋 Experiments

### Systematic Evaluation

Run comprehensive experiments across all model-dataset combinations:

```bash
# USILaughs dataset
python classification.py --config-name=usilaughs --multirun

# SEED dataset  
python classification.py --config-name=seed --multirun

# BiHeartS dataset
python classification.py --config-name=bihearts --multirun

# APSYNC dataset
python classification.py --config-name=apsync --multirun

# Workplace dataset
python classification.py --config-name=workplace --multirun
```

## 📊 Results Analysis

The repository includes Jupyter notebooks for result analysis in the `notebooks/` folder:

- `data_distribution.ipynb`: Dataset statistics and visualization
- `visualize_results.ipynb`: Comprehensive results comparison and visualization

Results are automatically saved to `outputs/` with timestamp and configuration details.

### Additional Notebooks

The `additional_notebook/` folder contains notebooks that were used during the initial development phases and are kept for reference purposes only. These notebooks are no longer actively maintained and should not be used for current analysis:

- `APSYNC_preparation.ipynb`: Legacy data preparation for APSYNC dataset
- `BiHeartS_prepatation.ipynb`: Legacy data preparation for BiHeartS dataset  
- `SEED_preparation.ipynb`: Legacy data preparation for SEED dataset
- `USILaughs_preparation.ipynb`: Legacy data preparation for USILaughs dataset
- `checking_data.ipynb`: Legacy data validation notebook
- `checking_results.ipynb`: Legacy results checking notebook
- `foundation_models_testing.ipynb`: Legacy model testing notebook

## 🔧 Extending the Framework

### Adding New Models

1. Create a new feature extractor in `edamame_downstream/feature_extraction/`:
```python
class NewModelExtractor:
    def __init__(self, model_name: str, **kwargs):
        self.model = load_model(model_name)
    
    def __call__(self, data: DataInfo) -> EDADataset:
        # Extract features using your model
        features = self.model.encode(data["values"])
        data["features"] = features
        return data
```

2. Add configuration file in `configs/classification/feature_extractor/`:
```yaml
_target_: edamame_downstream.feature_extraction.new_model.NewModelExtractor
model_name: "your-model-name"
device_map: ${device_map}
```

### Adding New Datasets

1. Prepare data in the required format (`.npz` file)
2. Create dataset configuration in `configs/classification/dataset/`
3. Create experiment configuration that imports the dataset:
```yaml
defaults:
  - _self_
  - default
  - sweeps@_here_: basic
  - override dataset: your_new_dataset
  - override label_processor: binarizer

label_name: your_label
segment_length: 10
device_map: "cpu"
```

### Adding New Sweep Configurations

Create custom sweep configurations in `configs/classification/sweeps/`:
```yaml
defaults:
  - _self_
  - override /hydra/sweeper: list

hydra:
  mode: MULTIRUN
  sweep:
    dir: "outputs/${dataset.dataset}-${dataset.side}/multirun_${now:%Y-%m-%d}-${now:%H-%M-%S}"
    subdir: "${hydra:runtime.choices.model}-${hydra:runtime.choices.feature_extractor}-${label_name}-${seed}"
  sweeper:
    grid_params: 
      seed: 42, 123, 456
      validation_method: tacv, lopo
    list_params:
      model: logistic_regression, xgboost, knn
      feature_extractor: handcrafted, timemixer, none
      trainable_feature_extractor: none, none, minirocket
```

### Adding Trainable Feature Extractors

1. Create a new trainable feature extractor in `edamame_downstream/model/trainable_feature_extraction/`:
```python
class NewTrainableExtractor:
    def __init__(self, random_state: int):
        self.random_state = random_state
    
    def fit(self, X: np.ndarray) -> "NewTrainableExtractor":
        # Train on the data (self-supervised)
        self.model_ = self._train_model(X)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        # Extract features using trained model
        return self.model_.extract_features(X)
```

2. Add configuration in `configs/classification/trainable_feature_extractor/`:
```yaml
_target_: edamame_downstream.model.trainable_feature_extraction.new_extractor.NewTrainableExtractor
random_state: ${seed}
```

3. Use in experiments:
```bash
python classification.py feature_extractor=none trainable_feature_extractor=new_extractor model=knn
```

## 🤝 Contributing

We welcome contributions!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MOMENT](https://github.com/moment-timeseries-foundation-model/moment) for the foundation time series model
- [Chronos](https://github.com/amazon-science/chronos-forecasting) for Amazon's time series forecasting framework
- [Hydra](https://hydra.cc/) for configuration management
- The contributors of the physiological datasets used in this research

---

⭐ If you find this work useful, please consider starring the repository!
