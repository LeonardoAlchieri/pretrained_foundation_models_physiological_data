# Pretrained Foundation Models for EDA Data

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)
[![Hydra](https://img.shields.io/badge/Hydra-1.3-green.svg)](https://hydra.cc)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the official implementation of research comparing pretrained foundation models for physiological data classification tasks. The codebase focuses on **electrodermal activity (EDA) signal analysis** using state-of-the-art foundation models including MOMENT, Chronos, PatchTSMixer, and Mantis.

## 🔬 Abstract

Recent advances in foundation models have demonstrated remarkable capabilities in natural language processing and computer vision. This work investigates their effectiveness in physiological signal analysis, specifically focusing on electrodermal activity (EDA) classification tasks. We evaluate multiple pretrained foundation models across four diverse datasets (USILaughs, SEED, BiHeartS, APSYNC) and compare their performance against traditional handcrafted features and machine learning approaches.

## 📊 Key Features

- **Multi-model evaluation**: Comparison of MOMENT, Chronos, PatchTSMixer, Mantis, and handcrafted features
- **Comprehensive datasets**: Four diverse EDA datasets for robust evaluation
- **Feature extraction pipeline**: Unified framework for extracting embeddings from foundation models
- **Cross-validation**: Support for Leave-One-Person-Out (LOPO) and Time-Aware Cross-Validation (TACV)
- **Modular design**: Easy extension for new models and datasets
- **Reproducible experiments**: Hydra configuration management for systematic experiments

## 🏗️ Architecture

The codebase is organized into several key components:

```
src/
├── data/               # Dataset loading and preprocessing
├── feature_extraction/ # Foundation model feature extractors
│   ├── chronos.py     # Amazon Chronos models
│   ├── moment.py      # MOMENT models
│   ├── mantis.py      # Mantis models
│   ├── timemixer.py   # PatchTSMixer models
│   └── handcrafted.py # Traditional handcrafted features
├── engine/            # Training and evaluation engine
├── validation/        # Cross-validation strategies
└── utils/             # Utility functions and configurations
```

## 🚀 Quick Start

### Prerequisites

```bash
# Create conda environment
conda env create -f env.yml
conda activate pff
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

Each dataset should be stored as `.npz` files with:
- `values`: EDA signal data (shape: [samples, time, channels])
- `labels`: Binary classification labels
- `groups`: Subject/session identifiers for cross-validation

## 🤖 Supported Models

### Foundation Models
- **MOMENT** (`AutonLab/MOMENT-1-large`): Time series foundation model
- **Chronos** (`amazon/chronos-t5-large/small`): Amazon's time series forecasting model
- **PatchTSMixer** (`ibm-granite/granite-timeseries-patchtsmixer`): IBM's patch-based model
- **Mantis** (`paris-noah/Mantis-8M`): Multi-modal foundation model

### Baseline Models
- **Handcrafted Features**: Traditional signal processing features (min, max, mean, std, slopes, peaks, spectral features)
- **Machine Learning**: Logistic Regression, XGBoost with hyperparameter optimization

## ⚙️ Configuration

The framework uses Hydra for configuration management with a modular, composable structure. Configuration files are organized as follows:

```
configs/classification/
├── default/
│   └── default.yaml        # Base configuration with all default settings
├── sweeps/
│   └── basic.yaml          # Sweep configuration for hyperparameter searches
├── model/                  # Model configurations
├── feature_extractor/      # Feature extractor configurations
├── dataset/               # Dataset-specific configurations
├── validation_method/     # Cross-validation strategies
├── aggregator/           # Feature aggregation methods
├── label_processor/      # Label processing methods
├── scaling_method/       # Data scaling methods
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
  - default@_here_: default      # Import base configuration
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

- **Models**: `logistic_regression`, `xgboost`, `dummy_classifier`, `random_baseline`
- **Feature Extractors**: `moment_large`, `chronos_large`, `chronos_small`, `mantis`, `timemixer`, `handcrafted`
- **Validation Methods**: `lopo` (Leave-One-Person-Out), `tacv` (Time-Aware Cross-Validation), `lnpo` (Leave-N-Persons-Out)
- **Aggregators**: `mean_chan`, `mean_time`, `concat`, `none`
- **Label Processors**: `binarizer`, `none`, `extreme_only`, `inside`
- **Scaling Methods**: `standard_scaler`, `min_max_scaler`, `robust_scaler`, `none`

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

1. Create a new feature extractor in `src/feature_extraction/`:
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
_target_: src.feature_extraction.new_model.NewModelExtractor
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
  - default@_here_: default
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
      model: logistic_regression, xgboost
      feature_extractor: handcrafted, moment_large
```

## 📝 Citation

If you use this codebase in your research, please cite:

```bibtex
@inproceedings{alchieri_exploring_2025,
	address = {Espoo, Finland},
	title = {Exploring Generalist Foundation Models for Time Series of Electrodermal Activity Data},
	isbn = {979-8-4007-1477-1},
	doi = {10.1145/3714394.3756186},
	language = {en},
	booktitle = {Companion of the 2025 {ACM} {International} {Joint} {Conference} on {Pervasive} and {Ubiquitous} {Computing}},
	author = {Alchieri, Leonardo and Candian, Lino and Alecci, Lidia and Abdalazim, Nouran},
	year = {2025},
}

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

## 📧 Contact

For questions or collaboration opportunities, please contact:
- Leonardo Alchieri: [leonardo.alchieri@usi.ch](mailto:leonardo.alchieri@usi.ch)
- Research Group: [https://pc.inf.usi.ch](https://pc.inf.usi.ch)

---

⭐ If you find this work useful, please consider starring the repository!
