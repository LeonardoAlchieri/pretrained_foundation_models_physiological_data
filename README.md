# Pretrained Foundation Models for Physiological Data Analysis

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
python classification.py --config-name=mainconf dataset=usilaughs side=right
```

2. **Hyperparameter sweep**:
```bash
python classification.py --config-name=usilaughs_sweep --multirun
```

3. **Custom configuration**:
```bash
python classification.py feature_extractor=moment_large model=logistic_regression validation_method=lopo
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

The framework uses Hydra for configuration management. Key configuration components:

```yaml
# Main configuration
dataset: usilaughs          # Dataset selection
side: right                 # Data subset
device_map: "mps"          # Device for model inference

# Model configuration
model: logistic_regression  # Classifier type
feature_extractor: moment_large  # Foundation model
validation_method: tacv     # Cross-validation strategy
aggregator: mean_chan      # Feature aggregation method
```

### Available Configurations

- **Models**: `logistic_regression`, `xgboost`, `dummy_classifier`
- **Feature Extractors**: `moment_large`, `chronos_large`, `chronos_small`, `mantis`, `timemixer`, `handcrafted`
- **Validation Methods**: `lopo` (Leave-One-Person-Out), `tacv` (Time-Aware Cross-Validation)
- **Aggregators**: `mean_chan`, `mean_time`, `concat`, `none`

## 📋 Experiments

### Systematic Evaluation

Run comprehensive experiments across all model-dataset combinations:

```bash
# USILaughs dataset
python classification.py --config-name=usilaughs_sweep --multirun

# SEED dataset  
python classification.py --config-name=seed_sweep --multirun

# BiHeartS dataset
python classification.py --config-name=bihearts_sweep --multirun

# APSYNC dataset
python classification.py --config-name=apsync_sweep --multirun
```

## 📊 Results Analysis

The repository includes Jupyter notebooks for result analysis:

- `check_results_final.ipynb`: Comprehensive results comparison
- `data_distribution.ipynb`: Dataset statistics and visualization

Results are automatically saved to `outputs/` with timestamp and configuration details.

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
2. Create sweep configuration in `configs/classification/`
3. Update dataset paths in configuration files

## 📝 Citation

If you use this codebase in your research, please cite:

```bibtex
...
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
- Leonardo Alchieri: [leonardo.alchieri@usi.cj](mailto:leonardo.alchieri@usi.cj)
- Research Group: [https://pc.inf.usi.ch](https://pc.inf.usi.ch)

---

⭐ If you find this work useful, please consider starring the repository!
