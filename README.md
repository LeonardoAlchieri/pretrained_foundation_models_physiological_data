# Pretrained Foundation Models for EDA Data

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)
[![Hydra](https://img.shields.io/badge/Hydra-1.3-green.svg)](https://hydra.cc)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the official implementation of research comparing pretrained foundation models for physiological data classification tasks. The codebase focuses on **electrodermal activity (EDA) signal analysis** using state-of-the-art foundation models including UME, MOMENT, Chronos, PatchTSMixer, and Mantis — used both as *frozen* feature extractors (linear probing) and as *end-to-end fine-tuned* classifiers.

## 🔬 Abstract

Recent advances in foundation models have demonstrated remarkable capabilities in natural language processing and computer vision. This work investigates their effectiveness in physiological signal analysis, specifically focusing on electrodermal activity (EDA) classification tasks. We evaluate multiple pretrained foundation models across nine diverse EDA datasets — six recorded with Empatica E4 wristbands (APSync, HeartS, USILaughs, WESAD, DREAMT, HHISS) and three recorded with **non-Empatica sensors**, i.e. Shimmer 2R and Shimmer3 GSR+ (AMIGOS, EDABE, Multi-physio) — and compare their performance against traditional handcrafted features, trainable feature extractors (e.g., MiniRocket), and machine learning approaches. In addition to frozen-feature evaluation, foundation models (UME and Mantis) can be **supervised fine-tuned** end-to-end on each downstream task, so that linear probing and full fine-tuning are measured under exactly the same cross-validation protocol.

## 📊 Key Features

- **Multi-model evaluation**: Comparison of UME, MOMENT, Chronos, PatchTSMixer, Mantis, and handcrafted features
- **Frozen *and* fine-tuned evaluation**: The same foundation model can be used as a frozen encoder or fine-tuned end-to-end on the downstream labels (see [Fine-Tuning Foundation Models](#-fine-tuning-foundation-models))
- **Trainable feature extractors**: Support for self-supervised feature learning methods like MiniRocket
- **Comprehensive datasets**: Nine EDA datasets for robust evaluation, spanning multiple recording devices (Empatica E4, Shimmer 2R, Shimmer3 GSR+) and sensor placements
- **Feature extraction pipeline**: Unified framework for both fixed and trainable feature extractors
- **Cross-validation**: Support for Leave-One-Person-Out (LOPO) and Time-Aware Cross-Validation (TACV)
- **Modular design**: Easy extension for new models, datasets, and feature extractors
- **Reproducible experiments**: Hydra configuration management for systematic experiments
- **Fair evaluation**: Trainable extractors and fine-tuned models are fitted per-fold to prevent data leakage

## 🏗️ Architecture

The codebase is organized into several key components:

```
edamame_downstream/
├── data/               # Dataset loading and preprocessing
├── feature_extraction/ # Frozen (fixed) feature extractors
│   ├── chronos.py     # Amazon Chronos models
│   ├── moment.py      # MOMENT models
│   ├── mantis.py      # Mantis models (frozen encoder)
│   ├── edamame.py     # UME / EDAMAME encoders (frozen encoder)
│   ├── timemixer.py   # PatchTSMixer models
│   ├── handcrafted.py # Traditional handcrafted features
│   └── none.py        # Pass-through for raw data (flat or shape-preserving)
├── model/             # Model implementations (the `model` config slot)
│   ├── customKernelSVC.py           # Custom kernel SVC classifier
│   ├── ume.py                       # Fine-tunable UME classifier
│   ├── mantis.py                    # Fine-tunable Mantis classifier
│   └── trainable_feature_extraction/ # Trainable feature extractors
│       └── minirocket.py            # MiniRocket implementation
├── label_processor/   # Label selection/binarization strategies
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

The environment is also described by `pixi.toml` / `pixi.lock` (`pixi install`), which is the reproducible path and the one that resolves the CUDA-enabled PyTorch builds.

> **UME requires a sibling checkout.** `pixi.toml` declares the [`edamame`](https://github.com/LeonardoAlchieri/eda-foundation-models) package as an editable local dependency at `../eda-foundation-models`, so that repository must be cloned next to this one for any UME run (frozen or fine-tuned). UME checkpoints are expected under `ume_weights.nosync/`. Mantis instead comes from PyPI (`mantis-tsfm`) and downloads its weights from the Hugging Face hub.

### Basic Usage

1. **Single experiment**:
```bash
python classification.py --config-name=default dataset=usilaughs
```

2. **Hyperparameter sweep**:
```bash
python classification.py --config-name=usilaughs --multirun
```

3. **Custom configuration with overrides** (frozen encoder + linear probe):
```bash
python classification.py --config-name=default feature_extractor=moment_large model=logistic_regression validation_method=lopo dataset=dreamt
```

4. **Fine-tuning a foundation model end-to-end** (see [Fine-Tuning Foundation Models](#-fine-tuning-foundation-models)):
```bash
python classification.py --config-name=apsync model=ume feature_extractor=none-flat aggregator=none
```

## 📈 Datasets

The evaluation covers nine EDA datasets and their binary tasks. Six were recorded with **Empatica E4** wristbands; the three most recent additions were recorded with **Shimmer** sensors, and test whether the models generalise across recording hardware and electrode placement:

| Dataset | Config | Binary task(s) | Device |
|---------|--------|----------------|--------|
| APSync | `apsync` | Low/High engagement | Empatica E4 |
| HeartS | `hearts` | Sleep/Wake | Empatica E4 |
| USILaughs | `usilaughs` | Cognitive load/Relaxation | Empatica E4 |
| WESAD | `wesad` | High/Low arousal, High/Low valence (self-report) | Empatica E4 |
| DREAMT | `dreamt` | Sleep/Wake, Deep sleep/REM | Empatica E4 |
| HHISS | `hhiss` | Low/High stress | Empatica E4 |
| AMIGOS | `amigos` | Arousal High/Low, Liking High/Low | Shimmer 2R |
| EDABE | `edabe` | Artifact | Shimmer3 GSR+ |
| Multi-physio | `multiphysio_2` | Relax/Video | Shimmer3 GSR+ |

Each row maps to an experiment config in `configs/classification/` (which fixes the task, the segment length and the label handling) plus a data config in `configs/classification/dataset/` (which fixes the data location and sampling rate). Additional dataset configs exist in the repository but are not part of this evaluation.

These datasets can be shared, either in raw format or in the pre-processed format used in this work, upon signing a data sharing agreement.

### Non-Empatica datasets

The Shimmer-based datasets are handled by the same pipeline, with two practical differences:

1. **The `side` slot carries the sensor/placement, not the body side.** For datasets recorded with more than one sensor it is what selects the data file, so the same subjects and task can be re-run per sensor:
   ```bash
   # Multi-physio: Shimmer3 GSR+ electrodes on the wrist...
   python classification.py --config-name=multiphysio_2 dataset.side=shimmer_wrist

   # ...and on the fingers
   python classification.py --config-name=multiphysio_2 dataset.side=shimmer_fingers
   ```
   For AMIGOS the slot is simply `shimmer`; EDABE has a single recording per segment, so its slot stays `unknown`.

   > ⚠️ Some experiment configs carry a root-level `dataset.side: ...` line. That is a **literal dotted key**, not a nested override, so it has no effect: the value actually used comes from `configs/classification/dataset/<name>.yaml` (e.g. `shimmer_wrist` for Multi-physio). Change the sensor either on the command line, as above, or in the dataset config.

2. **Binary tasks are carved out of multi-class or ordinal labels** by the label processors, not by a fixed threshold:
   - `custom_selector` keeps two explicit sets of label values and drops every other sample — Multi-physio maps `video1`/`video2` to the positive class and `relax` to the negative one (the recording also contains `squat` and `sit`), and EDABE maps `1`/`2` (artifact) against `0` (clean).
   - `extreme_only` keeps only the extremes of a self-reported scale and drops the middle — AMIGOS uses the bottom and top quartiles (`lower_threshold: 0.25`, `upper_threshold: 0.75`, `threshold_type: percentile`; absolute cut-offs are available with `threshold_type: value`).

### Data Format

Each dataset should be stored as `.npz` files following this naming convention:
```
data_{side}_{label_name}_{segment_length}s.npz
```

Where `{side}` is the body side (`left`, `right`, `unknown`/`Unknown`) for the wrist-worn Empatica datasets, and the sensor/placement (e.g. `shimmer`, `shimmer_wrist`, `shimmer_fingers`) for the Shimmer datasets. The full path is composed in `configs/classification/default.yaml` as `./data.nosync/${dataset.dataset}/data_${dataset.side}_${dataset.label_name}_${dataset.segment_length}s.npz`, and can be overridden with the `DATA_PATH` environment variable.

Examples:
- `data_right_engagement_10s.npz`
- `data_left_enjoyment_5s.npz`
- `data_unknown_performance_30s.npz`
- `data_shimmer_Arousal_60s.npz`
- `data_shimmer_wrist_task_60s.npz`

Each `.npz` file must contain exactly 4 keys with numpy arrays:
- `values`: EDA signal data (shape: [N, T, A] where N=samples, T=time points, A=channels)
- `labels`: Classification labels (shape: [N] - one label per sample)
- `groups`: Subject/session identifiers for cross-validation (shape: [N] - one group ID per sample)
- `name`: Dataset name (numpy array containing the dataset name as string)

## 🤖 Supported Models

### Foundation Models (Fixed Feature Extractors)
- **UME** (local checkpoint, e.g. `ume_weights.nosync/ume.ckpt`): EDA-specific foundation model (EfficientNet-style 1D encoder trained contrastively), provided by the [`edamame`](https://github.com/LeonardoAlchieri/eda-foundation-models) package
- **MOMENT** (`AutonLab/MOMENT-1-large`): Time series foundation model
- **Chronos** (`amazon/chronos-t5-large/small`): Amazon's time series forecasting model
- **PatchTSMixer** (`ibm-granite/granite-timeseries-patchtsmixer`): IBM's patch-based model
- **Mantis** (`paris-noah/Mantis-8M`): Multi-modal foundation model

### Fine-Tunable Foundation Models (End-to-End Classifiers)

The same encoders can also be **supervised fine-tuned** on the downstream labels instead of being frozen. These live in the `model` config slot (not `feature_extractor`), because they *are* the classifier:

- **UME** (`model=ume`): [`UMEClassifier`](edamame_downstream/model/ume.py) — fine-tunes the UME encoder with a classification head
- **Mantis** (`model=mantis`): [`MantisClassifier`](edamame_downstream/model/mantis.py) — fine-tunes Mantis-8M via `MantisTrainer`

See [Fine-Tuning Foundation Models](#-fine-tuning-foundation-models) for the full documentation.

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
│   ├── cksvc.yaml         # Custom Kernel SVC
│   ├── ume.yaml           # UME fine-tuned end-to-end
│   └── mantis.yaml        # Mantis fine-tuned end-to-end
├── feature_extractor/      # Feature extractor configurations (fixed/frozen)
│   ├── ume.yaml           # Frozen UME encoder
│   ├── mantis.yaml        # Frozen Mantis encoder
│   ├── mantis-finetune.yaml # Frozen, but from a fine-tuned checkpoint
│   └── none-flat.yaml     # Raw signal flattened to 2D (for fine-tuning)
├── trainable_feature_extractor/  # Trainable feature extractor configs
│   ├── minirocket.yaml    # MiniRocket configuration
│   └── none.yaml          # No trainable extractor
├── dataset/               # Dataset-specific configurations
├── validation_method/     # Cross-validation strategies
├── aggregator/           # Feature aggregation methods
├── label_processor/      # Label processing methods
├── feature_scaling_method/       # Feature-level scaling methods
├── sample_scaling_method/        # Per-sample (signal-level) scaling methods
├── resampling/           # Resampling strategies
├── usilaughs.yaml        # Dataset-specific experiment configs
├── apsync.yaml
├── hearts.yaml
├── wesad.yaml
├── dreamt.yaml
├── hhiss.yaml
├── amigos.yaml
├── edabe.yaml
└── multiphysio_2.yaml
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

- **Models** (shallow classifiers): `logistic_regression`, `svm`, `xgboost`, `knn`, `cksvc`, `random_forest`, `decision_tree`, `adaboost`, `gp`, `random_baseline`, plus the time-series estimators `tsknn`, `tssvc`, `shapelets`
- **Models** (fine-tuned foundation models): `ume`, `mantis` — see [Fine-Tuning Foundation Models](#-fine-tuning-foundation-models)
- **Feature Extractors** (Fixed/frozen): `ume`, `moment_large`, `chronos_large`, `chronos_small`, `mantis`, `mantis-finetune`, `timemixer`, `handcrafted`, `handcrafted_baseline_small`, `handcrafted_baseline_big`, `none`, `none-flat`; UME variants for ablations: `ume-base-75`, `edamame_efficientnet_{tiny,small,large,random}`, `edamame_efficientnet-userpair`, `edamame_mae`
- **Trainable Feature Extractors**: `minirocket`, `none`
- **Validation Methods**: `lopo` (Leave-One-Person-Out), `tacv` (Time-Aware Cross-Validation), `lnpo` (Leave-N-Persons-Out), `kfold`
- **Aggregators**: `mean_chan`, `mean_time`, `concat`, `none`
- **Label Processors**: `binarizer`, `label_binarizer`, `custom_selector`, `extreme_only`, `inside`, `none`
- **Feature Scaling Methods**: `standard_scaler`, `none`
- **Sample Scaling Methods**: `min_max_scale`, `min_max_scale_per_user`, `z_score`, `none`

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

## 🎓 Fine-Tuning Foundation Models

A foundation model can enter the pipeline in two different roles, and the distinction is which config slot it occupies:

| Slot | Role | Pretrained weights | Example |
|------|------|--------------------|---------|
| `feature_extractor` | **Frozen encoder.** Embeddings are computed once and fed to a shallow classifier (linear probing) | never updated | `feature_extractor=ume model=logistic_regression` |
| `model` | **End-to-end classifier.** Encoder + classification head are trained on the downstream labels | updated, independently per fold | `model=ume feature_extractor=none-flat` |

Two models are currently fine-tunable:

- [`model=ume`](configs/classification/model/ume.yaml) → [`UMEClassifier`](edamame_downstream/model/ume.py), which fine-tunes the UME (EfficientNet) encoder through `EdamamePipeline.fit`
- [`model=mantis`](configs/classification/model/mantis.yaml) → [`MantisClassifier`](edamame_downstream/model/mantis.py), which fine-tunes Mantis-8M through `MantisTrainer.fit`

### How fine-tuning plugs into the pipeline

Both classifiers implement the scikit-learn estimator API (`fit` / `predict` / `predict_proba`), so [`Engine`](edamame_downstream/engine/__init__.py) treats them exactly like logistic regression. This has three consequences worth knowing:

1. **Same evaluation protocol as linear probing.** The outer cross-validation (`lopo`, `tacv`, `lnpo`, `kfold`) is unchanged, and the model is re-instantiated **from the pretrained checkpoint and fine-tuned again on every fold** — no weights leak across folds, and frozen vs fine-tuned numbers are directly comparable.
2. **Hyperparameters come from `param_grid`.** If any entry of the model config's `param_grid` holds more than one value, `Engine` wraps the estimator in a `GridSearchCV` with `engine.inner_cv_folds` (default 3) inner folds; otherwise the model is fitted directly. Every grid point therefore costs a *full fine-tuning run per inner fold*.
3. **Metrics and outputs are unchanged**: the usual per-fold `reports.csv` is written to the Hydra run directory.

### Required pairing: raw signal in, no aggregation

A fine-tuned model consumes the **raw signal**, not features. It must therefore be paired with the flattening pass-through extractor and with aggregation disabled:

```bash
python classification.py --config-name=<dataset> model=ume feature_extractor=none-flat aggregator=none
```

`none-flat` is `NoneFeatureExtractor(preserve_shape=false)`, which hands over a `(n_samples, seq_len * n_channels)` matrix. Each classifier reshapes it internally:

| Model | Internal layout | Time axis |
|-------|-----------------|-----------|
| `ume` | `(n_samples, seq_len, n_channels)` | kept at native resolution (60 s @ 4 Hz = 240 samples) |
| `mantis` | `(n_samples, n_channels, seq_len)` | resampled (`scipy.signal.resample`) or zero-padded to the network's `seq_len` (512) |

> ⚠️ **`n_channels` must match `len(datamodule.channels)`** (default `[0, 1, 2]`, i.e. mixed/phasic/tonic). The flattening is not self-describing, so a mismatch either raises a divisibility error or silently reinterprets the signal.

### UME fine-tuning (`model=ume`)

```yaml
model:
  _target_: edamame_downstream.model.ume.UMEClassifier
  model_name: "efficientnet"        # fine-tuning is supported for the UME encoder only
  weights_path: "ume_weights.nosync/ume.ckpt"
  n_channels: 3                     # must match len(datamodule.channels)
  device: ${device_map}
  fine_tuning_type: full            # full | head | scratch
  learning_rate: 6.8e-5
  use_layer_norm: true              # LayerNorm -> Linear head
  class_weight: "balanced"          # inverse-frequency cross-entropy weights
  batch_size: 16
  num_epochs: 30
  early_stopping: true
  patience: 5
  monitor: val_macro_f1
  verbose: 0
param_grid:
  fine_tuning_type: ["full"]
  batch_size: [4, 16]
  learning_rate: [1e-4, 6.8e-5, 0.5e-5]
```

| `fine_tuning_type` | Behaviour |
|--------------------|-----------|
| `full` | Fine-tune encoder + head. `backbone_lr` sets a separate (smaller) encoder learning rate; it defaults to `learning_rate` |
| `head` | Freeze the encoder and train the head only (linear probe with a trained head) |
| `scratch` | Re-initialise the encoder and train from random weights — the "no pretraining" ablation |

Additional notes:

- **Validation split**: `UMEClassifier.fit` always holds out a stratified 20% of the fold's *training* data (`validation_split=0.2`) to drive early stopping; the test fold is never touched. With `early_stopping: true`, training stops after `patience` epochs without improvement in `monitor` (`val_macro_f1`, `val_acc` or `val_loss`) and the best epoch's weights are restored.
- **Class imbalance**: `class_weight: "balanced"` is usually the right choice for these datasets, since most tasks are skewed.
- **Weights**: `weights_path` is mandatory and must point at a local UME checkpoint (`.ckpt`); it is loaded via `EdamamePipeline.from_pretrained`.

### Mantis fine-tuning (`model=mantis`)

```yaml
model:
  _target_: edamame_downstream.model.mantis.MantisClassifier
  model_name: "paris-noah/Mantis-8M"  # HF hub id, or a path to a local .ckpt
  n_channels: 3                       # must match len(datamodule.channels)
  device: ${device_map}
  fine_tuning_type: full              # full | adapter_head | head | scratch
  num_epochs: 10
  batch_size: 32
  base_learning_rate: 2e-4
param_grid:
  fine_tuning_type: ["full"]
```

- `fine_tuning_type` is forwarded to `MantisTrainer.fit`: `full` (encoder + head), `adapter_head` (adapter + head), `head` (frozen encoder) or `scratch` (random init).
- `learning_rate_adjusting: true` (default) enables Mantis' built-in learning-rate schedule on top of `base_learning_rate`.
- Checkpoint state dicts are cached per `model_name` (`_STATE_DICT_CACHE`), so the repeated re-loading caused by per-fold fitting and grid search does not re-read the checkpoint every time.
- Known-good starting point for Multi-physio, recorded in `configs/classification/multiphysio_2.yaml`: `base_learning_rate=7e-5`, `batch_size=8`, `num_epochs=7`.

### Evaluating an adapted checkpoint as a frozen encoder

Training changes the representation, not just the classifier, and that can be measured separately: point a *frozen* extractor at the adapted checkpoint and linear-probe it under the usual protocol. `feature_extractor=mantis-finetune` does exactly this — the same `MantisExtractor`, but loading a local `.ckpt` (`artifacts.nosync/mantis-finetune-samplepair-EPOCH0.ckpt`) instead of the hub weights. Any local Mantis checkpoint works: `MantisExtractor` and `MantisClassifier` share the same checkpoint-vs-hub loading logic, so `model_name` may be either a hub id or a path.

```bash
python classification.py --config-name=<dataset> feature_extractor=mantis-finetune model=logistic_regression aggregator=mean_time
```

### Requirements and caveats

- **UME** requires the [`edamame`](https://github.com/LeonardoAlchieri/eda-foundation-models) package (declared in `pixi.toml` as an editable local dependency, `../eda-foundation-models`) plus a checkpoint under `ume_weights.nosync/`. **Mantis** requires `mantis-tsfm` (PyPI); its weights are pulled from the HF hub on first use.
- **Device**: set `device_map` (`cpu`, `cuda`, `mps`). Several dataset configs default to `mps`, but the configs also record that the UME (edamame) path does not support MPS — use `cpu` or `cuda` when fine-tuning UME.
- **Cost**: fine-tuning runs once per outer fold, times inner folds, times grid points. LOPO on a dataset with many subjects and a 6-point grid means hundreds of fine-tuning runs. Also note that the inner `GridSearchCV` is currently created with `n_jobs=40`, hard-coded in [`edamame_downstream/engine/__init__.py`](edamame_downstream/engine/__init__.py) — harmless for cheap sklearn models, but it will oversubscribe a single GPU here. Prefer a single-valued `param_grid` (which skips `GridSearchCV` entirely) when fine-tuning on GPU.

## 📋 Experiments

### Systematic Evaluation

Run comprehensive experiments across all model-dataset combinations:

```bash
# Empatica E4 datasets
python classification.py --config-name=usilaughs --multirun
python classification.py --config-name=apsync --multirun
python classification.py --config-name=hearts --multirun
python classification.py --config-name=wesad --multirun
python classification.py --config-name=dreamt --multirun
python classification.py --config-name=hhiss --multirun

# Shimmer (non-Empatica) datasets
python classification.py --config-name=amigos --multirun
python classification.py --config-name=edabe --multirun
python classification.py --config-name=multiphysio_2 --multirun
```

### Fine-Tuning Experiments

Fine-tuning uses the same configs, swapping the `model` slot and disabling fixed feature extraction:

```bash
# Fine-tune UME end-to-end, with the per-fold protocol of the frozen runs
python classification.py --config-name=wesad model=ume feature_extractor=none-flat aggregator=none

# Fine-tune Mantis end-to-end
python classification.py --config-name=multiphysio_2 model=mantis feature_extractor=none-flat aggregator=none \
    model.model.base_learning_rate=7e-5 model.model.batch_size=8 model.model.num_epochs=7

# Same dataset and task, but a different recording sensor
python classification.py --config-name=multiphysio_2 dataset.side=shimmer_fingers \
    model=ume feature_extractor=none-flat aggregator=none

# "No pretraining" ablation: identical architecture, random initialisation
python classification.py --config-name=wesad model=ume feature_extractor=none-flat aggregator=none \
    model.param_grid.fine_tuning_type='[scratch]'
```

> **Override paths:** a model config group exposes both the estimator and its grid, so hyperparameters live at `model.model.<param>` and candidates at `model.param_grid.<param>`. When `param_grid` holds more than one candidate anywhere, `GridSearchCV` runs and the grid values win over `model.model.*` for the parameters it contains — hence `model.param_grid.fine_tuning_type` in the ablation above.

Comparing the fine-tuned run against `feature_extractor=ume model=logistic_regression` on the same config gives the fine-tuning vs linear-probing contrast for that dataset and task.

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

### Adding a Fine-Tunable Model

A fine-tunable foundation model is just a scikit-learn estimator that owns the encoder. Follow the pattern of [`UMEClassifier`](edamame_downstream/model/ume.py) / [`MantisClassifier`](edamame_downstream/model/mantis.py):

1. Create the estimator in `edamame_downstream/model/`:
```python
class NewFineTunedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, weights_path: str, *, n_channels: int = 3, device: str = "cpu",
                 fine_tuning_type: str = "full", num_epochs: int = 30):
        # Store constructor args verbatim: sklearn clones estimators by re-calling __init__
        ...

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NewFineTunedClassifier":
        # X arrives flattened as (n_samples, seq_len * n_channels) from `none-flat`;
        # reshape it back to whatever layout the encoder expects, then load the
        # pretrained weights *here* so every fold starts from the checkpoint.
        ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
```

2. Add a config in `configs/classification/model/`, exposing both the estimator and its grid:
```yaml
model:
  _target_: edamame_downstream.model.new_model.NewFineTunedClassifier
  weights_path: "ume_weights.nosync/your_checkpoint.ckpt"
  n_channels: 3          # must match len(datamodule.channels)
  device: ${device_map}
  fine_tuning_type: full
  num_epochs: 30
param_grid:
  fine_tuning_type: ["full"]   # single-valued -> no inner GridSearchCV
```

3. Run it against the raw signal:
```bash
python classification.py --config-name=<dataset> model=new_model feature_extractor=none-flat aggregator=none
```

Implement `to_dict()` as well if you want the estimator's configuration recorded in the saved results.

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

- [UME / EDAMAME](https://github.com/LeonardoAlchieri/eda-foundation-models) for the EDA foundation model and its fine-tuning pipeline
- [MOMENT](https://github.com/moment-timeseries-foundation-model/moment) for the foundation time series model
- [Chronos](https://github.com/amazon-science/chronos-forecasting) for Amazon's time series forecasting framework
- [Mantis](https://huggingface.co/paris-noah/Mantis-8M) (`mantis-tsfm`) for the Mantis foundation model and its fine-tuning trainer
- [Hydra](https://hydra.cc/) for configuration management
- The contributors of the physiological datasets used in this research

---

⭐ If you find this work useful, please consider starring the repository!
