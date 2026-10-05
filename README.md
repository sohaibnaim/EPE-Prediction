# EPE-Prediction

Hyperparameter optimization framework for conventional machine learning (ML) algorithms used for preoperative prediction of extraprostatic extension (EPE) in prostate cancer.

## Overview

This repository provides the algorithm hyperparameter configurations and optimization search spaces used to train and evaluate conventional ML models for EPE prediction. Hyperparameter optimization was applied to only conventional ML algorithms evaluated in our study. Tabular foundation models (TabPFN and TabFM), which were evaluated without task-specific hyperparameter tuning, are not included in this optimization framework.

### `model_hyperparameter_optimization.py`

This file contains all baseline (`init`) hyperparameters for each conventional ML algorithm along with pre-defined hyperparameter search spaces for optimization using either `GridSearchCV` or `RandomizedSearchCV` from the scikit-learn library.

Hyperparameter optimization was performed independently within each fold of the outer 5-fold cross-validation framework to minimize information leakage and ensure consistent model comparison across bootstrapped iterations. Randomized search was prioritized for the primary experiments.

For each outer cross-validation fold, `RandomizedSearchCV` evaluated 50 randomly sampled hyperparameter combinations using an internal 5-fold cross-validation procedure, resulting in 250 model fits per outer fold. The optimal hyperparameter configuration that yielded the optimal cross-validation performance was subsequently used to fit the corresponding model and evaluate its performance on the held-out bootstrapped test set.

For reproducibility, a `random_state` parameter was assigned to the same predefined random seed used for the corresponding training, validation, and test splits within each bootstrap iteration.

## Hyperparameter Configuration

The provided parameter dictionaries can be used in three configurations:

- **Baseline:** `init`
- **Grid search:** `init` + `grid`
- **Randomized search:** `init` + `random`

The `init` parameters define the baseline model configuration, while the `grid` and `random` dictionaries define the parameter spaces evaluated using either `GridSearchCV` or `RandomizedSearchCV`, respectively.

### Requirements

- Python 3.11
- scikit-learn
- XGBoost
- LightGBM
- CatBoost

## Citation

If you use this code or implement this hyperparameter optimization framework as a benchmark for comparison, please cite:

> [citation]
