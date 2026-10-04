"""

Extraprostatic Extension (EPE) Prediction
Hyperparameter Optimization

"""

from scipy.stats import randint, uniform, loguniform

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# -----------------------------------------
# XGBoost Setup
# -----------------------------------------

xgb_init_params = {
    'objective': 'binary:logistic', 
    'eval_metric': 'logloss', 
    'booster': 'gbtree', 
    'n_estimators': 400, 
    'learning_rate': 0.05, 
    'max_depth': 3, 
    'min_child_weight': 3, 
    'gamma': 0.0, 
    'subsample': 0.8, 
    'colsample_bytree': 0.8, 
    'reg_alpha': 0.0, 
    'reg_lambda': 1.0, 
    'scale_pos_weight': 1.0, 
    'n_jobs': 1, 
}

xgb_grid_params = {
    'n_estimators': [300, 600], 
    'learning_rate': [0.03, 0.08], 
    'max_depth': [2, 3, 4], 
    'min_child_weight': [1, 5], 
    'subsample': [0.75, 1.0], 
    'colsample_bytree': [0.7, 1.0], 
    'reg_alpha': [0.0, 0.5], 
    'reg_lambda': [1.0, 5.0], 
}

xgb_random_params = {
    'n_estimators': randint(200, 801), 
    'learning_rate': loguniform(0.01, 0.15), 
    'max_depth': randint(2, 6), 
    'min_child_weight': randint(1, 11), 
    'gamma': uniform(0.0, 3.0), 
    'subsample': uniform(0.65, 0.35), 
    'colsample_bytree': uniform(0.60, 0.40), 
    'reg_alpha': [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0], 
    'reg_lambda': loguniform(0.1, 20.0), 
}

# -----------------------------------------
# LightGBM Setup
# -----------------------------------------
    
lgbm_init_params = {
    'objective': 'binary', 
    'boosting_type': 'gbdt', 
    'n_estimators': 400, 
    'learning_rate': 0.05, 
    'num_leaves': 15, 
    'max_depth': 4, 
    'min_child_samples': 30, 
    'min_child_weight': 1e-3, 
    'min_split_gain': 0.0, 
    'subsample': 0.8, 
    'subsample_freq': 1, 
    'colsample_bytree': 0.8, 
    'colsample_bynode': 1.0, 
    'reg_alpha': 0.0, 
    'reg_lambda': 1.0, 
    'scale_pos_weight': 1.0, 
    'n_jobs': 1, 
    'verbosity': -1, 
}

lgbm_grid_params = [
    
    # Shallow trees
    {'max_depth': [3], 
     'num_leaves': [7], 
     'n_estimators': [300, 600], 
     'learning_rate': [0.03, 0.08], 
     'min_child_samples': [20, 50], 
     'min_child_weight': [1e-3], 
     'min_split_gain': [0.0, 0.2], 
     'subsample': [0.75, 1.0], 
     'subsample_freq': [1], 
     'colsample_bytree': [0.7, 1.0], 
     'reg_alpha': [0.0, 0.5], 
     'reg_lambda': [1.0, 5.0], 
    }, 
    
    
    # Moderately deep trees
    {'max_depth': [4], 
     'num_leaves': [12, 15], 
     'n_estimators': [300, 600], 
     'learning_rate': [0.03, 0.08], 
     'min_child_samples': [30, 60], 
     'min_child_weight': [1e-3], 
     'min_split_gain': [0.0, 0.2], 
     'subsample': [0.75, 1.0], 
     'subsample_freq': [1], 
     'colsample_bytree': [0.7, 1.0], 
     'reg_alpha': [0.0, 0.5], 
     'reg_lambda': [1.0, 5.0], 
     }
]

lgbm_random_params = [
    
    # Shallow trees
    {'max_depth': [3], 
     'num_leaves': randint(5, 9), 
     'n_estimators': randint(200, 801), 
     'learning_rate': loguniform(0.01, 0.15), 
     'min_child_samples': randint(20, 81), 
     'min_child_weight': loguniform(1e-4, 1.0), 
     'min_split_gain': uniform(0.0, 1.0), 
     'subsample': uniform(0.65, 0.35), 
     'subsample_freq': [1], 
     'colsample_bytree': uniform(0.60, 0.40), 
     'reg_alpha': [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0], 
     'reg_lambda': loguniform(0.1, 20.0), 
    }, 
    
    # Moderately deep trees
    {'max_depth': [4], 
     'num_leaves': randint(8, 17), 
     'n_estimators': randint(200, 801), 
     'learning_rate': loguniform(0.01, 0.15), 
     'min_child_samples': randint(20, 81), 
     'min_child_weight': loguniform(1e-4, 1.0), 
     'min_split_gain': uniform(0.0, 1.0), 
     'subsample': uniform(0.65, 0.35), 
     'subsample_freq': [1], 
     'colsample_bytree': uniform(0.60, 0.40), 
     'reg_alpha': [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0], 
     'reg_lambda': loguniform(0.1, 20.0), 
    }, 
    
    # Slightly more flexible trees
    {'max_depth': [5], 
     'num_leaves': randint(12, 25), 
     'n_estimators': randint(200, 801), 
     'learning_rate': loguniform(0.01, 0.15), 
     'min_child_samples': randint(25, 101), 
     'min_child_weight': loguniform(1e-4, 1.0), 
     'min_split_gain': uniform(0.0, 1.0), 
     'subsample': uniform(0.65, 0.35), 
     'subsample_freq': [1], 
     'colsample_bytree': uniform(0.60, 0.40), 
     'reg_alpha': [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0], 
     'reg_lambda': loguniform(0.1, 20.0), 
    }
]

# -----------------------------------------
# CatBoost Setup
# -----------------------------------------

cb_init_params = {
    'loss_function': 'Logloss', 
    'eval_metric': 'AUC', 
    'iterations': 500, 
    'learning_rate': 0.03, 
    'depth': 5, 
    'l2_leaf_reg': 3.0, 
    'random_strength': 1.0, 
    'bootstrap_type': 'Bayesian', 
    'bagging_temperature': 1.0, 
    'rsm': 0.8, 
    'thread_count': 1, 
    'verbose': False, 
    'allow_writing_files': False, 
}

cb_grid_params = {
    'iterations': [400, 700], 
    'learning_rate': [0.03, 0.08], 
    'depth': [4, 6], 
    'l2_leaf_reg': [1.0, 5.0, 15.0], 
    'random_strength': [0.0, 2.0], 
    'bagging_temperature': [0.0, 1.0], 
    'rsm': [0.8, 1.0], 
}

cb_random_params = {
    'iterations': randint(250, 1001), 
    'learning_rate': loguniform(0.01, 0.15), 
    'depth': randint(3, 7), 
    'l2_leaf_reg': loguniform(0.1, 30.0), 
    'random_strength': [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0], 
    'bagging_temperature': uniform(0.0, 3.0), 
    'rsm': uniform(0.6, 0.4), 
}

# -----------------------------------------
# RandomForest Setup
# -----------------------------------------

rf_init_params = {
    'n_estimators': 500, 
    'criterion': 'gini', 
    'max_depth': None, 
    'min_samples_split': 5, 
    'min_samples_leaf': 2, 
    'max_features': 'sqrt', 
    'bootstrap': True, 
    'max_samples': None, 
    'class_weight': None, 
    'n_jobs': 1, 
}

rf_grid_params = {
    'n_estimators': [400, 800], 
    'max_depth': [None, 8], 
    'min_samples_split': [2, 10], 
    'min_samples_leaf': [2, 8], 
    'max_features': ['sqrt', 0.7], 
    'max_samples': [None, 0.75], 
    'class_weight': [None, 'balanced_subsample'], 
}

rf_random_params = {
    'n_estimators': randint(300, 1001), 
    'max_depth': [None, 4, 6, 8, 10, 12, 16], 
    'min_samples_split': randint(2, 21), 
    'min_samples_leaf': randint(1, 16), 
    'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7, 1.0], 
    'max_samples': [None, 0.6, 0.75, 0.9], 
    'criterion': ['gini', 'entropy', 'log_loss'], 
    'class_weight': [None, 'balanced_subsample'], 
}

# -----------------------------------------
# GradientBoost Setup
# -----------------------------------------

gb_init_params = {
    'loss': 'log_loss', 
    'n_estimators': 300, 
    'learning_rate': 0.05, 
    'subsample': 0.8, 
    'criterion': 'friedman_mse', 
    'max_depth': 2, 
    'min_samples_split': 5, 
    'min_samples_leaf': 5, 
    'max_features': None, 
    'min_weight_fraction_leaf': 0.0, 
    'n_iter_no_change': None, 
}

gb_grid_params = {
    'n_estimators': [250, 500], 
    'learning_rate': [0.03, 0.08], 
    'subsample': [0.7, 1.0], 
    'max_depth': [1, 3], 
    'min_samples_split': [2, 10], 
    'min_samples_leaf': [2, 8], 
    'max_features': [None, 'sqrt'], 
}

gb_random_params = {
    'n_estimators': randint(100, 801), 
    'learning_rate': loguniform(0.01, 0.15), 
    'subsample': uniform(0.6, 0.4), 
    'max_depth': randint(1, 5), 
    'min_samples_split': randint(2, 21), 
    'min_samples_leaf': randint(1, 21), 
    'max_features': [None, 'sqrt', 'log2', 0.5, 0.75], 
}

# -----------------------------------------
# AdaBoost Setup
# -----------------------------------------

weak_learner = DecisionTreeClassifier(
    criterion = 'gini', 
    max_depth = 1, 
    min_samples_split = 5, 
    min_samples_leaf = 5, 
    max_features = None, 
    class_weight = None, 
)

ab_init_params = {
    'estimator': weak_learner, 
    'n_estimators': 300, 
    'learning_rate': 0.05, 
}

ab_grid_params = {
    'n_estimators': [200, 500], 
    'learning_rate': [0.05, 0.2], 
    'estimator__max_depth': [1, 2], 
    'estimator__min_samples_leaf': [2, 8], 
    'estimator__max_features': [None, 0.75], 
}

ab_random_params = {
    'n_estimators': randint(100, 801), 
    'learning_rate': loguniform(0.01, 1.0), 
    'estimator__max_depth': [1, 2, 3], 
    'estimator__min_samples_split': randint(2, 21), 
    'estimator__min_samples_leaf': randint(1, 16), 
    'estimator__max_features': [None, 'sqrt', 0.5, 0.75], 
}

# -----------------------------------------
# LogisticRegression (ElasticNet) Setup
# -----------------------------------------

lr_init_params = {
    'penalty': 'elasticnet', 
    'solver': 'saga', 
    'C': 1.0, 
    'l1_ratio': 0.5, 
    'fit_intercept': True, 
    'class_weight': None, 
    'max_iter': 5000, 
    'tol': 1e-4, 
}

lr_grid_params = {
    'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], 
    'l1_ratio': [0.0, 0.25, 0.50, 0.75, 1.0], 
    'class_weight': [None, 'balanced'], 
}

lr_random_params = {
    'C': loguniform(1e-4, 1e3), 
    'l1_ratio': [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0], 
    'class_weight': [None, 'balanced'], 
}

# -----------------------------------------
# SupportVectorMachine Setup
# -----------------------------------------

svm_init_params = {
    'kernel': 'rbf', 
    'C': 1.0, 
    'gamma': 'scale', 
    'class_weight': None, 
    'probability': False, 
    'tol': 1e-3, 
    'shrinking': True, 
    'cache_size': 1000, 
    'max_iter': -1, 
}

svm_grid_params = {
    'C': [0.01, 0.1, 1.0, 10.0, 100.0], 
    'gamma': [0.0001, 0.001, 0.01, 0.1, 1.0], 
    'class_weight': [None, 'balanced'], 
}

svm_random_params = {
    'C': loguniform(1e-3, 1e3), 
    'gamma': loguniform(1e-4, 1e1), 
    'class_weight': [None, 'balanced'], 
}

# -----------------------------------------
# kNearestNeighbors Setup
# -----------------------------------------

knn_init_params = {
    'n_neighbors': 11, 
    'weights': 'distance', 
    'algorithm': 'auto', 
    'leaf_size': 30, 
    'metric': 'minkowski', 
    'p': 2, 
    'n_jobs': 1, 
}

knn_grid_params = {
    'n_neighbors': [3, 5, 7, 9, 11, 15, 21, 27, 35, 43, 51], 
    'weights': ['uniform', 'distance'], 
    'p': [1, 2], 
}

knn_random_params = {
    'n_neighbors': randint(3, 52), 
    'weights': ['uniform', 'distance'], 
    'p': [1, 2], 
}

# -----------------------------------------
# MultilayerPerceptron Setup
# -----------------------------------------

mlp_init_params = {
    'hidden_layer_sizes': (32,), 
    'activation': 'relu', 
    'solver': 'adam', 
    'alpha': 1e-3, 
    'learning_rate_init': 1e-3, 
    'batch_size': 'auto', 
    'max_iter': 1000, 
    'tol': 1e-4, 
    'n_iter_no_change': 20, 
    'early_stopping': False, 
    'shuffle': True, 
    'verbose': False, 
}

mlp_grid_params = [
    
    # L-BFGS (often effective for smaller datasets)
    {'solver': ['lbfgs'], 
     'hidden_layer_sizes': [(16,), (32,), (16, 8)], 
     'activation': ['tanh', 'relu'], 
     'alpha': [1e-4, 1e-3, 1e-2], 
     'max_fun': [15000, 30000], 
    }, 
    
    # Adam
    {'solver': ['adam'], 
     'hidden_layer_sizes': [(16,), (32,), (16, 8)], 
     'activation': ['tanh', 'relu'], 
     'alpha': [1e-4, 1e-3, 1e-2], 
     'learning_rate_init': [1e-4, 1e-3, 1e-2], 
     'batch_size': [32, 'auto'], 
    }
]
    

mlp_random_params = [
    
    # L-BFGS (often effective for smaller datasets)
    {'solver': ['lbfgs'], 
     'hidden_layer_sizes': [(8,), (16,), (32,), (64,), (16, 8), (32, 16)], 
     'activation': ['tanh', 'relu'], 'alpha': loguniform(1e-5, 1e-1), 
     'max_fun': [15000, 30000, 50000], 
    }, 
    
    # Adam
    {'solver': ['adam'], 
     'hidden_layer_sizes': [(8,), (16,), (32,), (64,), (16, 8), (32, 16)], 
     'activation': ['tanh', 'relu'], 
     'alpha': loguniform(1e-5, 1e-1), 
     'learning_rate_init': loguniform(1e-4, 1e-2), 
     'batch_size': [32, 64, 128, 'auto'], 
     'beta_1': [0.9], 
     'beta_2': [0.999], 
    }
]