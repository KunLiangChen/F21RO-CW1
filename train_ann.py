"""
Task 3: trains the small, medium and large inverse kinematics networks.

One network serves both arms: input [is_scara, is_ned, x, y, z], output the three
normalised joint values.

Workflow per dataset size:
    1. Load the merged Ned + SCARA training and validation sets.
    2. Grid-search the hyperparameters of a scikit-learn MLPRegressor, scored on the
       validation set only (PredefinedSplit). The test sets are never touched.
    3. Train the best configuration on the training set only and report its validation
       MSE and MAE (the validation set is never used for fitting).
    4. Save the model to models/model_<size>.pkl and the search result, metrics and loss
       curve to logs/log_<size>.json (used for the poster and video).

Run from the project root: python train_ann.py
"""

import os
import sys
import json
import time
import pickle
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, PredefinedSplit

# Make the project root importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from scripts.dataset_loader import get_combined_dataset


def train_single_model_scale(size_label, param_grid, dataset_dir, models_dir, logs_dir):
    """
    Runs the grid search and training for one dataset size.

    Args:
        size_label (str): 'small', 'medium' or 'large'.
        param_grid (dict): Hyperparameter grid for scikit-learn's MLPRegressor.
        dataset_dir (str): Directory of the datasets.
        models_dir (str): Directory the model is saved to.
        logs_dir (str): Directory the JSON log is saved to.
    """
    print(f"\n========================================================")
    print(f"  Starting Training Pipeline for Scale: [{size_label.upper()}]")
    print(f"========================================================")

    # 1. Load the merged data
    X_train, y_train, X_val, y_val = get_combined_dataset(dataset_dir, size=size_label)
    print(f"Dataset Loaded Successfully:")
    print(f"  - Training samples:   {X_train.shape[0]} (Input Dim: {X_train.shape[1]}, Target Dim: {y_train.shape[1]})")
    print(f"  - Validation samples: {X_val.shape[0]}")

    # 2. PredefinedSplit: every candidate is fitted on train and scored on the validation set
    X_all = np.vstack([X_train, X_val])
    y_all = np.vstack([y_train, y_val])

    # -1 = always in train, 0 = validation fold
    split_indices = np.concatenate([-np.ones(len(X_train)), np.zeros(len(X_val))])
    custom_cv = PredefinedSplit(test_fold=split_indices)

    # 3. Base network: adam solver; MLPRegressor minimises the squared error.
    # max_iter is only an upper limit: with the default tol / n_iter_no_change the training
    # stops as soon as the training loss no longer improves (see epochs in logs/log_<size>.json).
    base_mlp = MLPRegressor(
        solver='adam',
        max_iter=600,
        random_state=42,
        early_stopping=False,  # no hold-out split inside the training; the grid search uses the validation set
        verbose=False
    )

    # 4. Hyperparameter grid search
    print(f"\n[Info] Launching GridSearchCV on Validation Split...")
    start_time = time.time()
    
    grid_search = GridSearchCV(
        estimator=base_mlp,
        param_grid=param_grid,
        cv=custom_cv,
        scoring='neg_mean_squared_error',
        refit=False,  # the final model is trained on the training set only, see below
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_all, y_all)
    elapsed_time = time.time() - start_time

    best_params = grid_search.best_params_

    # Train the best configuration on the training set alone. GridSearchCV's default
    # refit would also fit on the validation data, which would make the validation
    # metrics below optimistic.
    best_model = clone(base_mlp).set_params(**best_params)
    best_model.fit(X_train, y_train)

    print(f"\n[Results] Best Hyperparameters for [{size_label.upper()}]:")
    for param, val in best_params.items():
        print(f"  - {param}: {val}")
    print(f"  - Grid search execution time: {elapsed_time:.2f} seconds")

    # 5. Validation metrics of the best model
    y_val_pred = best_model.predict(X_val)
    val_mse = float(mean_squared_error(y_val, y_val_pred))
    val_mae = float(mean_absolute_error(y_val, y_val_pred))

    print(f"\n[Validation Performance]:")
    print(f"  - Mean Squared Error (MSE): {val_mse:.6e}")
    print(f"  - Mean Absolute Error (MAE): {val_mae:.6e}")

    # 6. Save the model
    os.makedirs(models_dir, exist_ok=True)
    model_file_path = os.path.join(models_dir, f"model_{size_label}.pkl")
    with open(model_file_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\n[Saved] Model saved to: {os.path.relpath(model_file_path, PROJECT_ROOT)}")

    # 7. Save the training log and loss curve
    os.makedirs(logs_dir, exist_ok=True)
    log_file_path = os.path.join(logs_dir, f"log_{size_label}.json")
    
    loss_curve = [float(l) for l in best_model.loss_curve_] if hasattr(best_model, 'loss_curve_') else []
    
    log_data = {
        "scale": size_label,
        "dataset_info": {
            "num_train_samples": int(X_train.shape[0]),
            "num_val_samples": int(X_val.shape[0]),
            "feature_dim": int(X_train.shape[1]),
            "target_dim": int(y_train.shape[1])
        },
        "best_hyperparameters": best_params,
        "validation_metrics": {
            "mse": val_mse,
            "mae": val_mae
        },
        "training_time_seconds": round(elapsed_time, 2),
        "loss_curve": loss_curve
    }

    with open(log_file_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=4)
    print(f"[Saved] Training log saved to: {os.path.relpath(log_file_path, PROJECT_ROOT)}")


def main():
    """
    Trains the three models; the grids grow with the dataset size.
    """
    dataset_dir = os.path.join(PROJECT_ROOT, 'datasets')
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    logs_dir = os.path.join(PROJECT_ROOT, 'logs')

    # Hyperparameter grid per dataset size; larger datasets get larger networks
    param_grids = {
        "small": {
            "hidden_layer_sizes": [(64, 64), (128, 64)],
            "activation": ["relu", "tanh"],
            "alpha": [1e-4, 1e-3],
            "learning_rate_init": [0.001, 0.005]
        },
        "medium": {
            "hidden_layer_sizes": [(128, 128), (128, 128, 64)],
            "activation": ["relu", "tanh"],
            "alpha": [1e-4, 1e-3],
            "learning_rate_init": [0.001]
        },
        "large": {
            "hidden_layer_sizes": [(256, 128, 64), (256, 256, 128)],
            "activation": ["relu", "tanh"],
            "alpha": [1e-4, 1e-3],
            "learning_rate_init": [0.001]
        }
    }

    # Train all three sizes
    for size in ["small", "medium", "large"]:
        train_single_model_scale(
            size_label=size,
            param_grid=param_grids[size],
            dataset_dir=dataset_dir,
            models_dir=models_dir,
            logs_dir=logs_dir
        )

    print("\n========================================================")
    print("  Task 3 Training Pipeline Completed Successfully!")
    print("========================================================\n")


if __name__ == "__main__":
    main()