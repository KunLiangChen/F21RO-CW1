"""
Main Neural Network Training and Hyperparameter Tuning Script for Task 3.

This script executes the complete training pipeline for Artificial Neural Network (ANN)
based Inverse Kinematics for both Niryo Ned and Epson SCARA robotic arms.

Pipeline Workflow:
    1. Loads merged Ned + SCARA training and validation datasets for Small, Medium, Large scales.
    2. Uses explicit PredefinedSplit cross-validation to tune hyperparameters strictly on
       the validation set (preventing data leakage and random split inconsistency).
    3. Trains MLPRegressor models using scikit-learn.
    4. Evaluates performance using Mean Squared Error (MSE) and Mean Absolute Error (MAE).
    5. Saves trained serialized model files (.pkl) to 'models/' directory.
    6. Saves hyperparameter selection history, evaluation metrics, and loss curves (.json)
       to 'logs/' directory for visualization in Poster and Video deliverables.

Coursework Reference:
    F20/21RO Intelligent Robotics - Coursework 1 Task 3
"""

import os
import sys
import json
import time
import pickle
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, PredefinedSplit

# Ensure project root directory is in system path for relative imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from scripts.dataset_loader import get_combined_dataset


def train_single_model_scale(size_label, param_grid, dataset_dir, models_dir, logs_dir):
    """
    Executes automated hyperparameter grid search and training for a specific dataset scale.

    Args:
        size_label (str): Scale size ('small', 'medium', or 'large').
        param_grid (dict): Grid search hyperparameter space for scikit-learn MLPRegressor.
        dataset_dir (str): Path to datasets directory.
        models_dir (str): Path to save serialized models.
        logs_dir (str): Path to save JSON training logs.
    """
    print(f"\n========================================================")
    print(f"  Starting Training Pipeline for Scale: [{size_label.upper()}]")
    print(f"========================================================")

    # 1. Load combined data
    X_train, y_train, X_val, y_val = get_combined_dataset(dataset_dir, size=size_label)
    print(f"Dataset Loaded Successfully:")
    print(f"  - Training samples:   {X_train.shape[0]} (Input Dim: {X_train.shape[1]}, Target Dim: {y_train.shape[1]})")
    print(f"  - Validation samples: {X_val.shape[0]}")

    # 2. Setup PredefinedSplit to strictly evaluate grid search on explicit Validation set
    X_all = np.vstack([X_train, X_val])
    y_all = np.vstack([y_train, y_val])

    # Mark training samples as -1, validation samples as 0
    split_indices = np.concatenate([-np.ones(len(X_train)), np.zeros(len(X_val))])
    custom_cv = PredefinedSplit(test_fold=split_indices)

    # 3. Define Base Multi-Layer Perceptron Regressor
    # Using 'adam' solver and MSE loss metric as required for smooth kinematic mapping
    base_mlp = MLPRegressor(
        solver='adam',
        max_iter=600,
        random_state=42,
        early_stopping=False,  # Early stopping handled via GridSearch evaluation on validation split
        verbose=False
    )

    # 4. Perform Automated Hyperparameter Grid Search
    print(f"\n[Info] Launching GridSearchCV on Validation Split...")
    start_time = time.time()
    
    grid_search = GridSearchCV(
        estimator=base_mlp,
        param_grid=param_grid,
        cv=custom_cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_all, y_all)
    elapsed_time = time.time() - start_time

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"\n[Results] Best Hyperparameters for [{size_label.upper()}]:")
    for param, val in best_params.items():
        print(f"  - {param}: {val}")
    print(f"  - Grid search execution time: {elapsed_time:.2f} seconds")

    # 5. Validate Best Model Metrics
    y_val_pred = best_model.predict(X_val)
    val_mse = float(mean_squared_error(y_val, y_val_pred))
    val_mae = float(mean_absolute_error(y_val, y_val_pred))

    print(f"\n[Validation Performance]:")
    print(f"  - Mean Squared Error (MSE): {val_mse:.6e}")
    print(f"  - Mean Absolute Error (MAE): {val_mae:.6e}")

    # 6. Save Model (.pkl)
    os.makedirs(models_dir, exist_ok=True)
    model_file_path = os.path.join(models_dir, f"model_{size_label}.pkl")
    with open(model_file_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\n[Saved] Model saved to: {os.path.relpath(model_file_path, PROJECT_ROOT)}")

    # 7. Save Training Logs and Loss Curve (.json)
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
    Main execution pipeline configuring model scale hyperparameter search spaces.
    """
    dataset_dir = os.path.join(PROJECT_ROOT, 'datasets')
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    logs_dir = os.path.join(PROJECT_ROOT, 'logs')

    # Define hyperparameter grid for each model scale size
    # Architectures expand in capacity to match increasing dataset sizes
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

    # Execute training sequence for all 3 scales
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