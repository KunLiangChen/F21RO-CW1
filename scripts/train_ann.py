"""
Task 3: trains the small, medium and large inverse kinematics networks.

One network serves both arms: input [is_scara, is_ned, x, y, z], output the three
normalised joint values.

Workflow per dataset size:
    1. Load the merged Ned + SCARA training and validation sets.
    2. Grid-search the hyperparameters of a scikit-learn MLPRegressor, scored on the
       validation set only (PredefinedSplit). The test sets are never touched.
       Score = mean TCP position error (FK of the predicted joints) of the two arms, with
       both arms weighted equally; every configuration is trained with several random
       seeds and the scores are averaged, so a lucky initialisation does not win.
    3. Train the best configuration on the training set only, once per seed, keep the seed
       with the lowest validation position error and report its validation MSE and MAE
       (the validation set is never used for fitting).
    4. Save the model to models/model_<size>.pkl and the search result, metrics and loss
       curve to logs/log_<size>.json (used for the poster and video).

Run from the project root: python scripts/train_ann.py
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
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from scripts.dataset_loader import get_combined_dataset
from scripts.preprocess import denormalize_joints
from scripts.scara_fk import scara_forward_kinematics
from scripts.ned_fk import ned_forward_kinematics

# Seeds the network is initialised with during the search (scores are averaged over them);
# the final model is the best of these seeds on the validation set
SEARCH_SEEDS = [42, 43, 44]
FINAL_SEED = 42  # default of the base network


def _load_bounds(arm):
    """q_min, q_max (normalisation bounds written by preprocess.py) of 'scara' or 'ned'."""
    with open(os.path.join(PROJECT_ROOT, "normalisation", f"{arm}.json"), encoding="utf-8") as f:
        config = json.load(f)
    return np.array(config["q_min"]), np.array(config["q_max"])


def mean_position_error(estimator, X, y):
    """
    Search score (higher is better): minus the mean TCP position error [m] of the two arms.

    The predicted normalised joints are de-normalised and passed through FK (Task 1.B), and
    the result is compared with the target position that was fed to the network (columns
    2:5 of X). Each arm's mean error counts once, so neither arm dominates the choice.
    `y` is not needed: the target position is already part of X.
    """
    predicted = estimator.predict(X)
    is_scara = X[:, 0] == 1.0
    errors = []
    for mask, arm, fk in ((is_scara, "scara", scara_forward_kinematics),
                          (~is_scara, "ned", ned_forward_kinematics)):
        q_min, q_max = _load_bounds(arm)
        joints = denormalize_joints(predicted[mask], q_min, q_max)
        errors.append(np.mean(np.linalg.norm(fk(joints) - X[mask, 2:5], axis=1)))
    return -float(np.mean(errors))


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
    # Stopping rule: training ends when the training loss has not improved by more than
    # tol for n_iter_no_change epochs in a row; max_iter is only an upper limit.
    # The scikit-learn defaults (tol=1e-4, n_iter_no_change=10) are too coarse here: the loss
    # is about 1e-3, so an absolute tol of 1e-4 stops the training after 60 to 100 epochs
    # while it is still improving. With tol=1e-7 and 50 epochs of patience the validation
    # MSE drops by up to 3x (e.g. medium 3.8e-3 -> 1.2e-3), and training still takes seconds.
    base_mlp = MLPRegressor(
        solver='adam',
        max_iter=3000,
        tol=1e-7,
        n_iter_no_change=50,
        random_state=FINAL_SEED,
        early_stopping=False,  # no hold-out split inside the training; the grid search uses the validation set
        verbose=False
    )

    # 4. Hyperparameter grid search. The seed is part of the grid, so every configuration is
    # trained once per seed; the seeds are averaged below.
    seeded_grid = dict(param_grid, random_state=SEARCH_SEEDS)
    print(f"\n[Info] Launching GridSearchCV on Validation Split...")
    start_time = time.time()

    grid_search = GridSearchCV(
        estimator=base_mlp,
        param_grid=seeded_grid,
        cv=custom_cv,
        scoring=mean_position_error,
        refit=False,  # the final model is trained on the training set only, see below
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_all, y_all)
    elapsed_time = time.time() - start_time

    # Average the score of every configuration over the seeds and take the best one
    results = grid_search.cv_results_
    scores_by_config = {}
    for params, score in zip(results["params"], results["mean_test_score"]):
        key = json.dumps({k: v for k, v in params.items() if k != "random_state"}, sort_keys=True)
        scores_by_config.setdefault(key, []).append(-score)
    search_results = sorted(
        ({"params": json.loads(k), "mean_position_error_m": float(np.mean(v)),
          "std_over_seeds_m": float(np.std(v))} for k, v in scores_by_config.items()),
        key=lambda row: row["mean_position_error_m"])
    best_params = {k: (tuple(v) if isinstance(v, list) else v)
                   for k, v in search_results[0]["params"].items()}

    # Train the best configuration on the training set alone. GridSearchCV's default
    # refit would also fit on the validation data, which would make the validation
    # metrics below optimistic. One network is kept: the one of the seeds with the lowest
    # validation position error (a single unlucky initialisation can leave one joint much
    # worse, e.g. the SCARA shaft of the medium model was 7x worse with seed 42).
    best_model, best_seed, best_score = None, None, -np.inf
    for seed in SEARCH_SEEDS:
        candidate = clone(base_mlp).set_params(**best_params, random_state=seed)
        candidate.fit(X_train, y_train)
        score = mean_position_error(candidate, X_val, y_val)
        print(f"  - final training, seed {seed}: validation position error {-1000 * score:.2f} mm")
        if score > best_score:
            best_model, best_seed, best_score = candidate, seed, score
    print(f"  - kept seed {best_seed}")

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
        "search_results": search_results,
        "search_seeds": SEARCH_SEEDS,
        "final_model_seed": best_seed,
        "validation_metrics": {
            "mse": val_mse,
            "mae": val_mae,
            "mean_position_error_m": -float(mean_position_error(best_model, X_val, y_val))
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
    # Same hyperparameters for every size, only the layer sizes grow with the data
    common = {
        "activation": ["relu", "tanh"],
        "alpha": [1e-4, 1e-3],
        "learning_rate_init": [0.001, 0.003]
    }
    param_grids = {
        "small": dict(common, hidden_layer_sizes=[(64, 64), (128, 64), (128, 128), (64, 64, 64)]),
        "medium": dict(common, hidden_layer_sizes=[(128, 128), (128, 128, 64), (256, 128), (128, 128, 128)]),
        "large": dict(common, hidden_layer_sizes=[(256, 128, 64), (256, 256, 128), (256, 256), (256, 256, 256)])
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