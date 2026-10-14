"""
Task 4.A: Offline Inverse Kinematics Evaluation Script using Analytical Forward Kinematics.

This script evaluates the performance of the 3 trained ANN models (Small, Medium, Large)
on the independent test sets for both Niryo Ned and Epson SCARA robotic arms.

Workflow per Arm and Model Scale:
    1. Input target TCP position (x, y, z)_true with robot ID flag into the ANN model.
    2. Predict normalized joint values q_norm_pred in [-1.0, 1.0].
    3. De-normalize q_norm_pred into physical joint units (radians / meters).
    4. Compute Analytical Forward Kinematics (FK) to obtain (x, y, z)_pred.
    5. Compute error statistics across test samples:
       - Position Error: Euclidean distance ||(x,y,z)_true - (x,y,z)_pred||
       - Revolute Joint Error: Wrapped angular difference atan2(sin(d_theta), cos(d_theta))
       - Prismatic Joint Error: Squared linear error (d_true - d_pred)^2

Coursework Reference:
    F20/21RO Intelligent Robotics - Coursework 1 Task 4.A
"""

import os
import sys
import json
import pickle
import numpy as np

# Ensure project root is in system path for relative imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

# Import FK modules from Task 1.B
from scripts.ned_fk import ned_forward_kinematics
from scripts.scara_fk import scara_forward_kinematics
from scripts.dataset_loader import load_csv_dataset, prepare_robot_features


def load_normalisation_config(json_path):
    """
    Loads joint min/max array bounds from normalisation JSON configuration.
    
    Expected format:
    {
        "robot": "ned",
        "q_min": [-2.7999, -0.7997, 0.00006],
        "q_max": [2.7984, 0.7998, 1.2996]
    }
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Normalisation config file not found at: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    q_min = np.array(config["q_min"], dtype=np.float64)
    q_max = np.array(config["q_max"], dtype=np.float64)
    return q_min, q_max


def denormalize_joints_batch(q_norm_batch, q_min, q_max):
    """
    De-normalizes a batch of joint values (N, 3) from [-1.0, 1.0] back to physical values:
    q = q_min + ((q_norm + 1) / 2) * (q_max - q_min)

    Args:
        q_norm_batch (np.ndarray): Normalized joint array of shape (N, 3).
        q_min (np.ndarray): Minimum joint limit array (3,).
        q_max (np.ndarray): Maximum joint limit array (3,).

    Returns:
        np.ndarray: Physical joint values of shape (N, 3).
    """
    return q_min + ((q_norm_batch + 1.0) / 2.0) * (q_max - q_min)


def calculate_wrapped_angular_error(theta_true, theta_pred):
    """
    Computes wrapped angular difference: atan2(sin(theta_true - theta_pred), cos(theta_true - theta_pred)).
    
    Args:
        theta_true (np.ndarray): True joint angles (N,).
        theta_pred (np.ndarray): Predicted joint angles (N,).

    Returns:
        np.ndarray: Absolute wrapped angular error array (N,) in radians.
    """
    diff = theta_true - theta_pred
    wrapped_diff = np.arctan2(np.sin(diff), np.cos(diff))
    return np.abs(wrapped_diff)


def evaluate_model_on_robot(model, test_pos, test_joints_norm, is_scara, q_min, q_max):
    """
    Evaluates a trained model on a specific robot test dataset using vectorized computation.

    Args:
        model: Trained scikit-learn MLPRegressor model (.pkl).
        test_pos (np.ndarray): Target TCP positions (N, 3).
        test_joints_norm (np.ndarray): True normalized joint values (N, 3).
        is_scara (bool): True for SCARA, False for Ned.
        q_min (np.ndarray): Minimum physical joint bounds (3,).
        q_max (np.ndarray): Maximum physical joint bounds (3,).

    Returns:
        dict: Summary statistics containing mean and std for position and joint errors.
    """
    # 1. Prepare input feature vectors: [is_scara, is_ned, x, y, z] (N, 5)
    X_test = prepare_robot_features(test_pos, is_scara=is_scara)

    # 2. Predict normalized joint values (N, 3)
    q_norm_pred = model.predict(X_test)

    # 3. De-normalize to physical joint values (N, 3)
    q_true_phys = denormalize_joints_batch(test_joints_norm, q_min, q_max)
    q_pred_phys = denormalize_joints_batch(q_norm_pred, q_min, q_max)

    # 4. Compute Forward Kinematics (FK) on predicted physical joints (N, 3)
    fk_func = scara_forward_kinematics if is_scara else ned_forward_kinematics
    tcp_pred = fk_func(q_pred_phys)

    # 5. Calculate Metrics
    # A. Position Error: Euclidean distance ||(x,y,z)_true - (x,y,z)_pred||
    pos_errors = np.linalg.norm(test_pos - tcp_pred, axis=1)

    # B. Active Joint Errors
    joint_names = (
        ["base_arm_motor", "arm_motor", "shaft_linear_motor"] if is_scara
        else ["joint_1", "joint_2", "joint_3"]
    )
    
    joint_stats = []
    for j in range(3):
        is_prismatic = (is_scara and j == 2)  # SCARA 3rd joint is linear motor
        
        if is_prismatic:
            # Prismatic Motor Error: e_d = (d_true - d_pred)^2
            j_err = (q_true_phys[:, j] - q_pred_phys[:, j]) ** 2
            unit = "m^2"
        else:
            # Revolute Motor Error: wrapped angular difference
            j_err = calculate_wrapped_angular_error(q_true_phys[:, j], q_pred_phys[:, j])
            unit = "rad"

        joint_stats.append({
            "joint_index": j + 1,
            "joint_name": joint_names[j],
            "is_prismatic": is_prismatic,
            "unit": unit,
            "mean": float(np.mean(j_err)),
            "std": float(np.std(j_err))
        })

    return {
        "position_error": {
            "mean": float(np.mean(pos_errors)),
            "std": float(np.std(pos_errors))
        },
        "joint_errors": joint_stats
    }


def main():
    print("\n========================================================")
    print("  Task 4.A Evaluation: Analytical Forward Kinematics")
    print("========================================================\n")

    models_dir = os.path.join(PROJECT_ROOT, "models")
    datasets_dir = os.path.join(PROJECT_ROOT, "datasets")
    norm_dir = os.path.join(PROJECT_ROOT, "normalisation")
    logs_dir = os.path.join(PROJECT_ROOT, "logs")

    # Load Normalisation Configurations
    ned_q_min, ned_q_max = load_normalisation_config(os.path.join(norm_dir, "ned.json"))
    scara_q_min, scara_q_max = load_normalisation_config(os.path.join(norm_dir, "scara.json"))

    # Load Test Sets
    ned_test_pos, ned_test_joints = load_csv_dataset(os.path.join(datasets_dir, "ned", "test.csv"))
    scara_test_pos, scara_test_joints = load_csv_dataset(os.path.join(datasets_dir, "scara", "test.csv"))

    print(f"[Loaded Test Set Data]")
    print(f"  - Ned Test Samples:   {len(ned_test_pos)}")
    print(f"  - SCARA Test Samples: {len(scara_test_pos)}")

    scales = ["small", "medium", "large"]
    robots = [
        {"name": "Ned", "is_scara": False, "pos": ned_test_pos, "joints": ned_test_joints, "q_min": ned_q_min, "q_max": ned_q_max},
        {"name": "SCARA", "is_scara": True, "pos": scara_test_pos, "joints": scara_test_joints, "q_min": scara_q_min, "q_max": scara_q_max}
    ]

    all_results = {}

    for scale in scales:
        model_path = os.path.join(models_dir, f"model_{scale}.pkl")
        if not os.path.exists(model_path):
            print(f"[Error] Model file not found: {model_path}. Please run 'python train_ann.py' first.")
            return

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        all_results[scale] = {}

        for bot in robots:
            res = evaluate_model_on_robot(
                model=model,
                test_pos=bot["pos"],
                test_joints_norm=bot["joints"],
                is_scara=bot["is_scara"],
                q_min=bot["q_min"],
                q_max=bot["q_max"]
            )
            all_results[scale][bot["name"]] = res

    # ------------------------------------------------------------------
    # Display Formatted Summary Tables (Task 4.A Required Outputs)
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(" 6 SETS OF POSITION-ERROR STATISTICS (Metres)")
    print("=" * 72)
    print(f"{'Model Scale':<12} | {'Robot':<8} | {'Mean Position Error (m)':<25} | {'Std Dev (m)':<15}")
    print("-" * 72)

    for scale in scales:
        for bot_name in ["Ned", "SCARA"]:
            pos_stat = all_results[scale][bot_name]["position_error"]
            print(f"{scale.capitalize():<12} | {bot_name:<8} | {pos_stat['mean']:<25.6e} | {pos_stat['std']:<15.6e}")

    print("\n" + "=" * 85)
    print(" 18 SETS OF JOINT-ERROR STATISTICS")
    print("=" * 85)
    print(f"{'Model':<8} | {'Robot':<6} | {'Active Joint':<20} | {'Unit':<8} | {'Mean Joint Error':<18} | {'Std Dev':<15}")
    print("-" * 85)

    for scale in scales:
        for bot_name in ["Ned", "SCARA"]:
            joint_stats = all_results[scale][bot_name]["joint_errors"]
            for j_stat in joint_stats:
                print(
                    f"{scale.capitalize():<8} | {bot_name:<6} | {j_stat['joint_name']:<20} | "
                    f"{j_stat['unit']:<8} | {j_stat['mean']:<18.6e} | {j_stat['std']:<15.6e}"
                )

    # ------------------------------------------------------------------
    # Save Full Evaluation Metrics to JSON for Poster/Video Plotting
    # ------------------------------------------------------------------
    os.makedirs(logs_dir, exist_ok=True)
    out_json_path = os.path.join(logs_dir, "eval_task4a_results.json")
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)

    print(f"\n[Saved] Task 4.A evaluation metrics successfully saved to: {os.path.relpath(out_json_path, PROJECT_ROOT)}")
    print("========================================================\n")


if __name__ == "__main__":
    main()