"""
Task 4.A: evaluates the three trained networks on the test sets, using our FK.

No simulator needed. Per arm and model size:
    1. Feed the true TCP position (x, y, z) plus the robot flag into the network.
    2. Predict the normalised joint values and de-normalise them to rad / m.
    3. Apply FK to the predicted joints to get (x, y, z)_pred.
    4. Compute error statistics (mean and std) over the test set:
       - position error: Euclidean distance between (x, y, z)_pred and (x, y, z)_true
       - revolute joint error: absolute value of the wrapped angular difference, so errors
         of opposite sign do not cancel in the mean
       - prismatic joint error: squared difference (d_true - d_pred)^2

The position error is also given for the first 100 test samples, the ones Task 4.B
runs in Webots, so that both evaluations can be compared on the same inputs.

Results are printed and saved to logs/eval_task4a_results.json.
Run from the project root after train_ann.py: python scripts/evaluate_fk.py
"""

import os
import sys
import json
import pickle
import numpy as np

# Make the project root importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# FK from Task 1.B
from scripts.ned_fk import ned_forward_kinematics
from scripts.scara_fk import scara_forward_kinematics
from scripts.dataset_loader import load_csv_dataset, prepare_robot_features
from scripts.preprocess import denormalize_joints

# Task 4.B uses the first 100 test samples; the same subset is reported here
REDUCED_TEST_SIZE = 100


def load_normalisation_config(json_path):
    """
    Loads the joint bounds used for normalisation (written by preprocess.py).

    Expected format:
    {
        "robot": "ned",
        "q_min": [-2.8, -0.8, 0.0],
        "q_max": [2.8, 0.8, 1.3]
    }

    Returns:
        tuple: (q_min, q_max), each of shape (3,).
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Normalisation config file not found at: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    q_min = np.array(config["q_min"], dtype=np.float64)
    q_max = np.array(config["q_max"], dtype=np.float64)
    return q_min, q_max


def calculate_wrapped_angular_error(theta_true, theta_pred):
    """
    Computes the wrapped angular difference atan2(sin(d), cos(d)) with d = theta_true - theta_pred.

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
    Evaluates one trained model on the test set of one arm.

    Args:
        model: Trained scikit-learn MLPRegressor model (.pkl).
        test_pos (np.ndarray): Target TCP positions (N, 3).
        test_joints_norm (np.ndarray): True normalized joint values (N, 3).
        is_scara (bool): True for SCARA, False for Ned.
        q_min (np.ndarray): Lower normalisation bounds (3,).
        q_max (np.ndarray): Upper normalisation bounds (3,).

    Returns:
        dict: Mean and std of the position error and of each active joint error.
    """
    # 1. Network input [is_scara, is_ned, x, y, z], shape (N, 5)
    X_test = prepare_robot_features(test_pos, is_scara=is_scara)

    # 2. Predicted normalised joints, shape (N, 3)
    q_norm_pred = model.predict(X_test)

    # 3. De-normalise to rad / m
    q_true_phys = denormalize_joints(test_joints_norm, q_min, q_max)
    q_pred_phys = denormalize_joints(q_norm_pred, q_min, q_max)

    # 4. FK of the predicted joints gives the position the arm would reach
    fk_func = scara_forward_kinematics if is_scara else ned_forward_kinematics
    tcp_pred = fk_func(q_pred_phys)

    # 5. Metrics
    # A. Position error: Euclidean distance between true and reached position
    pos_errors = np.linalg.norm(test_pos - tcp_pred, axis=1)

    # B. Joint errors of the three active joints
    joint_names = (
        ["base_arm_motor", "arm_motor", "shaft_linear_motor"] if is_scara
        else ["joint_1", "joint_2", "joint_3"]
    )
    
    joint_stats = []
    for j in range(3):
        is_prismatic = (is_scara and j == 2)  # SCARA's third active joint is the linear shaft
        
        if is_prismatic:
            # Prismatic joint: squared difference
            j_err = (q_true_phys[:, j] - q_pred_phys[:, j]) ** 2
            unit = "m^2"
        else:
            # Revolute joint: absolute wrapped angular difference
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
        "position_error_first_100": {
            "mean": float(np.mean(pos_errors[:REDUCED_TEST_SIZE])),
            "std": float(np.std(pos_errors[:REDUCED_TEST_SIZE]))
        },
        "joint_errors": joint_stats
    }


def main():
    """
    Evaluates all three models on both test sets and saves the statistics.
    """
    print("\n========================================================")
    print("  Task 4.A Evaluation: Analytical Forward Kinematics")
    print("========================================================\n")

    models_dir = os.path.join(PROJECT_ROOT, "models")
    datasets_dir = os.path.join(PROJECT_ROOT, "datasets")
    norm_dir = os.path.join(PROJECT_ROOT, "normalisation")
    logs_dir = os.path.join(PROJECT_ROOT, "logs")

    # Normalisation bounds
    ned_q_min, ned_q_max = load_normalisation_config(os.path.join(norm_dir, "ned.json"))
    scara_q_min, scara_q_max = load_normalisation_config(os.path.join(norm_dir, "scara.json"))

    # Test sets
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
            print(f"[Error] Model file not found: {model_path}. Please run 'python scripts/train_ann.py' first.")
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

    # Summary tables (3 models x 2 arms for position, x 3 joints for joint errors)
    print("\n" + "=" * 72)
    print(" 6 SETS OF POSITION-ERROR STATISTICS (Metres)")
    print("=" * 72)
    print(f"{'Model Scale':<12} | {'Robot':<8} | {'Mean Position Error (m)':<25} | {'Std Dev (m)':<15}")
    print("-" * 72)

    for scale in scales:
        for bot_name in ["Ned", "SCARA"]:
            pos_stat = all_results[scale][bot_name]["position_error"]
            print(f"{scale.capitalize():<12} | {bot_name:<8} | {pos_stat['mean']:<25.6e} | {pos_stat['std']:<15.6e}")

    print("\n" + "=" * 72)
    print(f" POSITION ERROR ON THE FIRST {REDUCED_TEST_SIZE} TEST SAMPLES (the samples of Task 4.B, metres)")
    print("=" * 72)
    print(f"{'Model Scale':<12} | {'Robot':<8} | {'Mean Position Error (m)':<25} | {'Std Dev (m)':<15}")
    print("-" * 72)

    for scale in scales:
        for bot_name in ["Ned", "SCARA"]:
            pos_stat = all_results[scale][bot_name]["position_error_first_100"]
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

    # Save everything for the poster and video plots
    os.makedirs(logs_dir, exist_ok=True)
    out_json_path = os.path.join(logs_dir, "eval_task4a_results.json")
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)

    print(f"\n[Saved] Task 4.A evaluation metrics successfully saved to: {os.path.relpath(out_json_path, PROJECT_ROOT)}")
    print("========================================================\n")


if __name__ == "__main__":
    main()