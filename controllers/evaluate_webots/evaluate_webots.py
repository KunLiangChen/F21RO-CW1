"""
Task 4.B: evaluates the three trained networks in Webots.

Supervisor controller for the arm in the open world (worlds/articulated.wbt or
worlds/scara.wbt; the arm's controller has to be set to evaluate_webots). For the first
100 test samples of that arm and each model size it predicts the joints, commands the
motors, waits until the joints have settled (JointProbe in scripts/webots_arm.py), and
compares the TCP position measured by the supervisor, in the ARM frame, with the
test-set position.

Mean and std of the position error are merged into logs/eval_task4b_results.json,
so run it once per world.
"""

import os
import sys
import json
import pickle
import numpy as np
from controller import Supervisor

# Make the project root importable
CONTROLLER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CONTROLLER_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

from scripts.dataset_loader import load_csv_dataset, prepare_robot_features
from scripts.preprocess import denormalize_joints
from scripts.webots_arm import detect_arm, make_probe


def run_evaluation_for_active_robot(supervisor, robot_name):
    """
    Runs the Task 4.B simulation for the arm in the open world, for all three models.

    Returns:
        dict: {size: {"mean": ..., "std": ...}} of the position error in metres,
              or None if a model file is missing.
    """
    is_scara = (robot_name == "SCARA")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    datasets_dir = os.path.join(PROJECT_ROOT, "datasets")
    norm_dir = os.path.join(PROJECT_ROOT, "normalisation")

    # 1. Normalisation bounds
    config_file = "scara.json" if is_scara else "ned.json"
    with open(os.path.join(norm_dir, config_file), 'r', encoding='utf-8') as f:
        norm_conf = json.load(f)
    q_min = np.array(norm_conf["q_min"], dtype=np.float64)
    q_max = np.array(norm_conf["q_max"], dtype=np.float64)

    # 2. First 100 test samples (the coursework reduced test set)
    test_csv_path = os.path.join(datasets_dir, "scara" if is_scara else "ned", "test.csv")
    test_pos_all, _ = load_csv_dataset(test_csv_path)
    test_pos_100 = test_pos_all[:100]

    # 3. Commands the joints and measures the TCP (also used by the Task 1.A controllers)
    probe = make_probe(supervisor, robot_name)
    print(f"[Webots 4.B] Evaluating {robot_name}.")

    scales = ["small", "medium", "large"]
    robot_results = {}

    for scale in scales:
        model_path = os.path.join(models_dir, f"model_{scale}.pkl")
        if not os.path.exists(model_path):
            print(f"[Error] Model missing: {model_path}. Please run 'python scripts/train_ann.py' first.")
            return None

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Predict the normalised joints and de-normalise them
        X_test_100 = prepare_robot_features(test_pos_100, is_scara=is_scara)
        q_norm_pred_100 = model.predict(X_test_100)
        q_pred_phys_100 = denormalize_joints(q_norm_pred_100, q_min, q_max)

        position_errors = []
        print(f"[Webots 4.B] Simulating {robot_name} on [{scale.upper()}] model...")

        for i in range(100):
            true_pos = test_pos_100[i]

            # Command the joints, wait until they have settled and measure the TCP
            sim_pos = probe.move_to(q_pred_phys_100[i])

            # Position error: Euclidean distance to the test-set position
            pos_err = np.linalg.norm(true_pos - sim_pos)
            position_errors.append(pos_err)

        robot_results[scale] = {
            "mean": float(np.mean(position_errors)),
            "std": float(np.std(position_errors))
        }

    return robot_results


def main():
    supervisor = Supervisor()

    active_robot = detect_arm(supervisor)
    print(f"\n========================================================")
    print(f"  Starting Task 4.B Webots Simulation for: [{active_robot}]")
    print(f"========================================================\n")

    robot_results = run_evaluation_for_active_robot(supervisor, active_robot)

    if robot_results is not None:
        logs_dir = os.path.join(PROJECT_ROOT, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        out_json_path = os.path.join(logs_dir, "eval_task4b_results.json")

        existing_data = {}
        if os.path.exists(out_json_path):
            with open(out_json_path, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = {}

        # Merge this arm's results into the JSON file
        for scale, stats in robot_results.items():
            if scale not in existing_data:
                existing_data[scale] = {}
            existing_data[scale][active_robot] = stats

        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=4)

        print(f"\n[Updated] Real 4.B simulation errors for {active_robot} exported to: logs/eval_task4b_results.json")
        print("========================================================\n")

    # Pause the simulation when finished
    supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)


if __name__ == "__main__":
    main()