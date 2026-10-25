"""
Task 4.B: evaluates the three trained networks in Webots.

Supervisor controller for the arm in the open world (worlds/articulated.wbt or
worlds/scara.wbt; the arm's controller has to be set to evaluate_webots). For the first
100 test samples of that arm and each model size it predicts the joints, commands the
motors, waits, and compares the TCP position measured by the supervisor with the
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


def get_official_tcp_node(supervisor):
    """
    Finds the TCP node with the lookup of the course tcp_pose.py: the global DEF "TCP"
    first, otherwise the first node in ARM.tcpSlot.

    Returns:
        Node: The TCP node, or None if not found.
    """
    # 1. Global DEF
    tcp = supervisor.getFromDef("TCP")
    if tcp is not None:
        return tcp

    # 2. First node of ARM.tcpSlot
    arm = supervisor.getFromDef("ARM")
    if arm is not None:
        tcp_slot = arm.getField("tcpSlot")
        if tcp_slot is not None and tcp_slot.getCount() > 0:
            return tcp_slot.getMFNode(0)

    return None


def detect_active_robot(supervisor):
    """
    Detects which arm the open world contains, from its motor names.

    Returns:
        str: "Ned" or "SCARA"
    """
    if supervisor.getDevice("joint_1") is not None:
        return "Ned"
    elif supervisor.getDevice("base_arm_motor") is not None:
        return "SCARA"
    else:
        raise RuntimeError("Could not find Ned or SCARA motors in the active Webots world.")


def denormalize_joints_batch(q_norm_batch, q_min, q_max):
    """
    De-normalises [-1.0, 1.0] back to rad / m; the input is clipped to [-1.0, 1.0] first.
    """
    q_norm_clamped = np.clip(q_norm_batch, -1.0, 1.0)
    return q_min + ((q_norm_clamped + 1.0) / 2.0) * (q_max - q_min)


def run_evaluation_for_active_robot(supervisor, robot_name, time_step):
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

    # 3. Motors of the three active joints
    if is_scara:
        m1 = supervisor.getDevice("base_arm_motor")
        m2 = supervisor.getDevice("arm_motor")
        m3 = supervisor.getDevice("shaft_linear_motor")
    else:
        m1 = supervisor.getDevice("joint_1")
        m2 = supervisor.getDevice("joint_2")
        m3 = supervisor.getDevice("joint_3")

    motors = [m1, m2, m3]

    # 4. TCP node
    tcp_node = get_official_tcp_node(supervisor)
    if tcp_node is None:
        raise RuntimeError(
            f"ERROR: Could not locate TCP node for {robot_name}! "
            f"Ensure either DEF 'TCP' or 'ARM.tcpSlot' exists in the world file."
        )

    print(f"[Webots 4.B] TCP Node successfully hooked for {robot_name}.")

    SETTLE_STEPS = 35  # simulation steps to wait for the arm to settle
    scales = ["small", "medium", "large"]
    robot_results = {}

    for scale in scales:
        model_path = os.path.join(models_dir, f"model_{scale}.pkl")
        if not os.path.exists(model_path):
            print(f"[Error] Model missing: {model_path}. Please run 'python train_ann.py' first.")
            return None

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Predict the normalised joints and de-normalise them
        X_test_100 = prepare_robot_features(test_pos_100, is_scara=is_scara)
        q_norm_pred_100 = model.predict(X_test_100)
        q_pred_phys_100 = denormalize_joints_batch(q_norm_pred_100, q_min, q_max)

        position_errors = []
        print(f"[Webots 4.B] Simulating {robot_name} on [{scale.upper()}] model...")

        for i in range(100):
            target_joints = q_pred_phys_100[i]
            true_pos = test_pos_100[i]

            # Command the motors
            for j in range(3):
                motors[j].setPosition(float(target_joints[j]))

            # Let the motors settle
            for _ in range(SETTLE_STEPS):
                if supervisor.step(time_step) == -1:
                    break

            # TCP position measured by the supervisor
            sim_pos = np.array(tcp_node.getPosition(), dtype=np.float64)

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
    time_step = int(supervisor.getBasicTimeStep())

    active_robot = detect_active_robot(supervisor)
    print(f"\n========================================================")
    print(f"  Starting Task 4.B Webots Simulation for: [{active_robot}]")
    print(f"========================================================\n")

    robot_results = run_evaluation_for_active_robot(supervisor, active_robot, time_step)

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