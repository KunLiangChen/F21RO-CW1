"""
Task 4.B: Webots Supervisor Evaluation Controller.

This supervisor controller automatically identifies the active robot in Webots 
('articulated.wbt' or 'scara.wbt'), retrieves the TCP node using the official 
two-stage lookup algorithm from 'tcp_pose.py', and calculates the REAL physical 
simulation position errors across Small, Medium, and Large models.

Coursework Reference:
    F20/21RO Intelligent Robotics - Coursework 1 Task 4.B
"""

import os
import sys
import json
import pickle
import numpy as np
from controller import Supervisor

# Ensure project root directory is accessible for relative imports
CONTROLLER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CONTROLLER_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

from scripts.dataset_loader import load_csv_dataset, prepare_robot_features


def get_official_tcp_node(supervisor):
    """
    Retrieves the TCP node using the exact logic from coursework 'tcp_pose.py'.
    
    1. First tries global DEF "TCP".
    2. If that fails, retrieves the first node inside ARM.tcpSlot.
    
    Returns:
        Node: Webots TCP Node object, or None if not found.
    """
    # 1. Try global DEF lookup
    tcp = supervisor.getFromDef("TCP")
    if tcp is not None:
        return tcp

    # 2. Try retrieving from ARM.tcpSlot
    arm = supervisor.getFromDef("ARM")
    if arm is not None:
        tcp_slot = arm.getField("tcpSlot")
        if tcp_slot is not None and tcp_slot.getCount() > 0:
            return tcp_slot.getMFNode(0)

    return None


def detect_active_robot(supervisor):
    """
    Detects which robot is present in the current active Webots world file.
    
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
    De-normalizes [-1.0, 1.0] bounds back to physical units with domain clamping.
    """
    q_norm_clamped = np.clip(q_norm_batch, -1.0, 1.0)
    return q_min + ((q_norm_clamped + 1.0) / 2.0) * (q_max - q_min)


def run_evaluation_for_active_robot(supervisor, robot_name, time_step):
    """
    Runs Task 4.B simulation for the active robot across Small, Medium, and Large models.
    """
    is_scara = (robot_name == "SCARA")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    datasets_dir = os.path.join(PROJECT_ROOT, "datasets")
    norm_dir = os.path.join(PROJECT_ROOT, "normalisation")

    # 1. Load Normalisation Config
    config_file = "scara.json" if is_scara else "ned.json"
    with open(os.path.join(norm_dir, config_file), 'r', encoding='utf-8') as f:
        norm_conf = json.load(f)
    q_min = np.array(norm_conf["q_min"], dtype=np.float64)
    q_max = np.array(norm_conf["q_max"], dtype=np.float64)

    # 2. Load First 100 Test Samples
    test_csv_path = os.path.join(datasets_dir, "scara" if is_scara else "ned", "test.csv")
    test_pos_all, _ = load_csv_dataset(test_csv_path)
    test_pos_100 = test_pos_all[:100]  # Strictly evaluate first 100 samples[cite: 13]

    # 3. Get Motor Devices
    if is_scara:
        m1 = supervisor.getDevice("base_arm_motor")
        m2 = supervisor.getDevice("arm_motor")
        m3 = supervisor.getDevice("shaft_linear_motor")
    else:
        m1 = supervisor.getDevice("joint_1")
        m2 = supervisor.getDevice("joint_2")
        m3 = supervisor.getDevice("joint_3")

    motors = [m1, m2, m3]

    # 4. Get TCP Node using Official Lookup
    tcp_node = get_official_tcp_node(supervisor)
    if tcp_node is None:
        raise RuntimeError(
            f"ERROR: Could not locate TCP node for {robot_name}! "
            f"Ensure either DEF 'TCP' or 'ARM.tcpSlot' exists in the world file."
        )

    print(f"[Webots 4.B] TCP Node successfully hooked for {robot_name}.")

    SETTLE_STEPS = 35  # Simulation steps to wait for mechanical settling[cite: 9]
    scales = ["small", "medium", "large"]
    robot_results = {}

    for scale in scales:
        model_path = os.path.join(models_dir, f"model_{scale}.pkl")
        if not os.path.exists(model_path):
            print(f"[Error] Model missing: {model_path}. Please run 'python train_ann.py' first.")
            return None

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Predict normalized joints and de-normalize to physical ranges
        X_test_100 = prepare_robot_features(test_pos_100, is_scara=is_scara)
        q_norm_pred_100 = model.predict(X_test_100)
        q_pred_phys_100 = denormalize_joints_batch(q_norm_pred_100, q_min, q_max)

        position_errors = []
        print(f"[Webots 4.B] Simulating {robot_name} on [{scale.upper()}] model...")

        for i in range(100):
            target_joints = q_pred_phys_100[i]
            true_pos = test_pos_100[i]

            # Command Motors
            for j in range(3):
                motors[j].setPosition(float(target_joints[j]))

            # Step simulation for motors to settle
            for _ in range(SETTLE_STEPS):
                if supervisor.step(time_step) == -1:
                    break

            # Read TRUE simulated TCP position from Webots
            sim_pos = np.array(tcp_node.getPosition(), dtype=np.float64)

            # Compute Euclidean Position Error ||(x,y,z)_true - (x,y,z)_sim||
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

        # Merge active robot results into JSON
        for scale, stats in robot_results.items():
            if scale not in existing_data:
                existing_data[scale] = {}
            existing_data[scale][active_robot] = stats

        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=4)

        print(f"\n[Updated] Real 4.B simulation errors for {active_robot} exported to: logs/eval_task4b_results.json")
        print("========================================================\n")

    # Automatically pause simulation upon completion
    supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)


if __name__ == "__main__":
    main()