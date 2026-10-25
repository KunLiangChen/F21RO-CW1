"""
Task 5 demo: the trained network moves the arm to the red target sphere.

Supervisor controller for worlds/articulated.wbt (Ned) and worlds/scara.wbt (SCARA);
the arm's controller has to be set to nn_sphere_tracker. In every simulation step:
    1. Read the TARGET sphere position in the ARM frame.
    2. Build the network input [is_scara, is_ned, x, y, z].
    3. Predict the normalised joints with the model selected by MODEL_SCALE.
    4. De-normalise (clipped to the bounds) and command the motors.
    5. Periodically print the measured TCP position and tracking error.
"""

import os
import sys
import json
import math
import pickle
import numpy as np
from controller import Supervisor

# Make the project root importable
CONTROLLER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CONTROLLER_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

# ============================================================
# Webots DEF names and Configurations
# ============================================================
ARM_DEF_NAME = "ARM"
TARGET_DEF_NAME = "TARGET"
MODEL_SCALE = "large"  # Options: 'small', 'medium', 'large'


def get_official_tcp_node(supervisor):
    """
    Retrieves the TCP node using the official lookup logic.
    """
    tcp = supervisor.getFromDef("TCP")
    if tcp is not None:
        return tcp

    arm = supervisor.getFromDef("ARM")
    if arm is not None:
        tcp_slot = arm.getField("tcpSlot")
        if tcp_slot is not None and tcp_slot.getCount() > 0:
            return tcp_slot.getMFNode(0)

    return None


def detect_active_robot(supervisor):
    """
    Detects active robot based on device availability.
    """
    if supervisor.getDevice("joint_1") is not None:
        return "Ned"
    elif supervisor.getDevice("base_arm_motor") is not None:
        return "SCARA"
    else:
        raise RuntimeError("Could not find Ned or SCARA motors in the active Webots world.")


def get_xyz_relative_to_arm(node, arm_node):
    """
    Returns 3D translation (x, y, z) of a node relative to the ARM frame.
    """
    pose = node.getPose(arm_node)
    return pose[3], pose[7], pose[11]


def denormalize_joints(q_norm, q_min, q_max):
    """
    De-normalizes [-1.0, 1.0] predictions back to physical ranges:
    q = q_min + ((q_norm + 1) / 2) * (q_max - q_min)
    """
    q_clamped = np.clip(q_norm, -1.0, 1.0)
    return q_min + ((q_clamped + 1.0) / 2.0) * (q_max - q_min)


def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())

    # 1. Detect Active Robot
    robot_type = detect_active_robot(robot)
    is_scara = (robot_type == "SCARA")

    print("\n========================================================")
    print(f"  ANN Real-Time Target Tracker Initialized [{robot_type.upper()}]")
    print(f"  Loaded Model Scale: [{MODEL_SCALE.upper()}]")
    print("========================================================\n")

    # 2. Load Model and Normalisation Config
    models_dir = os.path.join(PROJECT_ROOT, "models")
    norm_dir = os.path.join(PROJECT_ROOT, "normalisation")

    model_path = os.path.join(models_dir, f"model_{MODEL_SCALE}.pkl")
    config_file = "scara.json" if is_scara else "ned.json"
    norm_path = os.path.join(norm_dir, config_file)

    if not os.path.exists(model_path):
        print(f"ERROR: Model file missing at {model_path}. Please run train_ann.py first.")
        return

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(norm_path, 'r', encoding='utf-8') as f:
        norm_conf = json.load(f)

    q_min = np.array(norm_conf["q_min"], dtype=np.float64)
    q_max = np.array(norm_conf["q_max"], dtype=np.float64)

    # 3. Get Webots Nodes
    arm_node = robot.getFromDef(ARM_DEF_NAME)
    target_node = robot.getFromDef(TARGET_DEF_NAME)
    tcp_node = get_official_tcp_node(robot)

    if arm_node is None:
        print(f"ERROR: Could not find DEF {ARM_DEF_NAME}")
        return
    if target_node is None:
        print(f"ERROR: Could not find DEF {TARGET_DEF_NAME}")
        return

    # 4. Get Motors and Set Velocities
    if is_scara:
        motor_names = ["base_arm_motor", "arm_motor", "shaft_linear_motor"]
    else:
        motor_names = ["joint_1", "joint_2", "joint_3"]

    motors = [robot.getDevice(name) for name in motor_names]
    for m in motors:
        if m is not None:
            m.setVelocity(1.0)  # Smooth tracking speed

    # Optional Wrist Motors (Keep fixed for Ned)
    wrist_motors = [robot.getDevice(f"joint_{i}") for i in [4, 5, 6]]
    for wm in wrist_motors:
        if wm is not None:
            wm.setVelocity(1.0)
            wm.setPosition(0.0)

    # 5. Timing Variables
    last_print_time = -999.0
    print_period = 0.5  # Print debug info every 0.5 seconds
    
    flag = [1.0, 0.0] if is_scara else [0.0, 1.0]

    # 6. Real-Time Tracking Control Loop
    while robot.step(timestep) != -1:
        now = robot.getTime()

        # A. Read TARGET position relative to ARM frame
        tx, ty, tz = get_xyz_relative_to_arm(target_node, arm_node)

        # B. Prepare ANN feature vector: [is_scara, is_ned, x, y, z]
        X_input = np.array([flag + [tx, ty, tz]], dtype=np.float32)

        # C. ANN Model Inference
        q_norm_pred = model.predict(X_input)[0]

        # D. De-normalize & Clamp to physical limits
        q_pred_phys = denormalize_joints(q_norm_pred, q_min, q_max)

        # E. Send Joint Commands to Motors
        for j in range(3):
            if motors[j] is not None:
                motors[j].setPosition(float(q_pred_phys[j]))

        # Keep wrist motors fixed
        for wm in wrist_motors:
            if wm is not None:
                wm.setPosition(0.0)

        # F. Print Debug and Real-Time Error Log (Ideal for Video Recording)
        if now - last_print_time >= print_period:
            last_print_time = now

            print("\n------------------------------------------------")
            print(f"[{robot_type.upper()} Real-Time ANN Tracker]")
            print(f"Target Pose (ARM Frame) : x={tx: .4f}, y={ty: .4f}, z={tz: .4f}")
            print(f"Predicted Joints (Phys) : q1={q_pred_phys[0]: .4f}, q2={q_pred_phys[1]: .4f}, q3={q_pred_phys[2]: .4f}")

            if tcp_node is not None:
                tcpx, tcpy, tcpz = get_xyz_relative_to_arm(tcp_node, arm_node)
                error_m = math.sqrt((tcpx - tx)**2 + (tcpy - ty)**2 + (tcpz - tz)**2)
                print(f"Measured TCP Position   : x={tcpx: .4f}, y={tcpy: .4f}, z={tcpz: .4f}")
                print(f"Real-Time Tracking Error: {error_m * 1000.0: .2f} mm")
            print("------------------------------------------------")


if __name__ == "__main__":
    main()