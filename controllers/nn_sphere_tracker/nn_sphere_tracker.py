"""
Task 5 demo: the trained network moves the arm to the red target sphere.

Supervisor controller for worlds/articulated.wbt (Ned) and worlds/scara.wbt (SCARA);
the arm's controller has to be set to nn_sphere_tracker. In every simulation step:
    1. Read the TARGET sphere position in the ARM frame.
    2. Build the network input [is_scara, is_ned, x, y, z].
    3. Predict the normalised joints with the model selected by MODEL_SCALE.
    4. De-normalise (clipped to the bounds) and check the prediction with our FK: if the
       predicted joints do not reach the target within REACH_TOLERANCE, the target is outside
       what the network learned (e.g. out of reach); the arm then keeps its last good pose
       instead of following a wild extrapolation.
    5. Otherwise command the motors.
    6. Show the target, the tracking error and the reachability in the 3D view (for the video)
       and periodically print the measured TCP position and tracking error.
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

from scripts.ned_fk import ned_forward_kinematics
from scripts.preprocess import denormalize_joints
from scripts.scara_fk import scara_forward_kinematics
from scripts.webots_arm import detect_arm, make_probe, position_in_arm_frame

# ============================================================
# Webots DEF names and Configurations
# ============================================================
ARM_DEF_NAME = "ARM"
TARGET_DEF_NAME = "TARGET"
MODEL_SCALE = "large"  # Options: 'small', 'medium', 'large'
REACH_TOLERANCE = 0.04  # [m] max distance between FK of the prediction and the target


def show_overlay(robot, robot_type, target, error_m, reachable, reach_error):
    """
    Writes the demo status as text into the Webots 3D view (setLabel, positions are 0 to 1
    from the top-left corner; colors are 0xRRGGBB).
    """
    status = "target reachable" if reachable else "target NOT reachable: holding the last pose"
    status_color = 0x008000 if reachable else 0xC00000
    robot.setLabel(0, f"{robot_type} - {MODEL_SCALE} model", 0.02, 0.02, 0.07, 0x000000, 0.0, "Arial")
    robot.setLabel(1, f"target  x={target[0]: .3f}  y={target[1]: .3f}  z={target[2]: .3f} m",
                   0.02, 0.07, 0.06, 0x000000, 0.0, "Arial")
    robot.setLabel(2, f"tracking error: {error_m * 1000.0:.1f} mm", 0.02, 0.12, 0.06, 0x000000, 0.0, "Arial")
    robot.setLabel(3, f"{status} (FK check: {reach_error * 1000.0:.0f} mm)", 0.02, 0.17, 0.06,
                   status_color, 0.0, "Arial")


def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())

    # 1. Detect Active Robot
    robot_type = detect_arm(robot)
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
        print(f"ERROR: Model file missing at {model_path}. Please run scripts/train_ann.py first.")
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

    if arm_node is None:
        print(f"ERROR: Could not find DEF {ARM_DEF_NAME}")
        return
    if target_node is None:
        print(f"ERROR: Could not find DEF {TARGET_DEF_NAME}")
        return

    # 4. Motors, fixed joints and TCP of the arm (shared with the Task 1.A and 4.B controllers)
    probe = make_probe(robot, robot_type)
    for motor in probe.motors:
        motor.setVelocity(1.0)  # Smooth tracking speed
    forward_kinematics = scara_forward_kinematics if is_scara else ned_forward_kinematics

    # 5. Timing Variables
    last_print_time = -999.0
    print_period = 0.5  # Print debug info every 0.5 seconds
    
    flag = [1.0, 0.0] if is_scara else [0.0, 1.0]

    # 6. Real-Time Tracking Control Loop
    while robot.step(timestep) != -1:
        now = robot.getTime()

        # A. Read TARGET position relative to ARM frame
        tx, ty, tz = position_in_arm_frame(target_node, arm_node)

        # B. Prepare ANN feature vector: [is_scara, is_ned, x, y, z]
        X_input = np.array([flag + [tx, ty, tz]], dtype=np.float32)

        # C. ANN Model Inference
        q_norm_pred = model.predict(X_input)[0]

        # D. De-normalize & Clamp to physical limits
        q_pred_phys = denormalize_joints(q_norm_pred, q_min, q_max)

        # E. Only follow predictions that FK confirms; otherwise keep the last good pose
        reach_error = np.linalg.norm(forward_kinematics(q_pred_phys) - np.array([tx, ty, tz]))
        reachable = reach_error <= REACH_TOLERANCE
        if reachable:
            for motor, q in zip(probe.motors, q_pred_phys):
                motor.setPosition(float(q))

        # F. Measured TCP and tracking error, shown in the 3D view
        tcpx, tcpy, tcpz = probe.measure_tcp()
        error_m = math.sqrt((tcpx - tx)**2 + (tcpy - ty)**2 + (tcpz - tz)**2)
        show_overlay(robot, robot_type, (tx, ty, tz), error_m, reachable, reach_error)

        # G. Print Debug and Real-Time Error Log (Ideal for Video Recording)
        if now - last_print_time >= print_period:
            last_print_time = now

            print("\n------------------------------------------------")
            print(f"[{robot_type.upper()} Real-Time ANN Tracker]")
            print(f"Target Pose (ARM Frame) : x={tx: .4f}, y={ty: .4f}, z={tz: .4f}")
            print(f"Predicted Joints (Phys) : q1={q_pred_phys[0]: .4f}, q2={q_pred_phys[1]: .4f}, q3={q_pred_phys[2]: .4f}")
            print(f"Target reachable        : {'yes' if reachable else 'NO, holding the last pose'} "
                  f"(FK of the prediction is {reach_error * 1000.0:.1f} mm away)")

            print(f"Measured TCP Position   : x={tcpx: .4f}, y={tcpy: .4f}, z={tcpz: .4f}")
            print(f"Real-Time Tracking Error: {error_m * 1000.0: .2f} mm")
            print("------------------------------------------------")


if __name__ == "__main__":
    main()