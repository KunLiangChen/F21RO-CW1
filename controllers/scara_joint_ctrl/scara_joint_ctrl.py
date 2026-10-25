"""
Task 1.A: SCARA joint controller (Supervisor attached to DEF ARM in worlds/scara.wbt).

Active joints [q1, q2, q4] -> [base_arm_motor (rad), arm_motor (rad), shaft_linear_motor (m)].
shaft_rotation_motor is ignored in this coursework and held at 0.
Joint values come from logs/joint_input_scara.csv (rows of q1,q2,q4) or the demo poses below.
"""

import os
import sys
from controller import Supervisor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.append(PROJECT_ROOT)

from scripts.scara_fk import SCARA_JOINT_LIMITS
from scripts.webots_arm import JointProbe, run_joint_controller

ACTIVE_MOTORS = ["base_arm_motor", "arm_motor", "shaft_linear_motor"]
FIXED_MOTORS = {"shaft_rotation_motor": 0.0}

# [q1, q2, q4]; q2 >= 0 (elbow-up), q4 in [-0.2, 0] (0 = shaft fully up)
DEMO_POSES = [
    [0.0, 0.0, 0.0],
    [0.5, 0.4, -0.10],
    [-0.5, 0.8, -0.17],
]


def main():
    robot = Supervisor()
    probe = JointProbe(robot, ACTIVE_MOTORS, SCARA_JOINT_LIMITS, FIXED_MOTORS)
    run_joint_controller(probe, "scara", "q1,q2,q4", DEMO_POSES, PROJECT_ROOT)
    robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)


if __name__ == "__main__":
    main()
