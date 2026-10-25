"""
Task 1.A: Ned joint controller (Supervisor attached to DEF ARM in worlds/articulated.wbt).

Active joints [q1, q2, q3] -> [joint_1, joint_2, joint_3] (rad).
Wrist joints joint_4..joint_6 are fixed at 0.
Joint values come from logs/joint_input_ned.csv (rows of q1,q2,q3) or the demo poses below.
"""

import os
import sys
from controller import Supervisor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.append(PROJECT_ROOT)

from scripts.webots_arm import make_probe, run_joint_controller


# [q1, q2, q3]; q3 >= 0 (elbow-up)
DEMO_POSES = [
    [0.0, 0.0, 0.0],
    [-0.7854, 0.3504, 0.3554],
    [1.5, -0.5, 1.0],
]


def main():
    robot = Supervisor()
    probe = make_probe(robot, "Ned")
    run_joint_controller(probe, "ned", "q1,q2,q3", DEMO_POSES, PROJECT_ROOT)
    robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)


if __name__ == "__main__":
    main()
