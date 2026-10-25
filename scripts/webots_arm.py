"""
Task 1.A: shared Webots helper for the SCARA and Ned joint controllers.

Both controllers run as a Supervisor attached to the arm itself (DEF ARM), so the
same process can command the motors AND read the measured TCP pose.

Workflow of JointProbe.move_to(q):
    1. clip the 3 active joint values to the physical limits,
    2. command the 3 active motors,
    3. step the simulation until every position sensor has reached its target
       (and stayed there for a few steps) or a timeout expires,
    4. return the TCP position (x, y, z) measured by the supervisor, expressed in
       the ARM frame so it is directly comparable with our FK scripts.

This module does not import the Webots `controller` package itself; it only uses
the Supervisor object it is given, so it can be unit-tested with a mock.
"""

import os
import numpy as np

SETTLE_TOLERANCE = 1e-3   # [rad] or [m]: max joint error that counts as "arrived"
SETTLE_HOLD_STEPS = 5     # consecutive steps inside the tolerance before we trust it
SETTLE_TIMEOUT_S = 10.0   # simulated seconds before giving up on a single move


def find_tcp_node(robot):
    """Same two-stage lookup as the course tcp_pose.py: global DEF TCP, else ARM.tcpSlot."""
    tcp = robot.getFromDef("TCP")
    if tcp is not None:
        return tcp

    arm = robot.getFromDef("ARM")
    tcp_slot = arm.getField("tcpSlot") if arm is not None else None
    if tcp_slot is not None and tcp_slot.getCount() > 0:
        return tcp_slot.getMFNode(0)

    raise RuntimeError("Could not find the TCP node (DEF TCP or ARM.tcpSlot).")


class JointProbe:
    """Commands the 3 active joints of one arm and measures the resulting TCP."""

    def __init__(self, robot, active_motors, limits, fixed_motors=None):
        """
        Args:
            robot: Webots Supervisor attached to the arm.
            active_motors: 3 motor names in the order [q1, q2, q3 (or q4 for SCARA)].
            limits: dict of (min, max) per active joint, in the same order, e.g. the
                    SCARA_JOINT_LIMITS / NED_JOINT_LIMITS from the FK scripts.
            fixed_motors: {motor name: constant position} for joints we do not use.
        """
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep())
        self.limits = np.array(list(limits.values()), dtype=np.float64)  # shape (3, 2)

        self.motors = [robot.getDevice(name) for name in active_motors]
        self.sensors = []
        for motor in self.motors:
            sensor = motor.getPositionSensor()
            sensor.enable(self.timestep)
            self.sensors.append(sensor)

        for name, value in (fixed_motors or {}).items():
            robot.getDevice(name).setPosition(value)

        self.arm = robot.getFromDef("ARM")
        self.tcp = find_tcp_node(robot)

        # Position sensors only return valid values after the first simulation step.
        self._step()

    def _step(self):
        if self.robot.step(self.timestep) == -1:
            raise SystemExit("Simulation terminated.")

    def move_to(self, q):
        """Moves to the active joint vector q (3,), waits to settle, returns TCP (x, y, z)."""
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (3,):
            raise ValueError(f"Expected 3 active joint values, got shape {q.shape}")

        target = np.clip(q, self.limits[:, 0], self.limits[:, 1])
        if not np.array_equal(target, q):
            print(f"[warn] q={q.tolist()} outside joint limits, clipped to {target.tolist()}")

        for motor, value in zip(self.motors, target):
            motor.setPosition(float(value))

        self._wait_until_settled(target)
        return self.measure_tcp()

    def _wait_until_settled(self, target):
        deadline = self.robot.getTime() + SETTLE_TIMEOUT_S
        steps_in_tolerance = 0
        while steps_in_tolerance < SETTLE_HOLD_STEPS:
            self._step()
            measured = np.array([s.getValue() for s in self.sensors])
            if np.max(np.abs(measured - target)) < SETTLE_TOLERANCE:
                steps_in_tolerance += 1
            else:
                steps_in_tolerance = 0
            if self.robot.getTime() > deadline:
                print(f"[warn] arm did not settle within {SETTLE_TIMEOUT_S}s "
                      f"(target={target.tolist()}, measured={measured.tolist()})")
                return

    def measure_tcp(self):
        """TCP position in the ARM frame: translation part of the 4x4 pose of TCP relative to ARM."""
        pose = self.tcp.getPose(self.arm)
        return np.array([pose[3], pose[7], pose[11]])


def load_joint_csv(path):
    """Loads rows of 3 active joint values; a text header line is skipped if present."""
    try:
        data = np.loadtxt(path, delimiter=",", ndmin=2)
    except ValueError:
        data = np.loadtxt(path, delimiter=",", ndmin=2, skiprows=1)
    return data[:, :3]


def run_joint_controller(probe, name, joint_header, demo_poses, project_root):
    """
    Entry point shared by both controllers.

    If logs/joint_input_<name>.csv exists, every row (3 active joint values, in
    physical units) is executed; otherwise the built-in demo poses are used.
    Results are written to logs/tcp_measured_<name>.csv as [joints..., x, y, z].
    """
    logs_dir = os.path.join(project_root, "logs")
    input_path = os.path.join(logs_dir, f"joint_input_{name}.csv")
    output_path = os.path.join(logs_dir, f"tcp_measured_{name}.csv")

    if os.path.exists(input_path):
        joints = load_joint_csv(input_path)
        print(f"[{name}] Running {len(joints)} joint sets from {input_path}")
    else:
        joints = np.array(demo_poses, dtype=np.float64)
        print(f"[{name}] {input_path} not found, running {len(joints)} demo poses")

    rows = []
    for q in joints:
        tcp = probe.move_to(q)
        print(f"[{name}] q={np.round(q, 4).tolist()} -> TCP={np.round(tcp, 4).tolist()}")
        rows.append(np.concatenate([q, tcp]))

    os.makedirs(logs_dir, exist_ok=True)
    np.savetxt(output_path, np.array(rows), delimiter=",", header=joint_header + ",x,y,z", comments="")
    print(f"[{name}] Measured TCP positions saved to {output_path}")
