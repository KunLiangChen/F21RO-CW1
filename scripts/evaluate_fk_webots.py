"""
Task 1.C: precision of our forward kinematics (Task 1.B) against Webots.

For the SAME joint values we compare
    p_fk     = analytical FK (scripts/scara_fk.py, scripts/ned_fk.py)
    p_webots = TCP position measured in Webots by the Task 1.A controllers.
No neural network is involved, so the error is purely FK model error + settling error.

Usage (run from the project root):
    python scripts/evaluate_fk_webots.py generate   # writes logs/joint_input_{scara,ned}.csv
    # then run worlds/scara.wbt and worlds/articulated.wbt in Webots; the controllers
    # (scara_joint_ctrl / ned_joint_ctrl) write logs/tcp_measured_{scara,ned}.csv
    python scripts/evaluate_fk_webots.py report     # prints statistics, saves logs/fk_vs_webots.json
"""

import os
import sys
import json
import numpy as np

# Make the project root importable; the controllers write logs/ there
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from scripts.gen_sample import generate_scara_samples, generate_ned_samples, NED_MIN_TCP_Z
from scripts.scara_fk import scara_forward_kinematics
from scripts.ned_fk import ned_forward_kinematics

LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
NUM_RANDOM_SAMPLES = 200
RANDOM_SEED = 2026   # fixed so every group member generates the same joint sets

# Corners of the sampled joint box (same ranges as the training data): worst cases for FK.
SCARA_CORNERS = [(q1, q2, q4) for q1 in (-0.73, 0.73) for q2 in (0.0, 0.83) for q4 in (-0.17, 0.0)]
# Ned corners whose TCP would be below the floor limit of the dataset (NED_MIN_TCP_Z) are left out.
NED_CORNERS = [(q1, q2, q3) for q1 in (-2.8, 2.8) for q2 in (-0.8, 0.8) for q3 in (0.0, 1.3)
               if ned_forward_kinematics([q1, q2, q3])[2] >= NED_MIN_TCP_Z]

# Ned only: below the dataset's floor limit the wrist/hand touches the Webots floor and is pushed
# out of its kinematic pose (an earlier run without this limit: max error 7.8 mm, 0.7 mm with
# the floor removed). It is a physics-contact artefact, not an FK error. The poses generated
# here respect the limit, so this split only guards against the measured TCP being lower.
NED_FLOOR_CONTACT_Z = NED_MIN_TCP_Z

ARMS = {
    "scara": {"sampler": generate_scara_samples, "fk": scara_forward_kinematics,
              "corners": SCARA_CORNERS, "header": "q1,q2,q4", "floor_contact_z": None},
    "ned": {"sampler": generate_ned_samples, "fk": ned_forward_kinematics,
            "corners": NED_CORNERS, "header": "q1,q2,q3", "floor_contact_z": NED_FLOOR_CONTACT_Z},
}


def generate_joint_inputs():
    """Corner poses first, then uniformly random poses sampled exactly like the dataset."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    for name, arm in ARMS.items():
        random_joints = arm["sampler"](NUM_RANDOM_SAMPLES, RANDOM_SEED)
        joints = np.vstack([np.array(arm["corners"]), random_joints])
        path = os.path.join(LOGS_DIR, f"joint_input_{name}.csv")
        np.savetxt(path, joints, delimiter=",", header=arm["header"], comments="")
        print(f"[{name}] {len(joints)} joint sets written to {path}")


def summarise(errors):
    return {"mean": float(np.mean(errors)), "std": float(np.std(errors)),
            "median": float(np.median(errors)), "max": float(np.max(errors))}


def report():
    results = {}
    for name, arm in ARMS.items():
        path = os.path.join(LOGS_DIR, f"tcp_measured_{name}.csv")
        if not os.path.exists(path):
            sys.exit(f"Missing {path}: run the {name} world in Webots first.")
        data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        joints, measured = data[:, :3], data[:, 3:6]

        fk_pos = arm["fk"](joints)
        diff = measured - fk_pos                     # Webots - FK, per axis [m]
        position_error = np.linalg.norm(diff, axis=1)  # Euclidean distance [m]

        results[name] = {
            "samples": len(joints),
            "position_error_m": summarise(position_error),
            "axis_bias_m": dict(zip("xyz", diff.mean(axis=0).tolist())),  # systematic offset
            "worst_joint_values": joints[np.argmax(position_error)].tolist(),
        }
        if arm["floor_contact_z"] is not None:
            free = measured[:, 2] >= arm["floor_contact_z"]  # TCP clear of the floor
            results[name]["position_error_m_without_floor_contact"] = summarise(position_error[free])
            results[name]["floor_contact_samples"] = int((~free).sum())
        stats = results[name]["position_error_m"]
        print(f"\n[{name.upper()}] {len(joints)} samples, position error Webots vs FK [mm]:")
        print(f"  mean={1000 * stats['mean']:.3f}  std={1000 * stats['std']:.3f}  "
              f"median={1000 * stats['median']:.3f}  max={1000 * stats['max']:.3f}")
        print("  mean signed error per axis [mm]: "
              + ", ".join(f"{k}={1000 * v:+.3f}" for k, v in results[name]["axis_bias_m"].items()))
        if "floor_contact_samples" in results[name]:
            clear = results[name]["position_error_m_without_floor_contact"]
            print(f"  excluding {results[name]['floor_contact_samples']} floor-contact poses (TCP z < "
                  f"{arm['floor_contact_z']} m): mean={1000 * clear['mean']:.3f}  "
                  f"std={1000 * clear['std']:.3f}  max={1000 * clear['max']:.3f}")
        print(f"  worst case at joints {np.round(results[name]['worst_joint_values'], 3).tolist()}")

    out_path = os.path.join(LOGS_DIR, "fk_vs_webots.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("generate", "report"):
        sys.exit(__doc__)
    generate_joint_inputs() if sys.argv[1] == "generate" else report()
