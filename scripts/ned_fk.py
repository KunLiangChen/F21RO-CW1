"""
Niryo Ned Robot Forward Kinematics (FK) Module.

This script calculates the analytical forward kinematics for the Niryo Ned robot
arm as specified in Coursework_1_2627.pdf (Task 1.B).

Active Joint Vector Input:
    q = [q1, q2, q3]
    - q1 (joint_1): waist/base yaw [rad], range: [-2.8, 2.8]
    - q2 (joint_2): shoulder pitch [rad], range: [-0.8, 0.8]
    - q3 (joint_3): elbow pitch [rad], range: [-1.5, 1.3] (sampled as [0.0, 1.3] for elbow-up)
    Note: Wrist joints (joint_4, joint_5, joint_6) are fixed.

Special Rule:
    Waist 0 degrees points along the positive y-axis.

Output:
    p = [x, y, z]: 3D coordinates of TCP in meters.
"""

from typing import Union, List, Tuple
import numpy as np


# Ned Active Joint Physical Limits (Min, Max) in radians
NED_JOINT_LIMITS = {
    'q1': (-2.8, 2.8),
    'q2': (-0.8, 0.8),
    'q3': (-1.5, 1.3)  # Note: restricted to [0.0, 1.3] for elbow-up dataset generation
}

# Geometric Constant Offsets (Meters)
H = 0.1065 + 0.0650   # Base to shoulder height = 0.1715m
A, B = 0.2210, 0.0120  # Shoulder to elbow vector
U, V = 0.03185, 0.23000 # Elbow to TCP vector


def ned_forward_kinematics(
    q: Union[List[float], Tuple[float, ...], np.ndarray],
    validate_limits: bool = False
) -> np.ndarray:
    """
    Computes Forward Kinematics for the Niryo Ned robot arm.

    Supports both single joint vectors of shape (3,) and batch arrays of shape (N, 3).

    Args:
        q: Active joint vector [q1, q2, q3] or batch array of shape (N, 3).
        validate_limits: If True, checks whether joint inputs stay within physical limits.

    Returns:
        np.ndarray: Calculated 3D TCP position [x, y, z] or batch array of shape (N, 3).
    """
    q_arr = np.asarray(q, dtype=np.float64)
    
    is_single = False
    if q_arr.ndim == 1:
        if q_arr.shape[0] != 3:
            raise ValueError(f"Expected joint vector of length 3, got shape {q_arr.shape}")
        q_arr = q_arr.reshape(1, 3)
        is_single = True
    elif q_arr.ndim == 2 and q_arr.shape[1] == 3:
        pass
    else:
        raise ValueError(f"Expected shape (3,) or (N, 3), got {q_arr.shape}")

    if validate_limits:
        check_joint_limits(q_arr)

    q1 = q_arr[:, 0]
    q2 = q_arr[:, 1]
    q3 = q_arr[:, 2]

    # 1. Elbow transformation (2D arm plane)
    cos3, sin3 = np.cos(q3), np.sin(q3)
    wx = A + U * cos3 - V * sin3
    wy = B + U * sin3 + V * cos3

    # 2. Shoulder pitch transformation (radius r and height z)
    cos2, sin2 = np.cos(q2), np.sin(q2)
    r = wy * cos2 + wx * sin2
    z = H - wy * sin2 + wx * cos2

    # 3. Base yaw rotation (with 0 deg pointing along positive y-axis)
    sin1, cos1 = np.sin(q1), np.cos(q1)
    x = -r * sin1
    y =  r * cos1

    p = np.column_stack((x, y, z))
    return p[0] if is_single else p


def check_joint_limits(q_arr: np.ndarray) -> None:
    """Helper function to validate joint limits for Ned arm."""
    q1_min, q1_max = NED_JOINT_LIMITS['q1']
    q2_min, q2_max = NED_JOINT_LIMITS['q2']
    q3_min, q3_max = NED_JOINT_LIMITS['q3']

    if np.any((q_arr[:, 0] < q1_min) | (q_arr[:, 0] > q1_max)):
        raise ValueError(f"q1 values out of range [{q1_min}, {q1_max}]")
    if np.any((q_arr[:, 1] < q2_min) | (q_arr[:, 1] > q2_max)):
        raise ValueError(f"q2 values out of range [{q2_min}, {q2_max}]")
    if np.any((q_arr[:, 2] < q3_min) | (q_arr[:, 2] > q3_max)):
        raise ValueError(f"q3 values out of range [{q3_min}, {q3_max}]")


if __name__ == '__main__':
    # Unit Verification: Zero joint angles
    p_zero = ned_forward_kinematics([-0.7854, 0.3504, 0.3554])
    print("=== Ned Forward Kinematics Verification ===")
    print(f"TCP Position at Zero q=[0.0, 0.0, 0.0]:")
    print(f"  x: {p_zero[0]:.6f} m (Expected: 0.000000)")
    print(f"  y: {p_zero[1]:.6f} m (Expected: B + V = 0.012 + 0.230 = 0.242000)")
    print(f"  z: {p_zero[2]:.6f} m (Expected: H + A + U = 0.1715 + 0.221 + 0.03185 = 0.424350)")
