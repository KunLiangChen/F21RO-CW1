"""
Epson SCARA Robot Forward Kinematics (FK) Module.

This script implements the exact analytical forward kinematics for the Epson SCARA
robot arm as specified in Coursework_1_2627.pdf (Task 1.B).

Active Joint Vector Input:
    q = [q1, q2, q4]
    - q1 (base_arm_motor): base rotation [rad], range: [-0.73, 0.73]
    - q2 (arm_motor): second arm rotation [rad], range: [-0.83, 0.83]
    - q4 (shaft_linear_motor): vertical linear displacement [m], range: [-0.20, 0.00]
    Note: q3 (shaft_rotation_motor) is fixed/ignored in this coursework.

Output:
    p = [x, y, z]: 3D coordinates of TCP in meters.
"""

from typing import Union, List, Tuple
import numpy as np

# SCARA Active Joint Physical Limits (Min, Max) in radians / meters
SCARA_JOINT_LIMITS = {
    'q1': (-0.73, 0.73),
    'q2': (-0.83, 0.83),  #This would be [0,0.83] later
    'q4': (-0.20, 0.00)
}


def scara_forward_kinematics(
    q: Union[List[float], Tuple[float, ...], np.ndarray],
    validate_limits: bool = False
) -> np.ndarray:
    """
    Computes the Forward Kinematics for the Epson SCARA robot arm.

    Supports both single joint vector inputs of shape (3,) or (1, 3) 
    and vectorized batch inputs of shape (N, 3) for high-performance dataset generation.

    Args:
        q: Active joint vector [q1, q2, q4] or batch array of shape (N, 3).
        validate_limits: If True, checks whether joint inputs stay within physical limits.

    Returns:
        np.ndarray: Calculated 3D TCP position [x, y, z] or batch array of shape (N, 3).

    Raises:
        ValueError: If input shape is invalid or joint values exceed physical limits.
    """
    q_arr = np.asarray(q, dtype=np.float64)
    
    # Handle single input vector vs batch input
    is_single = False
    if q_arr.ndim == 1:
        if q_arr.shape[0] != 3:
            raise ValueError(f"Expected input array of length 3, got shape {q_arr.shape}")
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
    q4 = q_arr[:, 2]
    
    cos1 = np.cos(q1)
    sin1 = np.sin(q1)
    
    q12 = q1 + q2
    cos12 = np.cos(q12)
    sin12 = np.sin(q12)
    
    # -------------------------------------------------------------------------
    # Analytical FK Formula derivation:
    # We avoid 4x4 Transformation Matrices to maximize batch-processing speed.
    # The mathematical model used is:
    # p = v0 + Rz(q1) * (v1 + Rz(q2) * v2)
    #
    # Where:
    # v0 = [0.060,  0.000,  0.220 ]
    # v1 = [0.310,  0.005, -0.004 ]
    # v2 = [0.2545, -0.005, q4 - 0.034]
    #
    # By expanding the rotation matrices Rz algebraically, we get the optimized 
    # scalar operations below, saving substantial computational overhead.
    # -------------------------------------------------------------------------
     
    x = 0.060 + 0.310 * cos1 - 0.005 * sin1 + 0.2545 * cos12 + 0.005 * sin12
    y = 0.000 + 0.310 * sin1 + 0.005 * cos1 + 0.2545 * sin12 - 0.005 * cos12
    z = 0.182 + q4  # 0.220 - 0.004 - 0.034 + q4 = 0.182 + q4
    
    p = np.column_stack((x, y, z))

    return p.flatten() if is_single else p

def check_joint_limits(q_arr: np.ndarray) -> None:
    """Helper function to validate joint limits for SCARA arm."""
    q1_min, q1_max = SCARA_JOINT_LIMITS['q1']
    q2_min, q2_max = SCARA_JOINT_LIMITS['q2']
    q4_min, q4_max = SCARA_JOINT_LIMITS['q4']

    if np.any((q_arr[:, 0] < q1_min) | (q_arr[:, 0] > q1_max)):
        raise ValueError(f"q1 values out of range [{q1_min}, {q1_max}]")
    if np.any((q_arr[:, 1] < q2_min) | (q_arr[:, 1] > q2_max)):
        raise ValueError(f"q2 values out of range [{q2_min}, {q2_max}]")
    if np.any((q_arr[:, 2] < q4_min) | (q_arr[:, 2] > q4_max)):
        raise ValueError(f"q4 values out of range [{q4_min}, {q4_max}]")
    
if __name__ == '__main__':
    # --- Unit Verification ---
    print("=== SCARA Forward Kinematics Verification ===")
    
    # Test Case 1: Zero configuration [0.0, 0.0, 0.0]
    q_zero = [0.7, 0.8, -0.1]
    p_zero = scara_forward_kinematics(q_zero)
    print(f"TCP Position at Zero q={q_zero}:")
    print(f"  x: {p_zero[0]:.6f} m (Expected: 0.060 + 0.310 + 0.2545 = 0.6245)")
    print(f"  y: {p_zero[1]:.6f} m (Expected: 0.005 - 0.005 = 0.0000)")
    print(f"  z: {p_zero[2]:.6f} m (Expected: 0.1820)")

    # Test Case 2: Fully extended linear shaft [0.0, 0.0, -0.20]
    q_extended = [0.0, 0.0, -0.20]
    p_extended = scara_forward_kinematics(q_extended)
    print(f"\nTCP Position at Fully Down q={q_extended}:")
    print(f"  z: {p_extended[2]:.6f} m (Expected: 0.182 - 0.200 = -0.0180)")

    # Test Case 3: Batch input testing
    q_batch = np.array([
        [0.0, 0.0, 0.0],
        [0.7, 0.8, -0.1]
    ])
    p_batch = scara_forward_kinematics(q_batch)
    print(f"\nBatch processing output shape: {p_batch.shape}")
    