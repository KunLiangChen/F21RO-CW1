"""
Task 2: joint sampling for the SCARA and Ned arms.

Draws uniformly random joint values inside the allowed ranges and validates them.
No forward kinematics here; only joint space constraints.

Elbow-up restriction (one joint solution per TCP position): SCARA q2 and Ned q3 are
sampled from positive values only.

Floor restriction: the SCARA shaft stops at q4 = -0.173 (sampled down to -0.17); a Ned pose
whose TCP is below NED_MIN_TCP_Z = 0.12 m is rejected, because the hand then touches the
Webots floor and does not reach the FK position.
"""

import numpy as np
from scripts.ned_fk import ned_forward_kinematics

# Ned only: below this TCP height the hand touches the Webots floor. Measured in Webots
# (FK vs simulated TCP, same joints): about 0.2 mm error at z >= 0.12 m, 1.5 mm at
# 0.11-0.12 m and 5 to 8 mm below 0.10 m.
NED_MIN_TCP_Z = 0.12

def generate_scara_samples(num_samples: int, random_seed: int = None) -> np.ndarray:
    """
    Generates random joint samples for the SCARA arm.

    Args:
        num_samples (int): Number of joint vectors to draw.
        random_seed (int, optional): Seed for reproducibility.

    Returns:
        np.ndarray: Array of shape (num_samples, 3), every row is [q1, q2, q4].
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    # q1 (base_arm_motor): base rotation [rad]
    # limit in theory: [-0.73, 0.73]
    q1 = np.random.uniform(-0.73, 0.73, num_samples)
    
    # q2 (arm_motor): second arm rotation [rad]
    # elbow-up: only q2 >= 0 is sampled, so every TCP position has a single joint solution
    # (the motor limit is [-0.83, 0.83])
    q2 = np.random.uniform(0.0, 0.83, num_samples)
    
    # q4 (shaft_linear_motor): vertical linear displacement [m]
    # Below q4 = -0.173 the shaft touches the Webots floor and cannot move further (the
    # motor limit is -0.2): the measured TCP then differs from FK by 7 to 27 mm, against
    # < 0.1 mm at -0.17. The lower bound of the sampling is therefore -0.17.
    q4 = np.random.uniform(-0.17, 0.0, num_samples)
    
    
    samples = np.column_stack((q1, q2, q4))
    
    return samples


def generate_ned_samples(num_samples: int, random_seed: int = None) -> np.ndarray:
    """
    Generates random joint samples for the Ned arm.

    Args:
        num_samples (int): Number of joint vectors to draw.
        random_seed (int, optional): Seed for reproducibility.

    Returns:
        np.ndarray: Array of shape (num_samples, 3), every row is [q1, q2, q3].
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    # Rejection sampling: draw joints uniformly in the box and keep only those whose TCP is
    # clear of the floor (z >= NED_MIN_TCP_Z); draw again until num_samples are collected.
    kept = []
    num_kept = 0
    while num_kept < num_samples:
        # q1 (joint_1): waist/base yaw [rad]
        # limit: [-2.8, 2.8]
        q1 = np.random.uniform(-2.8, 2.8, num_samples)

        # q2 (joint_2): shoulder pitch [rad]
        # limit: [-0.8, 0.8]
        q2 = np.random.uniform(-0.8, 0.8, num_samples)

        # q3 (joint_3): elbow pitch [rad]
        # elbow-up: only q3 >= 0 is sampled, so every TCP position has a single joint solution
        # (the motor limit is [-1.5, 1.3])
        q3 = np.random.uniform(0.0, 1.3, num_samples)

        candidates = np.column_stack((q1, q2, q3))
        candidates = candidates[ned_forward_kinematics(candidates)[:, 2] >= NED_MIN_TCP_Z]
        kept.append(candidates)
        num_kept += len(candidates)

    return np.vstack(kept)[:num_samples]


def validate_scara_dataset(samples: np.ndarray) -> bool:
    """
    Checks that every SCARA sample complies with all constraints.

    Args:
        samples (np.ndarray): Joint samples of shape (N, 3), columns [q1, q2, q4].

    Returns:
        bool: True if all samples are valid.
    """
    if samples.shape[1] != 3:
        raise ValueError(f"Expected SCARA samples shape (N, 3), got {samples.shape}")

    q1, q2, q4 = samples[:, 0], samples[:, 1], samples[:, 2]
    
    # Motor limits + elbow-up (q2 >= 0) + Webots q4 restriction
    valid_q1 = (q1 >= -0.73) & (q1 <= 0.73)
    valid_q2 = (q2 >= 0.0) & (q2 <= 0.83)
    valid_q4 = (q4 >= -0.17) & (q4 <= 0.0)
    
    return np.all(valid_q1 & valid_q2 & valid_q4)

def validate_ned_dataset(samples: np.ndarray) -> bool:
    """
    Checks that every Ned sample complies with all constraints.

    Args:
        samples (np.ndarray): Joint samples of shape (N, 3), columns [q1, q2, q3].

    Returns:
        bool: True if all samples are valid.
    """
    if samples.shape[1] != 3:
        raise ValueError(f"Expected Ned samples shape (N, 3), got {samples.shape}")

    q1, q2, q3 = samples[:, 0], samples[:, 1], samples[:, 2]
    
    # Motor limits + elbow-up (q3 >= 0) + TCP above the floor
    valid_q1 = (q1 >= -2.8) & (q1 <= 2.8)
    valid_q2 = (q2 >= -0.8) & (q2 <= 0.8)
    valid_q3 = (q3 >= 0.0) & (q3 <= 1.3)
    valid_z = ned_forward_kinematics(samples)[:, 2] >= NED_MIN_TCP_Z
    
    return np.all(valid_q1 & valid_q2 & valid_q3 & valid_z)

def get_invalid_indices(samples: np.ndarray, robot_type: str) -> np.ndarray:
    """
    Returns the row indices of all invalid samples, for debugging.

    Args:
        samples (np.ndarray): Joint samples of shape (N, 3).
        robot_type (str): 'scara' or 'ned'.
    """
    q1, q2, q3_or_q4 = samples[:, 0], samples[:, 1], samples[:, 2]
    
    if robot_type.lower() == 'scara':
        valid = ((q1 >= -0.73) & (q1 <= 0.73) & 
                 (q2 >= 0.0) & (q2 <= 0.83) & 
                 (q3_or_q4 >= -0.17) & (q3_or_q4 <= 0.0))
    elif robot_type.lower() == 'ned':
        valid = ((q1 >= -2.8) & (q1 <= 2.8) & 
                 (q2 >= -0.8) & (q2 <= 0.8) & 
                 (q3_or_q4 >= 0.0) & (q3_or_q4 <= 1.3) &
                 (ned_forward_kinematics(samples)[:, 2] >= NED_MIN_TCP_Z))
    else:
        raise ValueError("robot_type must be either 'scara' or 'ned'")
        
    return np.where(~valid)[0]


if __name__ == "__main__":
    print("=== Joint Sampling Module Test ===")
    
    test_samples_num = 5
    seed = 42
    

    scara_data = generate_scara_samples(test_samples_num, random_seed=seed)
    print("\n[SCARA Samples] Shape:", scara_data.shape)
    for i, (q1, q2, q4) in enumerate(scara_data):
        print(f"  Sample {i+1}: q1={q1:+.4f}, q2={q2:+.4f}, q4={q4:+.4f}")
        
        assert -0.73 <= q1 <= 0.73, "SCARA q1 out of bounds!"
        assert 0.0 <= q2 <= 0.83, "SCARA q2 out of bounds (Elbow-up violation)!"
        assert -0.17 <= q4 <= 0.0, "SCARA q4 out of bounds (Webots limitation violation)!"
        
    
    ned_data = generate_ned_samples(test_samples_num, random_seed=seed)
    print("\n[Ned Samples] Shape:", ned_data.shape)
    for i, (q1, q2, q3) in enumerate(ned_data):
        print(f"  Sample {i+1}: q1={q1:+.4f}, q2={q2:+.4f}, q3={q3:+.4f}")
        
        assert -2.8 <= q1 <= 2.8, "Ned q1 out of bounds!"
        assert -0.8 <= q2 <= 0.8, "Ned q2 out of bounds!"
        assert 0.0 <= q3 <= 1.3, "Ned q3 out of bounds (Elbow-up violation)!"

    assert validate_ned_dataset(ned_data), "Ned TCP below the floor limit!"
        
    print("\nAll sampling constraints verified successfully. Ready for Pipeline Integration.")