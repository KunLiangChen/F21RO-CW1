"""
gen_sample.py

Milestone 3 - Task A: Joint Sampling
Responsible for generating valid and random joint configuration data for SCARA and Ned robotic arms. 
This module does not include forward kinematics (FK) calculations and focuses solely on joint space constraints and sampling.
"""

import numpy as np

def generate_scara_samples(num_samples: int, random_seed: int = None) -> np.ndarray:
    """
    Generate random joint samples for the SCARA robotic arm.
    
    param:
        num_samples (int): 
        random_seed (int, optional):
        
    return:
        np.ndarray: 2D-array shaped (num_samples, 3), every line is [q1, q2, q4]
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    # q1 (base_arm_motor): base rotation [rad]
    # limit in theory: [-0.73, 0.73]
    q1 = np.random.uniform(-0.73, 0.73, num_samples)
    
    # q2 (arm_motor): second arm rotation [rad]
    # elbow-config: [0.0, 0.83]
    q2 = np.random.uniform(0.0, 0.83, num_samples)
    
    # q4 (shaft_linear_motor): vertical linear displacement [m]
    # After testing in Webots, the simulation fails or becomes inaccurate when the shaft extends beyond -0.17. 
    # So we've tightened the lower bound of the sampling to -0.17.
    q4 = np.random.uniform(-0.17, 0.0, num_samples)
    
    
    samples = np.column_stack((q1, q2, q4))
    
    return samples


def generate_ned_samples(num_samples: int, random_seed: int = None) -> np.ndarray:
    """
    Generate random joint samples for the SCARA robotic arm.
    
    param:
        num_samples (int):
        random_seed (int, optional): 
        
    return:
        np.ndarray: 2D-array shaped (num_samples, 3), every line is [q1, q2, q3]
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    # q1 (joint_1): waist/base yaw [rad]
    # limit: [-2.8, 2.8]
    q1 = np.random.uniform(-2.8, 2.8, num_samples)
    
    # q2 (joint_2): shoulder pitch [rad]
    # limit: [-0.8, 0.8]
    q2 = np.random.uniform(-0.8, 0.8, num_samples)
    
    # q3 (joint_3): elbow pitch [rad]
    # elbow-config: [0.0, 1.3]
    q3 = np.random.uniform(0.0, 1.3, num_samples)
    
    samples = np.column_stack((q1, q2, q3))
    
    return samples


def validate_scara_dataset(samples: np.ndarray) -> bool:
    """
    Rapidly validates whether the entire SCARA dataset complies with all constraints.
    Expected samples shape: (N, 3), corresponding to [q1, q2, q4].
    """
    if samples.shape[1] != 3:
        raise ValueError(f"Expected SCARA samples shape (N, 3), got {samples.shape}")

    q1, q2, q4 = samples[:, 0], samples[:, 1], samples[:, 2]
    
    # Physical limits + Coursework positive value constraint + Webots failure zone avoidance
    valid_q1 = (q1 >= -0.73) & (q1 <= 0.73)
    valid_q2 = (q2 >= 0.0) & (q2 <= 0.83)
    valid_q4 = (q4 >= -0.17) & (q4 <= 0.0)
    
    return np.all(valid_q1 & valid_q2 & valid_q4)

def validate_ned_dataset(samples: np.ndarray) -> bool:
    """
    Rapidly validates whether the entire Ned dataset complies with all constraints.
    Expected samples shape: (N, 3), corresponding to [q1, q2, q3].
    """
    if samples.shape[1] != 3:
        raise ValueError(f"Expected Ned samples shape (N, 3), got {samples.shape}")

    q1, q2, q3 = samples[:, 0], samples[:, 1], samples[:, 2]
    
    # Physical limits + Coursework elbow-up forced positive constraint
    valid_q1 = (q1 >= -2.8) & (q1 <= 2.8)
    valid_q2 = (q2 >= -0.8) & (q2 <= 0.8)
    valid_q3 = (q3 >= 0.0) & (q3 <= 1.3)
    
    return np.all(valid_q1 & valid_q2 & valid_q3)

def get_invalid_indices(samples: np.ndarray, robot_type: str) -> np.ndarray:
    """
    Returns the row indices of all invalid samples for rapid debugging.
    """
    q1, q2, q3_or_q4 = samples[:, 0], samples[:, 1], samples[:, 2]
    
    if robot_type.lower() == 'scara':
        valid = ((q1 >= -0.73) & (q1 <= 0.73) & 
                 (q2 >= 0.0) & (q2 <= 0.83) & 
                 (q3_or_q4 >= -0.17) & (q3_or_q4 <= 0.0))
    elif robot_type.lower() == 'ned':
        valid = ((q1 >= -2.8) & (q1 <= 2.8) & 
                 (q2 >= -0.8) & (q2 <= 0.8) & 
                 (q3_or_q4 >= 0.0) & (q3_or_q4 <= 1.3))
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
        
    print("\nAll sampling constraints verified successfully. Ready for Pipeline Integration.")