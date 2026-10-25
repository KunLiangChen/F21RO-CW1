"""
Task 2: raw FK dataset generation.

Combines randomly sampled joint values with the TCP positions (x, y, z) computed by
our forward kinematics (Task 1.B). Sits between joint sampling (gen_sample.py) and
normalisation/splitting (preprocess.py); the output is not normalised yet.
"""

import numpy as np
from scripts.gen_sample import generate_scara_samples, generate_ned_samples
from scripts.scara_fk import scara_forward_kinematics
from scripts.ned_fk import ned_forward_kinematics

def generate_raw_scara_data(num_samples: int, random_seed: int = None) -> np.ndarray:
    """
    Generates the raw dataset for the SCARA arm (unnormalized).
    
    Args:
        num_samples (int): The number of samples to generate.
        random_seed (int, optional): Random seed for reproducibility.
        
    Returns:
        np.ndarray: An array of shape (num_samples, 6). 
                    Columns correspond to [q1, q2, q4, x, y, z].
    """
    # 1. Sample valid joint values, shape (N, 3)
    joints = generate_scara_samples(num_samples, random_seed)
    
    # 2. Compute the TCP coordinates with FK, shape (N, 3)
    # validate_limits=True is a second safety check on the joint ranges
    positions = scara_forward_kinematics(joints, validate_limits=True)
    
    # 3. Horizontally stack joints and positions into a complete (N, 6) data matrix
    raw_data = np.hstack((joints, positions))
    
    return raw_data

def generate_raw_ned_data(num_samples: int, random_seed: int = None) -> np.ndarray:
    """
    Generates the raw dataset for the Ned arm (unnormalized).
    
    Args:
        num_samples (int): The number of samples to generate.
        random_seed (int, optional): Random seed for reproducibility.
        
    Returns:
        np.ndarray: An array of shape (num_samples, 6). 
                    Columns correspond to [q1, q2, q3, x, y, z].
    """
    # 1. Sample valid joint values, shape (N, 3)
    joints = generate_ned_samples(num_samples, random_seed)
    
    # 2. Compute the TCP coordinates with FK, shape (N, 3)
    positions = ned_forward_kinematics(joints, validate_limits=True)
    
    # 3. Horizontally stack joints and positions into a complete (N, 6) data matrix
    raw_data = np.hstack((joints, positions))
    
    return raw_data

if __name__ == "__main__":
    # Quick self-check
    print("=== Raw Dataset Generation Test ===")
    
    test_samples_num = 5
    seed = 42
    
    # Test SCARA dataset generation
    scara_raw = generate_raw_scara_data(test_samples_num, random_seed=seed)
    print("\nSCARA Raw Data [q1, q2, q4, x, y, z]:")
    print(np.round(scara_raw, 4))
    assert scara_raw.shape == (test_samples_num, 6), "SCARA raw data shape mismatch!"
    
    # Test Ned dataset generation
    ned_raw = generate_raw_ned_data(test_samples_num, random_seed=seed)
    print("\nNed Raw Data [q1, q2, q3, x, y, z]:")
    print(np.round(ned_raw, 4))
    assert ned_raw.shape == (test_samples_num, 6), "Ned raw data shape mismatch!"
    
    print("\nDataset generation successful. Ready for Task C (Normalisation).")