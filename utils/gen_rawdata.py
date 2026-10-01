import numpy as np
from utils.gen_sample import generate_scara_samples, generate_ned_samples
from utils.scara_fk import scara_forward_kinematics
from utils.ned_fk import ned_forward_kinematics

"""
generate_rawdata.py

Milestone 3 - Task B: FK Dataset Generation
Responsible for generating the raw datasets by combining randomly sampled joint
values with their corresponding end-effector Cartesian coordinates (x, y, z)
computed via Forward Kinematics (FK).

This script acts as the bridge between Task A (Joint Sampling) and Task C 
(Normalisation & Splitting), providing unnormalized, physically accurate data 
prior to preprocessing.
"""

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
    # 1. Call Task A's sampling function to get valid joint values of shape (N, 3)
    joints = generate_scara_samples(num_samples, random_seed)
    
    # 2. Call Task 1's FK to calculate TCP coordinates of shape (N, 3)
    # Enabling validate_limits=True acts as a secondary safety check
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
    # 1. Call Task A's sampling function to get valid joint values of shape (N, 3)
    joints = generate_ned_samples(num_samples, random_seed)
    
    # 2. Call Task 1's FK to calculate TCP coordinates of shape (N, 3)
    positions = ned_forward_kinematics(joints, validate_limits=True)
    
    # 3. Horizontally stack joints and positions into a complete (N, 6) data matrix
    raw_data = np.hstack((joints, positions))
    
    return raw_data

if __name__ == "__main__":
    # --- Task B Unit Verification ---
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