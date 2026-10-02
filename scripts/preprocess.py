"""
preprocess.py

Milestone 3 - Task C: Normalisation & Splitting
Responsible for normalizing joint values to [-1, 1], saving normalization 
parameters (q_min, q_max), and splitting datasets into training and validation sets.
"""

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple

def normalize_joints(raw_data: np.ndarray, robot_name: str, save_dir: str = "normalisation") -> np.ndarray:
    """
    Normalizes the joint values (first 3 columns) of the raw dataset to [-1.0, 1.0].
    Saves the q_min and q_max parameters to a JSON file for future denormalization.
    
    Args:
        raw_data (np.ndarray): The raw dataset of shape (N, 6) -> [q1, q2, q3, x, y, z].
        robot_name (str): Name of the robot ('scara' or 'ned') for the JSON filename.
        save_dir (str): Directory to save the normalization parameters.
        
    Returns:
        np.ndarray: The processed dataset with normalized joint values.
    """
    if raw_data.shape[1] != 6:
        raise ValueError(f"Expected raw_data shape (N, 6), got {raw_data.shape}")
        
    # Create the directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract joint values (first 3 columns)
    joints = raw_data[:, :3]
    
    # Calculate min and max for each joint across the entire dataset
    q_min = np.min(joints, axis=0)
    q_max = np.max(joints, axis=0)
    
    # Apply normalization formula: q_norm = 2 * (q - q_min) / (q_max - q_min) - 1
    # Adding a tiny epsilon to prevent division by zero in edge cases
    epsilon = 1e-10
    joints_norm = 2 * (joints - q_min) / (q_max - q_min + epsilon) - 1
    
    # Copy data to avoid modifying the original array and replace with normalized joints
    processed_data = np.copy(raw_data)
    processed_data[:, :3] = joints_norm
    
    # Save parameters to JSON
    json_path = os.path.join(save_dir, f"{robot_name.lower()}.json")
    params = {
        "robot": robot_name.lower(),
        "q_min": q_min.tolist(),
        "q_max": q_max.tolist()
    }
    
    with open(json_path, 'w') as f:
        json.dump(params, f, indent=4)
        
    print(f"[{robot_name.upper()}] Normalization parameters saved to {json_path}")
    
    return processed_data

def split_train_val(dataset: np.ndarray, val_ratio: float = 0.2, random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits the dataset into training and validation sets using a fixed random seed.
    
    Args:
        dataset (np.ndarray): The normalized dataset.
        val_ratio (float): The proportion of the dataset to include in the validation split (~80/20).
        random_seed (int): Fixed seed to ensure reproducible splits across the group.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (train_data, validation_data)
    """
    # Use scikit-learn for robust, randomized splitting with a fixed seed
    train_data, val_data = train_test_split(
        dataset, 
        test_size=val_ratio, 
        random_state=random_seed, 
        shuffle=True
    )
    
    return train_data, val_data

if __name__ == "__main__":
    # --- Task C Unit Verification ---
    print("=== Preprocessing & Normalisation Test ===")
    
    # Create dummy raw data: 100 samples, 6 columns [q1, q2, q3, x, y, z]
    np.random.seed(42)
    dummy_raw_data = np.random.rand(100, 6) * 2.0  # Random values between 0 and 2
    
    # Test Normalization
    normalized_data = normalize_joints(dummy_raw_data, robot_name="test_robot", save_dir="test_normalisation")
    
    # Verify the normalized joints are strictly bounded within [-1, 1]
    joints_norm = normalized_data[:, :3]
    assert np.all(joints_norm >= -1.0) and np.all(joints_norm <= 1.0), "Normalization failed bounds check!"
    print(f"Normalization successful. Min: {np.min(joints_norm):.4f}, Max: {np.max(joints_norm):.4f}")
    
    # Test Splitting
    train_set, val_set = split_train_val(normalized_data, val_ratio=0.2, random_seed=42)
    
    print(f"Splitting successful.")
    print(f"Total samples: {len(normalized_data)}")
    print(f"Train set size: {len(train_set)} (~80%)")
    print(f"Validation set size: {len(val_set)} (~20%)")