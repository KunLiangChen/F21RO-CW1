"""
Task 2: normalisation and train/validation split.

Normalises the joint values to [-1, 1], saves the bounds (q_min, q_max) needed for
de-normalisation to normalisation/<robot>.json, and splits datasets ~80/20.
"""

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple

# Define exact theoretical physical bounds per coursework specifications
# SCARA: q1 in [-0.73, 0.73], q2 in [0.0, 0.83], q4 in [-0.17, 0.0]
# Ned:   q1 in [-2.8, 2.8],   q2 in [-0.8, 0.8],  q3 in [0.0, 1.3]
THEORETICAL_BOUNDS = {
    "scara": (
        np.array([-0.73, 0.0, -0.17], dtype=np.float64),
        np.array([0.73, 0.83, 0.0], dtype=np.float64)
    ),
    "ned": (
        np.array([-2.8, -0.8, 0.0], dtype=np.float64),
        np.array([2.8, 0.8, 1.3], dtype=np.float64)
    )
}

def normalize_joints(
    raw_data: np.ndarray, 
    robot_name: str, 
    bounds: tuple = None, 
    save_dir: str = "normalisation"
) -> Tuple[np.ndarray, tuple]:
    """
    Normalizes the joint values (first 3 columns) of the raw dataset to [-1.0, 1.0]
    using theoretical physical bounds.
    
    Saves the exact physical q_min and q_max parameters to JSON for future denormalization.
    
    Args:
        raw_data (np.ndarray): Raw dataset matrix of shape (N, 6) -> [q1, q2, q3/q4, x, y, z].
        robot_name (str): Robot identifier ('scara' or 'ned').
        bounds (tuple, optional): Explicit (q_min, q_max) tuple. If None, uses theoretical physical bounds.
        save_dir (str): Directory to store JSON configuration.
        
    Returns:
        Tuple[np.ndarray, tuple]: (Processed normalized dataset, (q_min, q_max) bounds tuple)
    """
    if raw_data.shape[1] != 6:
        raise ValueError(f"Expected raw_data shape (N, 6), got {raw_data.shape}")
        
    os.makedirs(save_dir, exist_ok=True)
    robot_key = robot_name.lower()

    # Determine normalization bounds
    if bounds is None:
        if robot_key in THEORETICAL_BOUNDS:
            q_min, q_max = THEORETICAL_BOUNDS[robot_key]
        else:
            # Fallback to dataset stats only if an unmapped robot is passed
            joints_raw = raw_data[:, :3]
            q_min, q_max = np.min(joints_raw, axis=0), np.max(joints_raw, axis=0)

        # Save exact physical parameters to JSON configuration
        json_path = os.path.join(save_dir, f"{robot_key}.json")
        params = {
            "robot": robot_key,
            "q_min": q_min.tolist(),
            "q_max": q_max.tolist()
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=4)
            
        print(f"[{robot_key.upper()}] Theoretical normalization parameters saved to {json_path}")
    else:
        q_min, q_max = bounds

    # Apply Normalization Formula: q_norm = 2 * (q - q_min) / (q_max - q_min) - 1.0
    joints = raw_data[:, :3]
    epsilon = 1e-10
    joints_norm = 2.0 * (joints - q_min) / (q_max - q_min + epsilon) - 1.0

    # Safety clamp strictly within [-1.0, 1.0] to prevent floating point boundary noise
    joints_norm = np.clip(joints_norm, -1.0, 1.0)

    # Reconstruct normalized output matrix
    processed_data = np.copy(raw_data)
    processed_data[:, :3] = joints_norm

    return processed_data, (q_min, q_max)

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