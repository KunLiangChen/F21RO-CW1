"""
Task 2: builds the final datasets for the SCARA and Ned arms.

Runs the whole pipeline: joint sampling (gen_sample.py), FK (gen_rawdata.py),
normalisation and split (preprocess.py). Produces nested small/medium/large
train+validation sets and an independent test set per arm under datasets/.

Run from the project root: python scripts/build_datasets.py
"""

import os
import sys
import numpy as np

# Make the project root importable; datasets/ and normalisation/ are written there
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
NORMALISATION_DIR = os.path.join(PROJECT_ROOT, "normalisation")

from scripts.gen_rawdata import generate_raw_scara_data, generate_raw_ned_data
from scripts.gen_sample import validate_scara_dataset, validate_ned_dataset, get_invalid_indices
from scripts.preprocess import normalize_joints, split_train_val

# Dataset sizes (the smallest set must have at least 1000 samples before the split)
TOTAL_POOL_SIZE = 10500
TEST_SET_SIZE = 500

# 10,000 samples remain after removing the test set; an ~80/20 split of them gives the
# Large set (8000 train / 2000 validation). Medium and small are prefixes of it.
SIZES = {
    "large": {"train": 8000, "val": 2000}, # 10000 total
    "medium": {"train": 4000, "val": 1000}, # 5000 total
    "small": {"train": 1000, "val": 250}    # 1250 total (>= 1000 limit)
}

def save_to_csv(data: np.ndarray, filepath: str, header: str):
    """Saves an array to a CSV file with the given header line."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savetxt(filepath, data, delimiter=',', header=header, comments='')

def build_robot_datasets(robot_name: str, raw_data_func, header: str, seed: int = 42):
    """
    Generates and saves the datasets of one arm.

    Args:
        robot_name (str): 'scara' or 'ned'.
        raw_data_func (callable): Raw data generator, e.g. generate_raw_scara_data.
        header (str): CSV header line, e.g. "q1,q2,q4,x,y,z".
        seed (int): Seed for sampling and for the train/validation split.
    """
    print(f"\n--- Building Datasets for {robot_name.upper()} ---")
    
    # Step 1: Generate the full raw data pool
    raw_pool = raw_data_func(TOTAL_POOL_SIZE, random_seed=seed)
    
    # Abort if any sampled joint value violates the allowed ranges
    if robot_name == "scara":
        if not validate_scara_dataset(raw_pool[:,:3]):
            bad_idx = get_invalid_indices(raw_pool[:,:3], 'scara')
            raise RuntimeError(f"SCARA validation failed! {len(bad_idx)} invalid samples found. "
                               f"First invalid sample: {raw_pool[bad_idx[0], :3]}")
    elif robot_name == "ned":
        if not validate_ned_dataset(raw_pool[:,:3]):
            bad_idx = get_invalid_indices(raw_pool[:,:3], 'ned')
            raise RuntimeError(f"Ned validation failed! {len(bad_idx)} invalid samples found. "
                               f"First invalid sample: {raw_pool[bad_idx[0], :3]}")
    print(f"[{robot_name.upper()}] Dataset validation passed. All joint limits compliant.")

    # Step 2: Split off the test set; it never appears in train/validation
    test_raw = raw_pool[:TEST_SET_SIZE]
    train_val_raw = raw_pool[TEST_SET_SIZE:]
    
    # Step 3: Normalise the train/validation pool (this writes normalisation/<robot>.json)
    # and normalise the test set with the very same bounds
    train_val_norm, train_val_bounds = normalize_joints(train_val_raw, robot_name, save_dir=NORMALISATION_DIR)
    test_norm, _ = normalize_joints(test_raw, robot_name, bounds=train_val_bounds, save_dir=NORMALISATION_DIR) 

    # Step 4: 80/20 train/validation split of the pool (Large set)
    large_train, large_val = split_train_val(train_val_norm, val_ratio=0.2, random_seed=seed)
    
    # Step 5: Take prefixes of the Large set so that Small ⊂ Medium ⊂ Large
    datasets = {
        "large": (large_train, large_val),
        "medium": (large_train[:SIZES["medium"]["train"]], large_val[:SIZES["medium"]["val"]]),
        "small": (large_train[:SIZES["small"]["train"]], large_val[:SIZES["small"]["val"]])
    }
    
    # Step 6: Save everything under datasets/<robot>/
    base_dir = os.path.join(PROJECT_ROOT, "datasets", robot_name.lower())
    
    # Save test set
    save_to_csv(test_norm, f"{base_dir}/test.csv", header)
    print(f"Saved {robot_name.upper()} test set: {len(test_norm)} samples")
    
    # Save nested train/val sets
    for scale, (train_data, val_data) in datasets.items():
        scale_dir = f"{base_dir}/{scale}"
        save_to_csv(train_data, f"{scale_dir}/train.csv", header)
        save_to_csv(val_data, f"{scale_dir}/validation.csv", header)
        print(f"Saved {robot_name.upper()} {scale} set -> Train: {len(train_data)}, Val: {len(val_data)}")

if __name__ == "__main__":
    print("Initializing Task 2 dataset pipeline...")
    
    # SCARA: q1, q2, q4 (shaft_linear_motor); q3 (shaft rotation) is ignored
    scara_header = "q1,q2,q4,x,y,z"
    build_robot_datasets("scara", generate_raw_scara_data, scara_header, seed=42)
    
    # Ned: q1, q2, q3
    ned_header = "q1,q2,q3,x,y,z"
    build_robot_datasets("ned", generate_raw_ned_data, ned_header, seed=42)
    
    print("\nDataset generation complete. Normalization parameters saved in 'normalisation/' directory.")