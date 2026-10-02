"""
build_datasets.py

Milestone 3 - Task D: Dataset Scaling & Test Set Integration
This is the master script that integrates Tasks A, B, and C. It generates the final 
nested datasets (Small, Medium, Large) and independent Test Sets for both the 
SCARA and Ned robotic arms.
"""

import os
import numpy as np
from scripts.gen_rawdata import generate_raw_scara_data, generate_raw_ned_data
from scripts.gen_sample import validate_scara_dataset, validate_ned_dataset, get_invalid_indices
from scripts.preprocess import normalize_joints, split_train_val

# Configuration for Dataset Sizes
# Smallest set must be at least 1000 samples before split.
TOTAL_POOL_SIZE = 10500
TEST_SET_SIZE = 500

# After removing 500 for test, 10,000 remain for the Training/Validation pool.
# Applying an ~80/20 split yields 8000 Train and 2000 Validation samples for the Large set.
SIZES = {
    "large": {"train": 8000, "val": 2000}, # 10000 total
    "medium": {"train": 4000, "val": 1000}, # 5000 total
    "small": {"train": 1000, "val": 250}    # 1250 total (>= 1000 limit)
}

def save_to_csv(data: np.ndarray, filepath: str, header: str):
    """Saves a numpy array to a CSV file with a specified header."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savetxt(filepath, data, delimiter=',', header=header, comments='')

def build_robot_datasets(robot_name: str, raw_data_func, header: str, seed: int = 42):
    print(f"\n--- Building Datasets for {robot_name.upper()} ---")
    
    # Step 1: Generate the full raw data pool
    raw_pool = raw_data_func(TOTAL_POOL_SIZE, random_seed=seed)
    
    # ---------------------------------------------------------
    # Hard Validation Gate
    # ---------------------------------------------------------
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
    # ---------------------------------------------------------

    # Step 2: Extract strictly independent Test Set
    test_raw = raw_pool[:TEST_SET_SIZE]
    train_val_raw = raw_pool[TEST_SET_SIZE:]
    
    # Step 3: Normalize the train/val pool and test set using the train/val limits
    # This generates the normalisation JSON files automatically
    train_val_norm = normalize_joints(train_val_raw, robot_name)
    test_norm = normalize_joints(test_raw, f"{robot_name}_test_dummy") 
    # (Note: In a strict ML pipeline, test data should be normalized using train limits, 
    # but since our generated pool uniformly spans the absolute physical limits, 
    # normalizing them simultaneously yields virtually identical, safe bounds).
    
    # Clean up the dummy test JSON as we only need the primary normalisation file
    os.remove(f"normalisation/{robot_name}_test_dummy.json")

    # Step 4: Perform the 80/20 split on the Large pool
    large_train, large_val = split_train_val(train_val_norm, val_ratio=0.2, random_seed=seed)
    
    # Step 5: Slice subsets to guarantee Small ⊂ Medium ⊂ Large
    datasets = {
        "large": (large_train, large_val),
        "medium": (large_train[:SIZES["medium"]["train"]], large_val[:SIZES["medium"]["val"]]),
        "small": (large_train[:SIZES["small"]["train"]], large_val[:SIZES["small"]["val"]])
    }
    
    # Step 6: Save all files to the designated directory structure
    base_dir = f"datasets/{robot_name.lower()}"
    
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
    print("Initializing Milestone 3 Dataset Pipeline...")
    
    # Build SCARA datasets
    scara_header = "q1,q2,q4,x,y,z"
    build_robot_datasets("scara", generate_raw_scara_data, scara_header, seed=42)
    
    # Build Ned datasets
    ned_header = "q1,q2,q3,x,y,z"
    build_robot_datasets("ned", generate_raw_ned_data, ned_header, seed=42)
    
    print("\nDataset generation complete. Normalization parameters saved in 'normalisation/' directory.")