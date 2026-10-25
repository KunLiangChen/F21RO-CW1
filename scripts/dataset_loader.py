"""
Task 3: dataset loading for the network.

Loads the normalised CSV files of both arms, adds the robot flag [is_scara, is_ned]
and merges them into one training and one validation set for the single network.

Network input:  [is_scara, is_ned, x, y, z]   (5 values)
Network target: [q1_norm, q2_norm, q3_norm]   (3 values in [-1.0, 1.0])
"""

import os
import csv
import numpy as np


def load_csv_dataset(file_path):
    """
    Loads one CSV dataset file and separates inputs and targets.

    Expected columns: q1_norm, q2_norm, q3_norm, x, y, z (as written by build_datasets.py).
    A text header line is skipped.

    Args:
        file_path (str): Relative or absolute path to the CSV dataset.

    Returns:
        tuple: (positions, normalized_joints)
            - positions: numpy array of shape (N, 3) containing [x, y, z]
            - normalized_joints: numpy array of shape (N, 3) containing [q1_norm, q2_norm, q3_norm]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    data = []
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            # Skip empty lines and the header line
            if not row:
                continue
            try:
                row_floats = [float(val.strip()) for val in row]
                data.append(row_floats)
            except ValueError:
                continue
    
    if not data:
        raise ValueError(
            f"No valid numerical data parsed from: {file_path}.\n"
            f"Please verify if the CSV file is empty or formatted correctly."
        )

    data_arr = np.array(data, dtype=np.float32)

    if data_arr.shape[1] != 6:
        raise ValueError(f"Expected 6 columns in {file_path}, but got {data_arr.shape[1]}")

    # Column order of the pipeline: joints first, then the TCP position
    normalized_joints = data_arr[:, 0:3]
    positions = data_arr[:, 3:6]

    return positions, normalized_joints


def prepare_robot_features(positions, is_scara=False):
    """
    Adds the one-hot robot flag in front of the TCP positions.

    Flag representation:
        - SCARA: [1.0, 0.0, x, y, z]
        - Ned:   [0.0, 1.0, x, y, z]

    Args:
        positions (np.ndarray): Array of TCP coordinates (N, 3).
        is_scara (bool): True for the SCARA arm, False for Ned.

    Returns:
        np.ndarray: Input features of shape (N, 5).
    """
    num_samples = len(positions)
    if is_scara:
        flags = np.tile(np.array([1.0, 0.0], dtype=np.float32), (num_samples, 1))
    else:
        flags = np.tile(np.array([0.0, 1.0], dtype=np.float32), (num_samples, 1))

    return np.hstack([flags, positions])


def get_combined_dataset(dataset_dir, size='small'):
    """
    Loads, flags and merges the Ned and SCARA training and validation sets of one size.

    Args:
        dataset_dir (str): Path to the 'datasets' directory.
        size (str): Dataset size: 'small', 'medium' or 'large'.

    Returns:
        tuple: (X_train, y_train, X_val, y_val)
            - X_train: Training features [is_scara, is_ned, x, y, z]
            - y_train: Training targets [q1_norm, q2_norm, q3_norm]
            - X_val: Validation features
            - y_val: Validation targets
    """
    # 1. File paths
    ned_train_path = os.path.join(dataset_dir, 'ned', size, 'train.csv')
    ned_val_path = os.path.join(dataset_dir, 'ned', size, 'validation.csv')
    scara_train_path = os.path.join(dataset_dir, 'scara', size, 'train.csv')
    scara_val_path = os.path.join(dataset_dir, 'scara', size, 'validation.csv')

    # 2. Load raw datasets
    ned_pos_train, ned_q_train = load_csv_dataset(ned_train_path)
    ned_pos_val, ned_q_val = load_csv_dataset(ned_val_path)

    scara_pos_train, scara_q_train = load_csv_dataset(scara_train_path)
    scara_pos_val, scara_q_val = load_csv_dataset(scara_val_path)

    # 3. Add the robot flags
    X_ned_train = prepare_robot_features(ned_pos_train, is_scara=False)
    X_ned_val = prepare_robot_features(ned_pos_val, is_scara=False)

    X_scara_train = prepare_robot_features(scara_pos_train, is_scara=True)
    X_scara_val = prepare_robot_features(scara_pos_val, is_scara=True)

    # 4. Merge Ned and SCARA
    X_train = np.vstack([X_ned_train, X_scara_train])
    y_train = np.vstack([ned_q_train, scara_q_train])

    X_val = np.vstack([X_ned_val, X_scara_val])
    y_val = np.vstack([ned_q_val, scara_q_val])

    return X_train, y_train, X_val, y_val