"""
Dataset Loading and Preprocessing Module for Task 3.

This module handles loading normalized dataset CSV files for both the Ned and SCARA
robotic arms, appends the target robot flag [is_scara, is_ned], and combines them into 
unified training and validation sets for multi-robot neural network training.

Dataset Input Features:
    - [is_scara, is_ned, x, y, z]  (Dim: 5)
Dataset Target Outputs:
    - [q1_norm, q2_norm, q3_norm]  (Dim: 3, range [-1.0, 1.0])

Coursework Reference:
    F20/21RO Intelligent Robotics - Coursework 1 Task 3
"""

import os
import csv
import numpy as np


def load_csv_dataset(file_path):
    """
    Load a single CSV dataset file and separate inputs and outputs.
    Assumes CSV format either contains header or raw floating point numbers.
    Expected columns: q1_norm, q2_norm, q3_norm, x, y, z
    or x, y, z, q1_norm, q2_norm, q3_norm.

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
            # Skip empty lines or headers containing non-numeric strings
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

    # Standard convention check: if joints are first 3 columns or last 3 columns
    # Here we assume the default pipeline format: q1_norm, q2_norm, q3_norm, x, y, z
    normalized_joints = data_arr[:, 0:3]
    positions = data_arr[:, 3:6]

    return positions, normalized_joints


def prepare_robot_features(positions, is_scara=False):
    """
    Appends one-hot robot identification flags to spatial TCP positions.

    Flag representation:
        - SCARA: [1.0, 0.0, x, y, z]
        - Ned:   [0.0, 1.0, x, y, z]

    Args:
        positions (np.ndarray): Array of TCP coordinates (N, 3).
        is_scara (bool): True if dataset corresponds to SCARA arm, False for Ned.

    Returns:
        np.ndarray: Combined input feature array of shape (N, 5).
    """
    num_samples = len(positions)
    if is_scara:
        flags = np.tile(np.array([1.0, 0.0], dtype=np.float32), (num_samples, 1))
    else:
        flags = np.tile(np.array([0.0, 1.0], dtype=np.float32), (num_samples, 1))

    return np.hstack([flags, positions])


def get_combined_dataset(dataset_dir, size='small'):
    """
    Loads, tags, and merges training and validation datasets for both Ned and SCARA
    robots for a specified dataset size ('small', 'medium', or 'large').

    Args:
        dataset_dir (str): Path to the base 'datasets' directory.
        size (str): Model scale size ('small', 'medium', 'large').

    Returns:
        tuple: (X_train, y_train, X_val, y_val)
            - X_train: Training features [is_scara, is_ned, x, y, z]
            - y_train: Training targets [q1_norm, q2_norm, q3_norm]
            - X_val: Validation features
            - y_val: Validation targets
    """
    # 1. Construct relative file paths
    ned_train_path = os.path.join(dataset_dir, 'ned', size, 'train.csv')
    ned_val_path = os.path.join(dataset_dir, 'ned', size, 'validation.csv')
    scara_train_path = os.path.join(dataset_dir, 'scara', size, 'train.csv')
    scara_val_path = os.path.join(dataset_dir, 'scara', size, 'validation.csv')

    # 2. Load raw datasets
    ned_pos_train, ned_q_train = load_csv_dataset(ned_train_path)
    ned_pos_val, ned_q_val = load_csv_dataset(ned_val_path)

    scara_pos_train, scara_q_train = load_csv_dataset(scara_train_path)
    scara_pos_val, scara_q_val = load_csv_dataset(scara_val_path)

    # 3. Add robot identification flags
    X_ned_train = prepare_robot_features(ned_pos_train, is_scara=False)
    X_ned_val = prepare_robot_features(ned_pos_val, is_scara=False)

    X_scara_train = prepare_robot_features(scara_pos_train, is_scara=True)
    X_scara_val = prepare_robot_features(scara_pos_val, is_scara=True)

    # 4. Merge Ned + SCARA datasets
    X_train = np.vstack([X_ned_train, X_scara_train])
    y_train = np.vstack([ned_q_train, scara_q_train])

    X_val = np.vstack([X_ned_val, X_scara_val])
    y_val = np.vstack([ned_q_val, scara_q_val])

    return X_train, y_train, X_val, y_val