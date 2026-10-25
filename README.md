# F20/21RO Coursework 1: ANN-based inverse kinematics for the Ned and SCARA arms

A single neural network predicts the joint values of two arms (Niryo Ned and Epson SCARA)
from a target TCP position (x, y, z). The training data is generated with our own forward
kinematics, and the results are compared with the simulated arms in Webots.

## Requirements

- Python 3.11 or newer, Webots R2023b
- numpy and scikit-learn, the only third-party libraries used (pinned in `requirements.txt`;
  the models were trained with scikit-learn 1.8.0):

```bash
pip install -r requirements.txt
```

All paths are relative: run the Python scripts from the project root. The Webots
controllers find the project root themselves.

## Project layout

| Path | Content |
|---|---|
| `scripts/scara_fk.py`, `scripts/ned_fk.py` | Task 1.B forward kinematics |
| `controllers/scara_joint_ctrl/`, `controllers/ned_joint_ctrl/`, `scripts/webots_arm.py` | Task 1.A Webots controllers |
| `scripts/evaluate_fk_webots.py` | Task 1.C FK vs Webots |
| `scripts/gen_sample.py`, `scripts/gen_rawdata.py`, `scripts/preprocess.py`, `scripts/build_datasets.py` | Task 2 dataset creation |
| `datasets/`, `normalisation/` | generated datasets and normalisation bounds (committed) |
| `scripts/dataset_loader.py`, `scripts/train_ann.py` | Task 3 training |
| `models/`, `logs/log_*.json` | trained models and training logs (committed) |
| `worlds/`, `protos/` | Webots worlds and the Ned robot from the course repository |

Order of the pipeline: `python scripts/build_datasets.py` (datasets), then `python scripts/train_ann.py`
(models). Both outputs are already committed, so the scripts only have to be rerun to
reproduce them.

## Task 1.A: joint controllers (Webots)

Each arm has a Supervisor controller attached to `DEF ARM`; it commands the 3 active
joints, waits until the position sensors settle, then reads the TCP pose (ARM frame).

- SCARA: open `worlds/scara.wbt` (controller `scara_joint_ctrl`)
- Ned: open `worlds/articulated.wbt` (controller `ned_joint_ctrl`)

Joint values are read from `logs/joint_input_scara.csv` / `logs/joint_input_ned.csv`
(rows of `q1,q2,q4` / `q1,q2,q3`, physical units, optional header line). If the file is
missing, built-in demo poses run. Results go to `logs/tcp_measured_<arm>.csv`.
Shared logic: `scripts/webots_arm.py`.


## Task 1.B: forward kinematics

`scripts/scara_fk.py` and `scripts/ned_fk.py` compute the TCP position (x, y, z) in the ARM
frame from the three active joints: SCARA `[q1, q2, q4]` (q3, the shaft rotation, is
ignored) and Ned `[q1, q2, q3]` (wrist joints fixed). They accept one joint vector or a
batch of shape (N, 3). For the Ned, q1 = 0 points along +y. Running a script prints a self-check
at the zero configuration, e.g. `python scripts/ned_fk.py`.

## Task 1.C: FK vs Webots precision

```bash
python scripts/evaluate_fk_webots.py generate   # 8 corner + 200 random joint sets per arm (seed 2026)
# run worlds/scara.wbt and worlds/articulated.wbt once each in Webots (controllers write logs/tcp_measured_*.csv)
python scripts/evaluate_fk_webots.py report     # mean/std/max position error, saved to logs/fk_vs_webots.json
```

Result (208 poses per arm, position error between Webots and FK): SCARA mean 0.34 mm,
max 1.04 mm. Ned mean 0.42 mm, max 7.8 mm; the larger Ned errors only occur when the TCP is
below 0.12 m, where the hand touches the floor. Without those 13 poses the Ned mean is
0.21 mm and the maximum 0.70 mm.


## Task 2: sampled joint ranges and normalisation

Create the datasets (run from the project root; needs numpy and scikit-learn):

```bash
python scripts/build_datasets.py
```

This writes `datasets/<arm>/{small,medium,large}/{train,validation}.csv` and
`datasets/<arm>/test.csv` (columns `q1,q2,q3|q4,x,y,z`, joints normalised to [-1, 1]) and the
normalisation bounds `normalisation/<arm>.json`. The generated files are committed, so
this only has to be rerun to reproduce them. Random seed: 42 (joint sampling and the
80/20 split, for both arms). Small and medium are the first rows of large, so they are
nested; the 500 test samples are taken out of the pool before the split.

The joints are sampled uniformly from ranges that are narrower than the motor limits
(`scripts/gen_sample.py`):

| Joint | Sampled range | Motor limit | Why |
|---|---|---|---|
| SCARA q2 (`arm_motor`) | [0, 0.83] rad | [-0.83, 0.83] | elbow-up: one joint solution per TCP position |
| Ned q3 (`joint_3`) | [0, 1.3] rad | [-1.5, 1.3] | elbow-up, same reason |
| SCARA q4 (`shaft_linear_motor`) | [-0.17, 0] m | [-0.2, 0] | the shaft touches the Webots floor below about -0.173 m |

The same ranges are the `q_min` / `q_max` used for normalisation (saved in
`normalisation/<arm>.json` and needed to de-normalise the network output), so the output
range [-1, 1] is fully used and the network cannot predict an elbow-down solution.

All other joints use their full motor range. Poses with the TCP close to the floor are
not removed: there the Ned hand can touch the floor too, which adds up to about 8 mm
between Webots and FK (measured in Task 1.C).

## Task 3: training the network

```bash
python scripts/train_ann.py  # trains the small, medium and large model
```

One `MLPRegressor` (scikit-learn) per dataset size serves both arms: the input is
`[is_scara, is_ned, x, y, z]` and the output the three normalised joint values q1, q2, q3
(for the SCARA the third one is q4). The small, medium and large model are trained on the
Ned and SCARA sets of the same size merged into one (`scripts/dataset_loader.py`).

- The hyperparameters are chosen by a grid search that fits every candidate on the training
  set and scores it on the validation set (`PredefinedSplit`). The final model is then
  trained on the training set only, so the validation metrics are not biased. The test
  sets are not used.
- Training stops by itself when the training loss has not improved by more than `tol=1e-7`
  for 50 epochs in a row (`max_iter=3000` is never reached; the models train for 290 to
  570 epochs). scikit-learn's default `tol=1e-4` is too coarse for losses of about 1e-3 and
  stopped the training after 60 to 100 epochs, with up to 3x higher validation error.
- Random seed: `random_state=42`. Results are reproducible with the same scikit-learn version.
- Outputs: `models/model_<size>.pkl` and `logs/log_<size>.json` (best hyperparameters,
  validation metrics, loss curve). The `.pkl` files depend on the scikit-learn version
  (1.8.0 here); with a different version, rerun `python scripts/train_ann.py`.

Selected hyperparameters and validation errors (normalised joint space):

| Model | Train / val samples | Hidden layers | Activation | alpha | Learning rate | Val MSE | Val MAE |
|---|---|---|---|---|---|---|---|
| small | 2000 / 500 | (128, 64) | relu | 0.001 | 0.005 | 2.88e-03 | 2.41e-02 |
| medium | 8000 / 2000 | (128, 128, 64) | relu | 0.0001 | 0.001 | 1.16e-03 | 1.84e-02 |
| large | 16000 / 4000 | (256, 256, 128) | relu | 0.001 | 0.001 | 8.53e-04 | 1.30e-02 |

## Reused and AI-assisted code

- From the course repository (Christian Dondrup): the Webots worlds (we set the arm's
  controller and the supervisor flag), the Ned PROTO and meshes in `protos/`, the example
  controllers in `controllers/` (`ned_simple`, `scara_simple`, `tcp_pose`,
  `discover_devices`, `ned_keyboard`, `ned_sphere_ik`) and `scripts/ik_exact.py`,
  `scripts/ik_simple.py`. They are not used by our pipeline except for the TCP lookup of
  `tcp_pose`, which `scripts/webots_arm.py` follows.
- Generative AI was used for parts of the code, debugging and comments, as the coursework
  allows. All code was checked by the group; the details are in the group's AI statement.
