# F21RO Coursework: ANN-based inverse kinematics for the Ned and SCARA arms

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

Webots starts its controllers with the `python3` found in the `PATH` (or the interpreter set
in Webots > Preferences > Python command). The Task 4.B evaluation and the demo load the
trained models, so that interpreter must be the one where `requirements.txt` was installed.

## Project layout

| Path | Content |
|---|---|
| `scripts/scara_fk.py`, `scripts/ned_fk.py` | Task 1.B forward kinematics |
| `controllers/scara_joint_ctrl/`, `controllers/ned_joint_ctrl/`, `scripts/webots_arm.py` | Task 1.A Webots controllers |
| `scripts/evaluate_fk_webots.py` | Task 1.C FK vs Webots |
| `scripts/gen_sample.py`, `scripts/gen_rawdata.py`, `scripts/preprocess.py`, `scripts/build_datasets.py` | Task 2 dataset creation |
| `datasets/`, `normalisation/` | generated datasets and normalisation bounds (committed) |
| `scripts/dataset_loader.py`, `scripts/train_ann.py` | Task 3 training |
| `scripts/evaluate_fk.py` | Task 4.A evaluation with FK |
| `controllers/evaluate_webots/` | Task 4.B evaluation in Webots |
| `controllers/nn_sphere_tracker/` | Task 5 demo: the network moves the arm to the red target sphere |
| `models/`, `logs/log_*.json`, `logs/eval_task4*.json` | trained models, training logs and evaluation results (committed) |
| `worlds/`, `protos/` | Webots worlds and the Ned robot from the course repository |

Order of the pipeline: `python scripts/build_datasets.py` (datasets), then `python scripts/train_ann.py`
(models). Both outputs are already committed, so the scripts only have to be rerun to
reproduce them.

## Controllers in the Webots worlds

The controller of `DEF ARM` decides what runs when a world is opened and started. Both worlds start
with the Task 5 demo; to run another one, select `DEF ARM` in the scene tree, change `controller` and
reload the world.

| Controller | Task |
|---|---|
| `nn_sphere_tracker` (default) | Task 5 demo: the network follows the red sphere |
| `scara_joint_ctrl` (`worlds/scara.wbt`), `ned_joint_ctrl` (`worlds/articulated.wbt`) | Task 1.A and Task 1.C |
| `evaluate_webots` | Task 4.B |

## Task 1.A: joint controllers (Webots)

Each arm has a Supervisor controller attached to `DEF ARM`; it commands the 3 active
joints, waits until the position sensors settle, then reads the TCP pose (ARM frame).

- SCARA: open `worlds/scara.wbt` and set the controller of `DEF ARM` to `scara_joint_ctrl`
- Ned: open `worlds/articulated.wbt` and set the controller of `DEF ARM` to `ned_joint_ctrl`

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
# run worlds/scara.wbt and worlds/articulated.wbt once each in Webots with the controllers of Task 1.A
# (they write logs/tcp_measured_*.csv)
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


## Task 4.A: evaluation with FK

```bash
python scripts/evaluate_fk.py     # all 500 test samples per arm, with FK; no simulator needed
```

It prints the 6 position-error and 18 joint-error statistics (mean and std) and saves them to
`logs/eval_task4a_results.json`. The error of a revolute joint is the absolute value of the
wrapped angular difference, so errors of opposite sign do not cancel in the mean; the error of the
prismatic SCARA joint is the squared difference in m².

## Task 4.B: evaluation in Webots

Open `worlds/scara.wbt` and `worlds/articulated.wbt` one after the other, set the controller of
`DEF ARM` to `evaluate_webots` and run the simulation. For each model it moves the arm to the joint
values predicted for the first 100 test samples, waits until the joints have settled, and compares
the measured TCP with the test-set position. The results of both arms are merged into
`logs/eval_task4b_results.json`, and the simulation pauses when it is done.

### Results of Task 4.A and Task 4.B

Position error in mm (mean ± std), network trained on the training sets of the same size:

| Model | Arm | 4.A: FK, 500 samples | 4.B: Webots, 100 samples |
|---|---|---|---|
| small | Ned | 9.73 ± 6.72 | 9.92 ± 8.01 |
| small | SCARA | 4.86 ± 3.30 | 5.02 ± 3.62 |
| medium | Ned | 6.96 ± 3.66 | 7.54 ± 3.78 |
| medium | SCARA | 3.30 ± 2.00 | 3.41 ± 2.04 |
| large | Ned | 5.44 ± 4.00 | 5.43 ± 3.89 |
| large | SCARA | 2.80 ± 1.57 | 2.75 ± 1.47 |

For the Ned, 7 of the first 100 test poses have the TCP below 0.12 m, where the hand touches the
floor in Webots (see Task 1.C).

## Task 5 demo: the network follows the red target sphere

Open `worlds/scara.wbt` or `worlds/articulated.wbt` (they start with `nn_sphere_tracker`), run the
simulation and drag the red sphere (`DEF TARGET`) with the mouse.
Every simulation step the controller feeds the sphere position (ARM frame) to the trained
network (`MODEL_SCALE` in the controller selects small, medium or large) and commands the joints.

- The prediction is checked with our FK before it is used. If it does not reach the target within
  4 cm, the sphere is outside what the network learned (for example out of reach) and the arm keeps
  its last good pose instead of swinging to an extrapolated one; the console then prints
  "Target reachable: NO".
- Limitation of the Ned: close to its base axis, or when the sphere is moved from one side of the
  robot to the other behind it, `joint_1` has to swing over a large part of its range (limit
  -2.8 to 2.8 rad at 1 rad/s) and the network is least accurate there, so the arm lags or holds.

## Reused and AI-assisted code

- From the course repository (Christian Dondrup): the Webots worlds (we set the arm's
  controller and the supervisor flag), the Ned PROTO and meshes in `protos/`, the example
  controllers in `controllers/` (`ned_simple`, `scara_simple`, `tcp_pose`,
  `discover_devices`, `ned_keyboard`, `ned_sphere_ik`) and `scripts/ik_exact.py`,
  `scripts/ik_simple.py`. They are not used by our pipeline except for the TCP lookup of
  `tcp_pose`, which `scripts/webots_arm.py` follows.
- Generative AI was used for parts of the code, debugging and comments, as the coursework
  allows. All code was checked by the group; the details are in the group's AI statement.
