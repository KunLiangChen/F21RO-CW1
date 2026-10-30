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

Result (position error between Webots and FK; 208 poses for the SCARA, 206 for the Ned
because two corner poses below the Ned floor limit of Task 2 are left out): SCARA mean 0.34 mm,
max 1.04 mm. Ned mean 0.20 mm, max 0.70 mm.

Floor contact of the Ned: measured with the same joints at TCP heights from 0.08 to 0.18 m, the
error is about 0.2 mm for z >= 0.12 m, 1.5 mm for 0.11 to 0.12 m and 5 to 8 mm below 0.10 m
(the hand touches the floor). An earlier run without a floor limit had a Ned maximum of 7.8 mm
for this reason; Task 2 now removes these poses.


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

All other joints use their full motor range. For the Ned, poses whose TCP is below
`NED_MIN_TCP_Z` = 0.12 m are rejected while sampling (rejection sampling with the FK of Task 1.B),
because the hand then touches the Webots floor and does not reach the FK position (about 1.5 mm
error at 0.11 to 0.12 m and up to 8 mm lower, see Task 1.C). The joint ranges, and so the
normalisation bounds, stay the same; only the combinations that reach the floor are missing.

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
- The grid has 32 candidates per size (4 layer layouts, relu/tanh, alpha 1e-4/1e-3, learning
  rate 1e-3/3e-3; the layouts grow with the data size). Every candidate is trained with
  3 random seeds (42, 43, 44) and the scores are averaged, so a lucky initialisation does
  not win. The seed matters: with the same configuration, the validation position error of
  the small model varies between 5.9 and 8.6 mm and one seed of the medium model had a 7x
  larger SCARA shaft error than the others. The final network is therefore trained with each
  of the 3 seeds and the one with the lowest validation position error is kept
  (`final_model_seed` in the log).
- The score of a candidate is the mean TCP position error on the validation set: the predicted
  joints are de-normalised and passed through the FK of Task 1.B, and the mean error of the
  Ned and the SCARA count equally. The plain validation MSE of the merged set is dominated by
  the SCARA, which has a 3x larger MSE, so it hardly looks at the Ned.
- Training stops by itself when the training loss has not improved by more than `tol=1e-7`
  for 50 epochs in a row (`max_iter=3000` is never reached; the models train for 210 to
  1600 epochs). scikit-learn's default `tol=1e-4` is too coarse for losses of about 1e-3 and
  stopped the training after 60 to 100 epochs, with up to 3x higher validation error.
- Random seeds: 42, 43, 44 (see above). Results are reproducible with the same scikit-learn version.
- Outputs: `models/model_<size>.pkl` and `logs/log_<size>.json` (best hyperparameters,
  validation metrics, the mean position error of all 32 candidates, loss curve). The `.pkl` files depend on the scikit-learn version
  (1.8.0 here); with a different version, rerun `python scripts/train_ann.py`.

Selected hyperparameters and validation errors (normalised joint space):

| Model | Train / val samples | Hidden layers | Activation | alpha | Learning rate | Seed | Val MSE | Val MAE | Val position error |
|---|---|---|---|---|---|---|---|---|---|
| small | 2000 / 500 | (128, 128) | relu | 0.0001 | 0.001 | 42 | 2.04e-03 | 2.17e-02 | 5.89 mm |
| medium | 8000 / 2000 | (128, 128, 64) | relu | 0.0001 | 0.001 | 43 | 7.72e-04 | 1.50e-02 | 4.08 mm |
| large | 16000 / 4000 | (256, 256, 128) | relu | 0.0001 | 0.001 | 43 | 9.61e-04 | 1.54e-02 | 3.03 mm |

(The position error is the mean of the Ned and the SCARA error, the quantity the search minimises.)
The seed is chosen by this position error, not by the MSE, so the MSE of a model is not
necessarily the lowest of its three seeds (large: 9.61e-04 with seed 43, 6.26e-04 with seed 42).


## Task 4.A: evaluation with FK

```bash
python scripts/evaluate_fk.py     # all 500 test samples per arm, with FK; no simulator needed
```

It prints the 6 position-error and 18 joint-error statistics (mean and std), plus the position error
of the first 100 test samples (the ones of Task 4.B, for a comparison on the same inputs), and saves them to
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

| Model | Arm | 4.A: FK, 500 samples | 4.A: FK, first 100 | 4.B: Webots, first 100 |
|---|---|---|---|---|
| small | Ned | 8.39 ± 6.01 | 9.39 ± 6.43 | 9.42 ± 6.43 |
| small | SCARA | 3.31 ± 1.93 | 3.34 ± 1.86 | 3.37 ± 1.85 |
| medium | Ned | 5.91 ± 4.24 | 5.96 ± 4.37 | 5.97 ± 4.37 |
| medium | SCARA | 2.37 ± 1.43 | 2.49 ± 1.41 | 2.56 ± 1.44 |
| large | Ned | 4.46 ± 3.26 | 4.50 ± 3.18 | 4.43 ± 3.13 |
| large | SCARA | 1.82 ± 1.00 | 1.81 ± 0.86 | 1.90 ± 0.81 |

On the same 100 samples, 4.A and 4.B differ by at most 0.1 mm on average. No test pose of the
Ned is below the floor limit of Task 2, so the arm does not touch the floor in Webots.

## Task 5 demo: the network follows the red target sphere

Open `worlds/scara.wbt` or `worlds/articulated.wbt` (they start with `nn_sphere_tracker`), run the
simulation and drag the red sphere (`DEF TARGET`) with the mouse.
Every simulation step the controller feeds the sphere position (ARM frame) to the trained
network (`MODEL_SCALE` in the controller selects small, medium or large) and commands the joints.

- The 3D view shows, in red, the model, the target position, the tracking error and whether the
  target is reachable, so a failed target is visible in a screen recording.
- The prediction is checked with our FK before it is used. If it does not reach the target within
  4 cm, the sphere is outside what the network learned (for example out of reach) and the arm keeps
  its last good pose instead of swinging to an extrapolated one; the console then prints
  "Target reachable: NO".
- A Ned target below z = 0.12 m (`NED_MIN_TCP_Z`, the floor limit of the Ned data) is refused the
  same way, because the hand would touch the floor and the network never saw such poses; the 3D
  view then says "target below floor limit".
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
