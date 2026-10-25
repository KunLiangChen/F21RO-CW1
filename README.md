See the wiki for a basic tutorial

## Task 1.A: joint controllers (Webots)

Each arm has a Supervisor controller attached to `DEF ARM`; it commands the 3 active
joints, waits until the position sensors settle, then reads the TCP pose (ARM frame).

- SCARA: open `worlds/scara.wbt` (controller `scara_joint_ctrl`)
- Ned: open `worlds/articulated.wbt` (controller `ned_joint_ctrl`)

Joint values are read from `logs/joint_input_scara.csv` / `logs/joint_input_ned.csv`
(rows of `q1,q2,q4` / `q1,q2,q3`, physical units, optional header line). If the file is
missing, built-in demo poses run. Results go to `logs/tcp_measured_<arm>.csv`.
Shared logic: `scripts/webots_arm.py`.
