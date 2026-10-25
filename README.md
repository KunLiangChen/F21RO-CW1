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


## Task 1.C: FK vs Webots precision

```bash
python evaluate_fk_webots.py generate   # 8 corner + 200 random joint sets per arm (seed 2026)
# run worlds/scara.wbt and worlds/articulated.wbt once each in Webots (controllers write logs/tcp_measured_*.csv)
python evaluate_fk_webots.py report     # mean/std/max position error, saved to logs/fk_vs_webots.json
```
