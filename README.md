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


## Task 2: sampled joint ranges and normalisation

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

