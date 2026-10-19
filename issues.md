# Automation & Evaluation Known Issues

This document records the architectural, algorithmic, and environment issues identified during training and evaluation of the autonomous Ackermann vehicle in Gazebo Harmonic and RLlib.

---

## 1. Policy Degeneration into Crawling Local Optimum (Halting Trap)

### Symptom
During evaluation in both analytical simulation and Gazebo 3D simulation (`eval_car.py`), the car commands initial steering and acceleration, but rapidly decays throttle down to near zero:
- **Step 1:** Throttle `+0.36`, Steering `-0.34` (Speed: `1.18 m/s`)
- **Step 50:** Throttle `+0.21`, Steering `-0.18` (Speed: `0.72 m/s`)
- **Step 100:** Throttle `+0.11`, Steering `-0.06` (Speed: `0.34 m/s`)
- **Step 150+:** Throttle `+0.02`, Steering `-0.05` (Speed: `0.09 m/s` — near standstill)

The car appears incompetent or broken compared to earlier evaluation videos recorded during training.

### Root Cause
1. **Reward Landscape Imbalance:** In `src/translate/config.py`, `COLLISION_PENALTY = 500.0`, whereas step penalties and crawl penalties (`CRAWL_PENALTY_WEIGHT = 0.3`) are orders of magnitude smaller.
2. **PPO Local Optimum:** As training progressed beyond iteration 15, the optimizer discovered that aggressively driving the track risks incurring the massive `-500.0` collision penalty. Crawling forward at $0.09\text{ m/s}$ avoids all collisions and maximizes survival time, collapsing the policy into a risk-averse failure mode.
3. **Training Log Evidence:**
   - **Early Peak (Archive Trial 15):**
     - Iteration 10: `eval_iter_010_ep_00_collision_rew_+0447.mp4` (**+447 reward**) — fast, confident driving.
     - Iteration 15: `eval_iter_015_ep_00_collision_rew_+0441.mp4` (**+441 reward**).
   - **Late Collapse (Latest Run ending Oct 18 23:15):**
     - Iteration 10: `eval_iter_010_ep_00_timeout_rew_-3721.mp4` (**-3721 reward**)
     - Iteration 15: `eval_iter_015_ep_00_timeout_rew_-4689.mp4` (**-4689 reward**)
     - Iteration 30: `eval_iter_030_ep_00_timeout_rew_-4918.mp4` (**-4918 reward**)
     - Iteration 50: `eval_iter_050_ep_00_timeout_rew_-2512.mp4` (**-2512 reward**)

### Action Items
- [ ] Increase `CRAWL_PENALTY_WEIGHT` or enforce a hard minimum cruise speed threshold (`MIN_CRUISE_SPEED_MPS = 0.5`) in `src/translate/config.py` so that halting/crawling accumulates worse penalties than collisions.
- [ ] Implement early stopping or return-to-best policy weighting when evaluation returns turn negative.

---

## 2. Checkpoint Retention Strategy (Overwriting Best Weights)

### Symptom
Only a single model checkpoint exists on disk: `models/car_rllib/final_checkpoint`. High-performing policies (such as the +447 reward policy from iterations 10–15) cannot be evaluated because their weights were not preserved.

### Root Cause
In `src/v2/train_car.py`, `algo.save_checkpoint()` was only called once at the conclusion of all iterations:
```python
save_dir = CHECKPOINTS_DIR / "final_checkpoint"
algo.save_checkpoint(str(save_dir))
```
Because the training run collapsed in its later half, `final_checkpoint` saved the degraded iteration 50 weights instead of the peak weights.

### Action Items
- [ ] Update `src/v2/train_car.py` to save `checkpoint_best` whenever `mean_return` exceeds the previous maximum.
- [ ] Save periodic checkpoint snapshots (`checkpoint_iter_{iter:03d}`) every $N$ iterations (e.g., every 5 or 10 iterations) to retain training checkpoints.

---

## 3. Gazebo GUI / OGRE Rendering Missing on CPU Host

### Symptom
Launching the Gazebo GUI client (`gz sim -g`) on machines without a dedicated NVIDIA GPU or without the Vulkan OGRE plugin failed with:
```
[Err] [Ogre2RenderEngine.cc:714] Unable to find Ogre Plugin[/usr/lib/x86_64-linux-gnu/OGRE-2.3/OGRE/RenderSystem_Vulkan]. Rendering will not be possible. Make sure you have installed OGRE properly.
```

### Root Cause
Gazebo Harmonic defaults to the OGRE2 Vulkan render engine when opening a 3D GUI window. On hosts with only integrated graphics or CPU rendering, Vulkan is unavailable.

### Resolution & Workaround
- Run evaluation headless or use software rasterization:
  - Export `LIBGL_ALWAYS_SOFTWARE=1` and `MESA_LOADER_DRIVER_OVERRIDE=llvmpipe` when running on CPU-only machines.
  - Alternatively, evaluate using `eval_car.py --sim-mode analytical --record-video` or Gazebo server-only (`gz sim -s -r`) without GUI.

---

## 4. Gz-ROS 2 Discovery & Topic Partition Isolation

### Symptom
The ROS 2 parameter bridge failed to receive `/ray/lidar_points` or transmit `/cmd_vel` to Gazebo Harmonic, causing `eval_car.py` to time out waiting for LiDAR point clouds (`RuntimeError: Gazebo 3D LiDAR clouds unavailable after 3.0s gate`).

### Root Cause
Gazebo Harmonic uses `GZ_PARTITION` and `GZ_IP` to isolate discovery domains. When the Gazebo server, ROS 2 parameter bridge, and GUI client ran under differing or unset partition environments, topics were isolated and unable to bridge.

### Resolution
- Standardized `GZ_PARTITION="los_podadores"` and `GZ_IP="127.0.0.1"` across `scripts/run_eval_gazebo.sh`, `scripts/gz`, and `scripts/ros2`.

---

## 5. Ackermann Actuation Plugin Compatibility in Model SDF

### Symptom
Gazebo simulation ignored `/cmd_vel` steering and velocity commands sent by the evaluation script; wheels did not steer and the vehicle remained stationary.

### Root Cause
The tractor SDF (`src/v2/model/tractor/model.sdf`) relied on legacy custom GzDRL plugins (`ackermann_plugin`) rather than Gazebo Harmonic's standard physics plugins.

### Resolution
- Replaced the custom plugin definition in `src/v2/model/tractor/model.sdf` with Gazebo Harmonic native `gz-sim-ackermann-steering-system` listening to `/cmd_vel` with matched wheel separation (`0.48 m`) and wheel base (`0.65 m`).
