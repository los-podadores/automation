# Assumptions — Gazebo-truth tractor (v2)

## GoalState
GoalState is 6 floats plus mission phase. IRL it requires centimeter-level global
pose via RTK-GPS plus a surveyed site map providing waypoints and
`corridor_half_widths`. Without both, no field test — replan sensor-relative instead.

## Privileged reward
Progress, centering, heading, and collision derive from perfect GoalState plus
`/model/tractor/contacts`. These are simulation-only claims. Run a sensor-noise
ablation eval before any field test.

## Legacy `src/rl` scope
The `src/rl/*` mower coverage env is excluded from the Gazebo-truth gate. No Gazebo
port is planned for it under this audit.

## Single world
Training uses the generated SDF+JSON pair. Regeneration requires a Gazebo restart.
The default track is `src/v2/model/procedural_track.json`.

## Sensors
RL observations use the 2×3D lidars (front/rear) only. Camera and `translate`
fusion outputs are excluded from RL obs until a new plan wires them in.
