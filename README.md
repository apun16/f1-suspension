# F1 Active Suspension: SAC-Tuned PID Control

This project trains a reinforcement learning policy to control a Formula 1-inspired active suspension system. The key design choice is that SAC and PID are used together:

- PID is the low-level stabilizer that turns ride-height and pitch errors into front/rear suspension forces.
- SAC is the adaptive layer that tunes the PID gains and small setpoint trims while the car experiences braking, acceleration, rough track segments, and repeated bumps.

The reward is intentionally narrow: it only rewards aerodynamic platform stability and tire contact quality. It penalizes deviation from optimum aerodynamic ride height, front-to-rear rocking under braking/acceleration, and loss of tire contact patch.

## Project hypothesis

F1 aerodynamic floors are highly ride-height sensitive. A controller that keeps the floor close to an optimal platform while preserving contact patch quality should produce a more stable car than a fixed PID controller under changing load transfer and track disturbances.

The project uses a simplified half-car model, not a full vehicle dynamics simulator. That is deliberate: the goal is a clear, inspectable control experiment rather than a black-box racing sim.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Train SAC

```bash
python3 scripts/train_sac.py --timesteps 120000
```

For a quicker smoke run:

```bash
python3 scripts/train_sac.py --timesteps 5000
```

The final model is saved to:

```text
models/sac_hybrid_suspension.zip
```

## Generate required demo videos

After training:

```bash
python3 scripts/make_demos.py --include-baseline
```

This creates videos for:

- `videos/sac_pid_sudden_braking.mp4`
- `videos/sac_pid_sudden_acceleration.mp4`
- `videos/sac_pid_rough_track.mp4`
- `videos/sac_pid_repeated_bumps.mp4`
- optional fixed-PID comparison videos
- `videos/summary_metrics.json`

I generated the current demo set with:

```bash
python3 scripts/train_sac.py --timesteps 10000 --model-path models/sac_hybrid_suspension --log-dir runs/sac_hybrid
python3 scripts/make_demos.py --model-path models/sac_hybrid_suspension.zip --include-baseline --seconds 8 --output-dir videos
```

For a stronger final policy, run a longer training job, for example:

```bash
python3 scripts/train_sac.py --timesteps 120000 --model-path models/sac_hybrid_suspension --log-dir runs/sac_hybrid_long
python3 scripts/make_demos.py --model-path models/sac_hybrid_suspension.zip --include-baseline --seconds 8 --output-dir videos
```

## Evaluate one scenario

```bash
python3 scripts/evaluate.py --scenario braking
python3 scripts/evaluate.py --baseline --scenario braking
```

Plot telemetry:

```bash
python3 scripts/plot_rollout.py --scenario rough --output results/rough_sac_pid.png
python3 scripts/plot_rollout.py --baseline --scenario rough --output results/rough_fixed_pid.png
```

## Environment

The environment is a simplified half-car suspension model:

- front and rear ride height
- chassis heave and pitch
- pitch rate and heave velocity
- load transfer during braking/acceleration
- road input at the front and delayed rear axle
- tire contact patch quality proxy

Observation includes ride-height errors, pitch state, contact quality, longitudinal acceleration, and road disturbances.

SAC action has six continuous values:

```text
[height_Kp_scale, height_Kd_scale, height_Ki_scale, pitch_gain_scale, pitch_trim, heave_trim]
```

The PID layer then calculates active front/rear suspension forces.

## Reward

The reward is:

```text
reward = 1
  - 0.42 * ride_height_error
  - 0.20 * pitch_error
  - 0.10 * pitch_rate_error
  - 1.15 * contact_patch_loss
  - 0.28 * acceleration_weighted_rocking
  - 0.025 * force_cost
  - 0.015 * action_smoothness
```

This directly matches the project target:

- aerodynamic stability: front/rear floor height near optimum
- anti-rocking: pitch and pitch-rate penalties, especially under braking/acceleration
- mechanical grip: contact patch quality penalty
- physical plausibility: small force and smoothness penalties

## Presentation structure

1. Explain the control problem: active suspension should protect floor aero height and tire contact.
2. Show the hybrid controller: SAC tunes PID, PID applies forces.
3. Show fixed PID baseline on braking or rough track.
4. Show SAC+PID on sudden braking, sudden acceleration, rough track, and repeated bumps.
5. Discuss failure modes: unstable gains, over-stiff behavior, reward hacking where contact quality is sacrificed to hold ride height.
6. Close with what worked: hybrid control is easier to stabilize and explain than pure RL force control.

## Good demo clips to record

- sudden braking: nose dive and pitch-rate suppression
- sudden acceleration: rear squat and platform recovery
- rough track: contact patch stability
- repeated bumps: generalization to repeated disturbances
- failure clip: early training policy that overreacts or oscillates

## Research notes

This project is inspired by three real control ideas:

- PID controllers are widely used in engineering because they are simple, interpretable feedback controllers.
- Active suspension is naturally a control problem because the suspension must respond to disturbances while satisfying competing objectives.
- F1 floor aerodynamics are sensitive to ride height and pitch, so platform control is directly tied to aerodynamic stability.

The implementation keeps the physics minimal enough to audit. The point is not to claim a production-grade F1 simulation; the point is to study how a learned adaptive controller can improve a classical PID controller on a focused motorsport control problem.
