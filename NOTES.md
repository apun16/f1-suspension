# Notes

I trained a SAC policy to adapt a PID active suspension controller for a simplified Formula 1 car, with the reward focused only on aero ride-height stability and tire contact patch quality.

## Why PID + RL

Pure RL force control is hard to trust and easy to destabilize. PID is stable and interpretable, but fixed gains are not ideal when the car moves between braking, acceleration, rough track, and repeated bumps. The hybrid setup gives each method a useful role:

- PID handles fast feedback control
- SAC learns how to retune the controller as conditions change

## What SAC controls

SAC does not directly push the suspension. It outputs:

- height PID gain scale
- height damping scale
- integral gain scale
- pitch controller gain scale
- pitch setpoint trim
- heave setpoint trim

Then PID turns those adaptive parameters into front/rear active suspension forces.

## Reward explanation

The reward is narrow:

- penalty if front or rear floor height deviates from the optimum aero ride height
- penalty for pitch and pitch rate
- stronger pitch penalty during braking/acceleration because that is when rocking matters most
- penalty when the tire contact quality proxy drops
- tiny force/smoothness penalties to avoid unrealistic thrashing

## Demos

Four SAC+PID demos:

- sudden braking
- sudden acceleration
- rough track segment
- repeated bumps

## Failure modes 

- Over-stiff suspension: holds ride height but destroys contact patch
- Over-aggressive gains: oscillation after bumps
- Reward hacking: trims the target height to reduce immediate error but worsens contact
- Under-trained policy: behaves like a noisy PID tuner and performs worse than fixed PID