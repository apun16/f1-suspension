from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from stable_baselines3 import SAC

from f1_suspension_rl.eval import FixedPIDPolicy, rollout_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot rollout telemetry.")
    parser.add_argument("--model-path", type=Path, default=Path("models/sac_hybrid_suspension.zip"))
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--scenario", default="braking", choices=["braking", "acceleration", "rough", "bumps", "mixed"])
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--output", type=Path, default=Path("results/rollout.png"))
    args = parser.parse_args()

    policy = FixedPIDPolicy() if args.baseline else SAC.load(args.model_path)
    rollout = rollout_policy(policy, args.scenario, episode_seconds=args.seconds)
    t = [info["time"] for info in rollout.infos]
    front = [1000.0 * info["front_height"] for info in rollout.infos]
    rear = [1000.0 * info["rear_height"] for info in rollout.infos]
    pitch = [info["pitch"] * 180.0 / 3.141592653589793 for info in rollout.infos]
    front_contact = [info["front_contact"] for info in rollout.infos]
    rear_contact = [info["rear_contact"] for info in rollout.infos]
    accel = [info["long_accel"] for info in rollout.infos]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, front, label="front")
    axes[0].plot(t, rear, label="rear")
    axes[0].axhline(55, color="black", linestyle="--", linewidth=1, label="target")
    axes[0].set_ylabel("ride height mm")
    axes[0].legend(loc="upper right")

    axes[1].plot(t, pitch)
    axes[1].set_ylabel("pitch deg")

    axes[2].plot(t, front_contact, label="front")
    axes[2].plot(t, rear_contact, label="rear")
    axes[2].set_ylabel("contact")
    axes[2].set_ylim(0, 1.05)
    axes[2].legend(loc="lower right")

    axes[3].plot(t, accel)
    axes[3].set_ylabel("long accel m/s^2")
    axes[3].set_xlabel("time s")

    fig.suptitle(f"{'Fixed PID' if args.baseline else 'SAC + PID'} on {args.scenario}")
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
