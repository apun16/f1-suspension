from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import SAC

from f1_suspension_rl.eval import FixedPIDPolicy, rollout_policy, summarize_rollout


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PID baseline or SAC+PID policy.")
    parser.add_argument("--model-path", type=Path, default=Path("models/sac_hybrid_suspension.zip"))
    parser.add_argument("--baseline", action="store_true", help="Use fixed PID gains instead of SAC.")
    parser.add_argument("--scenario", default="braking", choices=["braking", "acceleration", "rough", "bumps", "mixed"])
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=Path, default=Path("results/metrics.json"))
    args = parser.parse_args()

    policy = FixedPIDPolicy() if args.baseline else SAC.load(args.model_path)
    rollout = rollout_policy(policy, scenario=args.scenario, seed=args.seed, episode_seconds=args.seconds)
    summary = summarize_rollout(rollout)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
