from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import SAC

from f1_suspension_rl.eval import FixedPIDPolicy, rollout_policy, summarize_rollout
from f1_suspension_rl.visualizer import write_video


SCENARIOS = {
    "braking": "sudden_braking",
    "acceleration": "sudden_acceleration",
    "rough": "rough_track",
    "bumps": "repeated_bumps",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Render demo videos for all required test scenarios.")
    parser.add_argument("--model-path", type=Path, default=Path("models/sac_hybrid_suspension.zip"))
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", type=Path, default=Path("videos"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    policies = {"sac_pid": SAC.load(args.model_path)}
    if args.include_baseline:
        policies["fixed_pid"] = FixedPIDPolicy()

    summaries = {}
    for policy_name, policy in policies.items():
        for scenario, label in SCENARIOS.items():
            rollout = rollout_policy(policy, scenario=scenario, seed=args.seed, episode_seconds=args.seconds)
            video_path = args.output_dir / f"{policy_name}_{label}.mp4"
            write_video(rollout, video_path)
            summaries[f"{policy_name}_{label}"] = summarize_rollout(rollout)
            print(f"Wrote {video_path}")

    summary_path = args.output_dir / "summary_metrics.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
