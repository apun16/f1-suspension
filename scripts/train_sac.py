from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from f1_suspension_rl.env import F1SuspensionEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAC to adapt a PID active suspension controller.")
    parser.add_argument("--timesteps", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-path", type=Path, default=Path("models/sac_hybrid_suspension"))
    parser.add_argument("--log-dir", type=Path, default=Path("runs/sac_hybrid"))
    args = parser.parse_args()

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(F1SuspensionEnv(scenario="mixed", seed=args.seed), str(args.log_dir / "train_monitor.csv"))
    eval_env = Monitor(F1SuspensionEnv(scenario="mixed", seed=args.seed + 1), str(args.log_dir / "eval_monitor.csv"))

    model = SAC(
        "MlpPolicy",
        env,
        seed=args.seed,
        learning_rate=3e-4,
        buffer_size=120_000,
        batch_size=256,
        learning_starts=2_500,
        tau=0.02,
        gamma=0.985,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        policy_kwargs={"net_arch": [128, 128]},
        verbose=1,
        tensorboard_log=str(args.log_dir / "tensorboard"),
    )

    checkpoint = CheckpointCallback(
        save_freq=20_000,
        save_path=str(args.model_path.parent),
        name_prefix="sac_hybrid_checkpoint",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.model_path.parent / "best"),
        log_path=str(args.log_dir / "eval"),
        eval_freq=10_000,
        deterministic=True,
    )

    model.learn(total_timesteps=args.timesteps, callback=[checkpoint, eval_callback])
    model.save(args.model_path)
    print(f"Saved SAC hybrid policy to {args.model_path}.zip")


if __name__ == "__main__":
    main()
