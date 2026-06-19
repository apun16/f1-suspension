from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from f1_suspension_rl.env import F1SuspensionEnv


@dataclass
class Rollout:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    infos: list[dict]


class FixedPIDPolicy:
    def predict(self, obs, deterministic: bool = True):
        batch = np.asarray(obs)
        if batch.ndim == 2:
            return np.zeros((batch.shape[0], 6), dtype=np.float32), None
        return np.zeros(6, dtype=np.float32), None


def rollout_policy(
    policy,
    scenario: str,
    seed: int = 7,
    episode_seconds: float = 8.0,
    settled_start: bool = True,
) -> Rollout:
    env = F1SuspensionEnv(scenario=scenario, seed=seed, episode_seconds=episode_seconds)
    obs, _ = env.reset(seed=seed)
    if settled_start:
        env.z = 0.0
        env.z_dot = 0.0
        env.pitch = 0.0
        env.pitch_rate = 0.0
        env.prev_action[:] = 0.0
        env.last_forces[:] = 0.0
        env.front_pid.reset()
        env.rear_pid.reset()
        env.pitch_pid.reset()
        obs = env._obs()
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    infos: list[dict] = []

    done = False
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append(obs)
        actions.append(np.asarray(action, dtype=np.float32))
        rewards.append(float(reward))
        infos.append(info)
        done = terminated or truncated

    return Rollout(
        observations=np.asarray(observations),
        actions=np.asarray(actions),
        rewards=np.asarray(rewards),
        infos=infos,
    )


def summarize_rollout(rollout: Rollout) -> dict[str, float]:
    infos = rollout.infos
    front_h = np.array([i["front_height"] for i in infos])
    rear_h = np.array([i["rear_height"] for i in infos])
    pitch = np.array([i["pitch"] for i in infos])
    front_c = np.array([i["front_contact"] for i in infos])
    rear_c = np.array([i["rear_contact"] for i in infos])
    target = 0.055
    return {
        "mean_reward": float(np.mean(rollout.rewards)),
        "min_reward": float(np.min(rollout.rewards)),
        "front_ride_height_rmse_mm": float(np.sqrt(np.mean((front_h - target) ** 2)) * 1000.0),
        "rear_ride_height_rmse_mm": float(np.sqrt(np.mean((rear_h - target) ** 2)) * 1000.0),
        "max_abs_pitch_deg": float(np.max(np.abs(pitch)) * 180.0 / np.pi),
        "mean_contact_quality": float(np.mean([front_c, rear_c])),
        "min_contact_quality": float(np.min([front_c, rear_c])),
    }
