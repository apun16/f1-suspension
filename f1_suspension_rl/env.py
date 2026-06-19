from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from f1_suspension_rl.pid import PIDController
from f1_suspension_rl.scenarios import ScenarioGenerator


@dataclass
class SuspensionParams:
    dt: float = 0.02
    mass: float = 798.0
    pitch_inertia: float = 1650.0
    wheelbase: float = 3.6
    cg_height: float = 0.31
    target_height: float = 0.055
    max_force: float = 12500.0
    passive_k: float = 58000.0
    passive_c: float = 5200.0
    heave_c: float = 2300.0
    pitch_k: float = 18500.0
    pitch_c: float = 4200.0


class F1SuspensionEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        scenario: str = "mixed",
        episode_seconds: float = 12.0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.params = SuspensionParams()
        self.scenario_name = scenario
        self.scenario = ScenarioGenerator(scenario, seed=seed)
        self.max_steps = int(episode_seconds / self.params.dt)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(13,), dtype=np.float32)

        self.front_pid = PIDController()
        self.rear_pid = PIDController()
        self.pitch_pid = PIDController(integral_limit=0.04)
        self.rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    @property
    def half_wheelbase(self) -> float:
        return self.params.wheelbase * 0.5

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.scenario.reset(seed)
        self.step_count = 0
        self.t = 0.0
        self.z = float(self.rng.normal(0.0, 0.004))
        self.z_dot = 0.0
        self.pitch = float(self.rng.normal(0.0, 0.006))
        self.pitch_rate = 0.0
        self.prev_action = np.zeros(6, dtype=np.float32)
        self.last_forces = np.zeros(2, dtype=np.float32)
        self.last_disturbance = self.scenario.sample(0.0)
        self.front_pid.reset()
        self.rear_pid.reset()
        self.pitch_pid.reset()
        return self._obs(), self._info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        p = self.params
        d = self.scenario.sample(self.t)
        self.last_disturbance = d

        gains = self._map_action_to_pid(action)
        heights = self._ride_heights(d)
        rates = self._ride_height_rates(d)

        target = p.target_height + gains["heave_trim"]
        target_pitch = gains["pitch_trim"]

        front_error = target - heights[0]
        rear_error = target - heights[1]
        front_error_rate = -rates[0]
        rear_error_rate = -rates[1]

        front_active = self.front_pid.step(
            front_error, front_error_rate, p.dt, gains["kp"], gains["ki"], gains["kd"]
        )
        rear_active = self.rear_pid.step(
            rear_error, rear_error_rate, p.dt, gains["kp"], gains["ki"], gains["kd"]
        )
        pitch_error = target_pitch - self.pitch
        pitch_force = self.pitch_pid.step(
            pitch_error,
            -self.pitch_rate,
            p.dt,
            gains["pitch_kp"],
            gains["pitch_ki"],
            gains["pitch_kd"],
        )

        front_force = front_active + pitch_force
        rear_force = rear_active - pitch_force

        passive_front = p.passive_k * (p.target_height - heights[0]) + p.passive_c * (-rates[0])
        passive_rear = p.passive_k * (p.target_height - heights[1]) + p.passive_c * (-rates[1])

        front_force = float(np.clip(front_force + passive_front, -p.max_force, p.max_force))
        rear_force = float(np.clip(rear_force + passive_rear, -p.max_force, p.max_force))
        self.last_forces = np.array([front_force, rear_force], dtype=np.float32)

        heave_force = front_force + rear_force - p.heave_c * self.z_dot
        accel_z = heave_force / p.mass

        braking_pitch_torque = p.mass * d.long_accel * p.cg_height
        suspension_torque = self.half_wheelbase * (front_force - rear_force)
        pitch_torque = (
            suspension_torque
            + braking_pitch_torque
            - p.pitch_k * self.pitch
            - p.pitch_c * self.pitch_rate
        )
        pitch_accel = pitch_torque / p.pitch_inertia

        self.z_dot += accel_z * p.dt
        self.z += self.z_dot * p.dt
        self.pitch_rate += pitch_accel * p.dt
        self.pitch += self.pitch_rate * p.dt
        self.t += p.dt
        self.step_count += 1

        reward, reward_terms = self._reward(action)
        terminated = bool(
            abs(self.pitch) > 0.22
            or abs(self.z) > 0.18
            or min(self._ride_heights(d)) < 0.005
            or max(self._ride_heights(d)) > 0.17
        )
        truncated = self.step_count >= self.max_steps

        self.prev_action = action.copy()
        info = self._info()
        info["reward_terms"] = reward_terms
        return self._obs(), reward, terminated, truncated, info

    def _map_action_to_pid(self, action: np.ndarray) -> dict[str, float]:
        scale = lambda x, lo, hi: lo + (hi - lo) * ((float(x) + 1.0) * 0.5)
        return {
            "kp": 82000.0 * scale(action[0], 0.55, 1.40),
            "kd": 7200.0 * scale(action[1], 0.45, 1.55),
            "ki": 2600.0 * scale(action[2], 0.00, 0.75),
            "pitch_kp": 48000.0 * scale(action[3], 0.45, 1.65),
            "pitch_kd": 5200.0 * scale(action[1], 0.45, 1.45),
            "pitch_ki": 900.0 * scale(action[2], 0.00, 0.55),
            "pitch_trim": 0.016 * float(action[4]),
            "heave_trim": 0.006 * float(action[5]),
        }

    def _ride_heights(self, disturbance=None) -> np.ndarray:
        d = disturbance or self.last_disturbance
        front = self.params.target_height + self.z + self.pitch * self.half_wheelbase - d.road_front
        rear = self.params.target_height + self.z - self.pitch * self.half_wheelbase - d.road_rear
        return np.array([front, rear], dtype=np.float64)

    def _ride_height_rates(self, disturbance=None) -> np.ndarray:
        d = disturbance or self.last_disturbance
        front = self.z_dot + self.pitch_rate * self.half_wheelbase - d.road_vel_front
        rear = self.z_dot - self.pitch_rate * self.half_wheelbase - d.road_vel_rear
        return np.array([front, rear], dtype=np.float64)

    def _contact_quality(self, disturbance=None) -> np.ndarray:
        d = disturbance or self.last_disturbance
        heights = self._ride_heights(d)
        static_load = self.params.mass * 9.81 * 0.5
        load_transfer = -self.params.mass * d.long_accel * self.params.cg_height / self.params.wheelbase
        front_load = static_load + load_transfer + 0.18 * self.last_forces[0]
        rear_load = static_load - load_transfer + 0.18 * self.last_forces[1]
        loads = np.array([front_load, rear_load], dtype=np.float64)

        load_quality = np.clip(loads / static_load, 0.0, 1.22)
        load_quality = 1.0 - np.clip(np.abs(load_quality - 1.0) / 0.55, 0.0, 1.0)
        height_quality = np.exp(-((heights - self.params.target_height) / 0.045) ** 2)
        bottoming = np.where(heights < 0.020, 0.35, 1.0)
        return np.clip(load_quality * height_quality * bottoming, 0.0, 1.0)

    def _reward(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        heights = self._ride_heights(self.last_disturbance)
        contact = self._contact_quality(self.last_disturbance)
        height_error = np.mean(((heights - self.params.target_height) / 0.018) ** 2)
        pitch_error = (self.pitch / 0.045) ** 2
        pitch_rate_error = (self.pitch_rate / 0.85) ** 2
        contact_loss = np.mean((1.0 - contact) ** 2)
        accel_factor = min(abs(self.last_disturbance.long_accel) / 9.5, 1.0)
        rocking = accel_factor * (0.65 * pitch_error + 0.35 * pitch_rate_error)
        force_cost = np.mean((self.last_forces / self.params.max_force) ** 2)
        action_smoothness = np.mean((action - self.prev_action) ** 2)

        penalty = (
            0.42 * height_error
            + 0.20 * pitch_error
            + 0.10 * pitch_rate_error
            + 1.15 * contact_loss
            + 0.28 * rocking
            + 0.025 * force_cost
            + 0.015 * action_smoothness
        )
        reward = 1.0 - penalty
        terms = {
            "height_error": float(height_error),
            "pitch_error": float(pitch_error),
            "pitch_rate_error": float(pitch_rate_error),
            "contact_loss": float(contact_loss),
            "rocking": float(rocking),
            "force_cost": float(force_cost),
            "reward": float(reward),
        }
        return float(reward), terms

    def _obs(self) -> np.ndarray:
        heights = self._ride_heights(self.last_disturbance)
        rates = self._ride_height_rates(self.last_disturbance)
        contact = self._contact_quality(self.last_disturbance)
        obs = np.array(
            [
                (heights[0] - self.params.target_height) / 0.06,
                (heights[1] - self.params.target_height) / 0.06,
                np.mean(heights - self.params.target_height) / 0.06,
                self.pitch / 0.10,
                self.pitch_rate / 2.0,
                self.z_dot / 1.5,
                rates[0] / 1.8,
                rates[1] / 1.8,
                contact[0] * 2.0 - 1.0,
                contact[1] * 2.0 - 1.0,
                self.last_disturbance.long_accel / 10.0,
                self.last_disturbance.road_front / 0.05,
                self.last_disturbance.road_rear / 0.05,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -5.0, 5.0)

    def _info(self) -> dict:
        heights = self._ride_heights(self.last_disturbance)
        contact = self._contact_quality(self.last_disturbance)
        return {
            "time": self.t,
            "front_height": float(heights[0]),
            "rear_height": float(heights[1]),
            "pitch": float(self.pitch),
            "pitch_rate": float(self.pitch_rate),
            "front_contact": float(contact[0]),
            "rear_contact": float(contact[1]),
            "front_force": float(self.last_forces[0]),
            "rear_force": float(self.last_forces[1]),
            "long_accel": float(self.last_disturbance.long_accel),
            "road_front": float(self.last_disturbance.road_front),
            "road_rear": float(self.last_disturbance.road_rear),
        }

