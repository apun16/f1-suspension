from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Disturbance:
    long_accel: float
    road_front: float
    road_rear: float
    road_vel_front: float
    road_vel_rear: float


class ScenarioGenerator:
    """Produces braking, acceleration, rough road, and bump disturbances."""

    def __init__(self, name: str = "mixed", wheel_delay: float = 0.24, seed: int | None = None):
        self.name = name
        self.wheel_delay = wheel_delay
        self.rng = np.random.default_rng(seed)
        self.phase = float(self.rng.uniform(0.0, 2.0 * np.pi))
        self.bump_shift = float(self.rng.uniform(-0.15, 0.15))

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.phase = float(self.rng.uniform(0.0, 2.0 * np.pi))
        self.bump_shift = float(self.rng.uniform(-0.15, 0.15))

    def sample(self, t: float) -> Disturbance:
        if self.name == "braking":
            return self._braking(t)
        if self.name == "acceleration":
            return self._acceleration(t)
        if self.name == "rough":
            return self._rough(t, long_accel=0.0)
        if self.name == "bumps":
            return self._bumps(t, long_accel=0.0)
        return self._mixed(t)

    def _mixed(self, t: float) -> Disturbance:
        if t < 3.0:
            return self._acceleration(t)
        if t < 6.0:
            return self._braking(t - 3.0)
        if t < 10.0:
            return self._rough(t - 6.0, long_accel=0.8 * np.sin(1.8 * t))
        return self._bumps(t - 10.0, long_accel=0.0)

    def _braking(self, t: float) -> Disturbance:
        accel = -9.5 * self._smooth_pulse(t, 0.8, 2.6, edge=0.18)
        road_f, road_f_v = self._single_bump(t, center=1.55, amp=0.016, width=0.11)
        road_r, road_r_v = self._single_bump(t - self.wheel_delay, center=1.55, amp=0.016, width=0.11)
        return Disturbance(accel, road_f, road_r, road_f_v, road_r_v)

    def _acceleration(self, t: float) -> Disturbance:
        accel = 7.5 * self._smooth_pulse(t, 0.6, 2.8, edge=0.22)
        road_f, road_f_v = self._single_bump(t, center=2.2, amp=-0.010, width=0.18)
        road_r, road_r_v = self._single_bump(t - self.wheel_delay, center=2.2, amp=-0.010, width=0.18)
        return Disturbance(accel, road_f, road_r, road_f_v, road_r_v)

    def _rough(self, t: float, long_accel: float) -> Disturbance:
        road_f = (
            0.010 * np.sin(7.4 * t + self.phase)
            + 0.006 * np.sin(13.0 * t + 0.4 * self.phase)
            + 0.003 * np.sin(23.0 * t)
        )
        road_r = (
            0.010 * np.sin(7.4 * (t - self.wheel_delay) + self.phase)
            + 0.006 * np.sin(13.0 * (t - self.wheel_delay) + 0.4 * self.phase)
            + 0.003 * np.sin(23.0 * (t - self.wheel_delay))
        )
        vel_f = (
            0.010 * 7.4 * np.cos(7.4 * t + self.phase)
            + 0.006 * 13.0 * np.cos(13.0 * t + 0.4 * self.phase)
            + 0.003 * 23.0 * np.cos(23.0 * t)
        )
        vel_r = (
            0.010 * 7.4 * np.cos(7.4 * (t - self.wheel_delay) + self.phase)
            + 0.006 * 13.0 * np.cos(13.0 * (t - self.wheel_delay) + 0.4 * self.phase)
            + 0.003 * 23.0 * np.cos(23.0 * (t - self.wheel_delay))
        )
        return Disturbance(float(long_accel), float(road_f), float(road_r), float(vel_f), float(vel_r))

    def _bumps(self, t: float, long_accel: float) -> Disturbance:
        road_f, vel_f = self._repeated_bumps(t + self.bump_shift)
        road_r, vel_r = self._repeated_bumps(t - self.wheel_delay + self.bump_shift)
        return Disturbance(float(long_accel), road_f, road_r, vel_f, vel_r)

    @staticmethod
    def _smooth_pulse(t: float, start: float, end: float, edge: float) -> float:
        on = 1.0 / (1.0 + np.exp(-(t - start) / edge))
        off = 1.0 / (1.0 + np.exp((t - end) / edge))
        return float(on * off)

    @staticmethod
    def _single_bump(t: float, center: float, amp: float, width: float) -> tuple[float, float]:
        x = (t - center) / width
        height = amp * np.exp(-0.5 * x * x)
        velocity = height * (-(t - center) / (width * width))
        return float(height), float(velocity)

    @staticmethod
    def _repeated_bumps(t: float) -> tuple[float, float]:
        height = 0.0
        velocity = 0.0
        for center in np.arange(0.75, 7.5, 0.72):
            x = (t - center) / 0.075
            bump = 0.024 * np.exp(-0.5 * x * x)
            height += bump
            velocity += bump * (-(t - center) / (0.075 * 0.075))
        return float(height), float(velocity)

