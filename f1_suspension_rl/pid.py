from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PIDState:
    integral: float = 0.0
    previous_error: float = 0.0


class PIDController:
    """Small PID helper with anti-windup clamping."""

    def __init__(self, integral_limit: float = 0.08) -> None:
        self.state = PIDState()
        self.integral_limit = integral_limit

    def reset(self) -> None:
        self.state = PIDState()

    def step(
        self,
        error: float,
        error_rate: float,
        dt: float,
        kp: float,
        ki: float,
        kd: float,
    ) -> float:
        self.state.integral += error * dt
        self.state.integral = float(
            np.clip(self.state.integral, -self.integral_limit, self.integral_limit)
        )
        self.state.previous_error = error
        return (kp * error) + (ki * self.state.integral) + (kd * error_rate)

