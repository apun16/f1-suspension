from __future__ import annotations

import math
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from f1_suspension_rl.eval import Rollout


class PyBulletSuspensionRenderer:
    def __init__(self, width: int = 1280, height: int = 720) -> None:
        import pybullet as p

        self.p = p
        self.width = width
        self.height = height
        self.client = p.connect(p.DIRECT)
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        self._build_scene()

    def close(self) -> None:
        self.p.disconnect(self.client)

    def _build_scene(self) -> None:
        p = self.p
        self.chassis = self._body_box([3.7, 0.52, 0.13], [0.05, 0.05, 0.05, 1.0])
        self.floor = self._body_box([3.95, 0.72, 0.035], [0.08, 0.08, 0.08, 1.0])
        self.front_wheel = self._body_sphere(0.18, [0.02, 0.02, 0.02, 1.0])
        self.rear_wheel = self._body_sphere(0.18, [0.02, 0.02, 0.02, 1.0])
        self.front_patch = self._body_box([0.46, 0.12, 0.018], [0.0, 0.7, 0.25, 1.0])
        self.rear_patch = self._body_box([0.46, 0.12, 0.018], [0.0, 0.7, 0.25, 1.0])
        self.track = self._body_box([7.0, 1.2, 0.02], [0.25, 0.25, 0.25, 1.0])

    def _body_box(self, half_extents, rgba):
        p = self.p
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=rgba,
            physicsClientId=self.client,
        )
        return p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual, physicsClientId=self.client)

    def _body_sphere(self, radius, rgba):
        p = self.p
        visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=rgba,
            physicsClientId=self.client,
        )
        return p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual, physicsClientId=self.client)

    def frame(self, info: dict) -> np.ndarray:
        p = self.p
        pitch = info["pitch"]
        z = 0.55 + 0.5 * ((info["front_height"] + info["rear_height"]) - 0.11)
        quat = p.getQuaternionFromEuler([0.0, pitch, 0.0])
        p.resetBasePositionAndOrientation(self.chassis, [0, 0, z], quat, physicsClientId=self.client)
        p.resetBasePositionAndOrientation(self.floor, [0, 0, z - 0.14], quat, physicsClientId=self.client)

        front_x = 1.8
        rear_x = -1.8
        front_road = info["road_front"]
        rear_road = info["road_rear"]
        p.resetBasePositionAndOrientation(
            self.front_wheel, [front_x, 0.34, 0.18 + front_road], [0, 0, 0, 1], physicsClientId=self.client
        )
        p.resetBasePositionAndOrientation(
            self.rear_wheel, [rear_x, 0.34, 0.18 + rear_road], [0, 0, 0, 1], physicsClientId=self.client
        )

        front_alpha = max(0.15, min(1.0, info["front_contact"]))
        rear_alpha = max(0.15, min(1.0, info["rear_contact"]))
        p.changeVisualShape(self.front_patch, -1, rgbaColor=[0.0, 0.85, 0.22, front_alpha], physicsClientId=self.client)
        p.changeVisualShape(self.rear_patch, -1, rgbaColor=[0.0, 0.85, 0.22, rear_alpha], physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.front_patch, [front_x, 0.34, 0.012 + front_road], [0, 0, 0, 1], physicsClientId=self.client
        )
        p.resetBasePositionAndOrientation(
            self.rear_patch, [rear_x, 0.34, 0.012 + rear_road], [0, 0, 0, 1], physicsClientId=self.client
        )

        p.resetBasePositionAndOrientation(self.track, [0, 0.34, -0.012], [0, 0, 0, 1], physicsClientId=self.client)
        view = p.computeViewMatrix(
            cameraEyePosition=[4.3, -4.2, 2.0],
            cameraTargetPosition=[0.0, 0.15, 0.35],
            cameraUpVector=[0.0, 0.0, 1.0],
        )
        proj = p.computeProjectionMatrixFOV(
            fov=42.0,
            aspect=self.width / self.height,
            nearVal=0.1,
            farVal=20.0,
        )
        _, _, rgba, _, _ = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.client,
        )
        return np.reshape(rgba, (self.height, self.width, 4))[:, :, :3]


def write_video(rollout: Rollout, output_path: str | Path, fps: int = 50) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        renderer = PyBulletSuspensionRenderer()
    except Exception as exc:
        print(f"PyBullet renderer unavailable ({exc}); using Matplotlib fallback.")
        _write_matplotlib_video(rollout, output, fps=fps)
        return
    try:
        with imageio.get_writer(output, format="FFMPEG", fps=fps, codec="libx264", quality=8) as writer:
            for info in rollout.infos:
                writer.append_data(renderer.frame(info))
    finally:
        renderer.close()


def _write_matplotlib_video(rollout: Rollout, output_path: Path, fps: int = 50) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    with imageio.get_writer(output_path, format="FFMPEG", fps=fps, codec="libx264", quality=8) as writer:
        for info in rollout.infos:
            ax.clear()
            _draw_matplotlib_frame(ax, info)
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(rgba[:, :, :3])
    plt.close(fig)


def _draw_matplotlib_frame(ax, info: dict) -> None:
    wheelbase = 3.6
    target = 0.055
    front_x = wheelbase * 0.5
    rear_x = -wheelbase * 0.5
    front_h = info["front_height"]
    rear_h = info["rear_height"]
    pitch = info["pitch"]
    body_z = 0.55 + 0.5 * ((front_h + rear_h) - 2 * target)

    xs = np.linspace(-3.2, 3.2, 240)
    road = np.interp(xs, [rear_x, front_x], [info["road_rear"], info["road_front"]])
    ax.fill_between(xs, -0.03, road, color="#3b3b3b")
    ax.plot(xs, road, color="#111111", linewidth=2)

    length = 3.7
    height = 0.26
    corners = np.array(
        [
            [-length / 2, -height / 2],
            [length / 2, -height / 2],
            [length / 2, height / 2],
            [-length / 2, height / 2],
        ]
    )
    rot = np.array(
        [
            [math.cos(pitch), -math.sin(pitch)],
            [math.sin(pitch), math.cos(pitch)],
        ]
    )
    rotated = corners @ rot.T
    rotated[:, 1] += body_z
    ax.fill(rotated[:, 0], rotated[:, 1], color="#111111")
    ax.fill(rotated[:, 0], rotated[:, 1] - 0.13, color="#d9d9d9", alpha=0.9)

    for x, h, contact, label in [
        (front_x, front_h, info["front_contact"], "front"),
        (rear_x, rear_h, info["rear_contact"], "rear"),
    ]:
        road_z = info[f"road_{label}"]
        tire = plt.Circle((x, 0.18 + road_z), 0.18, color="#050505")
        ax.add_patch(tire)
        patch_color = (0.0, 0.75, 0.25, max(0.18, min(1.0, contact)))
        ax.plot([x - 0.24, x + 0.24], [road_z + 0.005, road_z + 0.005], color=patch_color, linewidth=8)
        top_z = body_z + pitch * x - 0.13
        ax.plot([x, x], [road_z + 0.32, top_z], color="#b11226", linewidth=3)

    ax.axhline(target, color="#2f6fed", linestyle="--", linewidth=1)
    ax.text(-3.05, 1.23, f"t={info['time']:.2f}s", fontsize=13, weight="bold")
    ax.text(-3.05, 1.08, f"pitch={info['pitch'] * 180 / math.pi:+.2f} deg", fontsize=11)
    ax.text(-3.05, 0.94, f"contact F/R={info['front_contact']:.2f}/{info['rear_contact']:.2f}", fontsize=11)
    ax.text(-3.05, 0.80, f"longitudinal accel={info['long_accel']:+.1f} m/s^2", fontsize=11)
    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(-0.08, 1.35)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
