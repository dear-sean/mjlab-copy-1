"""Play an RL Boy motion NPZ produced by ``csv_to_npz_rlboy.py``.

Example:
  uv run src/mjlab/scripts/play_npz.py motions/walk.npz
"""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import tyro

import mjlab
from mjlab.asset_zoo.robots import get_rlboy_robot_cfg
from mjlab.entity import Entity

REQUIRED_KEYS = ("fps", "joint_pos", "body_pos_w", "body_quat_w")


def _load_motion(
  motion_file: Path,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
  if not motion_file.is_file():
    raise FileNotFoundError(f"Motion file does not exist: {motion_file}")

  with np.load(motion_file, allow_pickle=False) as motion:
    missing = [key for key in REQUIRED_KEYS if key not in motion]
    if missing:
      raise ValueError(
        f"{motion_file} is missing required arrays: {', '.join(missing)}"
      )

    fps_values = np.asarray(motion["fps"]).reshape(-1)
    if fps_values.size != 1 or not np.isfinite(fps_values[0]):
      raise ValueError("'fps' must contain one finite value")
    fps = float(fps_values[0])
    joint_pos = np.asarray(motion["joint_pos"], dtype=np.float64)
    body_pos_w = np.asarray(motion["body_pos_w"], dtype=np.float64)
    body_quat_w = np.asarray(motion["body_quat_w"], dtype=np.float64)

  if fps <= 0:
    raise ValueError(f"'fps' must be positive, got {fps}")
  if joint_pos.ndim != 2:
    raise ValueError(
      f"'joint_pos' must have shape (frames, joints), got {joint_pos.shape}"
    )
  if body_pos_w.ndim != 3 or body_pos_w.shape[2] != 3:
    raise ValueError(
      f"'body_pos_w' must have shape (frames, bodies, 3), got {body_pos_w.shape}"
    )
  if body_quat_w.ndim != 3 or body_quat_w.shape[2] != 4:
    raise ValueError(
      f"'body_quat_w' must have shape (frames, bodies, 4), got {body_quat_w.shape}"
    )

  frame_counts = (joint_pos.shape[0], body_pos_w.shape[0], body_quat_w.shape[0])
  if len(set(frame_counts)) != 1 or frame_counts[0] == 0:
    raise ValueError(f"Motion arrays have invalid frame counts: {frame_counts}")
  if not all(
    np.isfinite(array).all() for array in (joint_pos, body_pos_w, body_quat_w)
  ):
    raise ValueError("Motion arrays contain NaN or infinity")

  return fps, joint_pos, body_pos_w[:, 0], body_quat_w[:, 0]


def main(
  motion_file: Path,
  speed: float = 1.0,
  loop: bool = True,
  check_only: bool = False,
) -> None:
  """Play an RL Boy motion file in the native MuJoCo viewer.

  Args:
    motion_file: NPZ file produced by csv_to_npz_rlboy.py.
    speed: Playback speed multiplier.
    loop: Restart the motion after its last frame.
    check_only: Validate the NPZ without opening a viewer.
  """
  if speed <= 0:
    raise ValueError(f"'speed' must be positive, got {speed}")

  fps, joint_pos, root_pos, root_quat = _load_motion(motion_file)
  robot = Entity(get_rlboy_robot_cfg())
  joint_names = robot.joint_names
  if joint_pos.shape[1] != len(joint_names):
    raise ValueError(
      "'joint_pos' joint count does not match RL Boy: "
      f"{joint_pos.shape[1]} in file, {len(joint_names)} in model"
    )

  duration = joint_pos.shape[0] / fps
  print(
    f"Loaded {motion_file}: {joint_pos.shape[0]} frames, "
    f"{len(joint_names)} joints, {fps:g} FPS, {duration:.2f} s"
  )
  if check_only:
    return

  model = robot.spec.compile()
  data = mujoco.MjData(model)

  free_joint_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
  if free_joint_ids.size != 1:
    raise RuntimeError(f"Expected one free joint, found {free_joint_ids.size}")
  root_qpos_adr = int(model.jnt_qposadr[free_joint_ids[0]])

  joint_qpos_adrs = []
  for name in joint_names:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
      raise RuntimeError(f"Joint is missing from compiled model: {name}")
    joint_qpos_adrs.append(int(model.jnt_qposadr[joint_id]))

  def apply_frame(frame: int) -> None:
    data.qpos[root_qpos_adr : root_qpos_adr + 3] = root_pos[frame]
    data.qpos[root_qpos_adr + 3 : root_qpos_adr + 7] = root_quat[frame]
    data.qpos[joint_qpos_adrs] = joint_pos[frame]
    mujoco.mj_forward(model, data)

  apply_frame(0)
  root_body_id = int(model.jnt_bodyid[free_joint_ids[0]])
  with mujoco.viewer.launch_passive(
    model, data, show_left_ui=False, show_right_ui=False
  ) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = root_body_id
    viewer.cam.distance = 2.5
    viewer.cam.elevation = -10
    viewer.cam.azimuth = 135
    viewer.sync()

    frame = 0
    frame_period = 1.0 / (fps * speed)
    deadline = time.monotonic()
    while viewer.is_running():
      apply_frame(frame)
      viewer.sync()

      frame += 1
      if frame == joint_pos.shape[0]:
        if not loop:
          break
        frame = 0

      deadline += frame_period
      remaining = deadline - time.monotonic()
      if remaining > 0:
        time.sleep(remaining)
      else:
        deadline = time.monotonic()


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
