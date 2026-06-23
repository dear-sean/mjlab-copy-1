import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort

NUM_ACTIONS = 20
NUM_OBS = 115
SIMULATION_DT = 0.005
CONTROL_DECIMATION = 4
CONTROL_DT = SIMULATION_DT * CONTROL_DECIMATION

TRACKED_BODY_NAMES = (
  "base_link",
  "left_hip_pitch_link",
  "left_knee_pitch_link",
  "left_ankle_pitch_link",
  "right_hip_pitch_link",
  "right_knee_pitch_link",
  "right_ankle_pitch_link",
  "waist_yaw_link",
  "left_shoulder_roll_link",
  "left_elbow_pitch_link",
  "left_wrist_link",
  "right_shoulder_roll_link",
  "right_elbow_pitch_link",
  "right_wrist_link",
)

STIFFNESS = np.array(
  [
    16.3440649,
    16.3440649,
    16.3440649,
    16.3440649,
    8.2430936,
    16.3440649,
    16.3440649,
    16.3440649,
    16.3440649,
    8.2430936,
    8.2430936,
    8.2430936,
    1.6829649,
    1.6829649,
    1.6829649,
    1.6829649,
    1.6829649,
    1.6829649,
    1.6829649,
    1.6829649,
  ],
  dtype=np.float32,
)
DAMPING = np.array(
  [
    1.0404955,
    1.0404955,
    1.0404955,
    1.0404955,
    0.5247716,
    1.0404955,
    1.0404955,
    1.0404955,
    1.0404955,
    0.5247716,
    0.5247716,
    0.5247716,
    0.1071409,
    0.1071409,
    0.1071409,
    0.1071409,
    0.1071409,
    0.1071409,
    0.1071409,
    0.1071409,
  ],
  dtype=np.float32,
)
EFFORT_LIMITS = np.array(
  [
    20.0,
    20.0,
    20.0,
    20.0,
    11.0,
    20.0,
    20.0,
    20.0,
    20.0,
    11.0,
    11.0,
    11.0,
    3.0,
    3.0,
    3.0,
    3.0,
    3.0,
    3.0,
    3.0,
    3.0,
  ],
  dtype=np.float32,
)
DEFAULT_ANGLES = np.array(
  [
    0.0,
    0.0,
    -0.2,
    0.4,
    -0.2,
    0.0,
    0.0,
    -0.2,
    0.4,
    -0.2,
    0.0,
    0.0,
    0.15,
    0.3,
    0.0,
    0.9,
    0.15,
    -0.3,
    0.0,
    0.9,
  ],
  dtype=np.float32,
)
ACTION_SCALES = np.array(
  [
    0.3059214,
    0.3059214,
    0.3059214,
    0.3059214,
    0.3336126,
    0.3059214,
    0.3059214,
    0.3059214,
    0.3059214,
    0.3336126,
    0.3336126,
    0.0,
    0.4456421,
    0.4456421,
    0.4456421,
    0.4456421,
    0.4456421,
    0.4456421,
    0.4456421,
    0.4456421,
  ],
  dtype=np.float32,
)


@dataclass(frozen=True)
class MotionFrame:
  joint_pos: np.ndarray
  joint_vel: np.ndarray
  body_pos_w: np.ndarray
  body_quat_w: np.ndarray
  body_lin_vel_w: np.ndarray
  body_ang_vel_w: np.ndarray


class MotionReference:
  def __init__(self, path: Path, model: mujoco.MjModel) -> None:
    data = np.load(path)
    required = {
      "fps",
      "joint_pos",
      "joint_vel",
      "body_pos_w",
      "body_quat_w",
      "body_lin_vel_w",
      "body_ang_vel_w",
    }
    missing = required.difference(data.files)
    if missing:
      raise ValueError(f"Motion file is missing fields: {sorted(missing)}")

    self.fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    expected_fps = 1.0 / CONTROL_DT
    if not np.isclose(self.fps, expected_fps):
      raise ValueError(
        f"Motion FPS is {self.fps}, but the trained controller expects {expected_fps}."
      )

    self.joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
    self.joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
    if self.joint_pos.ndim != 2 or self.joint_pos.shape[1] != NUM_ACTIONS:
      raise ValueError(f"Unexpected joint_pos shape: {self.joint_pos.shape}")
    if self.joint_vel.shape != self.joint_pos.shape:
      raise ValueError(f"Unexpected joint_vel shape: {self.joint_vel.shape}")

    robot_body_names = tuple(
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
      for i in range(1, model.nbody)
    )
    body_indices = []
    for name in TRACKED_BODY_NAMES:
      if name not in robot_body_names:
        raise ValueError(f"Tracked body {name!r} is missing from the MuJoCo model.")
      body_indices.append(robot_body_names.index(name))

    self.body_pos_w = self._select_bodies(data["body_pos_w"], body_indices)
    self.body_quat_w = self._select_bodies(data["body_quat_w"], body_indices)
    self.body_lin_vel_w = self._select_bodies(data["body_lin_vel_w"], body_indices)
    self.body_ang_vel_w = self._select_bodies(data["body_ang_vel_w"], body_indices)
    self.frame_count = self.joint_pos.shape[0]
    if self.frame_count < 1:
      raise ValueError("Motion file contains no frames.")

  def _select_bodies(
    self,
    values: np.ndarray,
    body_indices: list[int],
  ) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != self.joint_pos.shape[0]:
      raise ValueError(f"Unexpected body data shape: {values.shape}")
    if values.shape[1] <= max(body_indices):
      raise ValueError(
        f"Motion contains {values.shape[1]} bodies, but the model requires "
        f"body index {max(body_indices)}."
      )
    return values[:, body_indices]

  def frame(self, index: int) -> MotionFrame:
    index = int(np.clip(index, 0, self.frame_count - 1))
    return MotionFrame(
      joint_pos=self.joint_pos[index],
      joint_vel=self.joint_vel[index],
      body_pos_w=self.body_pos_w[index],
      body_quat_w=self.body_quat_w[index],
      body_lin_vel_w=self.body_lin_vel_w[index],
      body_ang_vel_w=self.body_ang_vel_w[index],
    )


class MotionPolicy:
  def __init__(self, policy_path: Path) -> None:
    if policy_path.suffix.lower() != ".onnx":
      raise ValueError(f"Expected an ONNX policy, got: {policy_path}")

    self.session = ort.InferenceSession(str(policy_path))
    inputs = {item.name: item for item in self.session.get_inputs()}
    outputs = {item.name: item for item in self.session.get_outputs()}
    if set(inputs) != {"obs", "time_step"}:
      raise ValueError(f"Unexpected ONNX inputs: {sorted(inputs)}")
    required_outputs = {
      "actions",
      "joint_pos",
      "joint_vel",
      "body_pos_w",
      "body_quat_w",
      "body_lin_vel_w",
      "body_ang_vel_w",
    }
    if not required_outputs.issubset(outputs):
      raise ValueError(f"Unexpected ONNX outputs: {sorted(outputs)}")

    obs_dim = inputs["obs"].shape[-1]
    action_dim = outputs["actions"].shape[-1]
    if isinstance(obs_dim, int) and obs_dim != NUM_OBS:
      raise ValueError(
        f"Policy expects {obs_dim} observations, but this task requires {NUM_OBS}."
      )
    if isinstance(action_dim, int) and action_dim != NUM_ACTIONS:
      raise ValueError(
        f"Policy outputs {action_dim} actions, but this task requires {NUM_ACTIONS}."
      )

    metadata = self.session.get_modelmeta().custom_metadata_map
    body_names = tuple(
      name for name in metadata.get("body_names", "").split(",") if name
    )
    if body_names and body_names != TRACKED_BODY_NAMES:
      raise ValueError(
        "Policy tracked bodies do not match this deployment script:\n"
        f"policy={body_names}\nexpected={TRACKED_BODY_NAMES}"
      )
    print(f"Loaded motion tracking ONNX policy from {policy_path}")

  def run(
    self,
    observation: np.ndarray,
    frame_index: int,
  ) -> tuple[np.ndarray, MotionFrame]:
    outputs = self.session.run(
      None,
      {
        "obs": observation[None, :],
        "time_step": np.array([[frame_index]], dtype=np.float32),
      },
    )
    values = {
      output.name: value
      for output, value in zip(self.session.get_outputs(), outputs, strict=True)
    }
    action = np.asarray(values["actions"], dtype=np.float32).reshape(-1)
    if action.shape != (NUM_ACTIONS,):
      raise ValueError(f"Unexpected policy output shape: {action.shape}")
    frame = MotionFrame(
      joint_pos=values["joint_pos"][0],
      joint_vel=values["joint_vel"][0],
      body_pos_w=values["body_pos_w"][0],
      body_quat_w=values["body_quat_w"][0],
      body_lin_vel_w=values["body_lin_vel_w"][0],
      body_ang_vel_w=values["body_ang_vel_w"][0],
    )
    return action, frame


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
  result = quat.copy()
  result[1:] *= -1.0
  return result


def quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
  lw, lx, ly, lz = lhs
  rw, rx, ry, rz = rhs
  return np.array(
    [
      lw * rw - lx * rx - ly * ry - lz * rz,
      lw * rx + lx * rw + ly * rz - lz * ry,
      lw * ry - lx * rz + ly * rw + lz * rx,
      lw * rz + lx * ry - ly * rx + lz * rw,
    ],
    dtype=np.float32,
  )


def quat_rotate_inverse(quat: np.ndarray, world_vec: np.ndarray) -> np.ndarray:
  w = quat[0]
  q_vec = quat[1:4]
  t = 2.0 * np.cross(q_vec, world_vec)
  return world_vec - w * t + np.cross(q_vec, t)


def matrix_from_quat(quat: np.ndarray) -> np.ndarray:
  quat = quat / np.linalg.norm(quat)
  w, x, y, z = quat
  return np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ],
    dtype=np.float32,
  )


def pd_control(
  target_q: np.ndarray,
  q: np.ndarray,
  target_dq: np.ndarray,
  dq: np.ndarray,
) -> np.ndarray:
  return (target_q - q) * STIFFNESS + (target_dq - dq) * DAMPING


def sensor_data(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  name: str,
) -> np.ndarray:
  sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
  if sensor_id < 0:
    raise ValueError(f"Sensor {name!r} is missing from the MuJoCo model.")
  address = model.sensor_adr[sensor_id]
  dimension = model.sensor_dim[sensor_id]
  return data.sensordata[address : address + dimension]


def reset_to_motion_frame(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  frame: MotionFrame,
) -> None:
  mujoco.mj_resetData(model, data)
  anchor_pos = frame.body_pos_w[0]
  anchor_quat = frame.body_quat_w[0]
  data.qpos[:3] = anchor_pos
  data.qpos[3:7] = anchor_quat
  data.qpos[7:] = frame.joint_pos
  data.qvel[:3] = frame.body_lin_vel_w[0]
  data.qvel[3:6] = quat_rotate_inverse(anchor_quat, frame.body_ang_vel_w[0])
  data.qvel[6:] = frame.joint_vel
  data.ctrl[:] = 0.0
  mujoco.mj_forward(model, data)


def build_observation(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  action: np.ndarray,
  reference: MotionFrame,
) -> np.ndarray:
  robot_anchor_pos = data.xpos[1]
  robot_anchor_quat = data.xquat[1]
  reference_anchor_pos = reference.body_pos_w[0]
  reference_anchor_quat = reference.body_quat_w[0]

  anchor_pos_b = quat_rotate_inverse(
    robot_anchor_quat,
    reference_anchor_pos - robot_anchor_pos,
  )
  anchor_quat_b = quat_multiply(
    quat_conjugate(robot_anchor_quat),
    reference_anchor_quat,
  )
  anchor_ori_b = matrix_from_quat(anchor_quat_b)[:, :2].reshape(-1)

  observation = np.concatenate(
    (
      reference.joint_pos,
      reference.joint_vel,
      anchor_pos_b,
      anchor_ori_b,
      sensor_data(model, data, "imu_lin_vel"),
      sensor_data(model, data, "imu_ang_vel"),
      data.qpos[7:] - DEFAULT_ANGLES,
      data.qvel[6:],
      action,
    ),
    dtype=np.float32,
  )
  if observation.shape != (NUM_OBS,):
    raise ValueError(f"Unexpected observation shape: {observation.shape}")
  return observation


def validate_motion_pair(
  policy: MotionPolicy,
  motion: MotionReference,
) -> None:
  zero_obs = np.zeros(NUM_OBS, dtype=np.float32)
  for index in (0, motion.frame_count - 1):
    _, embedded = policy.run(zero_obs, index)
    external = motion.frame(index)
    for name in (
      "joint_pos",
      "joint_vel",
      "body_pos_w",
      "body_quat_w",
      "body_lin_vel_w",
      "body_ang_vel_w",
    ):
      np.testing.assert_allclose(
        getattr(embedded, name),
        getattr(external, name),
        atol=1e-5,
        rtol=1e-5,
        err_msg=f"ONNX and NPZ differ for {name} at frame {index}.",
      )
  print(f"Validated ONNX motion against {motion.frame_count} NPZ frames.")


def main() -> None:
  repo_root = Path(__file__).resolve().parents[3]
  default_xml = repo_root / "src/mjlab/asset_zoo/robots/RL_BOY/RLBOY2sim.xml"

  parser = argparse.ArgumentParser()
  parser.add_argument("--xml-path", type=Path, default=default_xml)
  parser.add_argument("--policy-path", type=Path, required=True)
  parser.add_argument("--motion-path", type=Path, required=True)
  parser.add_argument(
    "--loop",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Restart from motion frame zero after the final frame.",
  )
  args = parser.parse_args()

  model = mujoco.MjModel.from_xml_path(str(args.xml_path))
  data = mujoco.MjData(model)
  model.opt.timestep = SIMULATION_DT
  if model.nu != NUM_ACTIONS or model.nq != NUM_ACTIONS + 7:
    raise ValueError(
      f"Unexpected model dimensions: nq={model.nq}, nu={model.nu}; "
      f"expected nq={NUM_ACTIONS + 7}, nu={NUM_ACTIONS}."
    )

  policy = MotionPolicy(args.policy_path)
  motion = MotionReference(args.motion_path, model)
  validate_motion_pair(policy, motion)

  frame_index = 0
  action = np.zeros(NUM_ACTIONS, dtype=np.float32)
  reference = motion.frame(frame_index)
  reset_to_motion_frame(model, data, reference)
  observation = build_observation(model, data, action, reference)
  action, _ = policy.run(observation, frame_index)
  target_dof_pos = DEFAULT_ANGLES + action * ACTION_SCALES
  target_dof_vel = np.zeros(NUM_ACTIONS, dtype=np.float32)
  counter = 0

  with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
      step_start = time.time()
      torque = pd_control(
        target_dof_pos,
        data.qpos[7:],
        target_dof_vel,
        data.qvel[6:],
      )
      data.ctrl[:] = np.clip(torque, -EFFORT_LIMITS, EFFORT_LIMITS)
      mujoco.mj_step(model, data)

      counter += 1
      if counter % CONTROL_DECIMATION == 0:
        frame_index += 1
        if frame_index >= motion.frame_count:
          if args.loop:
            frame_index = 0
            action.fill(0.0)
            reference = motion.frame(frame_index)
            reset_to_motion_frame(model, data, reference)
          else:
            frame_index = motion.frame_count - 1

        reference = motion.frame(frame_index)
        observation = build_observation(model, data, action, reference)
        action, _ = policy.run(observation, frame_index)
        target_dof_pos = DEFAULT_ANGLES + action * ACTION_SCALES

      viewer.sync()
      time_until_next_step = model.opt.timestep - (time.time() - step_start)
      if time_until_next_step > 0:
        time.sleep(time_until_next_step)


if __name__ == "__main__":
  main()
