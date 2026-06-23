import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort

NUM_ACTIONS = 20
NUM_OBS = 72
SIMULATION_DT = 0.005
CONTROL_DECIMATION = 4

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


def quat_rotate_inverse(quat: np.ndarray, world_vec: np.ndarray) -> np.ndarray:
  """Rotate a world-frame vector into the body frame."""
  w = quat[0]
  q_vec = quat[1:4]
  t = 2.0 * np.cross(q_vec, world_vec)
  return world_vec - w * t + np.cross(q_vec, t)


def projected_gravity(quat: np.ndarray) -> np.ndarray:
  return quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))


def pd_control(
  target_q: np.ndarray,
  q: np.ndarray,
  kp: np.ndarray,
  target_dq: np.ndarray,
  dq: np.ndarray,
  kd: np.ndarray,
) -> np.ndarray:
  return (target_q - q) * kp + (target_dq - dq) * kd


class RandomCommandSampler:
  """Random velocity commands within the RL_BOY play-mode range."""

  def __init__(self) -> None:
    self.lin_vel_x_range = (-1.0, 1.5)
    self.lin_vel_y_range = (-1, 1)
    self.ang_vel_z_range = (-0.8, 0.8)
    self.resampling_time_range = (3.0, 8.0)
    self._next_resample_time = 0.0
    self._current_cmd = np.zeros(3, dtype=np.float32)

  def start(self) -> None:
    print("\n=== Random Command Sampler ===")
    print(f"lin_vel_x: {self.lin_vel_x_range}")
    print(f"lin_vel_y: {self.lin_vel_y_range}")
    print(f"ang_vel_z: {self.ang_vel_z_range}")
    print(
      "resampling every "
      f"{self.resampling_time_range[0]}-{self.resampling_time_range[1]}s"
    )
    print("==============================\n")

  def _resample(self) -> None:
    self._current_cmd[0] = np.random.uniform(*self.lin_vel_x_range)
    self._current_cmd[1] = np.random.uniform(*self.lin_vel_y_range)
    self._current_cmd[2] = np.random.uniform(*self.ang_vel_z_range)

  def update(self, simulation_time: float) -> None:
    if simulation_time >= self._next_resample_time:
      self._resample()
      self._next_resample_time = simulation_time + np.random.uniform(
        *self.resampling_time_range
      )

  def get_command(self) -> np.ndarray:
    return self._current_cmd.copy()


def _static_dim(shape: list[int | str | None], axis: int) -> int | None:
  dim = shape[axis]
  return dim if isinstance(dim, int) else None


def load_policy(policy_path: Path) -> tuple[ort.InferenceSession, str, str]:
  if policy_path.suffix.lower() != ".onnx":
    raise ValueError(f"Expected an ONNX policy, got: {policy_path}")

  policy = ort.InferenceSession(str(policy_path))
  policy_input = policy.get_inputs()[0]
  policy_output = policy.get_outputs()[0]
  input_dim = _static_dim(policy_input.shape, -1)
  output_dim = _static_dim(policy_output.shape, -1)
  if input_dim is not None and input_dim != NUM_OBS:
    raise ValueError(
      f"Policy expects {input_dim} observations, but this task requires {NUM_OBS}."
    )
  if output_dim is not None and output_dim != NUM_ACTIONS:
    raise ValueError(
      f"Policy outputs {output_dim} actions, but this task requires {NUM_ACTIONS}."
    )

  print(f"Loaded ONNX policy from {policy_path}")
  return policy, policy_input.name, policy_output.name


def reset_to_training_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  mujoco.mj_resetData(model, data)
  data.qpos[:3] = (0.0, 0.0, 0.41)
  data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
  data.qpos[7:] = DEFAULT_ANGLES
  data.qvel[:] = 0.0
  data.ctrl[:] = 0.0
  mujoco.mj_forward(model, data)


def build_observation(
  data: mujoco.MjData,
  action: np.ndarray,
  command: np.ndarray,
) -> np.ndarray:
  quat = data.qpos[3:7]
  base_lin_vel = quat_rotate_inverse(quat, data.qvel[0:3])
  # MuJoCo free-joint angular velocity is already expressed in the body frame.
  base_ang_vel = data.qvel[3:6]
  gravity_orientation = projected_gravity(quat)
  joint_pos = data.qpos[7:] - DEFAULT_ANGLES
  joint_vel = data.qvel[6:]
  return np.concatenate(
    (
      base_lin_vel,
      base_ang_vel,
      gravity_orientation,
      joint_pos,
      joint_vel,
      action,
      command,
    ),
    dtype=np.float32,
  )


def main() -> None:
  repo_root = Path(__file__).resolve().parents[3]
  default_xml = repo_root / "src/mjlab/asset_zoo/robots/RL_BOY/RLBOY2sim.xml"

  parser = argparse.ArgumentParser()
  parser.add_argument("--xml-path", type=Path, default=default_xml)
  parser.add_argument("--policy-path", type=Path, required=True)
  args = parser.parse_args()

  model = mujoco.MjModel.from_xml_path(str(args.xml_path))
  data = mujoco.MjData(model)
  model.opt.timestep = SIMULATION_DT
  if model.nu != NUM_ACTIONS or model.nq != NUM_ACTIONS + 7:
    raise ValueError(
      f"Unexpected model dimensions: nq={model.nq}, nu={model.nu}; "
      f"expected nq={NUM_ACTIONS + 7}, nu={NUM_ACTIONS}."
    )

  policy, input_name, output_name = load_policy(args.policy_path)
  reset_to_training_pose(model, data)

  command = np.zeros(3, dtype=np.float32)
  action = np.zeros(NUM_ACTIONS, dtype=np.float32)
  target_dof_pos = DEFAULT_ANGLES.copy()
  target_dof_vel = np.zeros(NUM_ACTIONS, dtype=np.float32)
  counter = 0

  command_sampler = RandomCommandSampler()
  command_sampler.start()

  with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
      step_start = time.time()

      torque = pd_control(
        target_dof_pos,
        data.qpos[7:],
        STIFFNESS,
        target_dof_vel,
        data.qvel[6:],
        DAMPING,
      )
      data.ctrl[:] = np.clip(torque, -EFFORT_LIMITS, EFFORT_LIMITS)
      mujoco.mj_step(model, data)

      counter += 1
      if counter % CONTROL_DECIMATION == 0:
        command_sampler.update(data.time)
        command[:] = command_sampler.get_command()

        if counter % 400 == 0:
          print(f"cmd: [{command[0]:+.2f}, {command[1]:+.2f}, {command[2]:+.2f}]")

        observation = build_observation(data, action, command)
        # The mjlab-exported ONNX graph already contains the observation normalizer.
        policy_output = policy.run(
          [output_name],
          {input_name: observation[None, :]},
        )[0]
        action = np.asarray(policy_output, dtype=np.float32).reshape(-1)
        if action.shape != (NUM_ACTIONS,):
          raise ValueError(f"Unexpected policy output shape: {action.shape}")
        target_dof_pos = DEFAULT_ANGLES + action * ACTION_SCALES

      viewer.sync()
      time_until_next_step = model.opt.timestep - (time.time() - step_start)
      if time_until_next_step > 0:
        time.sleep(time_until_next_step)


if __name__ == "__main__":
  main()
