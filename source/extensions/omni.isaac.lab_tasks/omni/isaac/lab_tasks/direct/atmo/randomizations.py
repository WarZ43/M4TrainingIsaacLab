from __future__ import annotations

import torch


class M4Randomizer:
    """Initial-state, disturbance, and actuator randomization for M4 landing."""

    def __init__(self, env):
        self.env = env
        env._disturbance_force = torch.zeros(env.num_envs, 1, 3, device=env.device)
        env._disturbance_moment = torch.zeros(env.num_envs, 1, 3, device=env.device)
        env._push_time = torch.zeros(env.num_envs, device=env.device)
        env._push_duration = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor):
        if self.env.cfg.randomize:
            self._reset_randomized(env_ids)
        else:
            self._reset_deterministic(env_ids)

    def _reset_randomized(self, env_ids: torch.Tensor):
        env = self.env
        env.vehicle.reset_action_buffers(env_ids, fill_actions_with_ones=True)

        env._desired_pos_w[env_ids, :2] = torch.zeros_like(env._desired_pos_w[env_ids, :2]).uniform_(
            -env.box_extent,
            env.box_extent,
        )
        env._desired_pos_w[env_ids, :2] += env._terrain.env_origins[env_ids, :2]
        env._desired_pos_w[env_ids, 2] = torch.zeros_like(env._desired_pos_w[env_ids, 2])

        default_root_state = env._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += env._terrain.env_origins[env_ids]
        default_root_state[:, 2] = torch.zeros_like(default_root_state[:, 2]).uniform_(
            env.cfg.initial_height_range[0],
            env.cfg.initial_height_range[1],
        )
        default_root_state[:, 7:10] = torch.zeros_like(default_root_state[:, 7:10]).uniform_(
            env.cfg.initial_lin_vel_range[0],
            env.cfg.initial_lin_vel_range[1],
        )
        default_root_state[:, 10:13] = torch.zeros_like(default_root_state[:, 10:13]).uniform_(
            env.cfg.initial_ang_vel_range[0],
            env.cfg.initial_ang_vel_range[1],
        )
        default_root_state[:, 3:7], _ = env.random_quaternion(len(env_ids))

        joint_pos, joint_vel = env.vehicle.randomized_joint_state(env_ids)

        env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)
        env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)

        env.vehicle.reset_actuator_params(env_ids, randomized=True)

        disturbance_force_direction = torch.normal(0.0, 1.0, size=(env.num_envs, 1, 3), device=env.device)
        disturbance_force_direction = disturbance_force_direction / (
            torch.linalg.norm(disturbance_force_direction, dim=1).unsqueeze(dim=1) + 1e-6
        )
        disturbance_moment_direction = torch.normal(0.0, 1.0, size=(env.num_envs, 1, 3), device=env.device)
        disturbance_moment_direction = disturbance_moment_direction / (
            torch.linalg.norm(disturbance_moment_direction, dim=1).unsqueeze(dim=1) + 1e-6
        )

        env._push_time[env_ids] = env.cfg.episode_length_s * torch.zeros_like(env._push_time[env_ids]).uniform_(
            0.0,
            0.5,
        )
        env._push_duration[env_ids] = torch.zeros_like(env._push_duration[env_ids]).uniform_(0.0, 0.2)

        force_intensity = torch.normal(torch.tensor(0.0), env.cfg.disturbance_force_scale)
        moment_intensity = torch.normal(torch.tensor(0.0), env.cfg.disturbance_moment_scale)
        env._disturbance_force = force_intensity * disturbance_force_direction
        env._disturbance_moment = moment_intensity * disturbance_moment_direction

        if env.cfg.randomize_motor_dynamics:
            env._alpha = torch.zeros_like(env._alpha).uniform_(env.cfg.alpha_range[0], env.cfg.alpha_range[1])

        env._time_elapsed[env_ids] = 0.0
        env._current_impulse[env_ids] = 0.0

    def _reset_deterministic(self, env_ids: torch.Tensor):
        env = self.env
        env.vehicle.reset_action_buffers(env_ids, fill_actions_with_ones=False)

        env._desired_pos_w[env_ids, :2] = torch.zeros_like(env._desired_pos_w[env_ids, :2]).uniform_(-0.1, 0.1)
        env._desired_pos_w[env_ids, :2] += env._terrain.env_origins[env_ids, :2]
        env._desired_pos_w[env_ids, 2] = torch.zeros_like(env._desired_pos_w[env_ids, 2])

        default_root_state = env._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += env._terrain.env_origins[env_ids]

        joint_pos, joint_vel = env.vehicle.deterministic_joint_state(env_ids)

        env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)
        env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        env.vehicle.reset_actuator_params(env_ids, randomized=False)

        disturbance_force_direction = torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1, 1)
        disturbance_force_direction = disturbance_force_direction / (
            torch.linalg.norm(disturbance_force_direction, dim=1).unsqueeze(dim=1) + 1e-6
        )
        disturbance_moment_direction = torch.normal(0.0, 1.0, size=(1, 1, 3), device=env.device).repeat(
            env.num_envs,
            1,
            1,
        )
        disturbance_moment_direction = disturbance_moment_direction / (
            torch.linalg.norm(disturbance_moment_direction, dim=1).unsqueeze(dim=1) + 1e-6
        )

        env._push_time[env_ids] = 0.5 * torch.ones_like(env._push_time[env_ids])
        env._push_duration[env_ids] = 0.5 * torch.ones_like(env._push_duration[env_ids])

        force_intensity = env.vehicle.nominal_total_kT() * 0.15
        moment_intensity = torch.normal(torch.tensor(0.0), env.cfg.disturbance_moment_scale)
        env._disturbance_force = force_intensity * disturbance_force_direction
        env._disturbance_moment = moment_intensity * disturbance_moment_direction

        env._time_elapsed[env_ids] = 0.0
        env._current_impulse[env_ids] = 0.0
