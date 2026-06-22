from __future__ import annotations

import torch


class M4Randomizer:
    """Common disturbance and actuator randomization for M4 tasks."""

    def __init__(self, env):
        self.env = env
        env._disturbance_force = torch.zeros(env.num_envs, 1, 3, device=env.device)
        env._disturbance_moment = torch.zeros(env.num_envs, 1, 3, device=env.device)
        env._disturbance_force_cts = torch.zeros(env.num_envs, 1, 3, device=env.device)
        env._disturbance_moment_cts = torch.zeros(env.num_envs, 1, 3, device=env.device)
        env._push_time = torch.zeros(env.num_envs, device=env.device)
        env._push_duration = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor):
        if self.env.cfg.randomize:
            self._reset_randomized(env_ids)
        else:
            self._reset_deterministic(env_ids)

    def _disturbance_scales(self) -> tuple[float, float]:
        env = self.env
        force_scale = env.cfg.disturbance_force_scale
        moment_scale = env.cfg.disturbance_moment_scale
        if hasattr(env.task, "disturbance_force_scale"):
            force_scale = env.task.disturbance_force_scale()
        if hasattr(env.task, "disturbance_moment_scale"):
            moment_scale = env.task.disturbance_moment_scale()
        return force_scale, moment_scale

    def _continuous_disturbance_scales(self) -> tuple[float, float]:
        env = self.env
        force_scale = env.cfg.dist_force_cts_scale
        moment_scale = env.cfg.dist_moment_cts_scale
        if hasattr(env.task, "disturbance_cts_force_scale"):
            force_scale = env.task.disturbance_cts_force_scale()
        return force_scale, moment_scale

    def _sample_unit_vectors(self, shape: tuple[int, int, int]) -> torch.Tensor:
        direction = torch.normal(0.0, 1.0, size=shape, device=self.env.device)
        return direction / (torch.linalg.norm(direction, dim=-1, keepdim=True) + 1e-6)

    def _reset_randomized(self, env_ids: torch.Tensor):
        env = self.env
        num_resets = len(env_ids)
        env.vehicle.reset_action_buffers(env_ids, fill_actions_with_ones=True)
        env.task.reset_initial_state(env_ids, randomized=True)
        env.vehicle.reset_actuator_params(env_ids, randomized=True)

        disturbance_force_direction = self._sample_unit_vectors((num_resets, 1, 3))
        disturbance_moment_direction = self._sample_unit_vectors((num_resets, 1, 3))

        env._push_time[env_ids] = env.cfg.episode_length_s * torch.zeros_like(env._push_time[env_ids]).uniform_(
            0.0,
            0.5,
        )
        env._push_duration[env_ids] = torch.zeros_like(env._push_duration[env_ids]).uniform_(0.0, 0.2)

        disturbance_force_scale, disturbance_moment_scale = self._disturbance_scales()
        force_intensity = torch.zeros(num_resets, 1, 1, device=env.device).uniform_(
            -disturbance_force_scale,
            disturbance_force_scale,
        )
        moment_intensity = torch.zeros(num_resets, 1, 1, device=env.device).uniform_(
            -disturbance_moment_scale,
            disturbance_moment_scale,
        )
        env._disturbance_force[env_ids] = force_intensity * disturbance_force_direction
        env._disturbance_moment[env_ids] = moment_intensity * disturbance_moment_direction

        cts_force_scale, cts_moment_scale = self._continuous_disturbance_scales()
        cts_force_direction = self._sample_unit_vectors((num_resets, 1, 3))
        cts_moment_direction = self._sample_unit_vectors((num_resets, 1, 3))
        cts_force_intensity = torch.zeros(num_resets, 1, 1, device=env.device).uniform_(
            -cts_force_scale,
            cts_force_scale,
        )
        cts_moment_intensity = torch.zeros(num_resets, 1, 1, device=env.device).uniform_(
            -cts_moment_scale,
            cts_moment_scale,
        )
        env._disturbance_force_cts[env_ids] = cts_force_intensity * cts_force_direction
        env._disturbance_moment_cts[env_ids] = cts_moment_intensity * cts_moment_direction

        if env.cfg.randomize_motor_dynamics:
            env._alpha[env_ids] = torch.zeros_like(env._alpha[env_ids]).uniform_(
                env.cfg.alpha_range[0],
                env.cfg.alpha_range[1],
            )

        env._time_elapsed[env_ids] = 0.0
        env._current_impulse[env_ids] = 0.0

    def _reset_deterministic(self, env_ids: torch.Tensor):
        env = self.env
        num_resets = len(env_ids)
        env.vehicle.reset_action_buffers(env_ids, fill_actions_with_ones=False)
        env.task.reset_initial_state(env_ids, randomized=False)
        env.vehicle.reset_actuator_params(env_ids, randomized=False)

        disturbance_force_direction = torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(num_resets, 1, 1)
        disturbance_force_direction = disturbance_force_direction / (
            torch.linalg.norm(disturbance_force_direction, dim=-1, keepdim=True) + 1e-6
        )
        disturbance_moment_direction = torch.normal(0.0, 1.0, size=(num_resets, 1, 3), device=env.device)
        disturbance_moment_direction = disturbance_moment_direction / (
            torch.linalg.norm(disturbance_moment_direction, dim=-1, keepdim=True) + 1e-6
        )

        env._push_time[env_ids] = 0.5 * torch.ones_like(env._push_time[env_ids])
        env._push_duration[env_ids] = 0.5 * torch.ones_like(env._push_duration[env_ids])

        disturbance_force_scale, disturbance_moment_scale = self._disturbance_scales()
        force_intensity = disturbance_force_scale
        moment_intensity = torch.normal(
            torch.zeros(num_resets, 1, 1, device=env.device),
            disturbance_moment_scale,
        )
        env._disturbance_force[env_ids] = force_intensity * disturbance_force_direction
        env._disturbance_moment[env_ids] = moment_intensity * disturbance_moment_direction

        cts_force_scale, cts_moment_scale = self._continuous_disturbance_scales()
        env._disturbance_force_cts[env_ids] = (
            torch.tensor([cts_force_scale, 0.0, 0.0], device=env.device)
            .repeat(
                num_resets,
                1,
            )
            .unsqueeze(1)
        )
        env._disturbance_moment_cts[env_ids] = (
            torch.tensor([0.0, 0.0, cts_moment_scale], device=env.device)
            .repeat(
                num_resets,
                1,
            )
            .unsqueeze(1)
        )

        env._time_elapsed[env_ids] = 0.0
        env._current_impulse[env_ids] = 0.0
