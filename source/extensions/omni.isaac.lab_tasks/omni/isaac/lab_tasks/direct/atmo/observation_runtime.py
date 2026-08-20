from __future__ import annotations

import torch


class ObservationRuntime:
    """Generic observation packing delegated from the drone runtime."""

    def _pack_observations(self) -> dict:
        env = self.env
        self._observation_pass_cache.clear()

        self._append_action_history()
        obs_kinematic_current = env._obs_kinematic_current
        obs_command_current = (
            env._obs_command_current if self.current_observation_schema.dim > 0 else env._empty_obs_command
        )
        add_noise = bool(env.cfg.noise)
        for term in self.spec.observation_terms:
            if not term.history:
                continue
            value = self._observation_value(term)
            target = obs_kinematic_current[:, self.observation_schema.slices[term.name]]
            if value.shape != target.shape:
                raise ValueError(
                    f"Observation term '{term.name}' from source '{term.source}' returned "
                    f"shape {tuple(value.shape)}, expected {tuple(target.shape)}"
                )
            target.copy_(value)
            if add_noise:
                noise_scale = self._observation_noise_scales[term.name]
                if noise_scale != 0.0:
                    noise = torch.empty_like(target).uniform_(-noise_scale, noise_scale)
                    per_env_scale = self._observation_noise_env_scale.get(term.name)
                    if per_env_scale is not None:
                        noise = noise * per_env_scale
                    target.add_(noise)
        self._sanitize_observation_(obs_kinematic_current)
        self._append_observation_history(obs_kinematic_current)
        if env._reset_observation_history_dirty:
            reset_mask = env._reset_observation_history_pending.view(env.num_envs, 1, 1)
            current_history = obs_kinematic_current.unsqueeze(1).expand_as(env._observation_buffer)
            torch.where(reset_mask, current_history, env._observation_buffer, out=env._observation_buffer)
        for term in self.spec.observation_terms:
            if term.history or self.current_observation_schema.dim <= 0:
                continue
            value = self._observation_value(term)
            target = obs_command_current[:, self.current_observation_schema.slices[term.name]]
            if value.shape != target.shape:
                raise ValueError(
                    f"Observation term '{term.name}' from source '{term.source}' returned "
                    f"shape {tuple(value.shape)}, expected {tuple(target.shape)}"
                )
            target.copy_(value)
            if add_noise:
                noise_scale = self._observation_noise_scales[term.name]
                if noise_scale != 0.0:
                    noise = torch.empty_like(target).uniform_(-noise_scale, noise_scale)
                    per_env_scale = self._observation_noise_env_scale.get(term.name)
                    if per_env_scale is not None:
                        noise = noise * per_env_scale
                    target.add_(noise)
        self._sanitize_observation_(obs_command_current)
        env._current_observation_buffer[:, env._observation_history_index, :].copy_(obs_command_current)
        if env._reset_observation_history_dirty:
            reset_mask = env._reset_observation_history_pending.view(env.num_envs, 1, 1)
            current_frame = obs_command_current.unsqueeze(1).expand_as(env._current_observation_buffer)
            torch.where(reset_mask, current_frame, env._current_observation_buffer, out=env._current_observation_buffer)
            env._reset_observation_history_pending.zero_()
            env._reset_observation_history_dirty = False
        action_dim = self.action_schema.dim
        obs_history = env._obs_policy[:, self._obs_history_slice].view(
            env.num_envs,
            env.cfg.observation_history_length,
            self.observation_schema.dim,
        )
        history_len = env._observation_buffer.shape[1]
        history_indices = (
            env._observation_history_offsets
            + int(env._observation_history_index)
            + 1
            + env._observation_delay_steps.view(-1, 1)
        ) % history_len
        obs_history.copy_(
            env._observation_buffer[env._observation_history_env_indices, history_indices, :]
        )
        action_history = env._obs_policy[:, self._action_history_slice].view(
            env.num_envs,
            env.cfg.action_history_length,
            action_dim,
        )
        self._sanitize_action_(env._action_history)
        action_history_len = env._action_history.shape[1]
        action_tail_len = action_history_len - env._action_history_index
        action_history[:, :action_tail_len, :].copy_(env._action_history[:, env._action_history_index :, :])
        if env._action_history_index > 0:
            action_history[:, action_tail_len:, :].copy_(env._action_history[:, : env._action_history_index, :])
        if self.current_observation_schema.dim > 0:
            current_indices = (
                int(env._observation_history_index) + env._observation_delay_steps
            ) % history_len
            env._obs_command_current.copy_(
                env._current_observation_buffer[
                    env._observation_history_env_indices.squeeze(1), current_indices, :
                ]
            )
            env._obs_policy[:, self._base_command_slice].copy_(env._obs_command_current)
        task_observation_dim = int(getattr(env.cfg, "task_observation_dim", 0))
        if task_observation_dim:
            if self.task_observation_provider is None:
                raise ValueError("Task observation dimension is nonzero but the task provides no task_observation()")
            task_observation = self.task_observation_provider()
            expected_shape = (env.num_envs, task_observation_dim)
            if task_observation.shape != expected_shape:
                raise ValueError(
                    f"Task observation has shape {tuple(task_observation.shape)}, expected {expected_shape}"
                )
            env._obs_policy[:, self._task_observation_slice].copy_(task_observation)
        self._sanitize_observation_(env._obs_policy)

        return {"policy": env._obs_policy}
