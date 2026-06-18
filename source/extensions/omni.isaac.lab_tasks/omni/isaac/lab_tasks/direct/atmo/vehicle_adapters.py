from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import torch

from omni.isaac.lab.utils.math import matrix_from_quat, quat_rotate

try:
    from .vehicle_specs import JointGroupSpec, ObservationSourceSpec, VehicleSpec
except ImportError:
    from vehicle_specs import JointGroupSpec, ObservationSourceSpec, VehicleSpec


@dataclass
class JointGroupRuntime:
    spec: JointGroupSpec
    joint_ids: list[int]
    target_pos: torch.Tensor
    target_vel: torch.Tensor
    max_velocity: torch.Tensor


class GenericM4VehicleAdapter:
    """Reusable M4 vehicle adapter driven by a declarative VehicleSpec."""

    def __init__(self, env, spec: VehicleSpec):
        self.env = env
        self.spec = spec
        self.action_schema = spec.make_action_schema()
        self.observation_schema = spec.make_observation_schema()
        self.base_link: int = -1
        self.rotor_ids: list[int] = []
        self.joint_groups: dict[str, JointGroupRuntime] = {}

    @property
    def rotor_count(self) -> int:
        return len(self.spec.rotors)

    def initialize(self):
        env = self.env
        if self.action_schema.dim != env.cfg.action_space:
            raise ValueError(
                f"{self.spec.name} action schema dim {self.action_schema.dim} "
                f"does not match cfg.action_space={env.cfg.action_space}"
            )
        if self.observation_schema.dim != env.cfg.num_obs:
            raise ValueError(
                f"{self.spec.name} observation schema dim {self.observation_schema.dim} "
                f"does not match cfg.num_obs={env.cfg.num_obs}"
            )

        action_dim = gym.spaces.flatdim(env.single_action_space)
        env._observation_buffer = torch.zeros(
            env.num_envs,
            env.cfg.observation_buffer_length,
            self.observation_schema.dim,
            device=env.device,
        )
        env._actions = torch.zeros(env.num_envs, action_dim, device=env.device)
        env._actions_filtered = torch.zeros_like(env._actions)
        env._previous_actions = torch.zeros_like(env._actions)
        env._action_history = torch.zeros(
            env.num_envs,
            env.cfg.action_history_length,
            action_dim,
            device=env.device,
        )

        env.thrust = torch.zeros(env.num_envs, self.rotor_count, 3, device=env.device)
        env.moment = torch.zeros(env.num_envs, self.rotor_count, 3, device=env.device)
        env.thrust_loss = torch.zeros(env.num_envs, self.rotor_count, device=env.device)
        env._current_impulse = torch.zeros(env.num_envs, 1, device=env.device)
        env._previous_lin_vel_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._acceleration = torch.zeros(env.num_envs, 3, device=env.device)
        env._time_elapsed = torch.zeros(env.num_envs, device=env.device)
        env._alpha = env.cfg.alpha_0 * torch.ones(env.num_envs, 1, device=env.device)

        self._resolve_body_handles()
        self._resolve_joint_groups()
        self._allocate_actuator_params()
        self._resolve_contact_groups()
        self._cache_inertial_params()

    def pre_physics_step(self, actions: torch.Tensor):
        env = self.env
        env._time_elapsed += env.step_dt
        env._actions = actions.clone().clamp(0.0, 1.0)

        if env.cfg.actuator_dynamics:
            self._filter_actions()
        else:
            env._actions_filtered = env._actions

        self._quantize_joint_actions()
        filtered_terms = self.action_schema.split(env._actions_filtered)
        self._compute_rotor_forces(filtered_terms)
        self._compute_joint_targets(filtered_terms)

    def apply_action(self):
        env = self.env
        dist_force = torch.zeros(env.num_envs, 1, 3, device=env.device)
        dist_moment = torch.zeros(env.num_envs, 1, 3, device=env.device)
        dist_force_cts = torch.zeros(env.num_envs, 1, 3, device=env.device)
        dist_moment_cts = torch.zeros(env.num_envs, 1, 3, device=env.device)

        if env.cfg.disturb:
            push = torch.logical_and(
                env._time_elapsed >= env._push_time,
                env._time_elapsed <= env._push_time + env._push_duration,
            ).reshape(env.num_envs, 1, 1)
            dist_force = env._disturbance_force * push
            dist_moment = env._disturbance_moment * push
            dist_force_cts = torch.zeros(env.num_envs, 1, 3, device=env.device).uniform_(
                -env.cfg.dist_force_cts_scale,
                env.cfg.dist_force_cts_scale,
            )
            dist_moment_cts = torch.zeros(env.num_envs, 1, 3, device=env.device).uniform_(
                -env.cfg.dist_moment_cts_scale,
                env.cfg.dist_moment_cts_scale,
            )

        total_environmental_force = dist_force + dist_force_cts
        total_environmental_moment = dist_moment + dist_moment_cts

        if env._stage_value(env.cfg.rotate_rotor_thrust_by_stage, "rotate_rotor_thrust_by_stage"):
            rotor_quats = env._robot.data.body_quat_w[:, self.rotor_ids, :]
            q_flat = rotor_quats.reshape(-1, 4)
            thrust_flat = env.thrust.reshape(-1, 3)
            moment_flat = env.moment.reshape(-1, 3)

            thrust_world_flat = quat_rotate(q_flat, thrust_flat)
            moment_world_flat = quat_rotate(q_flat, moment_flat)

            thrust_world = thrust_world_flat.reshape(env.num_envs, self.rotor_count, 3)
            moment_world = moment_world_flat.reshape(env.num_envs, self.rotor_count, 3)
        else:
            thrust_world = env.thrust
            moment_world = env.moment

        total_force = torch.cat([thrust_world, total_environmental_force], dim=1)
        total_moment = torch.cat([moment_world, total_environmental_moment], dim=1)

        env._robot.set_external_force_and_torque(
            total_force,
            total_moment,
            body_ids=self.rotor_ids + [self.base_link],
        )
        for runtime in self.joint_groups.values():
            env._robot.set_joint_velocity_target(
                self._expand_joint_group(runtime, runtime.target_vel),
                joint_ids=runtime.joint_ids,
            )
            env._robot.set_joint_position_target(
                self._expand_joint_group(runtime, runtime.target_pos),
                joint_ids=runtime.joint_ids,
            )

    def get_observations(self) -> dict:
        env = self.env
        env._action_history = torch.cat(
            [env._actions.clone().unsqueeze(dim=1), env._action_history[:, :-1]],
            dim=1,
        )
        env._current_impulse = self._compute_contact_impulse()

        obs_values = {
            term.name: self._observation_value(term)
            for term in self.spec.observation_terms
        }
        noise_values = {
            term.name: self._observation_noise(term, obs_values[term.name])
            for term in self.spec.observation_terms
        }

        obs_kinematic_current = self.observation_schema.pack(obs_values) + env.cfg.noise * self.observation_schema.pack(
            noise_values
        )
        env._observation_buffer[:, 1:] = env._observation_buffer[:, :-1].clone()
        env._observation_buffer[:, 0] = obs_kinematic_current

        reset_env_ids = (env.episode_length_buf == 1).nonzero(as_tuple=True)[0]
        if len(reset_env_ids) > 0:
            env._observation_buffer[reset_env_ids] = obs_kinematic_current[reset_env_ids].unsqueeze(1).repeat(
                1,
                env.cfg.observation_buffer_length,
                1,
            )

        kinematic_history_flat = env._observation_buffer.view(env.num_envs, -1)
        obs_actions = torch.reshape(env._action_history, (env.num_envs, -1))
        obs_policy = torch.cat([kinematic_history_flat, obs_actions], dim=-1)

        obs_privileged = torch.cat(
            [
                env._disturbance_force[:, 0, :],
                env._disturbance_moment[:, 0, :],
                env._push_time.unsqueeze(dim=1),
                env._push_duration.unsqueeze(dim=1),
                env._time_elapsed.unsqueeze(dim=1),
                env._current_impulse,
                env._actions_filtered,
                env._alpha,
            ],
            dim=-1,
        )
        obs_critic = torch.cat([obs_policy, obs_privileged], dim=-1)

        return {"policy": obs_policy, "critic": obs_critic}

    def reset_action_buffers(self, env_ids: torch.Tensor, fill_actions_with_ones: bool):
        env = self.env
        if fill_actions_with_ones:
            env._actions[env_ids] = torch.ones_like(env._actions[env_ids])
        else:
            env._actions[env_ids] = torch.zeros_like(env._actions[env_ids])
        env._actions_filtered[env_ids] = torch.zeros_like(env._actions_filtered[env_ids])
        env._action_history[env_ids] = torch.zeros_like(env._action_history[env_ids])

    def randomized_joint_state(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        env = self.env
        joint_pos = env._robot.data.default_joint_pos[env_ids]
        joint_vel = env._robot.data.default_joint_vel[env_ids]

        for runtime in self.joint_groups.values():
            pos_sample = self._sample_group_range(env_ids, runtime.spec.initial_position_range, runtime.target_pos)
            vel_sample = self._sample_group_range(env_ids, runtime.spec.initial_velocity_range, runtime.target_vel)

            if pos_sample is not None:
                runtime.target_pos[env_ids] = pos_sample
                joint_pos[:, runtime.joint_ids] = self._expand_joint_group(runtime, pos_sample)
            if vel_sample is not None:
                joint_vel[:, runtime.joint_ids] = self._expand_joint_group(runtime, vel_sample)

        return joint_pos, joint_vel

    def deterministic_joint_state(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        env = self.env
        joint_pos = env._robot.data.default_joint_pos[env_ids]
        joint_vel = env._robot.data.default_joint_vel[env_ids]

        for runtime in self.joint_groups.values():
            actual_pos = joint_pos[:, runtime.joint_ids]
            runtime.target_pos[env_ids] = self._compress_joint_group(runtime, actual_pos)
            runtime.target_vel[env_ids] = torch.zeros_like(runtime.target_vel[env_ids])

        return joint_pos, joint_vel

    def reset_actuator_params(self, env_ids: torch.Tensor, randomized: bool):
        env = self.env
        nominal_kT = self._nominal_kT().to(env.device)
        nominal_kM = self._nominal_kM().to(env.device)

        if randomized:
            if self.spec.randomize_rotors_together:
                kT_error = torch.zeros(len(env_ids), 1, device=env.device).uniform_(-1.0, 1.0)
                kM_error = torch.zeros(len(env_ids), 1, device=env.device).uniform_(-1.0, 1.0)
            else:
                kT_error = torch.zeros(len(env_ids), self.rotor_count, device=env.device).uniform_(-1.0, 1.0)
                kM_error = torch.zeros(len(env_ids), self.rotor_count, device=env.device).uniform_(-1.0, 1.0)
            env.kT[env_ids] = nominal_kT * (1 + env.cfg.kT_error_scale * kT_error)
            env.kM[env_ids] = nominal_kM * (1 + env.cfg.kM_error_scale * kM_error)
        else:
            env.kT[env_ids] = nominal_kT
            env.kM[env_ids] = nominal_kM

        env.thrust_loss[env_ids] = 0.0
        if randomized and self._thrust_loss_enabled():
            loss_max = float(env._stage_value(env.cfg.thrust_loss_max, "thrust_loss_max"))
            if loss_max < 0.0 or loss_max > 1.0:
                raise ValueError(f"thrust_loss_max must be in [0, 1], got {loss_max}")
            if loss_max > 0.0:
                env.thrust_loss[env_ids] = torch.zeros(
                    len(env_ids),
                    self.rotor_count,
                    device=env.device,
                ).uniform_(0.0, loss_max)
                env.kT[env_ids] *= 1.0 - env.thrust_loss[env_ids]

        for runtime in self.joint_groups.values():
            if randomized:
                error = torch.zeros_like(runtime.max_velocity[env_ids]).uniform_(-1.0, 1.0)
                runtime.max_velocity[env_ids] = runtime.spec.max_velocity * (
                    1 + env.cfg.max_tilt_vel_error_scale * error
                )
            else:
                runtime.max_velocity[env_ids] = runtime.spec.max_velocity

    def rotor_action_rate(self) -> torch.Tensor:
        env = self.env
        current = self._rotor_action_values(env._actions)
        previous = self._rotor_action_values(env._action_history[:, 0, :])
        return torch.sum(torch.square(current - previous), dim=1)

    def rotor_action_magnitude(self) -> torch.Tensor:
        return torch.sum(torch.square(self._rotor_action_values(self.env._actions)), dim=1)

    def landing_tuck_position(self) -> torch.Tensor:
        if self.spec.landing_tuck_joint_group is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        runtime = self.joint_groups[self.spec.landing_tuck_joint_group]
        joint_pos = self.env._robot.data.joint_pos[:, runtime.joint_ids]
        return self._compress_joint_group(runtime, joint_pos).mean(dim=1)

    def landing_tuck_target(self) -> float:
        return self.spec.landing_tuck_target

    def nominal_total_kT(self) -> float:
        return sum(rotor.kT for rotor in self.spec.rotors)

    def _thrust_loss_enabled(self) -> bool:
        if self.spec.thrust_loss_start_stage is None:
            return False
        if self.spec.thrust_loss_start_stage < 1:
            raise ValueError(
                f"{self.spec.name} thrust_loss_start_stage must be >= 1 or None, "
                f"got {self.spec.thrust_loss_start_stage}"
            )
        return int(self.env.cfg.curriculum_stage) >= self.spec.thrust_loss_start_stage

    def _resolve_body_handles(self):
        env = self.env
        self.base_link = env._robot.find_bodies(self.spec.base_body_name)[0][0]
        self.rotor_ids = [
            env._robot.find_bodies(rotor.body_name)[0][0]
            for rotor in self.spec.rotors
        ]

    def _resolve_joint_groups(self):
        env = self.env
        for group in self.spec.joint_groups:
            joint_ids = [env._robot.find_joints(name)[0][0] for name in group.joint_names]
            action_width = self._action_width(group.action_name)
            target_pos = torch.zeros(env.num_envs, action_width, device=env.device)
            target_vel = torch.zeros_like(target_pos)
            max_velocity = group.max_velocity * torch.ones_like(target_pos)
            runtime = JointGroupRuntime(group, joint_ids, target_pos, target_vel, max_velocity)
            self.joint_groups[group.name] = runtime

    def _allocate_actuator_params(self):
        env = self.env
        env.kT = self._nominal_kT().to(env.device).repeat(env.num_envs, 1)
        env.kM = self._nominal_kM().to(env.device).repeat(env.num_envs, 1)
        self.spin_direction = torch.tensor(
            [rotor.spin_direction for rotor in self.spec.rotors],
            device=env.device,
        )

    def _resolve_contact_groups(self):
        env = self.env
        env._valid_contact_ids = [
            env._contact_sensor.body_names.index(name)
            for name in self.spec.contacts.valid_body_names
        ]
        env._invalid_contact_ids = [
            env._contact_sensor.body_names.index(name)
            for name in self.spec.contacts.invalid_body_names
        ]

    def _cache_inertial_params(self):
        env = self.env
        env._robot_mass = env._robot.root_physx_view.get_masses()[0].sum()
        env._gravity_magnitude = torch.tensor(env.sim.cfg.gravity, device=env.device).norm()
        env._robot_weight = (env._robot_mass * env._gravity_magnitude).item()

    def _filter_actions(self):
        env = self.env
        action_terms = self.action_schema.split(env._actions)
        filtered_terms = self.action_schema.split(env._actions_filtered)
        rotor_action_names = {rotor.action_name for rotor in self.spec.rotors}
        for term in self.action_schema.terms:
            if term.name not in rotor_action_names:
                filtered_terms[term.name][:] = action_terms[term.name]
        for action_name in rotor_action_names:
            filtered_terms[action_name][:] = (
                env._alpha * action_terms[action_name]
                + (1 - env._alpha) * filtered_terms[action_name]
            )

    def _quantize_joint_actions(self):
        env = self.env
        if not env.cfg.quantize_tilt_action:
            return
        filtered_terms = self.action_schema.split(env._actions_filtered)
        for runtime in self.joint_groups.values():
            if runtime.spec.quantize_action:
                filtered_terms[runtime.spec.action_name][:] = torch.round(filtered_terms[runtime.spec.action_name])

    def _compute_rotor_forces(self, filtered_terms: dict[str, torch.Tensor]):
        env = self.env
        rotor_actions = self._rotor_action_values_from_terms(filtered_terms)
        env.thrust[:, :, 2] = env.kT * rotor_actions
        env.moment[:, :, 2] = self.spin_direction * env.kM * env.thrust[:, :, 2]

    def _compute_joint_targets(self, filtered_terms: dict[str, torch.Tensor]):
        env = self.env
        for runtime in self.joint_groups.values():
            action = filtered_terms[runtime.spec.action_name]
            runtime.target_pos[:] = runtime.target_pos + runtime.max_velocity * action * env.physics_dt
            runtime.target_pos[:] = torch.clamp(runtime.target_pos, runtime.spec.lower, runtime.spec.upper)
            runtime.target_vel[:] = runtime.max_velocity * action

    def _compute_contact_impulse(self) -> torch.Tensor:
        env = self.env
        return env.step_dt * torch.sum(
            torch.linalg.norm(
                (
                    env._contact_sensor.data.net_forces_w_history[:, 1, :, :]
                    - env._contact_sensor.data.net_forces_w_history[:, 0, :, :]
                )
                * env.step_dt,
                dim=-1,
            ),
            dim=1,
        ).unsqueeze(dim=1)

    def _observation_value(self, term: ObservationSourceSpec) -> torch.Tensor:
        env = self.env
        if term.source == "relative_pos_w":
            return env._desired_pos_w - env._robot.data.root_link_pos_w
        if term.source == "root_rotation_matrix":
            return matrix_from_quat(env._robot.data.root_link_quat_w).reshape(-1, 9)
        if term.source == "root_lin_vel_w":
            return env._robot.data.root_com_lin_vel_w
        if term.source == "root_ang_vel_b":
            return env._robot.data.root_com_ang_vel_b
        if term.source == "joint_group_position":
            if term.joint_group is None:
                raise ValueError(f"Observation term {term.name} needs a joint_group")
            runtime = self.joint_groups[term.joint_group]
            joint_pos = env._robot.data.joint_pos[:, runtime.joint_ids]
            return self._compress_joint_group(runtime, joint_pos)
        raise ValueError(f"Unsupported observation source '{term.source}' for term '{term.name}'")

    def _observation_noise(self, term: ObservationSourceSpec, like: torch.Tensor) -> torch.Tensor:
        env = self.env
        scale = getattr(env.cfg, term.noise_scale) if isinstance(term.noise_scale, str) else term.noise_scale
        return scale * torch.zeros_like(like).uniform_(-1, 1)

    def _sample_group_range(
        self,
        env_ids: torch.Tensor,
        range_value: str | tuple[float, float] | None,
        like: torch.Tensor,
    ) -> torch.Tensor | None:
        if range_value is None:
            return None
        if isinstance(range_value, str):
            range_value = getattr(self.env.cfg, range_value)
        low, high = range_value
        return torch.zeros_like(like[env_ids]).uniform_(low, high)

    def _expand_joint_group(self, runtime: JointGroupRuntime, value: torch.Tensor) -> torch.Tensor:
        joint_count = len(runtime.joint_ids)
        if value.shape[1] == joint_count:
            return value
        if value.shape[1] == 1:
            return value.repeat(1, joint_count)
        raise ValueError(
            f"Joint group {runtime.spec.name} action width {value.shape[1]} cannot map to {joint_count} joints"
        )

    def _compress_joint_group(self, runtime: JointGroupRuntime, joint_values: torch.Tensor) -> torch.Tensor:
        expected_width = runtime.target_pos.shape[1]
        if joint_values.shape[1] == expected_width:
            return joint_values
        if expected_width == 1:
            return joint_values[:, :1]
        raise ValueError(
            f"Joint group {runtime.spec.name} has {joint_values.shape[1]} sim joints but needs width {expected_width}"
        )

    def _action_width(self, action_name: str) -> int:
        return self.action_schema.slices[action_name].stop - self.action_schema.slices[action_name].start

    def _rotor_action_values(self, actions: torch.Tensor) -> torch.Tensor:
        return self._rotor_action_values_from_terms(self.action_schema.split(actions))

    def _rotor_action_values_from_terms(self, action_terms: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack(
            [
                action_terms[rotor.action_name][:, rotor.action_index]
                for rotor in self.spec.rotors
            ],
            dim=1,
        )

    def _nominal_kT(self) -> torch.Tensor:
        return torch.tensor([[rotor.kT for rotor in self.spec.rotors]], dtype=torch.float)

    def _nominal_kM(self) -> torch.Tensor:
        return torch.tensor([[rotor.kM for rotor in self.spec.rotors]], dtype=torch.float)
