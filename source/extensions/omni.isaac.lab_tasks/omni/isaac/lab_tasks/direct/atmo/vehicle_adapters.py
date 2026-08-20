from __future__ import annotations

from dataclasses import dataclass
import math

import gymnasium as gym
import torch

from omni.isaac.lab.utils.math import matrix_from_quat

from .observation_runtime import ObservationRuntime
from .kinematics_runtime import KinematicsRuntime
from .task_utils import (
    heading_yaw_from_quat,
    rotation_matrix_to_heading_frame,
    to_heading_frame,
)
from .vehicle_specs import JointGroupSpec, ObservationSourceSpec, VehicleSpec


# Weight of the three balance columns in the morph hip basis, against the
# tilt_mean column's 1.0. The hip command is mixed then quantized with
# torch.round (round-half-to-even), and the deployment adapter reproduces that
# with np.rint -- so any corner whose mixed value lands exactly on +-0.5 is on a
# tie and quantizes to ZERO velocity while its neighbours clip to full speed.
#
# At 0.5 that is not an edge case: over the 16 fully saturated balance inputs,
# 32 of the 64 corner commands land exactly on the tie. Measured consequence on
# hardware -- the front arms flare less than the rear ones and stall at
# unrelated angles, and a 0.098 change in one balance action flips a corner
# between stopped and full speed.
#
# 0.4 puts every saturated corner at least 0.1 away from the tie (0 of 64 on
# it). Keep this in step with `joints.hip_basis` in the deployment repo's
# contracts/m4tii_combined_v1.json and fixtures/m4tii_action_mapping_v1.json;
# that contract is hand-maintained, not generated.
MORPH_BALANCE_SCALE = 0.4


@dataclass
class JointGroupRuntime:
    spec: JointGroupSpec
    joint_ids: list[int]
    target_pos: torch.Tensor
    target_vel: torch.Tensor
    max_velocity: torch.Tensor
    normalized_command: torch.Tensor
    effort_limit: torch.Tensor
    effort_command: torch.Tensor
    command_basis: torch.Tensor | None
    position_scale: torch.Tensor
    position_offset: torch.Tensor
    is_torque: bool
    is_thrust_center: bool
    is_morph_basis: bool
    # Velocity-setpoint group: the action commands runtime.target_vel directly
    # (through the sim velocity-target path); no position ramp, no effort.
    is_velocity: bool = False
    # Open-loop duty-cycle group. Implies is_torque (it writes effort_command),
    # so it must be tested first wherever the two branch.
    is_duty: bool = False
    # Per-joint servo zero offset, in SIM joint space, shape (num_envs, joints).
    # Added to the position target only, so the joint physically settles away
    # from where the policy asked -- an offset the policy can observe through
    # joint_group_position and must learn to trim out, which is what the real
    # servos do. None for torque groups, which have no position target.
    zero_offset: torch.Tensor | None = None


class DroneRuntime(KinematicsRuntime, ObservationRuntime):
    """Isaac articulation runtime shared by the two concrete M4 drones."""

    # Effort-path mappings: they write runtime.effort_command and drive no
    # position or velocity target. differential_duty is one of these -- it
    # ultimately commands a torque -- so it must appear here for joint
    # resolution, but it computes that torque differently (see below).
    _TORQUE_ACTION_MAPPINGS = ("torque", "differential_torque", "differential_duty")
    # Velocity-setpoint mappings drive the sim's joint velocity-target path
    # (runtime.target_vel) with zero torque command. differential_velocity
    # mixes the 2-dim [drive, turn] action through the differential wheel
    # basis to per-wheel normalized speeds, scaled by spec max_velocity.
    _VELOCITY_ACTION_MAPPINGS = ("differential_velocity",)
    # Open-loop duty-cycle mappings. The action is motor volts as a fraction of
    # supply, not a speed and not a torque, so the delivered torque carries a
    # back-EMF term and falls to zero at the no-load speed:
    #
    #     tau = effort_limit * (duty - joint_velocity / max_velocity)
    #
    # This is the encoderless RoboClaw transport. Checked BEFORE the plain
    # torque path in _compute_joint_targets, since these are torque mappings
    # too and would otherwise be swallowed by it.
    _DUTY_ACTION_MAPPINGS = ("differential_duty",)

    def __init__(self, env, spec: VehicleSpec):
        self.env = env
        self.spec = spec
        self.joint_direction = lambda _name: 1.0
        self.observation_provider = None
        self.action_schema = self.spec.make_action_schema()
        self.observation_schema = self.spec.make_observation_schema(history=True)
        self.current_observation_schema = self.spec.make_observation_schema(history=False)
        self.base_link = -1
        self.rotor_ids = []
        self.rotor_axis_ids = []
        self.wrench_body_ids = []
        self.torque_joint_ids = []
        self._torque_group_sim_slices = {}
        self.joint_groups = {}
        self.controlled_joint_ids = []
        self._joint_group_sim_slices = {}
        self._observation_pass_cache = {}
        self._leg_geometry_cache = None
        self._observation_noise_scales = {}
        # Populated in initialize(); defaulted here so a partially constructed
        # runtime cannot AttributeError inside the observation hot path.
        self._observation_noise_env_scale = {}
        self._nominal_total_kT_value = None
        self._nominal_total_rotor_moment_coeff_value = None

    @classmethod
    def _is_torque_mapping(cls, action_mapping: str) -> bool:
        return action_mapping in cls._TORQUE_ACTION_MAPPINGS

    @classmethod
    def _is_velocity_mapping(cls, action_mapping: str) -> bool:
        return action_mapping in cls._VELOCITY_ACTION_MAPPINGS

    @classmethod
    def _is_duty_mapping(cls, action_mapping: str) -> bool:
        return action_mapping in cls._DUTY_ACTION_MAPPINGS

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
        if self.current_observation_schema.dim != env.cfg.num_current_obs:
            raise ValueError(
                f"{self.spec.name} current observation schema dim {self.current_observation_schema.dim} "
                f"does not match cfg.num_current_obs={env.cfg.num_current_obs}"
            )
        self._observation_noise_scales = {
            term.name: self._observation_noise_scale(term) for term in self.spec.observation_terms
        }
        # Per-episode multiplier on the nominal noise scale, so the estimator's
        # quality varies between episodes instead of being one fixed number the
        # policy can tune itself to. Shape (num_envs, 1); broadcasts across each
        # term's width. Terms without an entry keep their fixed scale.
        #
        # PROVISIONAL ranges -- see the Phase 4 block in M4EnvCfg. Not measured
        # on this vehicle; OptiTrack has been unavailable.
        env._observation_noise_multiplier = {
            name: torch.ones(env.num_envs, 1, device=env.device)
            for name in ("pos_noise_scale", "lin_vel_noise_scale")
        }
        self._observation_noise_env_scale = {
            term.name: env._observation_noise_multiplier[term.noise_scale]
            for term in self.spec.observation_terms
            if isinstance(term.noise_scale, str)
            and term.noise_scale in env._observation_noise_multiplier
            and self._observation_noise_scales[term.name] != 0.0
        }

        action_dim = gym.spaces.flatdim(env.single_action_space)
        env._observation_buffer = torch.zeros(
            env.num_envs,
            env.cfg.observation_buffer_length,
            self.observation_schema.dim,
            device=env.device,
        )
        env._reset_observation_history_pending = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._reset_observation_history_dirty = False
        env._observation_history_index = 0
        env._observation_delay_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._observation_history_offsets = torch.arange(
            env.cfg.observation_history_length,
            device=env.device,
            dtype=torch.long,
        ).view(1, env.cfg.observation_history_length)
        env._observation_history_env_indices = torch.arange(env.num_envs, device=env.device, dtype=torch.long).view(
            env.num_envs,
            1,
        )
        env._current_observation_buffer = torch.zeros(
            env.num_envs,
            env.cfg.observation_buffer_length,
            self.current_observation_schema.dim,
            device=env.device,
        )
        env._actions = torch.zeros(env.num_envs, action_dim, device=env.device)
        env._policy_actions = torch.zeros_like(env._actions)
        env._actions_filtered = torch.zeros_like(env._actions)
        env._previous_actions_filtered = torch.zeros_like(env._actions)
        env._previous_actions = torch.zeros_like(env._actions)
        env._previous_action_delta = torch.zeros_like(env._actions)
        env.normalized_rotor_thrust = torch.zeros(env.num_envs, self.rotor_count, device=env.device)
        env.normalized_rotor_thrust_filtered = torch.full_like(env.normalized_rotor_thrust, 0.5)
        env.previous_normalized_rotor_thrust = torch.zeros_like(env.normalized_rotor_thrust)
        env.previous_normalized_rotor_thrust_filtered = torch.full_like(env.normalized_rotor_thrust, 0.5)
        # Phase 5 resilience state, per env.
        # Command dropout: the deployed backend zeroes rotors after 100 ms and
        # holds joint targets, so a stuttering stream is a state this vehicle
        # actually reaches. Held actions are replayed for the sampled window.
        env._command_dropout_remaining_steps = torch.zeros(
            env.num_envs, 1, dtype=torch.long, device=env.device
        )
        env._command_dropout_probability = torch.zeros(env.num_envs, 1, device=env.device)
        env._held_policy_actions = torch.zeros_like(env._actions)
        env._action_history = torch.zeros(
            env.num_envs,
            env.cfg.action_history_length,
            action_dim,
            device=env.device,
        )
        env._action_history_index = 0
        self._obs_history_dim = env.cfg.observation_history_length * self.observation_schema.dim
        self._action_history_dim = env.cfg.action_history_length * action_dim
        self._obs_history_slice = slice(0, self._obs_history_dim)
        self._action_history_slice = slice(
            self._obs_history_dim,
            self._obs_history_dim + self._action_history_dim,
        )
        self._obs_command_slice = slice(
            self._obs_history_dim + self._action_history_dim,
            env.cfg.observation_space,
        )
        task_observation_dim = int(getattr(env.cfg, "task_observation_dim", 0))
        expected_command_dim = self.current_observation_schema.dim + task_observation_dim
        if self._obs_command_slice.stop - self._obs_command_slice.start != expected_command_dim:
            raise ValueError(
                f"{self.spec.name} observation layout mismatch: command slice width "
                f"{self._obs_command_slice.stop - self._obs_command_slice.start} but schema has "
                f"{expected_command_dim}"
            )
        self._base_command_slice = slice(
            self._obs_command_slice.start,
            self._obs_command_slice.start + self.current_observation_schema.dim,
        )
        self._task_observation_slice = slice(
            self._base_command_slice.stop,
            self._obs_command_slice.stop,
        )
        env._obs_policy = torch.zeros(env.num_envs, env.cfg.observation_space, device=env.device)
        env._reset_policy_actions = torch.zeros(
            env.num_envs, self.action_schema.dim, device=env.device
        )
        env._reset_policy_actions[:, self.action_schema.slices["lift"]] = -1.0
        env._obs_kinematic_current = torch.zeros(env.num_envs, self.observation_schema.dim, device=env.device)
        env._obs_command_current = torch.zeros(env.num_envs, self.current_observation_schema.dim, device=env.device)
        env._empty_obs_command = torch.empty(env.num_envs, 0, device=env.device)

        env.rotor_thrust_magnitude = torch.zeros(env.num_envs, self.rotor_count, device=env.device)
        env.rotor_moment_magnitude = torch.zeros(env.num_envs, self.rotor_count, device=env.device)
        env._debug_thrust_w = torch.zeros(env.num_envs, self.rotor_count, 3, device=env.device)
        env._debug_moment_w = torch.zeros_like(env._debug_thrust_w)
        env._debug_disturbance_force_w = torch.zeros(env.num_envs, 1, 3, device=env.device)
        env._debug_disturbance_moment_w = torch.zeros(env.num_envs, 1, 3, device=env.device)
        env._scratch_base_force_w = torch.zeros(env.num_envs, 1, 3, device=env.device)
        env._scratch_base_moment_w = torch.zeros(env.num_envs, 1, 3, device=env.device)
        env._scratch_rotor_force_w = torch.zeros(env.num_envs, self.rotor_count, 3, device=env.device)
        env._scratch_rotor_moment_w = torch.zeros_like(env._scratch_rotor_force_w)
        env._total_force_w = torch.zeros(env.num_envs, self.rotor_count + 1, 3, device=env.device)
        env._total_moment_w = torch.zeros(env.num_envs, self.rotor_count + 1, 3, device=env.device)
        env._joint_position_target_sim = torch.empty(env.num_envs, 0, device=env.device)
        env._joint_velocity_target_sim = torch.empty(env.num_envs, 0, device=env.device)
        env._joint_effort_target_sim = torch.empty(env.num_envs, 0, device=env.device)
        env._time_elapsed = torch.zeros(env.num_envs, device=env.device)
        # One column PER ROTOR, not one shared column. A single tau for all four
        # rotors is a common-mode lag: it delays collective thrust but leaves the
        # differential response -- which is what attitude control actually rides
        # on -- perfectly matched, so the policy never trains against rotors that
        # answer at different speeds. Per-rotor draws exercise that directly.
        # Safe at every curriculum stage: differential rotor error is corrected
        # through roll_pitch_yaw, which is never gated by action_authority (see
        # the tilt_balance / thrust_center_xy gating in _map_policy_actions).
        env._alpha_rise = env.cfg.alpha_0 * torch.ones(env.num_envs, len(self.spec.rotors), device=env.device)
        env._alpha_fall = env.cfg.alpha_0 * torch.ones_like(env._alpha_rise)
        env._actuator_tau_rise = env.cfg.T_m_0 * torch.ones_like(env._alpha_rise)
        env._actuator_tau_fall = env.cfg.T_m_0 * torch.ones_like(env._alpha_rise)
        # Observed-tau channel (packed into the tail of the allocation-matrix
        # observation, see _allocation_matrix_with_tau): the policy sees the
        # drawn per-rotor lag as a hint, not gospel. A per-episode multiplicative
        # error models the deployed estimator, and a per-episode dropout swaps
        # the hint for the band midpoint (encoded 0) so the policy retains a
        # robust fallback when the estimator is cold.
        env._tau_obs_noise_rise = torch.ones_like(env._actuator_tau_rise)
        env._tau_obs_noise_fall = torch.ones_like(env._actuator_tau_fall)
        env._tau_obs_dropout = torch.zeros(env.num_envs, 1, dtype=torch.bool, device=env.device)

        self._resolve_body_handles()
        has_native_joint_effort = hasattr(env._robot, "set_joint_effort_target")
        self._resolve_joint_groups()
        if self.torque_joint_ids and not has_native_joint_effort:
            raise RuntimeError("Torque-controlled joint groups require native joint-effort targets")
        self._allocate_wrench_buffers()
        self._allocate_joint_target_buffers()
        self._allocate_joint_effort_buffers()
        self._validate_configured_joint_bounds()
        self._cache_inertial_params()
        self._allocate_actuator_params()
        self._resolve_contact_groups()

    def get_observations(self) -> dict:
        return self._pack_observations()

    def bind_task(self, task) -> None:
        self.joint_direction = getattr(task, "joint_action_direction", lambda _name: 1.0)
        self.observation_provider = task.observation_value
        self.task_observation_provider = getattr(task, "task_observation", None)
        # Deployment-parity actuation gates; tasks that do not define them run
        # ungated as before.
        self.rotor_thrust_gate_provider = getattr(task, "rotor_thrust_gate", None)
        self.wheel_speed_gate_provider = getattr(task, "wheel_speed_gate", None)

    def pre_physics_step(self, actions: torch.Tensor):
        env = self.env
        env._time_elapsed += env.step_dt
        env._previous_action_delta[:] = env._actions - env._previous_actions
        env._previous_actions[:] = env._actions
        env._previous_actions_filtered[:] = env._actions_filtered
        env.previous_normalized_rotor_thrust[:] = env.normalized_rotor_thrust
        env.previous_normalized_rotor_thrust_filtered[:] = env.normalized_rotor_thrust_filtered
        env._policy_actions.copy_(actions)
        self._sanitize_action_(env._policy_actions)
        env._policy_actions.clamp_(-1.0, 1.0)
        self._apply_command_dropout(env._policy_actions)
        self._map_policy_actions_into(env._policy_actions, env._actions)
        env.normalized_rotor_thrust.copy_(self._rotor_action_values(env._actions))
        self._filter_normalized_rotor_thrust()
        env._actions_filtered.copy_(env._actions)

        filtered_terms = self.action_schema.split(env._actions_filtered)
        self._compute_rotor_forces()
        self._compute_joint_targets(filtered_terms)

    def _apply_command_dropout(self, policy_actions: torch.Tensor):
        """Replay the last action while the command stream is 'dropped'.

        Modelled on the deployment backend's behaviour rather than on noise: it
        holds the last command for a window, so the policy must stay stable
        across a stretch where its own output has no effect. Envs that are not
        currently dropping may start a new window with per-env probability.
        """
        env = self.env
        remaining = getattr(env, "_command_dropout_remaining_steps", None)
        if remaining is None:
            return
        active = remaining > 0
        if bool(torch.any(active)):
            policy_actions.copy_(torch.where(active, env._held_policy_actions, policy_actions))
            remaining.sub_(active.long()).clamp_(min=0)
        probability = env._command_dropout_probability
        if float(probability.max()) <= 0.0:
            return
        begin = (remaining <= 0) & (torch.rand_like(probability) < probability)
        if not bool(torch.any(begin)):
            return
        hold_low, hold_high = (float(value) for value in env.cfg.command_dropout_hold_s)
        if hold_low < 0.0 or hold_high < hold_low:
            raise ValueError(f"command_dropout_hold_s must satisfy 0 <= min <= max, got {(hold_low, hold_high)}")
        hold_steps = torch.ceil(
            torch.empty_like(probability).uniform_(hold_low, hold_high) / float(env.step_dt)
        ).long()
        remaining.copy_(torch.where(begin, hold_steps.clamp_min(1), remaining))
        # The window starts NOW: this step's action is the one that gets held.
        env._held_policy_actions.copy_(torch.where(begin, policy_actions, env._held_policy_actions))

    def apply_action(self):
        env = self.env
        base_force_w = env._scratch_base_force_w
        base_moment_w = env._scratch_base_moment_w
        base_force_w.zero_()
        base_moment_w.zero_()

        if env.cfg.disturb:
            disturbance_weight = env.disturbance_weight()
            torch.logical_and(
                env._time_elapsed >= env._push_time,
                env._time_elapsed <= env._push_end_time,
                out=env._push_active,
            )
            push = env._push_active.view(env.num_envs, 1, 1)
            base_force_w.copy_(env._disturbance_force).mul_(push).mul_(disturbance_weight)
            base_moment_w.copy_(env._disturbance_moment).mul_(push).mul_(disturbance_weight)
            base_force_w.add_(env._disturbance_force_cts, alpha=disturbance_weight)
            base_moment_w.add_(env._disturbance_moment_cts, alpha=disturbance_weight)

        if env.cfg.thrust_vector_debug_vis:
            env._debug_disturbance_force_w.copy_(base_force_w)
            env._debug_disturbance_moment_w.copy_(base_moment_w)
        thrust_axis_w = self._current_rotor_thrust_axis_w()
        env._current_thrust_axis_w.copy_(thrust_axis_w)

        rotor_force_w = env._scratch_rotor_force_w
        rotor_moment_w = env._scratch_rotor_moment_w
        torch.mul(env.rotor_thrust_magnitude.unsqueeze(-1), thrust_axis_w, out=rotor_force_w)
        torch.mul(env.rotor_moment_magnitude.unsqueeze(-1), thrust_axis_w, out=rotor_moment_w)

        if env.cfg.thrust_vector_debug_vis:
            env._debug_thrust_w.copy_(rotor_force_w)
            env._debug_moment_w.copy_(rotor_moment_w)

        total_force = env._total_force_w
        total_moment = env._total_moment_w
        total_force[:, : self.rotor_count, :].copy_(rotor_force_w)
        total_force[:, self.rotor_count : self.rotor_count + 1, :].copy_(base_force_w)
        total_moment[:, : self.rotor_count, :].copy_(rotor_moment_w)
        total_moment[:, self.rotor_count : self.rotor_count + 1, :].copy_(base_moment_w)
        torch.nan_to_num(total_force, nan=0.0, posinf=1e6, neginf=-1e6, out=total_force)
        torch.nan_to_num(total_moment, nan=0.0, posinf=1e6, neginf=-1e6, out=total_moment)
        torch.clamp(total_force, -1e6, 1e6, out=total_force)
        torch.clamp(total_moment, -1e6, 1e6, out=total_moment)
        self._apply_external_wrench_world(
            total_force,
            total_moment,
            body_ids=self.wrench_body_ids,
        )
        self._apply_joint_targets()
        self._apply_joint_efforts()

    def reset_action_buffers(self, env_ids: torch.Tensor):
        env = self.env
        env._policy_actions[env_ids] = env._reset_policy_actions[env_ids]
        env._actions[env_ids].copy_(env._policy_actions[env_ids])
        self._map_policy_actions_into(env._actions[env_ids], env._actions[env_ids])
        env._previous_actions[env_ids] = env._actions[env_ids]
        env._previous_action_delta[env_ids] = 0.0
        env.normalized_rotor_thrust[env_ids] = self._rotor_action_values(env._actions[env_ids])
        env.previous_normalized_rotor_thrust[env_ids] = env.normalized_rotor_thrust[env_ids]
        env._actions_filtered[env_ids] = env._actions[env_ids]
        env._previous_actions_filtered[env_ids] = env._actions[env_ids]
        env.normalized_rotor_thrust_filtered[env_ids].fill_(0.5)
        env.previous_normalized_rotor_thrust_filtered[env_ids].fill_(0.5)
        env._action_history[env_ids] = env._reset_policy_actions[env_ids].unsqueeze(1)
        n = env_ids.shape[0]
        env._tau_obs_noise_rise[env_ids] = 0.85 + 0.30 * torch.rand(
            n, env._tau_obs_noise_rise.shape[1], device=env.device
        )
        env._tau_obs_noise_fall[env_ids] = 0.85 + 0.30 * torch.rand(
            n, env._tau_obs_noise_fall.shape[1], device=env.device
        )
        env._tau_obs_dropout[env_ids] = torch.rand(n, 1, device=env.device) < 0.10
        self._sample_observation_delay_steps(env_ids)
        for runtime in self.joint_groups.values():
            runtime.effort_command[env_ids] = 0.0

    def randomized_joint_state(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        env = self.env
        joint_pos = env._robot.data.default_joint_pos[env_ids]
        joint_vel = env._robot.data.default_joint_vel[env_ids]

        for runtime in self.joint_groups.values():
            pos_sample = self._sample_group_range(env_ids, runtime.spec.initial_position_range, runtime.target_pos)
            vel_sample = self._sample_group_range(env_ids, runtime.spec.initial_velocity_range, runtime.target_vel)

            if pos_sample is not None:
                runtime.target_pos[env_ids] = pos_sample
                joint_pos[:, runtime.joint_ids] = self._joint_group_position_to_sim(runtime, pos_sample)
            if vel_sample is not None:
                joint_vel[:, runtime.joint_ids] = self._joint_group_velocity_to_sim(runtime, vel_sample)

        return joint_pos, joint_vel

    def deterministic_joint_state(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        env = self.env
        joint_pos = env._robot.data.default_joint_pos[env_ids]
        joint_vel = env._robot.data.default_joint_vel[env_ids]

        for runtime in self.joint_groups.values():
            actual_pos = joint_pos[:, runtime.joint_ids]
            runtime.target_pos[env_ids] = self._joint_group_position_from_sim(runtime, actual_pos)
            runtime.target_vel[env_ids] = torch.zeros_like(runtime.target_vel[env_ids])

        return joint_pos, joint_vel

    def randomize_joint_group_position(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        env_ids: torch.Tensor,
        group_name: str | None,
    ):
        if group_name is None or group_name not in self.joint_groups:
            return
        runtime = self.joint_groups[group_name]
        if runtime.target_pos.shape[1] > 1:
            self._randomize_runtime_position(runtime, joint_pos, joint_vel, env_ids)

    def seed_thrust_center_joint_group(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        env_ids: torch.Tensor,
        cx_range: tuple[float, float],
        cy_range: tuple[float, float],
        max_angle: float,
    ):
        env = self.env
        stage = env.curriculum_stage_for_value("thrust_center")
        if stage < self.env.cfg.thrust_center_start_stage:
            return
        group_name = self.spec.leg_joint_group
        if group_name is None or group_name not in self.joint_groups:
            return
        max_angle = max(float(max_angle), 0.0)
        if max_angle <= 0.0:
            return
        cx_low, cx_high = float(cx_range[0]), float(cx_range[1])
        cy_low, cy_high = float(cy_range[0]), float(cy_range[1])
        if cx_high < cx_low or cy_high < cy_low:
            raise ValueError("initial thrust center seed ranges must be ordered low <= high")
        if cx_high == cx_low == 0.0 and cy_high == cy_low == 0.0:
            return

        runtime = self.joint_groups[group_name]
        if runtime.target_pos.shape[1] == 1:
            lr_basis = runtime.target_pos.new_zeros(1, 1)
            fb_basis = torch.zeros_like(lr_basis)
        elif runtime.target_pos.shape[1] == 2:
            lr_basis = runtime.target_pos.new_tensor((-1.0, 1.0)).reshape(1, 2)
            fb_basis = torch.zeros_like(lr_basis)
        elif runtime.target_pos.shape[1] == 4:
            lr_basis = runtime.target_pos.new_tensor((-1.0, 1.0, -1.0, 1.0)).reshape(1, 4)
            fb_basis = runtime.target_pos.new_tensor((1.0, 1.0, -1.0, -1.0)).reshape(1, 4)
        else:
            raise ValueError(f"Thrust-center seed expects 1, 2, or 4 joints, got {runtime.target_pos.shape[1]}")
        cx = torch.empty(len(env_ids), 1, device=env.device, dtype=runtime.target_pos.dtype).uniform_(cx_low, cx_high)
        cy = torch.empty(len(env_ids), 1, device=env.device, dtype=runtime.target_pos.dtype).uniform_(cy_low, cy_high)
        seed = max_angle * torch.clamp(cx * fb_basis + cy * lr_basis, min=0.0, max=1.0)
        self.set_joint_group_state_values(joint_pos, joint_vel, env_ids, group_name, seed)

    def _randomize_runtime_position(
        self,
        runtime: JointGroupRuntime,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        env_ids: torch.Tensor,
    ):
        pos_sample = self._sample_group_range(env_ids, runtime.spec.initial_position_range, runtime.target_pos)
        if pos_sample is None:
            return
        runtime.target_pos[env_ids] = pos_sample
        runtime.target_vel[env_ids] = 0.0
        joint_pos[:, runtime.joint_ids] = self._joint_group_position_to_sim(runtime, pos_sample)
        joint_vel[:, runtime.joint_ids] = 0.0

    def set_joint_group_state_values(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        env_ids: torch.Tensor,
        group_name: str | None,
        position_values: torch.Tensor | float | tuple[float, ...],
        velocity_values: torch.Tensor | float = 0.0,
    ):
        if group_name is None:
            return
        runtime = self.joint_groups[group_name]
        target_like = runtime.target_pos[env_ids]
        if isinstance(position_values, torch.Tensor):
            position_values = position_values.to(device=target_like.device, dtype=target_like.dtype)
        else:
            position_values = self._joint_value_like(target_like, position_values)
        if position_values.ndim == 1:
            position_values = position_values.unsqueeze(1)
        if position_values.shape[0] != target_like.shape[0]:
            raise ValueError(
                f"Joint group {group_name} got {position_values.shape[0]} position rows, "
                f"expected {target_like.shape[0]}"
            )
        if position_values.shape[1] == 1 and target_like.shape[1] > 1:
            position_values = position_values.expand_as(target_like)
        elif position_values.shape[1] != target_like.shape[1]:
            raise ValueError(
                f"Joint group {group_name} got {position_values.shape[1]} position columns, "
                f"expected {target_like.shape[1]}"
            )
        self._validate_joint_value_bounds(runtime, position_values, f"{group_name} joint state")
        position_values = torch.clamp(position_values, runtime.spec.lower, runtime.spec.upper)

        if isinstance(velocity_values, torch.Tensor):
            velocity = velocity_values.to(device=target_like.device, dtype=target_like.dtype)
            if velocity.ndim == 1:
                velocity = velocity.unsqueeze(1)
            if velocity.shape[1] == 1 and target_like.shape[1] > 1:
                velocity = velocity.expand_as(target_like)
            elif velocity.shape != target_like.shape:
                raise ValueError(
                    f"Joint group {group_name} velocity shape {tuple(velocity.shape)} does not match "
                    f"{tuple(target_like.shape)}"
                )
        else:
            velocity = float(velocity_values) * torch.ones_like(target_like)

        runtime.target_pos[env_ids] = position_values
        runtime.target_vel[env_ids] = velocity
        joint_pos[:, runtime.joint_ids] = self._joint_group_position_to_sim(runtime, position_values)
        joint_vel[:, runtime.joint_ids] = self._joint_group_velocity_to_sim(runtime, velocity)

    def set_joint_config_state(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        env_ids: torch.Tensor,
        joint_config: tuple[tuple[str, float | tuple[float, ...]], ...],
        velocity: float = 0.0,
    ):
        for group_name, position in joint_config:
            self.set_joint_group_state_values(joint_pos, joint_vel, env_ids, group_name, position, velocity)

    def rotor_action_rate(self) -> torch.Tensor:
        env = self.env
        current = env.normalized_rotor_thrust
        previous = env.previous_normalized_rotor_thrust
        return torch.sum(torch.square(current - previous), dim=1)

    def rotor_action_values(self, filtered: bool = False) -> torch.Tensor:
        return self.env.normalized_rotor_thrust_filtered if filtered else self.env.normalized_rotor_thrust

    def thrust_center_offset_body(self) -> torch.Tensor:
        env = self.env
        rotor_r_b = self._rotor_relative_pos_body()
        actual_center = self._weighted_rotor_center_xy(rotor_r_b, env.kT)
        nominal_kT = self._nominal_kT_tensor.to(device=env.kT.device, dtype=env.kT.dtype).expand_as(env.kT)
        nominal_center = self._weighted_rotor_center_xy(rotor_r_b, nominal_kT)
        return actual_center - nominal_center

    def action_rate(self) -> torch.Tensor:
        return self._normalized_action_difference(self.env._actions - self.env._previous_actions)

    def action_jerk(self) -> torch.Tensor:
        env = self.env
        action_delta = env._actions - env._previous_actions
        jerk = action_delta - env._previous_action_delta
        wheel_slice = self.action_schema.slices.get("wheel_speed")
        if wheel_slice is not None:
            wheel_group = next(
                group
                for group in self.spec.joint_groups
                if group.action_name == "wheel_speed"
            )
            # The two policy channels are virtual [drive, turn] wheel-speed
            # commands. Convert them to fractions of maximum physical wheel
            # speed before squaring: drive uses its full authority while turn
            # uses the smaller differential authority. Do not expand to four
            # wheel targets, which would count the same virtual command once
            # per wheel.
            jerk = jerk.clone()
            jerk[:, wheel_slice] *= jerk.new_tensor(
                (
                    float(wheel_group.differential_drive_scale),
                    float(wheel_group.differential_turn_scale),
                )
            )
        return self._normalized_action_difference(jerk)

    def _normalized_action_difference(self, difference: torch.Tensor) -> torch.Tensor:
        env = self.env
        active_dimensions = float(self.action_schema.dim)
        base_dimensions = float(self.action_schema.dim)
        for name, start_stage, ramp_epochs in (
            ("tilt_balance", env.cfg.morph_bias_start_stage, env.cfg.morph_balance_authority_ramp_epochs),
            ("thrust_center_xy", env.cfg.thrust_center_start_stage, env.cfg.thrust_center_authority_ramp_epochs),
        ):
            action_slice = self.action_schema.slices.get(name)
            if action_slice is None:
                continue
            width = action_slice.stop - action_slice.start
            authority = env.action_authority(start_stage, ramp_epochs)
            active_dimensions -= width * (1.0 - authority * authority)
            base_dimensions -= width
        return (
            torch.sum(torch.square(difference), dim=1)
            * max(base_dimensions, 1.0)
            / max(active_dimensions, 1.0)
        )

    def hover_collective_throttle(self) -> float:
        return self._nominal_hover_collective_throttle()

    def action_term_values(self, action_name: str, filtered: bool = False) -> torch.Tensor:
        actions = self.env._actions_filtered if filtered else self.env._actions
        return self.action_schema.split(actions)[action_name]

    def joint_group_positions(self, group_name: str) -> torch.Tensor:
        runtime = self.joint_groups[group_name]
        joint_pos = self.env._robot.data.joint_pos[:, runtime.joint_ids]
        return self._joint_group_position_from_sim(runtime, joint_pos)

    def joint_group_velocities(self, group_name: str) -> torch.Tensor:
        runtime = self.joint_groups[group_name]
        joint_vel = self.env._robot.data.joint_vel[:, runtime.joint_ids]
        return self._joint_group_velocity_from_sim(runtime, joint_vel)

    def joint_group_positions_from_joint_state(self, group_name: str, joint_pos: torch.Tensor) -> torch.Tensor:
        runtime = self.joint_groups[group_name]
        return self._joint_group_position_from_sim(runtime, joint_pos[:, runtime.joint_ids])

    def actuator_group_position_disagreement(self, group_name: str) -> torch.Tensor:
        actual_pos = self.joint_group_positions(group_name)
        if actual_pos.shape[1] <= 1:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        mean_pos = torch.mean(actual_pos, dim=1, keepdim=True)
        return torch.mean(torch.square(actual_pos - mean_pos), dim=1)

    def morph_joint_positions(self) -> torch.Tensor:
        if self.spec.morph_joint_group is None:
            return torch.zeros(self.env.num_envs, 1, device=self.env.device)
        runtime = self.joint_groups[self.spec.morph_joint_group]
        joint_pos = self.env._robot.data.joint_pos[:, runtime.joint_ids]
        return self._joint_group_position_from_sim(runtime, joint_pos)

    def morph_joint_positions_from_joint_state(self, joint_pos: torch.Tensor) -> torch.Tensor:
        if self.spec.morph_joint_group is None:
            return torch.zeros(joint_pos.shape[0], 1, device=joint_pos.device, dtype=joint_pos.dtype)
        return self.joint_group_positions_from_joint_state(self.spec.morph_joint_group, joint_pos)

    def joint_config_target_values(
        self,
        joint_config: tuple[tuple[str, float | tuple[float, ...]], ...],
        group_name: str | None,
        like: torch.Tensor,
        fallback: float | tuple[float, ...] | None = None,
    ) -> torch.Tensor | None:
        target = self._joint_config_value(joint_config, group_name)
        if target is None:
            if fallback is None:
                return None
            target = fallback
        return self._joint_value_like(like, target)

    def nominal_total_kT(self) -> float:
        if self._nominal_total_kT_value is not None:
            return self._nominal_total_kT_value
        return float(self._nominal_kT_tensor.sum().item())

    def nominal_total_rotor_moment_coeff(self) -> float:
        if self._nominal_total_rotor_moment_coeff_value is not None:
            return self._nominal_total_rotor_moment_coeff_value
        kT = self._nominal_kT_tensor.reshape(-1)
        kM = self._nominal_kM_tensor.reshape(-1)
        return float(torch.sum(torch.abs(kT * kM)).item())

    def _resolve_body_handles(self):
        env = self.env
        self.base_link = env._robot.find_bodies(self.spec.base_body_name)[0][0]
        self.rotor_ids = [env._robot.find_bodies(rotor.body_name)[0][0] for rotor in self.spec.rotors]
        self.rotor_axis_ids = [
            env._robot.find_bodies(rotor.axis_body_name or rotor.body_name)[0][0] for rotor in self.spec.rotors
        ]

    def _apply_external_wrench_world(
        self,
        force_w: torch.Tensor,
        moment_w: torch.Tensor,
        body_ids: list[int],
    ):
        env = self.env
        body_quats_w = torch.nan_to_num(self._body_quat_w(body_ids), nan=0.0, posinf=0.0, neginf=0.0)
        env._robot.set_external_force_and_torque(
            self._rotate_world_vectors_to_body(body_quats_w, force_w),
            self._rotate_world_vectors_to_body(body_quats_w, moment_w),
            body_ids=body_ids,
        )

    def _resolve_joint_groups(self):
        env = self.env
        self.controlled_joint_ids = []
        self.torque_joint_ids = []
        self._joint_group_sim_slices = {}
        self._torque_group_sim_slices = {}
        for group in self.spec.joint_groups:
            joint_ids = [env._robot.find_joints(name)[0][0] for name in group.joint_names]
            if not self._is_torque_mapping(group.action_mapping):
                start = len(self.controlled_joint_ids)
                self.controlled_joint_ids.extend(joint_ids)
                self._joint_group_sim_slices[group.name] = slice(start, start + len(joint_ids))
            else:
                start = len(self.torque_joint_ids)
                self.torque_joint_ids.extend(joint_ids)
                self._torque_group_sim_slices[group.name] = slice(start, start + len(joint_ids))
            action_slice = self.action_schema.slices[group.action_name]
            action_width = action_slice.stop - action_slice.start
            if (
                group.action_mapping
                in (
                    "morph_tilt_basis",
                    "thrust_center_shift",
                )
                or self._is_torque_mapping(group.action_mapping)
                or self._is_velocity_mapping(group.action_mapping)
            ):
                target_width = len(joint_ids)
            else:
                target_width = action_width
            target_pos = torch.zeros(env.num_envs, target_width, device=env.device)
            target_vel = torch.zeros_like(target_pos)
            max_velocity = group.max_velocity * torch.ones_like(target_pos)
            normalized_command = torch.zeros_like(target_pos)
            effort_limit = self._joint_group_param(group.effort_limit, group, target_pos)
            position_scale = self._joint_group_param(group.position_scale, group, target_pos)
            position_offset = self._joint_group_param(group.position_offset, group, target_pos)
            effort_command = torch.zeros_like(target_pos)
            runtime = JointGroupRuntime(
                group,
                joint_ids,
                target_pos,
                target_vel,
                max_velocity,
                normalized_command,
                effort_limit,
                effort_command,
                self._joint_command_basis(group, len(joint_ids), target_width, env.device, target_pos.dtype),
                position_scale,
                position_offset,
                self._is_torque_mapping(group.action_mapping),
                group.action_mapping == "thrust_center_shift",
                group.action_mapping == "morph_tilt_basis",
                is_velocity=self._is_velocity_mapping(group.action_mapping),
                is_duty=self._is_duty_mapping(group.action_mapping),
                # Sized in SIM joint space. Torque and velocity groups drive no
                # position target, so they carry no zero offset.
                zero_offset=None
                if (
                    self._is_torque_mapping(group.action_mapping)
                    or self._is_velocity_mapping(group.action_mapping)
                )
                else torch.zeros(env.num_envs, len(joint_ids), device=env.device),
            )
            self.joint_groups[group.name] = runtime

    def _joint_command_basis(
        self,
        group: JointGroupSpec,
        joint_count: int,
        target_width: int,
        device,
        dtype,
    ) -> torch.Tensor | None:
        if group.action_mapping == "morph_tilt_basis" and target_width > 1:
            basis = torch.zeros(target_width, 4, device=device, dtype=dtype)
            basis[:, 0] = 1.0
            if target_width == 2:
                basis[:, 1] = MORPH_BALANCE_SCALE * torch.tensor((-1.0, 1.0), device=device, dtype=dtype)
            elif target_width == 4:
                basis[:, 1:] = MORPH_BALANCE_SCALE * torch.tensor(
                    ((-1.0, 1.0, -1.0, 1.0), (1.0, 1.0, -1.0, -1.0), (1.0, -1.0, -1.0, 1.0)),
                    device=device,
                    dtype=dtype,
                ).T
            else:
                raise ValueError(f"Morph command basis expects 1, 2, or 4 joints, got {target_width}")
            return basis
        if group.action_mapping in (
            "differential_torque",
            "differential_velocity",
            "differential_duty",
        ):
            if joint_count == 1:
                lateral = torch.zeros(1, device=device, dtype=dtype)
            elif joint_count == 2:
                lateral = torch.tensor((-1.0, 1.0), device=device, dtype=dtype)
            elif joint_count == 4:
                lateral = torch.tensor((-1.0, 1.0, -1.0, 1.0), device=device, dtype=dtype)
            else:
                raise ValueError(f"Differential command basis expects 1, 2, or 4 joints, got {joint_count}")
            return torch.stack(
                (
                    torch.full_like(lateral, float(group.differential_drive_scale)),
                    float(group.differential_turn_scale) * lateral,
                ),
                dim=1,
            )
        return None

    def _allocate_wrench_buffers(self):
        env = self.env
        self.wrench_body_ids = self.rotor_ids + [self.base_link]
        wrench_count = len(self.wrench_body_ids)
        env._total_force_w = torch.zeros(env.num_envs, wrench_count, 3, device=env.device)
        env._total_moment_w = torch.zeros_like(env._total_force_w)

    def _allocate_joint_target_buffers(self):
        env = self.env
        joint_count = len(self.controlled_joint_ids)
        env._joint_position_target_sim = torch.zeros(env.num_envs, joint_count, device=env.device)
        env._joint_velocity_target_sim = torch.zeros_like(env._joint_position_target_sim)

    def _allocate_joint_effort_buffers(self):
        env = self.env
        joint_count = len(self.torque_joint_ids)
        env._joint_effort_target_sim = torch.zeros(env.num_envs, joint_count, device=env.device)

    def _apply_joint_targets(self):
        if not self.controlled_joint_ids:
            return

        env = self.env
        for runtime in self.joint_groups.values():
            target_slice = self._joint_group_sim_slices.get(runtime.spec.name)
            if target_slice is None:
                continue
            env._joint_velocity_target_sim[:, target_slice] = self._joint_group_velocity_to_sim(
                runtime,
                runtime.target_vel,
            )
            position_target = self._joint_group_position_to_sim(runtime, runtime.target_pos)
            if runtime.zero_offset is not None:
                # The servo's zero is not where the model thinks it is. Applied
                # to the TARGET only, never to the commanded value the policy
                # tracks, so the resulting steady-state error is real and
                # observable rather than cancelled out on both sides.
                #
                # Clamp in SIM space. spec.lower/upper are VIRTUAL-frame bounds
                # and position_scale may be negative (morph_tilt carries -1), so
                # the mapped bounds can arrive swapped -- derive both ends and
                # order them rather than assuming lower maps to lower.
                mapped_lower = runtime.position_offset + runtime.position_scale * float(runtime.spec.lower)
                mapped_upper = runtime.position_offset + runtime.position_scale * float(runtime.spec.upper)
                sim_low = torch.minimum(mapped_lower, mapped_upper)
                sim_high = torch.maximum(mapped_lower, mapped_upper)
                position_target = torch.clamp(
                    position_target + runtime.zero_offset, min=sim_low, max=sim_high
                )
            env._joint_position_target_sim[:, target_slice] = position_target
        torch.nan_to_num(
            env._joint_velocity_target_sim,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
            out=env._joint_velocity_target_sim,
        )
        torch.nan_to_num(
            env._joint_position_target_sim,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
            out=env._joint_position_target_sim,
        )
        env._robot.set_joint_velocity_target(env._joint_velocity_target_sim, joint_ids=self.controlled_joint_ids)
        env._robot.set_joint_position_target(env._joint_position_target_sim, joint_ids=self.controlled_joint_ids)

    def _apply_joint_efforts(self):
        if not self.torque_joint_ids:
            return

        env = self.env
        for runtime in self.joint_groups.values():
            target_slice = self._torque_group_sim_slices.get(runtime.spec.name)
            if target_slice is None:
                continue
            env._joint_effort_target_sim[:, target_slice] = self._joint_group_effort_to_sim(
                runtime,
                runtime.effort_command,
            )
        torch.nan_to_num(
            env._joint_effort_target_sim,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
            out=env._joint_effort_target_sim,
        )
        env._robot.set_joint_effort_target(env._joint_effort_target_sim, joint_ids=self.torque_joint_ids)

    def _validate_configured_joint_bounds(self):
        for config_name, joint_config in (
            ("landing_joint_config", self.spec.landing_joint_config),
            ("hover_joint_config", self.spec.hover_joint_config),
        ):
            for group_name, position in joint_config:
                if group_name not in self.joint_groups:
                    known = ", ".join(sorted(self.joint_groups))
                    raise ValueError(
                        f"{self.spec.name} {config_name} references unknown joint group "
                        f"'{group_name}'. Expected one of: {known}"
                    )
                runtime = self.joint_groups[group_name]
                self._bounded_joint_value(
                    runtime,
                    runtime.target_pos[:1],
                    position,
                    f"{config_name}.{group_name}",
                )

        if self.spec.morph_joint_group is not None:
            runtime = self.joint_groups[self.spec.morph_joint_group]
            self._bounded_joint_value(
                runtime,
                runtime.target_pos[:1],
                self.spec.morph_target,
                "morph_target",
            )

        for runtime in self.joint_groups.values():
            self._validate_joint_range_bounds(
                runtime,
                runtime.spec.initial_position_range,
                f"{runtime.spec.name}.initial_position_range",
            )

    def _validate_joint_range_bounds(
        self,
        runtime: JointGroupRuntime,
        range_value: str | tuple[float, float] | None,
        context: str,
    ):
        if range_value is None:
            return
        if isinstance(range_value, str):
            range_value = getattr(self.env.cfg, range_value)
        low, high = float(range_value[0]), float(range_value[1])
        if low > high:
            raise ValueError(f"{self.spec.name} {context} has low={low:.6g} greater than high={high:.6g}")
        self._validate_joint_value_bounds(
            runtime,
            torch.full_like(runtime.target_pos[:1], low),
            f"{context}.low",
        )
        self._validate_joint_value_bounds(
            runtime,
            torch.full_like(runtime.target_pos[:1], high),
            f"{context}.high",
        )

    def _bounded_joint_value(
        self,
        runtime: JointGroupRuntime,
        like: torch.Tensor,
        value: float | tuple[float, ...],
        context: str,
    ) -> torch.Tensor:
        joint_value = self._joint_value_like(like, value)
        self._validate_joint_value_bounds(runtime, joint_value, context)
        return torch.clamp(joint_value, runtime.spec.lower, runtime.spec.upper)

    def _validate_joint_value_bounds(self, runtime: JointGroupRuntime, value: torch.Tensor, context: str):
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{self.spec.name} {context} contains a non-finite joint target")
        lower = float(runtime.spec.lower)
        upper = float(runtime.spec.upper)
        eps = 1e-6
        min_value = float(torch.min(value).item())
        max_value = float(torch.max(value).item())
        if min_value < lower - eps or max_value > upper + eps:
            raise ValueError(
                f"{self.spec.name} {context} must stay within joint group '{runtime.spec.name}' "
                f"bounds [{lower:.6g}, {upper:.6g}], got range [{min_value:.6g}, {max_value:.6g}]"
            )

    def _allocate_actuator_params(self):
        env = self.env
        env.kT = self._nominal_kT_tensor.repeat(env.num_envs, 1)
        env.kM = self._nominal_kM_tensor.repeat(env.num_envs, 1)
        self.spin_direction = torch.tensor(
            [rotor.spin_direction for rotor in self.spec.rotors],
            device=env.device,
        )
        self.thrust_axis = torch.tensor(
            [rotor.thrust_axis for rotor in self.spec.rotors],
            device=env.device,
            dtype=torch.float,
        )
        env._current_thrust_axis_w = self.thrust_axis.unsqueeze(0).repeat(env.num_envs, 1, 1)
        self.rotor_control_mix = self._configured_rotor_control_mix().to(device=env.device, dtype=torch.float)
        self.rotor_roll_sign = self.rotor_control_mix[:, 0]
        self.rotor_pitch_sign = self.rotor_control_mix[:, 1]
        self.rotor_virtual_control_matrix = self._build_rotor_virtual_control_matrix()

    def _configured_rotor_control_mix(self) -> torch.Tensor:
        if "roll_pitch_yaw" not in self.action_schema.slices:
            return torch.zeros(self.rotor_count, 3)
        configured = [rotor.control_mix for rotor in self.spec.rotors]
        has_configured = [mix is not None for mix in configured]
        if any(has_configured) and not all(has_configured):
            missing = [rotor.body_name for rotor in self.spec.rotors if rotor.control_mix is None]
            names = ", ".join(missing)
            raise ValueError(
                f"{self.spec.name} must either configure control_mix for every rotor or omit it for every rotor. "
                f"Missing: {names}"
            )
        if not all(has_configured):
            missing = [rotor.body_name for rotor in self.spec.rotors if rotor.control_mix is None]
            raise ValueError(
                f"{self.spec.name} must configure control_mix for every rotor; missing {', '.join(missing)}"
            )
        return torch.tensor(configured, dtype=torch.float)

    def _build_rotor_virtual_control_matrix(self) -> torch.Tensor:
        if "roll_pitch_yaw" not in self.action_schema.slices:
            return torch.ones(1, self.rotor_count, device=self.env.device)
        roll_scale, pitch_scale, yaw_scale = self.spec.rotor_mix_scale
        control_mix = self.rotor_control_mix
        mix = torch.stack(
            (
                torch.ones(self.rotor_count, device=self.env.device),
                roll_scale * control_mix[:, 0],
                pitch_scale * control_mix[:, 1],
                yaw_scale * control_mix[:, 2],
            ),
            dim=0,
        )
        self._validate_rotor_virtual_control_matrix(mix)
        return mix

    def _resolve_contact_groups(self):
        env = self.env
        env._valid_contact_ids = [
            env._contact_sensor.body_names.index(name) for name in self.spec.contacts.valid_body_names
        ]
        env._invalid_contact_ids = [
            env._contact_sensor.body_names.index(name) for name in self.spec.contacts.invalid_body_names
        ]

    def _cache_inertial_params(self):
        env = self.env
        env._robot_mass = float(env._robot.root_physx_view.get_masses()[0].sum())
        env._gravity_magnitude = torch.tensor(env.sim.cfg.gravity, device=env.device).norm()
        env._robot_weight = env._robot_mass * env._gravity_magnitude.item()
        nominal_kT_values = self._nominal_kT_values()
        self._nominal_kT_tensor = torch.tensor([nominal_kT_values], device=env.device, dtype=torch.float)
        self._nominal_kM_tensor = torch.tensor(
            [[rotor.kM for rotor in self.spec.rotors]], device=env.device, dtype=torch.float
        )
        nominal_km = [rotor.kM for rotor in self.spec.rotors]
        self._nominal_total_kT_value = float(sum(nominal_kT_values))
        self._nominal_total_rotor_moment_coeff_value = float(
            sum(abs(kT * kM) for kT, kM in zip(nominal_kT_values, nominal_km))
        )

    def _filter_normalized_rotor_thrust(self):
        env = self.env
        alpha = torch.where(
            env.normalized_rotor_thrust >= env.normalized_rotor_thrust_filtered,
            env._alpha_rise,
            env._alpha_fall,
        )
        env.normalized_rotor_thrust_filtered[:] = (
            alpha * env.normalized_rotor_thrust + (1.0 - alpha) * env.normalized_rotor_thrust_filtered
        )
        torch.nan_to_num(
            env.normalized_rotor_thrust_filtered,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
            out=env.normalized_rotor_thrust_filtered,
        )
        torch.clamp(env.normalized_rotor_thrust_filtered, 0.0, 1.0, out=env.normalized_rotor_thrust_filtered)

    def _map_policy_actions_into(
        self,
        policy_actions: torch.Tensor,
        mapped_actions: torch.Tensor,
    ) -> torch.Tensor:
        env = self.env
        stage = env.curriculum_stage_for_value("action_authority")
        mapped_actions.copy_(policy_actions)
        policy_terms = self.action_schema.split(policy_actions)
        mapped_terms = self.action_schema.split(mapped_actions)
        for term in self.action_schema.terms:
            if term.name == "lift":
                policy_values = policy_terms[term.name]
                term_values = 0.5 * (policy_values + 1.0)
            else:
                term_values = policy_terms[term.name]
            term_values = torch.nan_to_num(term_values, nan=0.0, posinf=term.high, neginf=term.low)
            torch.clamp(term_values, term.low, term.high, out=mapped_terms[term.name])
        if "tilt_balance" in mapped_terms:
            mapped_terms["tilt_balance"] *= env.action_authority(
                env.cfg.morph_bias_start_stage,
                env.cfg.morph_balance_authority_ramp_epochs,
            )
        if "thrust_center_xy" in mapped_terms:
            mapped_terms["thrust_center_xy"] *= env.action_authority(
                env.cfg.thrust_center_start_stage,
                env.cfg.thrust_center_authority_ramp_epochs,
            )
        if "wheel_speed" in mapped_terms and stage < env.cfg.wheel_action_start_stage:
            mapped_terms["wheel_speed"][:] = 0.0
        return mapped_actions

    def _nominal_hover_collective_throttle(self) -> float:
        hover = self.spec.kT_hover_throttle
        if hover is None:
            total_kT = max(float(self._nominal_kT_tensor.sum().item()), 1e-6)
            hover = float(self.env._robot_weight) / total_kT
        return max(0.05, min(0.95, float(hover)))

    def _compute_rotor_forces(self):
        env = self.env
        rotor_actions = env.normalized_rotor_thrust_filtered
        torch.nan_to_num(rotor_actions, nan=0.0, posinf=1.0, neginf=0.0, out=rotor_actions)
        torch.clamp(rotor_actions, 0.0, 1.0, out=rotor_actions)
        thrust_magnitude = env.kT * rotor_actions
        # Gate the physical output only: the filter state and action history
        # keep integrating the policy's commands, matching the deployment.
        if self.rotor_thrust_gate_provider is not None:
            gate = self.rotor_thrust_gate_provider()
            if gate is not None:
                thrust_magnitude = thrust_magnitude * gate
        moment_magnitude = self.spin_direction * env.kM * thrust_magnitude
        env.rotor_thrust_magnitude[:] = thrust_magnitude
        env.rotor_moment_magnitude[:] = moment_magnitude

    def _joint_group_action(
        self,
        runtime: JointGroupRuntime,
        filtered_terms: dict[str, torch.Tensor],
        stage: int,
    ) -> torch.Tensor:
        if runtime.is_thrust_center:
            return self._thrust_center_shift_action(runtime, filtered_terms, stage)
        if runtime.command_basis is not None:
            command = filtered_terms[runtime.spec.action_name]
            if runtime.is_morph_basis and "tilt_balance" in filtered_terms:
                command = torch.cat((command, filtered_terms["tilt_balance"]), dim=1)
            return torch.matmul(command, runtime.command_basis.T)
        return filtered_terms[runtime.spec.action_name]

    def _thrust_center_shift_action(
        self,
        runtime: JointGroupRuntime,
        filtered_terms: dict[str, torch.Tensor],
        stage: int,
    ) -> torch.Tensor:
        if "thrust_center_xy" not in filtered_terms or stage < self.env.cfg.thrust_center_start_stage:
            return torch.zeros_like(runtime.target_pos)

        trim = filtered_terms["thrust_center_xy"]
        if trim.shape[1] != 2:
            raise ValueError("thrust_center_xy action must contain cx and cy channels")
        cx = trim[:, 0:1]
        cy = trim[:, 1:2]
        if runtime.target_pos.shape[1] == 1:
            return torch.zeros_like(runtime.target_pos)
        if runtime.target_pos.shape[1] == 2:
            lateral = runtime.target_pos.new_tensor((-1.0, 1.0)).reshape(1, 2)
            fore_aft = torch.zeros_like(lateral)
        elif runtime.target_pos.shape[1] == 4:
            lateral = runtime.target_pos.new_tensor((-1.0, 1.0, -1.0, 1.0)).reshape(1, 4)
            fore_aft = runtime.target_pos.new_tensor((1.0, 1.0, -1.0, -1.0)).reshape(1, 4)
        else:
            raise ValueError(f"Thrust-center command expects 1, 2, or 4 joints, got {runtime.target_pos.shape[1]}")
        return torch.clamp(cx * fore_aft + cy * lateral, -1.0, 1.0)

    def _compute_joint_targets(self, filtered_terms: dict[str, torch.Tensor]):
        env = self.env
        stage = env.curriculum_stage_for_value("joint_action")
        for runtime in self.joint_groups.values():
            runtime.effort_command.zero_()
            if runtime.is_thrust_center and stage < env.cfg.thrust_center_start_stage:
                runtime.normalized_command.zero_()
                runtime.target_vel.zero_()
                continue

            action = self._joint_group_action(runtime, filtered_terms, stage)
            action = torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
            action = torch.clamp(action, -1.0, 1.0)
            direction = float(self.joint_direction(runtime.spec.name))
            if env.cfg.quantize_tilt_action and runtime.spec.quantize_action:
                action = torch.round(action)
            if runtime.is_torque or runtime.is_velocity:
                action = self._expand_joint_group(runtime, action)
            runtime.normalized_command[:] = action
            # Before is_torque: duty groups are torque groups, and the plain
            # torque path would otherwise consume them and drop the back-EMF
            # term, which is the whole point of the mapping.
            if runtime.is_duty:
                self._compute_duty_joint_targets(runtime, action, float(direction))
                continue
            if runtime.is_torque:
                self._compute_torque_joint_targets(runtime, action, float(direction))
                continue
            if runtime.is_velocity:
                self._compute_velocity_joint_targets(runtime, action, float(direction))
                continue
            # pre_physics_step runs once per policy step, so the ramp must use
            # step_dt. Scaling by physics_dt makes the effective slew rate
            # max_velocity / decimation, which matches the timing baselines
            # (_morph_max_velocity() and the 2.67 s takeoff prep window, both
            # of which assume the full max_velocity) only at decimation 1.
            # Using step_dt keeps decimation a pure fidelity knob.
            runtime.target_pos[:] = runtime.target_pos + direction * runtime.max_velocity * action * env.step_dt
            runtime.target_pos[:] = torch.clamp(runtime.target_pos, runtime.spec.lower, runtime.spec.upper)
            runtime.target_vel[:] = direction * runtime.max_velocity * action
            if (
                runtime.is_morph_basis
                and stage < env.cfg.morph_bias_start_stage
                and runtime.target_pos.shape[1] > 1
            ):
                runtime.target_pos[:] = torch.mean(runtime.target_pos, dim=1, keepdim=True).expand_as(
                    runtime.target_pos
                )
                runtime.target_vel[:] = torch.mean(runtime.target_vel, dim=1, keepdim=True).expand_as(
                    runtime.target_vel
                )
            torch.nan_to_num(
                runtime.target_pos,
                nan=runtime.spec.lower,
                posinf=runtime.spec.upper,
                neginf=runtime.spec.lower,
                out=runtime.target_pos,
            )
            torch.nan_to_num(runtime.target_vel, nan=0.0, posinf=0.0, neginf=0.0, out=runtime.target_vel)

    def _compute_torque_joint_targets(
        self,
        runtime: JointGroupRuntime,
        action: torch.Tensor,
        direction: float,
    ):
        env = self.env
        action = self._expand_joint_group(runtime, action)
        effort = direction * runtime.effort_limit * action
        # Gate the physical output only, matching the deployment runtime.
        if self.wheel_speed_gate_provider is not None:
            gate = self.wheel_speed_gate_provider()
            if gate is not None:
                effort = effort * gate
        runtime.target_vel.zero_()
        runtime.effort_command[:] = torch.nan_to_num(effort, nan=0.0, posinf=0.0, neginf=0.0)

    def _compute_duty_joint_targets(
        self,
        runtime: JointGroupRuntime,
        action: torch.Tensor,
        direction: float,
    ):
        """Open-loop duty-cycle group: normalized volts to delivered torque.

        The encoderless RoboClaw takes a duty cycle, so the command is motor
        volts as a fraction of supply. A brushed DC motor under applied volts V
        at shaft speed w delivers

            tau = kt * (V - ke*w) / R = tau_stall * (duty - w / w_no_load)

        which is the straight torque-speed line: full torque from rest, zero
        torque at the no-load speed, and a braking torque if the wheel is
        driven faster than the command supports. ``effort_limit`` carries
        tau_stall and ``max_velocity`` carries w_no_load.

        Reading the ACTUAL joint velocity is what makes this a plant model
        rather than a relabelled torque command. It is read in the group's
        virtual frame, so the mirrored right-side sign flips in position_scale
        are already undone and the back-EMF term opposes the command on every
        wheel rather than reinforcing it on half of them.
        """
        action = self._expand_joint_group(runtime, action)
        duty = direction * action
        joint_velocity = torch.nan_to_num(
            self.joint_group_velocities(runtime.spec.name), nan=0.0, posinf=0.0, neginf=0.0
        )
        no_load_speed = torch.clamp(runtime.max_velocity, min=1e-6)
        effort = runtime.effort_limit * (duty - joint_velocity / no_load_speed)
        # Current limit. Plugging -- commanding one way while the wheel is
        # still rolling the other -- puts the back-EMF term in series with the
        # applied volts and would otherwise ask for up to twice stall torque.
        # The real driver current-limits there, so clamp rather than let the
        # sim deliver a torque the hardware cannot.
        effort = torch.clamp(effort, -runtime.effort_limit, runtime.effort_limit)
        # Gate the physical output only, matching the deployment runtime.
        if self.wheel_speed_gate_provider is not None:
            gate = self.wheel_speed_gate_provider()
            if gate is not None:
                effort = effort * gate
        runtime.target_vel.zero_()
        runtime.effort_command[:] = torch.nan_to_num(effort, nan=0.0, posinf=0.0, neginf=0.0)

    def _compute_velocity_joint_targets(
        self,
        runtime: JointGroupRuntime,
        action: torch.Tensor,
        direction: float,
    ):
        """Velocity-setpoint group: normalized per-joint speed to rad/s target.

        The target feeds the sim's velocity-target path through
        _apply_joint_targets (the actuator's damping term does the tracking);
        the torque command path stays zero -- effort_command was cleared at the
        top of _compute_joint_targets and is never written here.
        """
        velocity = direction * runtime.max_velocity * action
        # Gate the physical output only, matching the deployment runtime: while
        # airborne the wheel speed target is zeroed, the policy's command still
        # flows through the action history unchanged.
        if self.wheel_speed_gate_provider is not None:
            gate = self.wheel_speed_gate_provider()
            if gate is not None:
                velocity = velocity * gate
        # No position ramp: the wheel actuator runs stiffness 0, so target_pos
        # is inert and deliberately left at zero.
        runtime.target_vel[:] = torch.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _sanitize_action_(actions: torch.Tensor) -> torch.Tensor:
        torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0, out=actions)
        torch.clamp(actions, -1.0, 1.0, out=actions)
        return actions

    @staticmethod
    def _sanitize_observation_(obs: torch.Tensor) -> torch.Tensor:
        torch.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0, out=obs)
        torch.clamp(obs, -100.0, 100.0, out=obs)
        return obs

    def _append_action_history(self):
        env = self.env
        history_len = env.cfg.action_history_length
        env._action_history_index = (env._action_history_index - 1) % history_len
        env._action_history[:, env._action_history_index, :].copy_(env._actions)

    def _append_observation_history(self, obs_kinematic_current: torch.Tensor):
        env = self.env
        history_len = env.cfg.observation_buffer_length
        env._observation_history_index = (env._observation_history_index - 1) % history_len
        env._observation_buffer[:, env._observation_history_index, :].copy_(obs_kinematic_current)

    def _sample_observation_delay_steps(self, env_ids: torch.Tensor):
        env = self.env
        delay_min = int(env.cfg.observation_delay_min_steps)
        delay_max = int(env.cfg.observation_delay_max_steps)
        delay_min = max(delay_min, 0)
        delay_max = max(delay_max, delay_min)
        if bool(env.cfg.randomize) and delay_max > delay_min:
            sampled_delay = torch.randint(
                delay_min,
                delay_max + 1,
                (len(env_ids),),
                device=env.device,
                dtype=torch.long,
            )
            env._observation_delay_steps[env_ids] = sampled_delay
        else:
            env._observation_delay_steps[env_ids] = delay_min

    def _heading_yaw(self) -> torch.Tensor:
        """Heading yaw for this step, cached across the terms of one pack."""
        cached = self._observation_pass_cache.get("heading_yaw")
        if cached is None:
            cached = heading_yaw_from_quat(
                self.env._robot.data.root_link_quat_w, self.spec.forward_yaw_offset
            )
            self._observation_pass_cache["heading_yaw"] = cached
        return cached

    def _observation_value(self, term: ObservationSourceSpec) -> torch.Tensor:
        env = self.env
        # Vector observations are expressed in the vehicle's heading frame: yaw
        # rotated, z left in world. The actions are body referenced, so keeping
        # the observations in world frame left the policy to learn the rotation
        # before it could connect any error to any action.
        if term.source == "root_pos_local":
            root_pos_obs_w = env._robot.data.root_link_pos_w + env._virtual_xy_offset_w
            return to_heading_frame(
                root_pos_obs_w - env._terrain.env_origins, self._heading_yaw()
            )
        if term.source == "root_rotation_matrix":
            matrix_w = matrix_from_quat(env._robot.data.root_link_quat_w)
            return rotation_matrix_to_heading_frame(
                matrix_w, self._heading_yaw()
            ).reshape(-1, 9)
        if term.source == "root_lin_vel_w":
            return to_heading_frame(env._robot.data.root_com_lin_vel_w, self._heading_yaw())
        if term.source == "root_ang_vel_w":
            return to_heading_frame(env._robot.data.root_com_ang_vel_w, self._heading_yaw())
        if term.source == "joint_group_position":
            if term.joint_group is None:
                raise ValueError(f"Observation term {term.name} needs a joint_group")
            return self._joint_group_position_observation(term.joint_group)
        if term.source == "joint_group_velocity":
            if term.joint_group is None:
                raise ValueError(f"Observation term {term.name} needs a joint_group")
            return self._joint_group_velocity_observation(term.joint_group)
        if self.observation_provider is not None:
            value = self.observation_provider(term.source)
            if value is not None:
                expected_shape = (env.num_envs, term.size)
                if value.shape != expected_shape:
                    raise ValueError(
                        f"Observation source '{term.source}' for term '{term.name}' returned "
                        f"shape {tuple(value.shape)}, expected {expected_shape}"
                    )
                return value
        if term.source == "thrust_wrench_forward_matrix_w":
            return self._thrust_wrench_forward_matrix_world()
        if term.source == "thrust_wrench_allocation_matrix_w":
            return self._allocation_matrix_with_tau()
        if term.source == "thrust_center_xy_b":
            return self._thrust_center_xy_body()
        if term.source == "morph_trig":
            return self._morph_trig_observation()
        raise ValueError(f"Unsupported observation source '{term.source}' for term '{term.name}'")

    def _thrust_wrench_forward_matrix_world(
        self, joint_positions: dict[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        """Map mapped [lift, roll, pitch, yaw] controls to normalized world [force, torque]."""
        if joint_positions is None:
            cached = self._observation_pass_cache.get("thrust_wrench_forward_matrix_w")
            if cached is not None:
                return cached
        env = self.env
        mix = self.rotor_virtual_control_matrix.to(device=env.kT.device, dtype=env.kT.dtype)
        num_controls = mix.shape[0]
        nominal_kT = self._nominal_kT_tensor.to(device=env.device, dtype=env.kT.dtype).expand_as(env.kT)
        nominal_kM = self._nominal_kM_tensor.to(device=env.device, dtype=env.kM.dtype).expand_as(env.kM)
        thrust_axis_w = env._current_thrust_axis_w if joint_positions is None else None
        if thrust_axis_w is None:
            thrust_axis_w = self._current_rotor_thrust_axis_w(joint_positions)
        rotor_force_w = nominal_kT[:, None, :, None] * mix[None, :, :, None] * thrust_axis_w[:, None, :, :]

        rotor_moment_w = (
            self.spin_direction.reshape(1, 1, self.rotor_count, 1)
            * nominal_kM[:, None, :, None]
            * nominal_kT[:, None, :, None]
            * mix[None, :, :, None]
            * thrust_axis_w[:, None, :, :]
        )

        if joint_positions is None:
            rotor_r_w = self._rotor_relative_pos_world()
            rotor_r_b = self._rotor_relative_pos_body()
        else:
            rotor_r_b = self._rotor_relative_pos_body(joint_positions)
            base_quat = self._base_quat_w().expand(-1, self.rotor_count, -1)
            rotor_r_w = self._rotate_link_vectors_to_world(base_quat, rotor_r_b)
        rotor_torque_w = torch.cross(
            rotor_r_w[:, None, :, :].expand(-1, num_controls, -1, -1),
            rotor_force_w,
            dim=-1,
        )
        force_w = torch.sum(rotor_force_w, dim=2)
        torque_w = torch.sum(rotor_torque_w + rotor_moment_w, dim=2)

        force_scale = max(float(self.nominal_total_kT()), 1e-6)
        arm_length = torch.mean(torch.linalg.norm(rotor_r_b[:, :, :2], dim=-1), dim=1)
        moment_scale = force_scale * arm_length.clamp_min(1e-3)
        force_w = force_w / force_scale
        torque_w = torque_w / moment_scale.reshape(env.num_envs, 1, 1)
        value = torch.cat((force_w, torque_w), dim=-1).reshape(env.num_envs, num_controls * 6)
        if joint_positions is None:
            self._observation_pass_cache["thrust_wrench_forward_matrix_w"] = value
        return value

    def _allocation_matrix_with_tau(self) -> torch.Tensor:
        """Allocation-matrix observation with its last 8 entries repurposed.

        The allocation matrix is the pseudo-inverse of the forward matrix the
        policy already observes, so its tail carries no information the network
        cannot reconstruct. Entries [16:24] now carry the observed per-rotor
        actuator lag as normalized filter coefficients: rise alphas in [16:20],
        fall alphas in [20:24], each mapped over the active training tau band to
        [-1, 1]. Noise/dropout buffers are drawn per episode in
        reset_action_buffers; dropped-out envs read 0 (the band midpoint).
        """
        env = self.env
        mat = self._thrust_wrench_allocation_matrix_world().clone()
        tau_min, tau_max = env.actuator_tau_range()
        step_dt = float(env.cfg.step_dt)
        a_min = 1.0 - math.exp(-step_dt / max(tau_max, 1e-6))
        a_max = 1.0 - math.exp(-step_dt / max(tau_min, 1e-6))
        span = max(a_max - a_min, 1e-9)

        def encode(tau: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
            alpha = 1.0 - torch.exp(-step_dt / (tau * noise).clamp_min(1e-6))
            norm = (2.0 * (alpha - a_min) / span - 1.0).clamp_(-1.0, 1.0)
            return torch.where(env._tau_obs_dropout, torch.zeros_like(norm), norm)

        mat[:, 16:20] = encode(env._actuator_tau_rise, env._tau_obs_noise_rise)
        mat[:, 20:24] = encode(env._actuator_tau_fall, env._tau_obs_noise_fall)
        return mat

    def _thrust_wrench_allocation_matrix_world(
        self, joint_positions: dict[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        """Map normalized world [force, torque] targets to mapped [lift, roll, pitch, yaw] controls."""
        if joint_positions is None:
            cached = self._observation_pass_cache.get("thrust_wrench_allocation_matrix_w")
            if cached is not None:
                return cached
        env = self.env
        forward = self._thrust_wrench_forward_matrix_world(joint_positions).reshape(env.num_envs, 4, 6)
        gram = torch.matmul(forward, forward.transpose(1, 2))
        regularized = gram + 1e-6 * torch.eye(4, device=forward.device, dtype=forward.dtype).unsqueeze(0)
        value = torch.linalg.solve(regularized, forward).reshape(env.num_envs, 24)
        value.clamp_(-100.0, 100.0)
        if joint_positions is None:
            self._observation_pass_cache["thrust_wrench_allocation_matrix_w"] = value
        return value

    def _observation_noise_scale(self, term: ObservationSourceSpec) -> float:
        scale = getattr(self.env.cfg, term.noise_scale) if isinstance(term.noise_scale, str) else term.noise_scale
        return float(scale)

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

    def _joint_group_position_to_sim(self, runtime: JointGroupRuntime, value: torch.Tensor) -> torch.Tensor:
        expanded = self._expand_joint_group(runtime, value)
        return runtime.position_offset + runtime.position_scale * expanded

    def _joint_group_velocity_to_sim(self, runtime: JointGroupRuntime, value: torch.Tensor) -> torch.Tensor:
        expanded = self._expand_joint_group(runtime, value)
        return runtime.position_scale * expanded

    def _joint_group_effort_to_sim(self, runtime: JointGroupRuntime, value: torch.Tensor) -> torch.Tensor:
        expanded = self._expand_joint_group(runtime, value)
        safe_scale = torch.where(torch.abs(runtime.position_scale) > 1e-6, runtime.position_scale, torch.ones_like(runtime.position_scale))
        return expanded / safe_scale

    def _joint_group_position_from_sim(self, runtime: JointGroupRuntime, joint_values: torch.Tensor) -> torch.Tensor:
        virtual_joint_values = (joint_values - runtime.position_offset) / runtime.position_scale
        return self._compress_joint_group(runtime, virtual_joint_values)

    def _joint_group_velocity_from_sim(self, runtime: JointGroupRuntime, joint_values: torch.Tensor) -> torch.Tensor:
        virtual_joint_values = joint_values / runtime.position_scale
        return self._compress_joint_group(runtime, virtual_joint_values)

    def _joint_group_param(
        self,
        value: float | tuple[float, ...],
        group: JointGroupSpec,
        like: torch.Tensor,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=like.device, dtype=like.dtype)
        if tensor.ndim == 0:
            return tensor
        joint_count = len(group.joint_names)
        group_name = group.name
        if tensor.numel() != joint_count:
            raise ValueError(
                f"Joint group {group_name} transform has {tensor.numel()} values " f"but maps to {joint_count} joints"
            )
        return tensor.reshape(1, -1)

    @staticmethod
    def _joint_config_value(
        joint_config: tuple[tuple[str, float | tuple[float, ...]], ...],
        group_name: str | None,
    ) -> float | tuple[float, ...] | None:
        if group_name is None:
            return None
        for config_group_name, position in joint_config:
            if config_group_name == group_name:
                return position
        return None

    @staticmethod
    def _joint_value_like(like: torch.Tensor, value: float | tuple[float, ...]) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=like.device, dtype=like.dtype)
        if tensor.ndim == 0:
            return tensor * torch.ones_like(like)
        if tensor.numel() != like.shape[1]:
            raise ValueError(f"Joint target has {tensor.numel()} values but expected {like.shape[1]}")
        return tensor.reshape(1, -1).expand_as(like)

    def _rotor_action_values(self, actions: torch.Tensor) -> torch.Tensor:
        return self._rotor_action_values_from_terms(self.action_schema.split(actions))

    def _rotor_action_values_from_terms(self, action_terms: dict[str, torch.Tensor]) -> torch.Tensor:
        if "lift" not in action_terms:
            raise ValueError(f"{self.spec.name} needs a lift action")

        collective = action_terms["lift"].expand(-1, self.rotor_count)
        if "roll_pitch_yaw" not in action_terms:
            return torch.clamp(collective, 0.0, 1.0)

        rpy = action_terms["roll_pitch_yaw"]
        if rpy.shape[1] != 3:
            raise ValueError("roll_pitch_yaw action must contain roll, pitch, and yaw channels")
        controls = torch.cat((action_terms["lift"], rpy), dim=1)
        rotor_actions = torch.matmul(controls, self.rotor_virtual_control_matrix)
        return torch.clamp(rotor_actions, 0.0, 1.0)

    def _validate_rotor_virtual_control_matrix(self, mix: torch.Tensor):
        expected_rank = mix.shape[0]
        rank = int(torch.linalg.matrix_rank(mix.detach().cpu()).item())
        if rank < expected_rank:
            raise ValueError(
                f"{self.spec.name} rotor virtual control matrix is rank {rank}, "
                f"but needs rank {expected_rank}. Check RotorSpec.control_mix and rotor_mix_scale."
            )

    def _nominal_kT_values(self) -> list[float]:
        if self.spec.kT_hover_throttle is None:
            return [rotor.kT for rotor in self.spec.rotors]
        if self.spec.kT_hover_throttle <= 0.0:
            raise ValueError(f"{self.spec.name} kT_hover_throttle must be positive")
        weight = getattr(self.env, "_robot_weight", None)
        if weight is None:
            return [rotor.kT for rotor in self.spec.rotors]
        kT = float(weight) / (self.rotor_count * float(self.spec.kT_hover_throttle))
        return [kT] * self.rotor_count
