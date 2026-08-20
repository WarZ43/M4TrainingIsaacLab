from __future__ import annotations

import os

from numpy import pi
import torch

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import quat_from_euler_xyz

from .base import BaseTask
from .task_utils import heading_yaw_from_quat, to_heading_frame


@configclass
class TakeoffTaskCfg:
    target_pos = [0.0, 0.0, 1.0]
    target_height_range = [(1.0, 3.0), (1.0, 3.0), (1.0, 3.0)]
    target_xy_offset_range = [(0.0, 4.0), (0.0, 4.0), (0.0, 4.0)]
    target_heading_offset_deg = [15.0, 15.0, 15.0]
    start_xy_range = [2.0, 2.0, 2.0]
    # Root-link z offset, relative to the terrain origin, when the transformed vehicle just touches the ground.
    start_root_height = [0.22, 0.22, 0.22]
    start_morph_angle = [pi / 2, pi / 2, pi / 2]
    drive_duration_range_s = [(1.0, 4.0), (1.0, 4.0), (1.0, 4.0)]
    drive_speed_range = [(0.0, 1.5), (0.0, 1.5), (0.0, 1.5)]
    ground_reset_velocity_range = (-1.0, 1.0)
    takeoff_drive_completion_fraction = 0.50
    takeoff_drive_completion_min_tolerance_m = 0.35
    flight_xy_reference_max_speed = [0.84375, 0.84375, 0.84375]
    flight_z_reference_speed_range = [(0.25, 0.75), (0.25, 0.75), (0.25, 0.75)]
    flight_min_duration_s = [0.75, 0.75, 0.75]
    flight_duration_scale = [1.875, 1.875, 1.875]
    trajectory_pos_rew_scale = [2.0, 2.0, 2.0]
    trajectory_vel_rew_scale = [0.5, 0.5, 0.5]
    morph_config_rew_scale = [10.0, 10.0, 10.0]
    successful_takeoff_rew = [500.0, 500.0, 500.0]
    takeoff_reference_min_upward_speed = 0.05
    invalid_contact_pen = [-2.0, -2.0, -2.0]
    action_rate_pen_scale = [-0.5, -1.5, -0.5]
    airborne_rotor_agreement_rew_scale = [2.0, 2.0, 2.0]
    airborne_wheel_spin_pen_scale = [-10.0, -10.0, -10.0]
    # Recovery resets are enabled only by the final-stage overlay.
    hover_recovery_xy_error_max = [1.5, 4.0, 5.0]
    hover_recovery_velocity_max = [1.0, 2.5, 3.0]
    hover_recovery_yaw_error_max_deg = [45.0, 110.0, 135.0]
    hover_reset_tilt_deg = 20.0
    takeoff_height_threshold = [0.10, 0.10, 0.10]
    termination_min_vz = [-1.0, -1.0, -1.0]
    disturbance_force_scale = [5.0, 5.0, 5.0]
    disturbance_moment_scale = [5.0, 5.0, 5.0]
    disturbance_cts_force_scale = [1.0, 1.0, 1.0]
    disturbance_cts_moment_scale = [1.0, 1.0, 1.0]


class TakeoffTask(BaseTask):
    """Drive forward from the ground, then fly through a nonzero-velocity trajectory."""

    reward_keys = (
        "trajectory_pos_rew",
        "trajectory_vel_rew",
        "morph_config_rew",
        "action_rate_pen",
        "airborne_rotor_agreement_rew",
        "airborne_wheel_spin_penalty",
        "successful_takeoff_rew",
        "invalid_contact_penalty",
    )

    def __init__(self, env, cfg: TakeoffTaskCfg):
        super().__init__(env, cfg, self.reward_keys)

        env._desired_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._virtual_xy_offset_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._takeoff_spawn_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._takeoff_drive_duration = torch.zeros(env.num_envs, device=env.device)
        env._takeoff_drive_velocity_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._takeoff_liftoff_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._takeoff_flight_delta_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._takeoff_flight_duration = torch.ones(env.num_envs, 1, device=env.device)
        env._takeoff_flight_start_velocity_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._takeoff_reset_phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._takeoff_reset_airborne = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._takeoff_reset_throttle = torch.zeros(env.num_envs, 1, device=env.device)
        env._ep_airborne = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        # Latched copy of the env's OWN takeoff-success event, for
        # benchmarking. new_takeoff fires for a single step and is
        # otherwise unrecorded, so there is no way to ask afterwards
        # whether the env considered the takeoff successful. Write-only:
        # nothing in the env reads it, so dynamics are unchanged.
        env._ep_took_off = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._takeoff_trajectory_pos_error = torch.zeros(env.num_envs, device=env.device)
        env._initial_yaw = torch.zeros(env.num_envs, device=env.device)

    def disturbance_force_scale(self) -> float:
        return (
            self.env.vehicle.nominal_total_kT()
            * self.env.cfg.disturbance_force_scale
            * self.stage_value(
                self.cfg.disturbance_force_scale,
                "disturbance_force_scale",
            )
        )

    def disturbance_moment_scale(self) -> float:
        return (
            self.env.vehicle.nominal_total_rotor_moment_coeff()
            * self.env.cfg.disturbance_moment_scale
            * self.stage_value(
                self.cfg.disturbance_moment_scale,
                "disturbance_moment_scale",
            )
        )

    def disturbance_cts_force_scale(self) -> float:
        return (
            self.env.vehicle.nominal_total_kT()
            * self.env.cfg.dist_force_cts_scale
            * self.stage_value(
                self.cfg.disturbance_cts_force_scale,
                "disturbance_cts_force_scale",
            )
        )

    def disturbance_cts_moment_scale(self) -> float:
        return (
            self.env.vehicle.nominal_total_rotor_moment_coeff()
            * self.env.cfg.dist_moment_cts_scale
            * self.stage_value(
                self.cfg.disturbance_cts_moment_scale,
                "disturbance_cts_moment_scale",
            )
        )

    def reset_initial_state(self, env_ids: torch.Tensor, randomized: bool):
        self._observation_context_cache = None
        env = self.env
        num_resets = len(env_ids)
        final_stage_overlay = env.final_stage_overlay_active()
        phase = torch.randint(0, 4, (num_resets,), device=env.device) if randomized else torch.zeros(
            num_resets, dtype=torch.long, device=env.device
        )
        # Benchmark hook: pin every reset to one spawn bucket. The training mix
        # (0=drive, 1=takeoff, 2=flight, 3=hover) spawns most episodes already
        # airborne, which cannot be a test of taking off -- and those episodes
        # then score for merely staying up. Inert unless the env var is set.
        _forced_phase = os.environ.get("M4_BENCH_START_PHASE")
        if _forced_phase is not None:
            phase = torch.full_like(phase, int(_forced_phase))
        if final_stage_overlay:
            phase[phase == 3] = 0
        drive_phase = phase == 0
        takeoff_phase = phase == 1
        flight_phase = phase == 2
        hover_phase = phase == 3
        takeoff_airborne = takeoff_phase & (torch.rand(num_resets, device=env.device) < 0.5)
        takeoff_ground = takeoff_phase & ~takeoff_airborne
        env._takeoff_reset_phase[env_ids] = phase
        env._takeoff_reset_airborne[env_ids] = takeoff_airborne | flight_phase | hover_phase
        target = torch.tensor(self.cfg.target_pos, device=env.device, dtype=torch.float).repeat(num_resets, 1)
        if randomized:
            target_height_min, target_height_max = self.stage_value(
                self.cfg.target_height_range,
                "target_height_range",
            )
            target[:, 2] = torch.zeros(num_resets, device=env.device).uniform_(
                target_height_min,
                target_height_max,
            )
        env._desired_pos_w[env_ids] = env._terrain.env_origins[env_ids] + target
        env._virtual_xy_offset_w[env_ids] = 0.0

        root_state = env._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] = env._terrain.env_origins[env_ids]
        root_state[:, 2] += self.stage_value(self.cfg.start_root_height, "start_root_height")

        if randomized:
            xy_offset = torch.zeros(num_resets, 2, device=env.device).uniform_(
                -self.stage_value(self.cfg.start_xy_range, "start_xy_range"),
                self.stage_value(self.cfg.start_xy_range, "start_xy_range"),
            )
            roll = torch.zeros(num_resets, device=env.device)
            pitch = torch.zeros(num_resets, device=env.device)
        else:
            xy_offset = torch.zeros(num_resets, 2, device=env.device)
            roll = torch.zeros(num_resets, device=env.device)
            pitch = torch.zeros(num_resets, device=env.device)

        root_state[:, :2] += xy_offset
        drive_heading = torch.zeros(num_resets, device=env.device).uniform_(-pi, pi)
        yaw = self._wrap_to_pi(drive_heading - float(env.vehicle.spec.forward_yaw_offset))
        root_state[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
        root_state[:, 7:13] = 0.0
        env._takeoff_spawn_pos_w[env_ids] = root_state[:, :3]
        env._initial_yaw[env_ids] = yaw

        drive_duration_min, drive_duration_max = self.stage_value(
            self.cfg.drive_duration_range_s, "drive_duration_range_s"
        )
        drive_duration = torch.zeros(num_resets, device=env.device).uniform_(
            float(drive_duration_min), float(drive_duration_max)
        )
        drive_speed_min, drive_speed_max = self.stage_value(self.cfg.drive_speed_range, "drive_speed_range")
        drive_speed = torch.zeros(num_resets, device=env.device).uniform_(
            float(drive_speed_min), float(drive_speed_max)
        )
        forward_w = torch.stack(
            (torch.cos(drive_heading), torch.sin(drive_heading), torch.zeros_like(drive_heading)), dim=1
        )
        drive_velocity = drive_speed.unsqueeze(1) * forward_w
        liftoff_pos = root_state[:, :3] + drive_velocity * drive_duration.unsqueeze(1)
        target_xy_min, target_xy_max = self.stage_value(self.cfg.target_xy_offset_range, "target_xy_offset_range")
        target_xy_distance = torch.zeros(num_resets, device=env.device).uniform_(
            float(target_xy_min), float(target_xy_max)
        )
        heading_offset = (
            float(self.stage_value(self.cfg.target_heading_offset_deg, "target_heading_offset_deg")) * pi / 180.0
        )
        flight_heading = drive_heading + torch.zeros(num_resets, device=env.device).uniform_(
            -heading_offset, heading_offset
        )
        env._desired_pos_w[env_ids, :2] = liftoff_pos[:, :2] + target_xy_distance.unsqueeze(1) * torch.stack(
            (torch.cos(flight_heading), torch.sin(flight_heading)), dim=1
        )
        flight_delta = env._desired_pos_w[env_ids] - liftoff_pos
        env._takeoff_drive_duration[env_ids] = drive_duration
        env._takeoff_drive_velocity_w[env_ids] = drive_velocity
        env._takeoff_liftoff_pos_w[env_ids] = liftoff_pos
        env._takeoff_flight_delta_w[env_ids] = flight_delta
        env._takeoff_flight_start_velocity_w[env_ids] = drive_velocity
        flight_height = torch.abs(flight_delta[:, 2:3])
        flight_xy_distance = torch.linalg.norm(flight_delta[:, :2], dim=1, keepdim=True)
        max_xy_speed = max(
            float(self.stage_value(self.cfg.flight_xy_reference_max_speed, "flight_xy_reference_max_speed")), 1e-3
        )
        min_duration = max(
            float(self.stage_value(self.cfg.flight_min_duration_s, "flight_min_duration_s")), env.step_dt
        )
        z_speed_min, z_speed_max = self.stage_value(
            self.cfg.flight_z_reference_speed_range, "flight_z_reference_speed_range"
        )
        z_speed = torch.zeros(num_resets, 1, device=env.device).uniform_(
            max(float(z_speed_min), 1e-3), max(float(z_speed_max), float(z_speed_min) + 1e-3)
        )
        duration_scale = max(float(self.stage_value(self.cfg.flight_duration_scale, "flight_duration_scale")), 1.0)
        duration = torch.maximum(
            torch.maximum(flight_height / z_speed, duration_scale * flight_xy_distance / max_xy_speed),
            torch.full_like(flight_height, min_duration),
        )
        env._takeoff_flight_duration[env_ids] = duration
        phase_time = torch.zeros(num_resets, device=env.device)
        phase_time[drive_phase] = 0.0
        phase_time[takeoff_ground] = drive_duration[takeoff_ground] + 0.15 * torch.rand_like(
            drive_duration[takeoff_ground]
        ) * duration[takeoff_ground, 0]
        phase_time[takeoff_airborne] = drive_duration[takeoff_airborne] + (
            0.05 + 0.25 * torch.rand_like(drive_duration[takeoff_airborne])
        ) * duration[takeoff_airborne, 0]
        phase_time[flight_phase] = drive_duration[flight_phase] + (0.30 + 0.60 * torch.rand_like(
            drive_duration[flight_phase]
        )) * duration[flight_phase, 0]
        phase_time[hover_phase] = drive_duration[hover_phase] + duration[hover_phase, 0]
        env._time_elapsed[env_ids] = phase_time
        ref_pos_w, ref_vel_w, ref_accel_w = self._takeoff_reference_state()
        ref_pos_w = ref_pos_w[env_ids]
        ref_vel_w = ref_vel_w[env_ids]
        ref_accel_w = ref_accel_w[env_ids]
        takeoff_height_threshold = float(self.stage_value(self.cfg.takeoff_height_threshold, "takeoff_height_threshold"))
        insufficient_clearance = takeoff_airborne & (
            ref_pos_w[:, 2] <= env._takeoff_spawn_pos_w[env_ids, 2] + takeoff_height_threshold
        )
        if torch.any(insufficient_clearance):
            env._time_elapsed[env_ids[insufficient_clearance]] = (
                drive_duration[insufficient_clearance] + 0.30 * duration[insufficient_clearance, 0]
            )
            ref_pos_w, ref_vel_w, ref_accel_w = self._takeoff_reference_state()
            ref_pos_w = ref_pos_w[env_ids]
            ref_vel_w = ref_vel_w[env_ids]
            ref_accel_w = ref_accel_w[env_ids]
        root_state[drive_phase, :3] = ref_pos_w[drive_phase]
        reset_drive_speed = torch.empty(num_resets, device=env.device).uniform_(
            float(self.cfg.ground_reset_velocity_range[0]),
            float(self.cfg.ground_reset_velocity_range[1]),
        )
        root_state[drive_phase, 7:10] = reset_drive_speed[drive_phase].unsqueeze(1) * forward_w[drive_phase]
        root_state[takeoff_ground, :3] = liftoff_pos[takeoff_ground]
        root_state[takeoff_ground, 7:10] = 0.0
        airborne_phase = takeoff_airborne | flight_phase | hover_phase
        root_state[airborne_phase, :3] = ref_pos_w[airborne_phase]
        root_state[airborne_phase, 7:10] = ref_vel_w[airborne_phase]
        pose_noise = torch.zeros(num_resets, 3, device=env.device).uniform_(-0.08, 0.08)
        root_state[airborne_phase, :3] += pose_noise[airborne_phase]
        root_state[takeoff_airborne, 2] = torch.maximum(
            root_state[takeoff_airborne, 2],
            env._takeoff_spawn_pos_w[env_ids[takeoff_airborne], 2] + takeoff_height_threshold + 0.02,
        )
        root_state[hover_phase, :3] = env._desired_pos_w[env_ids[hover_phase]] + pose_noise[hover_phase]
        root_state[hover_phase, 7:10] = torch.empty_like(root_state[hover_phase, 7:10]).uniform_(-0.2, 0.2)
        recovery_probability = 1.0 if final_stage_overlay else 0.05
        recovery = hover_phase & (
            torch.rand(num_resets, device=env.device) < recovery_probability
        )
        if torch.any(recovery):
            xy_error_max = max(
                float(self.stage_value(self.cfg.hover_recovery_xy_error_max, "hover_recovery_xy_error_max")),
                0.0,
            )
            xy_error = torch.zeros(num_resets, device=env.device).uniform_(0.0, xy_error_max)
            xy_heading = torch.zeros(num_resets, device=env.device).uniform_(-pi, pi)
            root_state[recovery, 0] = env._desired_pos_w[env_ids[recovery], 0] + (
                xy_error[recovery] * torch.cos(xy_heading[recovery])
            )
            root_state[recovery, 1] = env._desired_pos_w[env_ids[recovery], 1] + (
                xy_error[recovery] * torch.sin(xy_heading[recovery])
            )
            velocity_max = max(
                float(self.stage_value(self.cfg.hover_recovery_velocity_max, "hover_recovery_velocity_max")),
                0.0,
            )
            velocity = torch.zeros(num_resets, device=env.device).uniform_(0.0, velocity_max)
            velocity_heading = torch.zeros(num_resets, device=env.device).uniform_(-pi, pi)
            root_state[recovery, 7] = velocity[recovery] * torch.cos(velocity_heading[recovery])
            root_state[recovery, 8] = velocity[recovery] * torch.sin(velocity_heading[recovery])
            root_state[recovery, 9] = torch.zeros(num_resets, device=env.device).uniform_(-0.5, 0.5)[recovery]
            yaw_error_max = max(
                float(
                    self.stage_value(
                        self.cfg.hover_recovery_yaw_error_max_deg,
                        "hover_recovery_yaw_error_max_deg",
                    )
                ),
                0.0,
            ) * pi / 180.0
            yaw[recovery] += torch.zeros(num_resets, device=env.device).uniform_(-yaw_error_max, yaw_error_max)[recovery]
        roll[airborne_phase] = torch.empty_like(roll[airborne_phase]).uniform_(-5.0 * pi / 180.0, 5.0 * pi / 180.0)
        pitch[airborne_phase] = torch.empty_like(pitch[airborne_phase]).uniform_(-5.0 * pi / 180.0, 5.0 * pi / 180.0)
        hover_tilt = float(self.cfg.hover_reset_tilt_deg) * pi / 180.0
        roll[hover_phase] = torch.empty_like(roll[hover_phase]).uniform_(-hover_tilt, hover_tilt)
        pitch[hover_phase] = torch.empty_like(pitch[hover_phase]).uniform_(-hover_tilt, hover_tilt)
        root_state[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)

        joint_pos, joint_vel = env.vehicle.deterministic_joint_state(env_ids)
        if env.vehicle.spec.landing_joint_config:
            env.vehicle.set_joint_config_state(
                joint_pos,
                joint_vel,
                env_ids,
                env.vehicle.spec.landing_joint_config,
                0.0,
            )
        else:
            env.vehicle.set_joint_group_state_values(
                joint_pos,
                joint_vel,
                env_ids,
                env.vehicle.spec.morph_joint_group,
                self.stage_value(self.cfg.start_morph_angle, "start_morph_angle"),
                0.0,
            )
        morph_angle = torch.full((num_resets, 1), pi / 2, device=env.device)
        morph_angle[drive_phase] = torch.empty_like(morph_angle[drive_phase]).uniform_(70.0 * pi / 180.0, pi / 2)
        morph_angle[takeoff_ground] = torch.empty_like(morph_angle[takeoff_ground]).uniform_(0.0, pi / 2)
        morph_angle[takeoff_airborne] = torch.empty_like(morph_angle[takeoff_airborne]).uniform_(
            0.0, 45.0 * pi / 180.0
        )
        morph_angle[flight_phase | hover_phase] = torch.empty_like(morph_angle[flight_phase | hover_phase]).uniform_(
            0.0, 5.0 * pi / 180.0
        )
        env.vehicle.set_joint_group_state_values(
            joint_pos, joint_vel, env_ids, env.vehicle.spec.morph_joint_group, morph_angle, 0.0
        )

        reset_actions = env._reset_policy_actions[env_ids]
        reset_actions.fill_(-1.0)
        action_terms = env.vehicle.action_schema.split(reset_actions)
        action_terms["roll_pitch_yaw"].zero_()
        if "wheel_torque" in action_terms:
            action_terms["wheel_torque"].zero_()
        reset_throttle = torch.zeros(num_resets, 1, device=env.device)
        hover_throttle = env.vehicle.hover_collective_throttle()
        reset_throttle[takeoff_ground] = hover_throttle * torch.cos(morph_angle[takeoff_ground])
        reset_throttle[airborne_phase] = (
            hover_throttle * (1.0 + ref_accel_w[airborne_phase, 2:3] / 9.81) / torch.cos(morph_angle[airborne_phase])
        ).clamp(0.05, 0.95)
        warm_start = takeoff_phase | flight_phase | hover_phase
        action_terms["lift"][warm_start] = 2.0 * reset_throttle[warm_start] - 1.0
        action_terms["tilt_mean"][warm_start] = 0.0
        if "tilt_balance" in action_terms:
            action_terms["tilt_balance"][warm_start] = 0.0
        if "thrust_center_xy" in action_terms:
            action_terms["thrust_center_xy"][warm_start] = 0.0
        env._takeoff_reset_throttle[env_ids] = reset_throttle

        env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        env._robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        env._robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)

    def reference_state(
        self, time_offset_s: float | torch.Tensor = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._takeoff_reference_state(time_offset_s)

    def _trajectory_time(self, time_offset_s: float | torch.Tensor = 0.0) -> torch.Tensor:
        env = self.env
        if isinstance(time_offset_s, torch.Tensor):
            time_offset = time_offset_s.to(device=env.device, dtype=env._time_elapsed.dtype)
            if time_offset.ndim == 1:
                time_offset = time_offset.unsqueeze(1)
            time_offset = torch.clamp(time_offset, min=0.0)
        else:
            time_offset = torch.full(
                (env.num_envs, 1),
                max(float(time_offset_s), 0.0),
                device=env.device,
                dtype=env._time_elapsed.dtype,
            )
        return env._time_elapsed.unsqueeze(1) + time_offset

    def _takeoff_reference_state(
        self, time_offset_s: float | torch.Tensor = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        env = self.env
        time = self._trajectory_time(time_offset_s)
        drive_time = env._takeoff_drive_duration.unsqueeze(1)
        drive_elapsed = torch.clamp(time, min=0.0)
        drive_pos = env._takeoff_spawn_pos_w + env._takeoff_drive_velocity_w * drive_elapsed
        drive_vel = env._takeoff_drive_velocity_w
        drive_accel = torch.zeros_like(drive_vel)

        flight_time = torch.clamp(time - drive_time, min=0.0)
        duration = env._takeoff_flight_duration.clamp_min(env.step_dt)
        accel_duration = (0.2 * duration + 0.2).expand(-1, 3).clone()
        decel_duration = (0.2 * duration + 0.5).expand(-1, 3).clone()
        accel_duration[:, 2] *= 0.25
        decel_duration[:, 2] *= 0.25
        cruise_duration = torch.clamp(duration - accel_duration - decel_duration, min=env.step_dt)
        start = env._takeoff_liftoff_pos_w
        end = torch.nan_to_num(env._desired_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
        delta = env._takeoff_flight_delta_w
        start_velocity = env._takeoff_flight_start_velocity_w
        end_velocity = torch.zeros_like(start_velocity)
        cruise_velocity = (delta - 0.5 * accel_duration * start_velocity - 0.5 * decel_duration * end_velocity) / (
            0.5 * accel_duration + cruise_duration + 0.5 * decel_duration
        )
        accel_end = start + 0.5 * accel_duration * (start_velocity + cruise_velocity)
        decel_start = accel_end + cruise_duration * cruise_velocity
        accel_time = torch.minimum(flight_time, accel_duration)
        cruise_time = torch.minimum(torch.clamp(flight_time - accel_duration, min=0.0), cruise_duration)
        decel_time = torch.minimum(torch.clamp(flight_time - accel_duration - cruise_duration, min=0.0), decel_duration)
        accel_pos, accel_vel, accel_accel = self._seventh_order_segment(
            start, accel_end, start_velocity, cruise_velocity, accel_time, accel_duration
        )
        cruise_pos = accel_end + cruise_velocity * cruise_time
        cruise_vel = cruise_velocity.expand_as(cruise_pos)
        cruise_accel = torch.zeros_like(cruise_pos)
        decel_pos, decel_vel, decel_accel = self._seventh_order_segment(
            decel_start, end, cruise_velocity, end_velocity, decel_time, decel_duration
        )
        flight_pos = torch.where(
            flight_time <= accel_duration,
            accel_pos,
            torch.where(flight_time <= accel_duration + cruise_duration, cruise_pos, decel_pos),
        )
        flight_vel = torch.where(
            flight_time <= accel_duration,
            accel_vel,
            torch.where(flight_time <= accel_duration + cruise_duration, cruise_vel, decel_vel),
        )
        flight_accel = torch.where(
            flight_time <= accel_duration,
            accel_accel,
            torch.where(flight_time <= accel_duration + cruise_duration, cruise_accel, decel_accel),
        )
        post_elapsed = torch.clamp(flight_time - duration, min=0.0)
        post_pos = end
        post_vel = torch.zeros_like(end)
        post_accel = torch.zeros_like(post_pos)
        flight_done = flight_time >= duration
        flight_pos = torch.where(flight_done, post_pos, flight_pos)
        flight_vel = torch.where(flight_done, post_vel, flight_vel)
        flight_accel = torch.where(flight_done, post_accel, flight_accel)
        in_drive = time < drive_time
        return (
            torch.nan_to_num(torch.where(in_drive, drive_pos, flight_pos), nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(torch.where(in_drive, drive_vel, flight_vel), nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(torch.where(in_drive, drive_accel, flight_accel), nan=0.0, posinf=0.0, neginf=0.0),
        )

    @staticmethod
    def _seventh_order_segment(start, end, start_velocity, end_velocity, time, duration):
        tau = torch.clamp(time / duration.clamp_min(1e-6), min=0.0, max=1.0)
        tau2, tau3 = tau * tau, tau * tau * tau
        tau4, tau5 = tau2 * tau2, tau2 * tau3
        tau6, tau7 = tau5 * tau, tau5 * tau2
        shape = 35.0 * tau4 - 84.0 * tau5 + 70.0 * tau6 - 20.0 * tau7
        shape_rate = 140.0 * tau3 - 420.0 * tau4 + 420.0 * tau5 - 140.0 * tau6
        shape_accel = 420.0 * tau2 - 1680.0 * tau3 + 2100.0 * tau4 - 840.0 * tau5
        start_shape = tau - 20.0 * tau4 + 45.0 * tau5 - 36.0 * tau6 + 10.0 * tau7
        start_rate = 1.0 - 80.0 * tau3 + 225.0 * tau4 - 216.0 * tau5 + 70.0 * tau6
        start_accel = -240.0 * tau2 + 900.0 * tau3 - 1080.0 * tau4 + 420.0 * tau5
        end_shape = -15.0 * tau4 + 39.0 * tau5 - 34.0 * tau6 + 10.0 * tau7
        end_rate = -60.0 * tau3 + 195.0 * tau4 - 204.0 * tau5 + 70.0 * tau6
        end_accel = -180.0 * tau2 + 780.0 * tau3 - 1020.0 * tau4 + 420.0 * tau5
        delta = end - start
        return (
            start + delta * shape + start_velocity * duration * start_shape + end_velocity * duration * end_shape,
            delta * (shape_rate / duration) + start_velocity * start_rate + end_velocity * end_rate,
            delta * (shape_accel / torch.square(duration))
            + start_velocity * (start_accel / duration)
            + end_velocity * (end_accel / duration),
        )

    def _yaw_target_and_error(self, root_quat_w: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        env = self.env
        if root_quat_w is None:
            root_quat_w = env._robot.data.root_link_quat_w
        yaw = self._yaw_from_quat(root_quat_w)
        yaw_target = torch.nan_to_num(env._initial_yaw, nan=0.0, posinf=0.0, neginf=0.0)
        yaw_error = self._wrap_to_pi(yaw_target - yaw)
        return yaw_target.unsqueeze(1), yaw_error.unsqueeze(1)

    def observation_value(self, source: str) -> torch.Tensor | None:
        ctx = self._observation_context()
        if source == "task_ref_accel_w":
            return ctx["ref_accel_w"]
        if source == "task_ref_pos_error_w":
            return ctx["ref_pos_error_w"]
        if source == "task_vel_error_w":
            return ctx["vel_error_w"]
        if source == "task_ref_yaw_accel":
            return ctx["reference_yaw_accel"]
        if source == "task_yaw_error":
            return ctx["yaw_error"]
        if source == "task_ref_yaw_rate_error":
            return ctx["yaw_rate_error"]
        return None

    def _observation_context(self) -> dict:
        env = self.env
        cache_step = env._global_env_step
        cache = self._observation_context_cache
        if cache is not None and cache.get("step") == cache_step:
            return cache

        ref_pos_w, ref_vel_w, ref_accel_w = self._takeoff_reference_state()
        root_pos_w = env._robot.data.root_link_pos_w
        root_quat_w = env._robot.data.root_link_quat_w
        root_lin_vel_w = env._robot.data.root_com_lin_vel_w
        root_ang_vel_w = env._robot.data.root_com_ang_vel_w
        ref_pos_error_w = ref_pos_w - root_pos_w
        vel_error_w = ref_vel_w - root_lin_vel_w
        _, yaw_error = self._yaw_target_and_error(root_quat_w)
        # Heading frame, matching the root observations: yaw rotated, z world.
        heading_yaw = heading_yaw_from_quat(root_quat_w, env.vehicle.spec.forward_yaw_offset)
        cache = {
            "step": cache_step,
            "ref_accel_w": to_heading_frame(ref_accel_w, heading_yaw),
            "ref_pos_error_w": to_heading_frame(ref_pos_error_w, heading_yaw),
            "vel_error_w": to_heading_frame(vel_error_w, heading_yaw),
            "reference_yaw_accel": torch.zeros_like(yaw_error),
            "yaw_error": yaw_error,
            "yaw_rate_error": -root_ang_vel_w[:, 2:3],
        }
        self._observation_context_cache = cache
        return cache

    def joint_action_direction(self, group_name: str) -> float:
        if group_name == self.env.vehicle.spec.morph_joint_group:
            return -1.0
        return 1.0

    def get_rewards(self) -> torch.Tensor:
        env = self.env
        died, time_out = env.get_cached_dones()
        root_pos_w = torch.nan_to_num(env._robot.data.root_link_pos_w, nan=0.0, posinf=1e6, neginf=-1e6)
        root_lin_vel_w = torch.nan_to_num(env._robot.data.root_com_lin_vel_w, nan=0.0, posinf=1e3, neginf=-1e3)
        root_quat_w = torch.nan_to_num(env._robot.data.root_link_quat_w, nan=0.0, posinf=0.0, neginf=0.0)

        ref_pos_w, ref_vel_w, _ = self._takeoff_reference_state()
        yaw_error = self._wrap_to_pi(self._yaw_from_quat(root_quat_w) - env._initial_yaw)
        angular_velocity_error = torch.nan_to_num(
            env._robot.data.root_com_ang_vel_b, nan=0.0, posinf=1e3, neginf=-1e3
        )
        pos_error = torch.sqrt(torch.sum(torch.square(ref_pos_w - root_pos_w), dim=1) + torch.square(yaw_error))
        vel_error = torch.sqrt(
            torch.sum(torch.square(ref_vel_w - root_lin_vel_w), dim=1)
            + torch.sum(torch.square(angular_velocity_error), dim=1)
        )
        env._takeoff_trajectory_pos_error[:] = pos_error.detach()
        pos_score = 50.0 / (1.0 + 2.0 * pos_error) + 25.0 / (1.0 + 10.0 * pos_error)
        velocity_weight = torch.clamp(1.0 - pos_error / 3.0, min=0.0, max=1.0)
        vel_score = 25.0 / (1.0 + 2.0 * vel_error) * velocity_weight

        contact_time = env.scene["contact_sensor"].data.current_contact_time[:, env._valid_contact_ids]
        any_ground_contact = torch.any(contact_time > 0.0, dim=1)
        flight_started = ref_vel_w[:, 2] > float(self.cfg.takeoff_reference_min_upward_speed)
        upright = self._body_up_z(root_quat_w) > 0.0
        clearance = root_pos_w[:, 2] - env._takeoff_spawn_pos_w[:, 2]
        takeoff_height_threshold = max(
            float(self.stage_value(self.cfg.takeoff_height_threshold, "takeoff_height_threshold")), 0.0
        )
        physically_airborne = (~any_ground_contact) & (clearance > takeoff_height_threshold)
        reference_driving = env._time_elapsed < env._takeoff_drive_duration
        trajectory_state_matches = (reference_driving & any_ground_contact) | (~reference_driving & physically_airborne)
        planned_drive_distance = torch.linalg.norm(
            env._takeoff_liftoff_pos_w[:, :2] - env._takeoff_spawn_pos_w[:, :2], dim=1
        )
        drive_completion_tolerance = torch.maximum(
            float(self.cfg.takeoff_drive_completion_fraction) * planned_drive_distance,
            torch.full_like(planned_drive_distance, float(self.cfg.takeoff_drive_completion_min_tolerance_m)),
        )
        drive_completion_error = torch.linalg.norm(
            root_pos_w[:, :2] - env._takeoff_liftoff_pos_w[:, :2], dim=1
        )
        drive_completed = drive_completion_error <= drive_completion_tolerance
        new_takeoff = flight_started & physically_airborne & upright & drive_completed & (~died) & (~env._ep_airborne)
        env._ep_airborne |= physically_airborne
        env._ep_took_off |= new_takeoff

        tilt_values = torch.nan_to_num(env.vehicle.morph_joint_positions(), nan=0.0, posinf=1e3, neginf=-1e3)
        flight_elapsed = torch.clamp(env._time_elapsed - env._takeoff_drive_duration, min=0.0)
        untuck_progress = torch.clamp(
            flight_elapsed / (0.30 * env._takeoff_flight_duration[:, 0]).clamp_min(env.step_dt), min=0.0, max=1.0
        )
        target_morph_angle = (pi / 2) * (1.0 - untuck_progress)
        morph_score = 1.0 - torch.mean(torch.abs(tilt_values - target_morph_angle.unsqueeze(1)), dim=1) / (pi / 2)

        action_rate = torch.nan_to_num(env.vehicle.action_rate(), nan=0.0, posinf=1e3, neginf=0.0)
        action_rate_scale = self.stage_value(self.cfg.action_rate_pen_scale, "action_rate_pen_scale")
        if env.final_stage_overlay_active():
            action_rate_scale *= 10.0
        wheel_speed = torch.nan_to_num(
            env.vehicle.joint_group_velocities("wheel_dof"), nan=0.0, posinf=12.0, neginf=-12.0
        )
        airborne_wheel_spin = physically_airborne.float() * torch.mean(torch.square(wheel_speed / 12.0), dim=1)
        filtered_rotor_action = torch.nan_to_num(
            env.vehicle.rotor_action_values(filtered=True), nan=0.0, posinf=1.0, neginf=0.0
        )
        rotor_thrust = env.kT * filtered_rotor_action
        total_rotor_thrust = torch.sum(rotor_thrust, dim=1, keepdim=True)
        rotor_share = rotor_thrust / total_rotor_thrust.clamp_min(1e-6)
        uniform_share = 1.0 / rotor_share.shape[1]
        rotor_disagreement = rotor_share.shape[1] ** 2 * torch.mean(
            torch.square(rotor_share - uniform_share), dim=1
        )
        rotor_agreement = torch.exp(-2.0 * rotor_disagreement)
        rotor_agreement *= (
            torch.sum(filtered_rotor_action, dim=1) > 0.05 * filtered_rotor_action.shape[1]
        ).float()
        airborne_rotor_agreement = rotor_agreement * physically_airborne.float()
        invalid_contact = torch.any(
            env.scene["contact_sensor"].data.current_contact_time[:, env._invalid_contact_ids] > 0.0, dim=1
        )
        invalid_contact &= env.final_stage_overlay_active()
        rewards = {
            "trajectory_pos_rew": pos_score
            * self.stage_value(self.cfg.trajectory_pos_rew_scale, "trajectory_pos_rew_scale")
            * env.step_dt,
            "trajectory_vel_rew": vel_score
            * trajectory_state_matches.float()
            * self.stage_value(self.cfg.trajectory_vel_rew_scale, "trajectory_vel_rew_scale")
            * env.step_dt,
            "morph_config_rew": morph_score.clamp_min(0.0)
            * self.stage_value(self.cfg.morph_config_rew_scale, "morph_config_rew_scale")
            * env.step_dt,
            "action_rate_pen": action_rate
            * action_rate_scale
            * env.step_dt,
            "airborne_rotor_agreement_rew": airborne_rotor_agreement
            * self.stage_value(
                self.cfg.airborne_rotor_agreement_rew_scale,
                "airborne_rotor_agreement_rew_scale",
            )
            * env.step_dt,
            "airborne_wheel_spin_penalty": airborne_wheel_spin
            * self.stage_value(self.cfg.airborne_wheel_spin_pen_scale, "airborne_wheel_spin_pen_scale")
            * env.step_dt,
            "successful_takeoff_rew": new_takeoff.float()
            * self.stage_value(self.cfg.successful_takeoff_rew, "successful_takeoff_rew"),
            "invalid_contact_penalty": invalid_contact.float()
            * self.stage_value(self.cfg.invalid_contact_pen, "invalid_contact_pen"),
        }
        rewards = {key: torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0) for key, value in rewards.items()}
        self.log_rewards(rewards, died | time_out)
        return self.reward_mixer.sum(rewards)

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        env = self.env
        time_out = env.episode_length_buf >= env.max_episode_length - 1
        root_lin_vel_w = env._robot.data.root_com_lin_vel_w
        finite_state = (
            torch.isfinite(env._robot.data.root_link_pos_w).all(dim=1)
            & torch.isfinite(env._robot.data.root_link_quat_w).all(dim=1)
            & torch.isfinite(root_lin_vel_w).all(dim=1)
            & torch.isfinite(env._robot.data.root_com_ang_vel_b).all(dim=1)
        )
        safe_root_lin_vel_w = torch.nan_to_num(root_lin_vel_w, nan=0.0, posinf=1e6, neginf=-1e6)
        vz_below_min = safe_root_lin_vel_w[:, 2] < self.stage_value(
            self.cfg.termination_min_vz,
            "termination_min_vz",
        )
        return (~finite_state) | vz_below_min, time_out

    def reset_episode_state(self, env_ids: torch.Tensor):
        self._observation_context_cache = None
        env = self.env
        env._takeoff_trajectory_pos_error[env_ids] = 0.0
        env._ep_airborne[env_ids] = env._takeoff_reset_airborne[env_ids]
        env._ep_took_off[env_ids] = False
        throttle = env._takeoff_reset_throttle[env_ids]
        env.normalized_rotor_thrust[env_ids] = throttle
        env.previous_normalized_rotor_thrust[env_ids] = throttle
        env.normalized_rotor_thrust_filtered[env_ids] = throttle
        env.previous_normalized_rotor_thrust_filtered[env_ids] = throttle

    @staticmethod
    def _yaw_from_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return torch.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _body_up_z(quat_wxyz: torch.Tensor) -> torch.Tensor:
        return 1.0 - 2.0 * (torch.square(quat_wxyz[:, 1]) + torch.square(quat_wxyz[:, 2]))

    @staticmethod
    def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))
