from __future__ import annotations

from numpy import pi
import torch

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import quat_from_euler_xyz

from .base import BaseTask
from .task_utils import heading_yaw_from_quat, to_heading_frame


@configclass
class LandingTaskCfg:
    target_pos = [0.0, 0.0, 0.20]
    virtual_xy_offset_range = [15.0, 15.0, 15.0]
    virtual_z_offset_range = [(0.8, 1.2), (0.8, 1.2), (0.8, 1.2)]
    initial_xy_range = [2.0, 2.0, 2.0]
    initial_z_range = [(0.5, 1.75), (0.5, 1.75), (0.5, 1.75)]
    initial_roll_pitch_range = [5.0 * pi / 180.0, 5.0 * pi / 180.0, 5.0 * pi / 180.0]
    initial_yaw_range = [30.0 * pi / 180.0, 30.0 * pi / 180.0, 30.0 * pi / 180.0]

    initial_tuck_seed_range = [(0.0, pi / 12), (0.0, pi / 6), (0.0, pi / 6)]
    contact_tuck_worst_joint_weight = [0.75, 0.75, 0.75]
    contact_speed_rew_free_speed = [0.30, 0.30]
    contact_speed_rew_zero_speed = [1.00, 1.00]
    contact_reward_min_time_remaining_s = [0.50, 0.50]
    contact_reward_min_valid_contacts_by_stage = [3, 3, 3]
    landing_config_rew_scale = [10.0, 10.0, 10.0]

    vertical_reference_speed_range = (0.4, 0.7)
    xy_reference_max_speed = [1.0, 1.0, 1.0]
    xy_reference_min_duration_s = [0.75, 0.75, 0.75]
    trajectory_duration_scale = 1.875
    trajectory_boundary_velocity_max = 1.0
    trajectory_boundary_velocity_delta_max = 0.5
    trajectory_boundary_velocity_cone_deg = 30.0
    ground_drive_bucket_probability = [0.20, 0.20, 0.20]
    ground_drive_position_error_range = 0.75
    ground_drive_velocity_range = (-1.0, 1.0)
    ground_drive_heading_error_deg = 15.0
    yaw_rate_pen_scale = [-6.00, -6.00]
    action_rate_pen_scale = [-0.5, -1.5, -0.5]
    airborne_rotor_agreement_rew_scale = [2.0, 2.0, 2.0]
    thrust_center_action_pen_scale = [0.0, 0.0, -0.25]
    thrust_center_loss_offset_pen_scale = [0.0, 0.0, -0.25]
    yaw_angle_pen_scale = [-3.00, -6.00, -6.00]
    joint_disagreement_pen_scale = [0.00, -60.00, -60.00]
    post_landing_thrust_pen_scale = [-40.0, -40.0, -60.0]

    invalid_contact_pen = [-2.0, -2.0]
    post_landing_invalid_contact_pen = [-500.0, -500.0]
    died_pen = [-25.0, -25.0, -45.0]
    died_excess_vel_pen_scale = [25.0, 25.0, 60.0]
    died_no_landing_pen = [-20.0, -20.0, -20.0]
    died_no_landing_time_pen_scale = [-20.0, -20.0, -20.0]
    timeout_pen = [-15.0, -15.0]

    tuck_absolute_linear_rew_scale = [300, 300]
    tuck_absolute_rew_cap = [180.0, 180.0]
    contact_in_acceptance_rew_scale = [1725.00, 1725.00, 1725.00]
    contact_in_acceptance_baseline_rew = [75.00, 75.00, 75.00]
    contact_in_acceptance_rew_cap = [800.0, 800.0, 800.0]
    disturbance_force_scale = [5.0, 5.0, 5.0]
    disturbance_moment_scale = [5.0, 5.0, 5.0]
    disturbance_cts_force_scale = [1.0, 1.0, 1.0]
    disturbance_cts_moment_scale = [1.0, 1.0, 1.0]

class LandingTask(BaseTask):
    """Landing-specific curriculum, rewards, termination, and metrics."""

    reward_keys = (
        "trajectory_pos_rew",
        "trajectory_vel_rew",
        "yaw_angle_penalty",
        "yaw_rate_penalty",
        "action_rate_pen",
        "airborne_rotor_agreement_rew",
        "thrust_center_penalty",
        "post_landing_thrust_penalty",
        "post_landing_takeoff_penalty",
        "joint_disagreement_penalty",
        "tuck_progress_rew",
        "contact_in_acceptance_rew",
        "final_landing_quality_rew",
        "landing_config_rew",
        "invalid_contact_penalty",
        "too_fast_penalty",
        "nonfinite_state_penalty",
        "no_landing_terminal_penalty",
        "timeout_high_penalty",
    )

    def __init__(self, env, cfg: LandingTaskCfg):
        super().__init__(env, cfg, self.reward_keys)

        env._desired_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._virtual_xy_offset_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._landing_start_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._landing_reference_delta_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._landing_reference_duration = torch.ones(env.num_envs, 1, device=env.device)
        env._landing_start_velocity_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._landing_ground_velocity_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._previous_tuck_completion = torch.zeros(env.num_envs, device=env.device)
        env._ep_contact_in_acceptance = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._ep_invalid_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._nonfinite_root_state = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._post_landing_takeoff = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._invalid_contact_pen_given = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._landing_yaw_heading_w = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        env._touchdown_tuck_multiplier = torch.zeros(env.num_envs, device=env.device)
        env._touchdown_tuck_recorded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._final_landing_quality_reward_given = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        env._ground_drive_bucket = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def _contact_state(self) -> dict[str, torch.Tensor]:
        env = self.env
        valid_contact_time = env.scene["contact_sensor"].data.current_contact_time[:, env._valid_contact_ids]
        valid_contact_mask = valid_contact_time > 0.0
        invalid_contact_time = env.scene["contact_sensor"].data.current_contact_time[:, env._invalid_contact_ids]
        any_valid_contacts = torch.any(valid_contact_mask, dim=1)
        valid_contact_count = torch.sum(valid_contact_mask, dim=1)

        state = {
            "valid_contact_time": valid_contact_time,
            "valid_contact_mask": valid_contact_mask,
            "any_valid_contacts": any_valid_contacts,
            "valid_contact_count": valid_contact_count,
            "invalid_contacts": torch.any(invalid_contact_time > 0.0, dim=1),
        }
        return state

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
            * self.stage_value(self.cfg.disturbance_cts_force_scale, "disturbance_cts_force_scale")
        )

    def disturbance_cts_moment_scale(self) -> float:
        return (
            self.env.vehicle.nominal_total_rotor_moment_coeff()
            * self.env.cfg.dist_moment_cts_scale
            * self.stage_value(self.cfg.disturbance_cts_moment_scale, "disturbance_cts_moment_scale")
        )

    def reset_initial_state(self, env_ids: torch.Tensor, randomized: bool):
        self._observation_context_cache = None
        env = self.env
        num_resets = len(env_ids)
        ground_drive_probability = (
            float(self.stage_value(self.cfg.ground_drive_bucket_probability, "ground_drive_bucket_probability"))
            if randomized
            else 0.0
        )
        env._ground_drive_bucket[env_ids] = torch.rand(num_resets, device=env.device) < ground_drive_probability
        target = torch.tensor(self.cfg.target_pos, device=env.device, dtype=torch.float).repeat(num_resets, 1)
        desired_pos_w = env._terrain.env_origins[env_ids] + target
        virtual_offset = torch.zeros(num_resets, 3, device=env.device)

        if randomized:
            virtual_offset[:, :2] = torch.zeros(num_resets, 2, device=env.device).uniform_(
                -self.stage_value(self.cfg.virtual_xy_offset_range, "virtual_xy_offset_range"),
                self.stage_value(self.cfg.virtual_xy_offset_range, "virtual_xy_offset_range"),
            )
            virtual_z_offset_min, virtual_z_offset_max = self.stage_value(
                self.cfg.virtual_z_offset_range,
                "virtual_z_offset_range",
            )
            virtual_offset[:, 2] = torch.zeros(num_resets, device=env.device).uniform_(
                virtual_z_offset_min,
                virtual_z_offset_max,
            )

            root_state = env._robot.data.default_root_state[env_ids].clone()
            root_state[:, :3] += env._terrain.env_origins[env_ids]
            initial_xy_offset = torch.zeros(num_resets, 2, device=env.device).uniform_(
                -self.stage_value(self.cfg.initial_xy_range, "initial_xy_range"),
                self.stage_value(self.cfg.initial_xy_range, "initial_xy_range"),
            )
            initial_z_min, initial_z_max = self.stage_value(
                self.cfg.initial_z_range,
                "initial_z_range",
            )
            root_state[:, :2] += initial_xy_offset
            root_state[:, 2] = torch.zeros_like(root_state[:, 2]).uniform_(
                initial_z_min,
                initial_z_max,
            )
            root_state[:, 2] += env._terrain.env_origins[env_ids, 2]
            roll_pitch_range = self.stage_value(self.cfg.initial_roll_pitch_range, "initial_roll_pitch_range")
            roll = torch.zeros(num_resets, device=env.device).uniform_(
                -roll_pitch_range,
                roll_pitch_range,
            )
            pitch = torch.zeros(num_resets, device=env.device).uniform_(
                -roll_pitch_range,
                roll_pitch_range,
            )
            root_state[:, 10:13] = torch.zeros_like(root_state[:, 10:13]).uniform_(
                env.cfg.initial_ang_vel_range[0],
                env.cfg.initial_ang_vel_range[1],
            )
            joint_pos, joint_vel = env.vehicle.randomized_joint_state(env_ids)
        else:
            root_state = env._robot.data.default_root_state[env_ids].clone()
            root_state[:, :3] += env._terrain.env_origins[env_ids]
            joint_pos, joint_vel = env.vehicle.deterministic_joint_state(env_ids)
        if env.vehicle.spec.hover_joint_config:
            env.vehicle.set_joint_config_state(
                joint_pos,
                joint_vel,
                env_ids,
                env.vehicle.spec.hover_joint_config,
                0.0,
            )
        if randomized:
            self._apply_initial_tuck_seed(joint_pos, joint_vel, env_ids)
        if randomized:
            if env.curriculum_stage_for_value("morph_bias") >= env.cfg.morph_bias_start_stage:
                env.vehicle.randomize_joint_group_position(
                    joint_pos,
                    joint_vel,
                    env_ids,
                    env.vehicle.spec.morph_joint_group,
                )
        landing_delta_w = desired_pos_w - root_state[:, :3]
        env._desired_pos_w[env_ids] = desired_pos_w
        env._landing_start_pos_w[env_ids] = root_state[:, :3]
        env._virtual_xy_offset_w[env_ids] = virtual_offset
        self._update_reference_constants(env_ids, randomized)
        if randomized:
            start_velocity_w = self._sample_boundary_velocity(
                landing_delta_w, num_resets, float(self.cfg.trajectory_boundary_velocity_max)
            )
            start_speed = torch.linalg.norm(start_velocity_w, dim=1, keepdim=True)
            velocity_delta = torch.empty_like(start_speed).uniform_(
                -float(self.cfg.trajectory_boundary_velocity_delta_max),
                float(self.cfg.trajectory_boundary_velocity_delta_max),
            )
            ground_speed = torch.clamp(
                start_speed + velocity_delta,
                min=0.0,
                max=float(self.cfg.trajectory_boundary_velocity_max),
            )
            boundary_direction_w = start_velocity_w / start_speed.clamp_min(1e-6)
            zero_speed = start_speed <= 1e-6
            displacement_xy_direction = landing_delta_w[:, :2] / torch.linalg.norm(
                landing_delta_w[:, :2], dim=1, keepdim=True
            ).clamp_min(1e-6)
            displacement_direction_w = torch.cat(
                (displacement_xy_direction, torch.zeros_like(displacement_xy_direction[:, :1])), dim=1
            )
            boundary_direction_w = torch.where(zero_speed, displacement_direction_w, boundary_direction_w)
            ground_velocity_w = boundary_direction_w * ground_speed
            duration = env._landing_reference_duration[env_ids]
            accel_duration = 0.2 * duration + 0.2
            decel_duration = 0.2 * duration + 0.5
            xy_distance = torch.linalg.norm(landing_delta_w[:, :2], dim=1, keepdim=True)
            xy_direction = landing_delta_w[:, :2] / xy_distance.clamp_min(1e-6)
            start_progress_speed = torch.sum(start_velocity_w[:, :2] * xy_direction, dim=1, keepdim=True)
            ground_progress_speed = torch.sum(ground_velocity_w[:, :2] * xy_direction, dim=1, keepdim=True)
            boundary_progress = (
                0.5 * accel_duration * start_progress_speed
                + 0.5 * decel_duration * ground_progress_speed
            )
            boundary_scale = torch.clamp(
                0.95 * xy_distance / boundary_progress.clamp_min(1e-6),
                max=1.0,
            )
            boundary_scale = torch.where(xy_distance > 1e-6, boundary_scale, torch.zeros_like(boundary_scale))
            start_velocity_w *= boundary_scale
            ground_velocity_w *= boundary_scale
            root_state[:, 7:10] = start_velocity_w
        else:
            start_velocity_w = torch.zeros_like(landing_delta_w)
            ground_velocity_w = torch.zeros_like(landing_delta_w)
            root_state[:, 7:10] = 0.0
        env._landing_start_velocity_w[env_ids] = start_velocity_w
        env._landing_ground_velocity_w[env_ids] = ground_velocity_w
        displacement_heading_w = torch.atan2(landing_delta_w[:, 1], landing_delta_w[:, 0])
        start_xy_speed = torch.linalg.norm(start_velocity_w[:, :2], dim=1)
        start_heading_w = torch.where(
            start_xy_speed > 1e-6,
            torch.atan2(start_velocity_w[:, 1], start_velocity_w[:, 0]),
            displacement_heading_w,
        )
        ground_xy_speed = torch.linalg.norm(ground_velocity_w[:, :2], dim=1)
        extension_heading_w = torch.where(
            ground_xy_speed > 1e-6,
            torch.atan2(ground_velocity_w[:, 1], ground_velocity_w[:, 0]),
            start_heading_w,
        )
        forward_offset = float(env.vehicle.spec.forward_yaw_offset)
        if randomized:
            yaw_range = self.stage_value(self.cfg.initial_yaw_range, "initial_yaw_range")
            initial_vehicle_heading_w = start_heading_w + torch.zeros(
                num_resets, device=env.device
            ).uniform_(-yaw_range, yaw_range)
            root_yaw = self._wrap_to_pi(initial_vehicle_heading_w - forward_offset)
            root_state[:, 3:7] = quat_from_euler_xyz(roll, pitch, root_yaw)
        else:
            zero_rp = torch.zeros(num_resets, device=env.device)
            root_yaw = self._wrap_to_pi(start_heading_w - forward_offset)
            root_state[:, 3:7] = quat_from_euler_xyz(zero_rp, zero_rp, root_yaw)
        env._landing_yaw_heading_w[env_ids] = extension_heading_w

        ground_drive_mask = env._ground_drive_bucket[env_ids]
        if torch.any(ground_drive_mask):
            ground_env_ids = env_ids[ground_drive_mask]
            ground_count = len(ground_env_ids)
            ground_target = desired_pos_w[ground_drive_mask]
            ground_heading = torch.empty(ground_count, device=env.device).uniform_(-pi, pi)
            heading_cos = torch.cos(ground_heading)
            heading_sin = torch.sin(ground_heading)
            ground_speed = torch.empty(ground_count, device=env.device).uniform_(
                float(self.cfg.ground_drive_velocity_range[0]),
                float(self.cfg.ground_drive_velocity_range[1]),
            )
            ground_velocity = torch.stack(
                (ground_speed * heading_cos, ground_speed * heading_sin, torch.zeros_like(ground_speed)), dim=1
            )
            position_error_angle = torch.empty(ground_count, device=env.device).uniform_(-pi, pi)
            position_error_radius = torch.sqrt(torch.rand(ground_count, device=env.device)) * float(
                self.cfg.ground_drive_position_error_range
            )
            position_error_b = torch.stack(
                (
                    position_error_radius * torch.cos(position_error_angle),
                    position_error_radius * torch.sin(position_error_angle),
                ),
                dim=1,
            )
            position_error_w = torch.stack(
                (
                    heading_cos * position_error_b[:, 0] - heading_sin * position_error_b[:, 1],
                    heading_sin * position_error_b[:, 0] + heading_cos * position_error_b[:, 1],
                ),
                dim=1,
            )
            root_state[ground_drive_mask, :2] = ground_target[:, :2] - position_error_w
            root_state[ground_drive_mask, 2] = ground_target[:, 2]
            root_state[ground_drive_mask, 7:13] = 0.0
            yaw_range = float(self.cfg.ground_drive_heading_error_deg) * pi / 180.0
            vehicle_heading = ground_heading + torch.empty(ground_count, device=env.device).uniform_(
                -yaw_range, yaw_range
            )
            zero_angle = torch.zeros(ground_count, device=env.device)
            root_state[ground_drive_mask, 3:7] = quat_from_euler_xyz(
                zero_angle,
                zero_angle,
                self._wrap_to_pi(vehicle_heading - forward_offset),
            )

            ground_joint_pos = joint_pos[ground_drive_mask].clone()
            ground_joint_vel = joint_vel[ground_drive_mask].clone()
            env.vehicle.set_joint_config_state(
                ground_joint_pos,
                ground_joint_vel,
                ground_env_ids,
                env.vehicle.spec.landing_joint_config,
                0.0,
            )
            joint_pos[ground_drive_mask] = ground_joint_pos
            joint_vel[ground_drive_mask] = ground_joint_vel

            env._landing_start_pos_w[ground_env_ids] = ground_target
            env._landing_reference_delta_w[ground_env_ids] = 0.0
            env._landing_reference_duration[ground_env_ids] = 0.0
            env._landing_start_velocity_w[ground_env_ids] = ground_velocity
            env._landing_ground_velocity_w[ground_env_ids] = ground_velocity
            env._landing_yaw_heading_w[ground_env_ids] = ground_heading
            env._virtual_xy_offset_w[ground_env_ids] = 0.0

        env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        env._robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        env._robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)

    def _apply_initial_tuck_seed(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor, env_ids: torch.Tensor):
        env = self.env
        group_name = env.vehicle.spec.morph_joint_group
        if group_name is None or group_name not in env.vehicle.joint_groups:
            return
        seed_range = self.stage_value(self.cfg.initial_tuck_seed_range, "initial_tuck_seed_range")
        low, high = float(seed_range[0]), float(seed_range[1])
        if high <= low:
            return
        runtime = env.vehicle.joint_groups[group_name]
        seed = torch.empty(len(env_ids), 1, device=env.device, dtype=runtime.target_pos.dtype).uniform_(low, high)
        env.vehicle.set_joint_group_state_values(joint_pos, joint_vel, env_ids, group_name, seed)

    def _sample_boundary_velocity(
        self, displacement_w: torch.Tensor, count: int, max_speed: float
    ) -> torch.Tensor:
        env = self.env
        cone = float(self.cfg.trajectory_boundary_velocity_cone_deg) * pi / 180.0
        heading = torch.atan2(displacement_w[:, 1:2], displacement_w[:, 0:1])
        heading += torch.zeros(count, 1, device=env.device).uniform_(-cone, cone)
        direction = torch.cat(
            (torch.cos(heading), torch.sin(heading), torch.zeros_like(heading)), dim=1
        )
        speed = torch.zeros(count, 1, device=env.device).uniform_(0.0, max(max_speed, 0.0))
        return speed * direction

    def _yaw_target_and_error(self, root_quat_w: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        env = self.env
        if root_quat_w is None:
            root_quat_w = env._robot.data.root_link_quat_w
        yaw = self._yaw_from_quat(root_quat_w)
        forward_offset = float(env.vehicle.spec.forward_yaw_offset)
        vehicle_forward_yaw = self._wrap_to_pi(yaw + forward_offset)
        yaw_target = torch.nan_to_num(env._landing_yaw_heading_w, nan=0.0, posinf=0.0, neginf=0.0)
        yaw_error = self._wrap_to_pi(yaw_target - vehicle_forward_yaw)
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

        ref_pos_w, ref_vel_w, ref_accel_w = self._reference_state()
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

    def reference_state(
        self, time_offset_s: float | torch.Tensor = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._reference_state(time_offset_s)

    def _update_reference_constants(self, env_ids: torch.Tensor, randomized: bool):
        env = self.env
        start = torch.nan_to_num(env._landing_start_pos_w[env_ids], nan=0.0, posinf=0.0, neginf=0.0)
        target = torch.nan_to_num(env._desired_pos_w[env_ids], nan=0.0, posinf=0.0, neginf=0.0)
        delta = target - start
        height = torch.abs(delta[:, 2:3])
        xy_distance = torch.linalg.norm(delta[:, :2], dim=1, keepdim=True)
        max_xy_speed = max(float(self.stage_value(self.cfg.xy_reference_max_speed, "xy_reference_max_speed")), 1e-3)
        min_duration = max(
            float(self.stage_value(self.cfg.xy_reference_min_duration_s, "xy_reference_min_duration_s")), env.step_dt
        )
        vertical_speed_min, vertical_speed_max = self.cfg.vertical_reference_speed_range
        if randomized:
            vertical_speed = torch.empty_like(height).uniform_(vertical_speed_min, vertical_speed_max)
        else:
            vertical_speed = torch.full_like(height, 0.5 * (vertical_speed_min + vertical_speed_max))
        vertical_duration = height / vertical_speed.clamp_min(1e-3)
        horizontal_duration = xy_distance / max_xy_speed
        duration = torch.maximum(
            torch.maximum(vertical_duration, horizontal_duration), torch.full_like(vertical_duration, min_duration)
        )
        duration = duration * max(float(self.cfg.trajectory_duration_scale), 1.0)
        env._landing_reference_delta_w[env_ids] = delta
        env._landing_reference_duration[env_ids] = duration

    def _reference_state(
        self, time_offset_s: float | torch.Tensor = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        env = self.env
        if isinstance(time_offset_s, torch.Tensor):
            time_offset = time_offset_s.to(device=env.device, dtype=env._time_elapsed.dtype)
            if time_offset.ndim == 1:
                time_offset = time_offset.unsqueeze(1)
        else:
            time_offset = torch.full(
                (env.num_envs, 1), max(float(time_offset_s), 0.0), device=env.device, dtype=env._time_elapsed.dtype
            )
        time = env._time_elapsed.unsqueeze(1) + torch.clamp(time_offset, min=0.0)
        duration = torch.nan_to_num(
            env._landing_reference_duration, nan=env.step_dt, posinf=env.cfg.episode_length_s, neginf=env.step_dt
        ).clamp_min(env.step_dt)
        start = torch.nan_to_num(env._landing_start_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
        delta = torch.nan_to_num(env._landing_reference_delta_w, nan=0.0, posinf=0.0, neginf=0.0)
        start_velocity = env._landing_start_velocity_w
        ground_velocity = env._landing_ground_velocity_w
        airborne_end_velocity = ground_velocity
        accel_duration = 0.2 * duration + 0.2
        decel_duration = 0.2 * duration + 0.5
        cruise_duration = torch.clamp(duration - accel_duration - decel_duration, min=env.step_dt)
        cruise_velocity = (
            delta
            - 0.5 * accel_duration * start_velocity
            - 0.5 * decel_duration * airborne_end_velocity
        ) / (
            0.5 * accel_duration + cruise_duration + 0.5 * decel_duration
        )
        accel_end = start + 0.5 * accel_duration * (start_velocity + cruise_velocity)
        decel_start = accel_end + cruise_duration * cruise_velocity
        accel_time = torch.minimum(torch.clamp(time, min=0.0), accel_duration)
        cruise_time = torch.minimum(torch.clamp(time - accel_duration, min=0.0), cruise_duration)
        decel_time = torch.minimum(
            torch.clamp(time - accel_duration - cruise_duration, min=0.0),
            decel_duration,
        )
        accel_pos, accel_vel, accel_accel = self._seventh_order_segment(
            start, accel_end, start_velocity, cruise_velocity, accel_time, accel_duration
        )
        cruise_pos = accel_end + cruise_velocity * cruise_time
        cruise_vel = cruise_velocity.expand_as(cruise_pos)
        cruise_accel = torch.zeros_like(cruise_pos)
        decel_pos, decel_vel, decel_accel = self._seventh_order_segment(
            decel_start, start + delta, cruise_velocity, airborne_end_velocity, decel_time, decel_duration
        )
        airborne_pos = torch.where(
            time <= accel_duration,
            accel_pos,
            torch.where(time <= accel_duration + cruise_duration, cruise_pos, decel_pos),
        )
        airborne_vel = torch.where(
            time <= accel_duration,
            accel_vel,
            torch.where(time <= accel_duration + cruise_duration, cruise_vel, decel_vel),
        )
        airborne_accel = torch.where(
            time <= accel_duration,
            accel_accel,
            torch.where(time <= accel_duration + cruise_duration, cruise_accel, decel_accel),
        )
        ground_elapsed = torch.clamp(time - duration, min=0.0)
        ground_pos = env._desired_pos_w + ground_velocity * ground_elapsed
        ground_accel = torch.zeros_like(ground_velocity)
        ground_phase = time >= duration
        ref_pos = torch.where(ground_phase, ground_pos, airborne_pos)
        ref_vel = torch.where(ground_phase, ground_velocity, airborne_vel)
        ref_accel = torch.where(ground_phase, ground_accel, airborne_accel)
        ground_drive_bucket = env._ground_drive_bucket.unsqueeze(1)
        ref_pos = torch.where(
            ground_drive_bucket,
            env._desired_pos_w + ground_velocity * time,
            ref_pos,
        )
        ref_vel = torch.where(ground_drive_bucket, ground_velocity, ref_vel)
        ref_accel = torch.where(ground_drive_bucket, torch.zeros_like(ground_velocity), ref_accel)
        return (
            torch.nan_to_num(ref_pos, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(ref_vel, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(ref_accel, nan=0.0, posinf=0.0, neginf=0.0),
        )

    @staticmethod
    def _seventh_order_segment(
        start: torch.Tensor,
        end: torch.Tensor,
        start_velocity: torch.Tensor,
        end_velocity: torch.Tensor,
        time: torch.Tensor,
        duration: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tau = torch.clamp(time / duration, min=0.0, max=1.0)
        tau2 = tau * tau
        tau3 = tau2 * tau
        tau4 = tau2 * tau2
        tau5 = tau4 * tau
        tau6 = tau5 * tau
        tau7 = tau6 * tau
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

    def _landing_config_score(self) -> torch.Tensor:
        env = self.env
        errors = []
        for group_name, _ in env.vehicle.spec.landing_joint_config:
            if group_name not in env.vehicle.joint_groups:
                continue
            positions = env.vehicle.joint_group_positions(group_name)
            target = env.vehicle.joint_config_target_values(
                env.vehicle.spec.landing_joint_config, group_name, positions
            )
            if target is not None:
                errors.append(torch.abs(positions - target))

        if not errors:
            return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

        worst_error = torch.max(torch.cat(errors, dim=1), dim=1).values
        return 5.0 * torch.clamp(1.0 - worst_error / (pi / 2.0), min=0.0, max=1.0)

    def get_rewards(self) -> torch.Tensor:
        env = self.env
        termination = self._termination_masks()
        died = termination["died"]
        time_out = termination["time_out"]
        too_fast_vertical = termination["too_fast_vertical"]
        nonfinite_state = termination["nonfinite_state"]
        invalid_contacts = termination["invalid_contacts"]
        terminating_invalid_contact = termination["terminating_invalid_contact"]
        post_landing_invalid_contact = termination["post_landing_invalid_contact"]
        contact_state = self._contact_state()
        valid_contact_time = contact_state["valid_contact_time"]
        valid_contact_mask = contact_state["valid_contact_mask"]
        any_valid_contacts = contact_state["any_valid_contacts"]
        valid_contact_count = contact_state["valid_contact_count"]
        min_valid_contacts = min(
            max(
                int(
                    self.stage_value(
                        self.cfg.contact_reward_min_valid_contacts_by_stage,
                        "contact_reward_min_valid_contacts_by_stage",
                    )
                ),
                1,
            ),
            len(env._valid_contact_ids),
        )
        enough_valid_contacts = valid_contact_count >= min_valid_contacts
        env._ep_invalid_contact |= invalid_contacts
        new_invalid_contacts = invalid_contacts & (~env._invalid_contact_pen_given)
        env._invalid_contact_pen_given |= invalid_contacts
        root_pos_w = torch.nan_to_num(env._robot.data.root_link_pos_w, nan=0.0, posinf=0.0, neginf=0.0)
        root_lin_vel_w = torch.nan_to_num(env._robot.data.root_com_lin_vel_w, nan=0.0, posinf=0.0, neginf=0.0)
        root_quat_w = torch.nan_to_num(env._robot.data.root_link_quat_w, nan=0.0, posinf=0.0, neginf=0.0)
        reward_contacts = enough_valid_contacts
        vertical_speed = torch.abs(root_lin_vel_w[:, 2])
        yaw = self._yaw_from_quat(root_quat_w)
        forward_offset = float(env.vehicle.spec.forward_yaw_offset)
        vehicle_forward_yaw = self._wrap_to_pi(yaw + forward_offset)
        yaw_abs_error = torch.abs(self._wrap_to_pi(vehicle_forward_yaw - env._landing_yaw_heading_w))
        yaw_error = torch.square(yaw_abs_error)
        root_ang_vel_w = torch.nan_to_num(env._robot.data.root_com_ang_vel_w, nan=0.0, posinf=0.0, neginf=0.0)
        yaw_rate_error = torch.square(root_ang_vel_w[:, 2])

        tilt_values = torch.nan_to_num(env.vehicle.morph_joint_positions(), nan=0.0, posinf=1e3, neginf=-1e3)
        landing_tilt_target_values = torch.nan_to_num(
            env.vehicle.joint_config_target_values(
                env.vehicle.spec.landing_joint_config,
                env.vehicle.spec.morph_joint_group,
                tilt_values,
                env.vehicle.spec.morph_target,
            ),
            nan=0.0,
            posinf=1e3,
            neginf=-1e3,
        )
        airborne_tuck_target_values = torch.minimum(
            landing_tilt_target_values,
            torch.full_like(landing_tilt_target_values, 80.0 * pi / 180.0),
        )
        tuck_completion = torch.mean(
            torch.clamp(
                tilt_values / airborne_tuck_target_values.clamp_min(1e-6),
                min=0.0,
                max=1.0,
            ),
            dim=1,
        )
        tuck_progress = tuck_completion - env._previous_tuck_completion
        height_progress = torch.clamp(
            (env._landing_start_pos_w[:, 2] - root_pos_w[:, 2])
            / (env._landing_start_pos_w[:, 2] - env._desired_pos_w[:, 2]).clamp_min(1e-6),
            min=0.0,
            max=1.0,
        )
        tuck_height_gate = 0.2 + 0.8 * height_progress
        above_target_height = root_pos_w[:, 2] > env._desired_pos_w[:, 2]
        tuck_absolute_reward = (
            tuck_progress
            * tuck_height_gate
            * self.stage_value(
                self.cfg.tuck_absolute_linear_rew_scale,
                "tuck_absolute_linear_rew_scale",
            )
            * (~env._ep_contact_in_acceptance).float()
            * (~invalid_contacts).float()
            * above_target_height.float()
        )
        env._previous_tuck_completion[:] = tuck_completion.detach()

        ref_pos_w, ref_vel_w, _ = self._reference_state()
        trajectory_pos_error = ref_pos_w - root_pos_w
        trajectory_vel_error = ref_vel_w - root_lin_vel_w
        trajectory_pos_error_scaled = torch.linalg.norm(
            torch.cat(
                (
                    trajectory_pos_error[:, :2] / 1.25,
                    trajectory_pos_error[:, 2:3] / 0.35,
                ),
                dim=1,
            ),
            dim=1,
        )
        trajectory_pos_base_score = torch.exp(-trajectory_pos_error_scaled)
        trajectory_pos_distance = torch.linalg.norm(trajectory_pos_error, dim=1)
        trajectory_pos_proximity = torch.clamp(
            1.0 - trajectory_pos_distance,
            min=0.0,
            max=1.0,
        )
        trajectory_pos_reward_per_second = 100.0 * trajectory_pos_proximity
        trajectory_vel_error_scaled = (
            torch.sum(torch.square(trajectory_vel_error[:, :2]), dim=1) / (0.75 * 0.75)
            + torch.square(trajectory_vel_error[:, 2]) / (0.30 * 0.30)
        )
        trajectory_vel_score = (
            3.0
            * torch.exp(-trajectory_vel_error_scaled)
            * (0.2 + 0.8 * trajectory_pos_base_score)
        )
        trajectory_finished = env._time_elapsed >= env._landing_reference_duration.squeeze(1)
        ground_contact = valid_contact_count >= min(3, len(env._valid_contact_ids))
        airborne_trajectory_gate = (
            (~trajectory_finished)
            & (~any_valid_contacts)
            & (~invalid_contacts)
            & (~died)
        ).to(root_pos_w.dtype)
        ground_trajectory_gate = (
            trajectory_finished
            & env._ep_contact_in_acceptance
            & ground_contact
            & (~died)
        ).to(root_pos_w.dtype)

        accepted_contact = reward_contacts & ~died
        accepted_contact &= ~invalid_contacts
        min_time_remaining_s = max(
            float(
                self.stage_value(
                    self.cfg.contact_reward_min_time_remaining_s,
                    "contact_reward_min_time_remaining_s",
                )
            ),
            0.0,
        )
        time_remaining = max(float(env.cfg.episode_length_s), 0.0) - env._time_elapsed
        landing_reward_ready = (
            accepted_contact
            & (time_remaining > min_time_remaining_s)
            & (~env._ep_contact_in_acceptance)
        )
        has_landed = env._ep_contact_in_acceptance | landing_reward_ready
        env._ep_contact_in_acceptance = has_landed
        timeout_high = time_out & (~has_landed) & (~env._ep_invalid_contact)
        timeout_no_contact_penalty = timeout_high.float() * self.stage_value(self.cfg.timeout_pen, "timeout_pen")
        excess_vel = torch.clamp(vertical_speed - 2.0, min=0.0)
        too_fast_termination = died & too_fast_vertical & ~nonfinite_state
        too_fast_penalty = too_fast_termination.float() * (
            self.stage_value(self.cfg.died_pen, "died_pen")
            - excess_vel * self.stage_value(self.cfg.died_excess_vel_pen_scale, "died_excess_vel_pen_scale")
        )
        nonfinite_termination = died & nonfinite_state
        nonfinite_state_penalty = nonfinite_termination.float() * self.stage_value(self.cfg.died_pen, "died_pen")
        no_landing_terminal_debt = (
            died.float()
            * (~has_landed).float()
            * (
                self.stage_value(self.cfg.died_no_landing_pen, "died_no_landing_pen")
                + torch.clamp(time_remaining / max(float(env.cfg.episode_length_s), 1e-6), min=0.0, max=1.0)
                * self.stage_value(self.cfg.died_no_landing_time_pen_scale, "died_no_landing_time_pen_scale")
            )
        )

        worst_joint_weight = max(
            0.0,
            min(
                1.0,
                float(
                    self.stage_value(
                        self.cfg.contact_tuck_worst_joint_weight,
                        "contact_tuck_worst_joint_weight",
                    )
                ),
            ),
        )
        contact_tuck_per_joint_score = torch.clamp(
            tilt_values / airborne_tuck_target_values.clamp_min(1e-6),
            min=0.0,
            max=1.0,
        )
        contact_tuck_score = (1.0 - worst_joint_weight) * torch.mean(
            contact_tuck_per_joint_score, dim=1
        ) + worst_joint_weight * torch.min(contact_tuck_per_joint_score, dim=1).values
        current_tuck_multiplier = contact_tuck_score
        first_valid_contact = any_valid_contacts & (~env._touchdown_tuck_recorded)
        env._touchdown_tuck_multiplier = torch.where(
            first_valid_contact,
            current_tuck_multiplier.detach(),
            env._touchdown_tuck_multiplier,
        )
        env._touchdown_tuck_recorded |= any_valid_contacts
        contact_rew_factor = torch.where(
            landing_reward_ready,
            env._touchdown_tuck_multiplier,
            torch.zeros_like(current_tuck_multiplier),
        )
        contact_speed_free = max(
            float(self.stage_value(self.cfg.contact_speed_rew_free_speed, "contact_speed_rew_free_speed")),
            0.0,
        )
        contact_speed_zero = max(
            float(self.stage_value(self.cfg.contact_speed_rew_zero_speed, "contact_speed_rew_zero_speed")),
            contact_speed_free + 1e-6,
        )
        vertical_speed = torch.abs(root_lin_vel_w[:, 2])
        contact_speed_multiplier = torch.clamp(
            (contact_speed_zero - vertical_speed) / max(contact_speed_zero - contact_speed_free, 1e-6),
            min=0.0,
            max=1.0,
        )
        invalid_contact_penalty = torch.where(
            post_landing_invalid_contact,
            torch.full_like(
                root_pos_w[:, 0],
                float(
                    self.stage_value(
                        self.cfg.post_landing_invalid_contact_pen,
                        "post_landing_invalid_contact_pen",
                    )
                ),
            ),
            torch.where(
                terminating_invalid_contact,
                torch.full_like(
                    root_pos_w[:, 0],
                    float(self.stage_value(self.cfg.timeout_pen, "timeout_pen")),
                ),
                new_invalid_contacts.float()
                * self.stage_value(self.cfg.invalid_contact_pen, "invalid_contact_pen"),
            ),
        )
        touchdown_timing_score = torch.exp(
            -torch.square(
                (env._time_elapsed - env._landing_reference_duration.squeeze(1)) / 0.25
            )
        )
        first_to_last_contact_delay = torch.max(valid_contact_time, dim=1).values
        contact_sync_score = torch.exp(
            -torch.square(torch.clamp(first_to_last_contact_delay - 0.05, min=0.0) / 0.15)
        )
        touchdown_xy_velocity_error = torch.linalg.norm(
            root_lin_vel_w[:, :2] - env._landing_ground_velocity_w[:, :2], dim=1
        )
        touchdown_xy_velocity_score = torch.exp(
            -torch.square(touchdown_xy_velocity_error / 0.40)
        )
        contact_baseline = self.stage_value(
            self.cfg.contact_in_acceptance_baseline_rew,
            "contact_in_acceptance_baseline_rew",
        )
        contact_quality_scale = self.stage_value(
            self.cfg.contact_in_acceptance_rew_scale,
            "contact_in_acceptance_rew_scale",
        )
        gated_contact_rew = landing_reward_ready.float() * (
            contact_baseline
            + trajectory_pos_proximity
            * torch.square(contact_rew_factor)
            * contact_speed_multiplier
            * contact_quality_scale
        )
        final_landing_quality_ready = torch.zeros_like(landing_reward_ready)
        if env.curriculum_stage_for_value("contact_quality") >= 3:
            final_landing_quality_ready = (
                has_landed
                & (valid_contact_count >= len(env._valid_contact_ids))
                & (~invalid_contacts)
                & (~died)
                & (~env._final_landing_quality_reward_given)
            )
        final_landing_quality_rew = (
            final_landing_quality_ready.float()
            * current_tuck_multiplier
            * contact_speed_multiplier
            * touchdown_timing_score
            * contact_sync_score
            * touchdown_xy_velocity_score
            * 700.0
        )
        env._final_landing_quality_reward_given |= final_landing_quality_ready
        contact_reward_cap = max(
            float(self.stage_value(self.cfg.contact_in_acceptance_rew_cap, "contact_in_acceptance_rew_cap")),
            0.0,
        )
        if contact_reward_cap > 0.0:
            gated_contact_rew = torch.clamp(gated_contact_rew, max=contact_reward_cap)
        tuck_progress_rew = tuck_absolute_reward
        tuck_absolute_rew_cap = max(
            float(
                self.stage_value(
                    self.cfg.tuck_absolute_rew_cap,
                    "tuck_absolute_rew_cap",
                )
            ),
            0.0,
        )
        if tuck_absolute_rew_cap > 0.0:
            tuck_progress_rew = torch.clamp(
                tuck_progress_rew,
                min=-tuck_absolute_rew_cap,
                max=tuck_absolute_rew_cap,
            )
        valid_contact_fraction = valid_contact_count.float() / max(float(len(env._valid_contact_ids)), 1.0)
        landing_config_stage = env.curriculum_stage_for_value("landing_config")
        landing_config_contact = has_landed & any_valid_contacts
        if landing_config_stage >= 2:
            landing_config_contact_multiplier = torch.square(valid_contact_fraction)
        else:
            landing_config_contact_multiplier = torch.ones_like(valid_contact_fraction)
        landing_config_rew = (
            landing_config_contact.float()
            * (~died).float()
            * landing_config_contact_multiplier
            * self._landing_config_score()
            * self.stage_value(self.cfg.landing_config_rew_scale, "landing_config_rew_scale")
            * env.step_dt
        )
        action_rate = torch.nan_to_num(env.vehicle.action_rate(), nan=0.0, posinf=1e3, neginf=0.0)
        action_rate_scale = self.stage_value(self.cfg.action_rate_pen_scale, "action_rate_pen_scale")
        if env.final_stage_overlay_active():
            action_rate_scale *= 10.0
        thrust_center_action_mag = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        thrust_center_action_name = next(
            (name for name in ("thrust_center_xy", "thrust_center") if name in env.vehicle.action_schema.slices),
            None,
        )
        if thrust_center_action_name is not None:
            policy_terms = env.vehicle.action_schema.split(env._policy_actions)
            thrust_center_action = torch.nan_to_num(
                policy_terms[thrust_center_action_name],
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            )
            thrust_center_action_mag = torch.sum(torch.square(thrust_center_action), dim=1)
        thrust_center_offset = torch.nan_to_num(
            env.vehicle.thrust_center_offset_body(), nan=0.0, posinf=1e3, neginf=-1e3
        )
        thrust_center_error = torch.linalg.norm(thrust_center_offset, dim=1)
        thrust_center_penalty = thrust_center_action_mag * self.stage_value(
            self.cfg.thrust_center_action_pen_scale, "thrust_center_action_pen_scale"
        ) + thrust_center_error * self.stage_value(
            self.cfg.thrust_center_loss_offset_pen_scale,
            "thrust_center_loss_offset_pen_scale",
        )
        filtered_rotor_action = torch.nan_to_num(
            env.vehicle.rotor_action_values(filtered=True),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
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
        airborne_rotor_agreement = rotor_agreement * (~any_valid_contacts & ~has_landed & ~died).float()
        post_landing_thrust_score = torch.sum(torch.square(filtered_rotor_action), dim=1)
        joint_disagreement = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        tuck_group_name = env.vehicle.spec.morph_joint_group
        if tuck_group_name is not None and tuck_group_name in env.vehicle.joint_groups:
            joint_disagreement = torch.nan_to_num(
                env.vehicle.actuator_group_position_disagreement(tuck_group_name),
                nan=0.0,
                posinf=1e3,
                neginf=0.0,
            )
        yaw_reward_gate = torch.ones(env.num_envs, dtype=root_pos_w.dtype, device=env.device)
        rewards = {
            "trajectory_pos_rew": trajectory_pos_reward_per_second
            * (airborne_trajectory_gate + ground_trajectory_gate)
            * env.step_dt,
            "trajectory_vel_rew": trajectory_vel_score
            * (airborne_trajectory_gate + ground_trajectory_gate)
            * env.step_dt,
            "yaw_angle_penalty": yaw_error
            * yaw_reward_gate
            * self.stage_value(self.cfg.yaw_angle_pen_scale, "yaw_angle_pen_scale")
            * env.step_dt,
            "yaw_rate_penalty": yaw_rate_error
            * yaw_reward_gate
            * self.stage_value(self.cfg.yaw_rate_pen_scale, "yaw_rate_pen_scale")
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
            "thrust_center_penalty": thrust_center_penalty * env.step_dt,
            "post_landing_thrust_penalty": post_landing_thrust_score
            * has_landed.float()
            * self.stage_value(self.cfg.post_landing_thrust_pen_scale, "post_landing_thrust_pen_scale")
            * env.step_dt,
            "post_landing_takeoff_penalty": -1200.0 * env._post_landing_takeoff.float(),
            "joint_disagreement_penalty": joint_disagreement
            * self.stage_value(self.cfg.joint_disagreement_pen_scale, "joint_disagreement_pen_scale")
            * env.step_dt,
            "tuck_progress_rew": tuck_progress_rew,
            "contact_in_acceptance_rew": gated_contact_rew,
            "final_landing_quality_rew": final_landing_quality_rew,
            "landing_config_rew": landing_config_rew,
            "invalid_contact_penalty": invalid_contact_penalty,
            "too_fast_penalty": too_fast_penalty,
            "nonfinite_state_penalty": nonfinite_state_penalty,
            "no_landing_terminal_penalty": no_landing_terminal_debt,
            "timeout_high_penalty": timeout_no_contact_penalty,
        }
        rewards = {key: torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0) for key, value in rewards.items()}
        self.log_rewards(rewards, died | time_out)
        reward = self.reward_mixer.sum(rewards)

        return reward

    @staticmethod
    def _yaw_from_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return torch.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def _termination_masks(self):
        env = self.env
        time_out = env.episode_length_buf >= env.max_episode_length - 1
        root_pos_w = env._robot.data.root_link_pos_w
        root_lin_vel_w = env._robot.data.root_com_lin_vel_w
        root_quat_w = env._robot.data.root_link_quat_w
        root_ang_vel_w = env._robot.data.root_com_ang_vel_w
        nonfinite_root_state = (
            ~torch.isfinite(root_pos_w).all(dim=1)
            | ~torch.isfinite(root_lin_vel_w).all(dim=1)
            | ~torch.isfinite(root_quat_w).all(dim=1)
            | ~torch.isfinite(root_ang_vel_w).all(dim=1)
        )
        env._nonfinite_root_state = nonfinite_root_state
        too_fast_vertical = torch.abs(
            torch.nan_to_num(root_lin_vel_w[:, 2], nan=0.0, posinf=1e6, neginf=-1e6)
        ) > 2.0
        contact_state = self._contact_state()
        invalid_contacts = contact_state["invalid_contacts"]
        post_landing_invalid_contact = invalid_contacts & env._ep_contact_in_acceptance
        terminating_invalid_contact = invalid_contacts & (
            env._ep_contact_in_acceptance | env.final_stage_overlay_active()
        )
        if not env.cfg.terminate:
            terminating_invalid_contact = torch.zeros_like(terminating_invalid_contact)
            post_landing_invalid_contact = torch.zeros_like(post_landing_invalid_contact)
        env._post_landing_takeoff[:] = (
            env._ep_contact_in_acceptance
            & (~contact_state["any_valid_contacts"])
            & (root_pos_w[:, 2] > env._desired_pos_w[:, 2] + 0.05)
        )
        died = (
            nonfinite_root_state | too_fast_vertical | terminating_invalid_contact | env._post_landing_takeoff
            if env.cfg.terminate
            else torch.zeros_like(nonfinite_root_state)
        )
        return {
            "died": died,
            "time_out": time_out,
            "too_fast_vertical": too_fast_vertical,
            "nonfinite_state": nonfinite_root_state,
            "invalid_contacts": invalid_contacts,
            "terminating_invalid_contact": terminating_invalid_contact,
            "post_landing_invalid_contact": post_landing_invalid_contact,
        }

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        termination = self._termination_masks()
        return termination["died"], termination["time_out"]

    def reset_episode_state(self, env_ids: torch.Tensor):
        self._observation_context_cache = None
        env = self.env
        ground_drive_bucket = env._ground_drive_bucket[env_ids]
        env._ep_contact_in_acceptance[env_ids] = ground_drive_bucket
        env._ep_invalid_contact[env_ids] = False
        env._nonfinite_root_state[env_ids] = False
        env._post_landing_takeoff[env_ids] = False
        env._previous_tuck_completion[env_ids] = ground_drive_bucket.float()
        env._touchdown_tuck_multiplier[env_ids] = ground_drive_bucket.float()
        env._touchdown_tuck_recorded[env_ids] = ground_drive_bucket
        env._final_landing_quality_reward_given[env_ids] = ground_drive_bucket
        env._invalid_contact_pen_given[env_ids] = False

        ground_env_ids = env_ids[ground_drive_bucket]
        if len(ground_env_ids) > 0:
            env._disturbance_force[ground_env_ids] = 0.0
            env._disturbance_moment[ground_env_ids] = 0.0
            env._disturbance_force_cts[ground_env_ids] = 0.0
            env._disturbance_moment_cts[ground_env_ids] = 0.0
            env._disturbance_batch_scale[ground_env_ids] = 0.0
            env._thrust_loss_batch_scale[ground_env_ids] = 0.0
            env._push_active[ground_env_ids] = False
            env.kT[ground_env_ids] = env.vehicle._nominal_kT_tensor
            env.kM[ground_env_ids] = env.vehicle._nominal_kM_tensor
            env.normalized_rotor_thrust_filtered[ground_env_ids] = 0.0
            env.previous_normalized_rotor_thrust_filtered[ground_env_ids] = 0.0
