from __future__ import annotations

from numpy import pi
import torch

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import quat_from_euler_xyz

from .landing_task import RewardMixer


@configclass
class TakeoffTaskCfg:
    target_pos = [0.0, 0.0, 1.0]
    target_height_range = [(0.8, 1.2), (0.8, 1.2)]
    target_xy_offset_range = [15.0, 15.0]
    start_xy_range = [0.5, 0.5]
    # Root-link z offset, relative to the terrain origin, when the transformed vehicle just touches the ground.
    start_root_height = [0.2, 0.2]
    start_roll_pitch_range = [0.0, 0.0]
    start_yaw_range = [pi, pi]
    start_morph_angle = [pi / 2, pi / 2]

    hover_height_rew_scale = [6.0, 6.0]
    hover_height_flat_bonus = [1.0, 1.0]
    ground_contact_pen_scale = [-0.50, -0.50]
    low_rotor_thrust_threshold = [0.55, 0.00]
    low_rotor_thrust_pen_scale = [-0.50, -0.20]
    yaw_angle_pen_scale = [-1.00, -2.00]
    roll_pitch_angle_pen_scale = [0.0, -0.05]
    xy_dist_pen_scale = [-1.2, -3.20]
    xy_dist_pen_error_cap = [2.0, 2.0]
    target_lin_vel_pen_scale = [-0.07, -0.15]
    action_mag_pen_scale = [0.0, -0.6]
    wasted_thrust_pen_scale = [0.0, -0.002]
    action_rate_pen_scale = [-0.20, -0.2]
    tilt_rew_scale = [4.00, 0.80]
    invalid_contact_pen = [0.0, -2.0]
    disturbance_force_scale = [4.0, 2.0]
    disturbance_moment_scale = [4.0, 2.0]
    disturbance_cts_force_scale = [2.0, 2.0]

    hover_pos_tolerance = [0.35, 0.15]
    hover_lin_vel_tolerance = [0.35, 0.15]
    hover_ang_vel_tolerance = [0.50, 0.20]


class TakeoffTask:
    """Takeoff task: start transformed on the ground and recover to hover at a target pose."""

    reward_keys = (
        "ground_contact_pen",
        "low_rotor_thrust_pen",
        "yaw_angle_pen",
        "roll_pitch_angle_pen",
        "hover_height_rew",
        "xy_dist_pen",
        "target_lin_vel_pen",
        "action_mag_pen",
        "wasted_thrust_pen",
        "action_rate_pen",
        "tilt_rew",
        "invalid_contact_pen",
    )

    def __init__(self, env, cfg: TakeoffTaskCfg):
        self.env = env
        self.cfg = cfg
        self.reward_mixer = RewardMixer(self.reward_keys)

        env._desired_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._target_pos_obs_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._virtual_xy_offset_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._takeoff_spawn_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._initial_yaw = torch.zeros(env.num_envs, device=env.device)
        env._initial_roll_pitch = torch.zeros(env.num_envs, 2, device=env.device)
        env._previous_yaw = torch.zeros(env.num_envs, device=env.device)
        env._unwrapped_yaw_error = torch.zeros(env.num_envs, device=env.device)
        env._previous_roll_pitch = torch.zeros(env.num_envs, 2, device=env.device)
        env._unwrapped_roll_pitch_error = torch.zeros(env.num_envs, 2, device=env.device)
        env._ep_hover_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._episode_sums = {
            key: torch.zeros(env.num_envs, dtype=torch.float, device=env.device) for key in self.reward_keys
        }

    def stage_value(self, values, name: str):
        try:
            stage = int(self.env.cfg.curriculum_stage)
            if stage < 1:
                raise ValueError(f"curriculum_stage must be >= 1, got {self.env.cfg.curriculum_stage}")
            return values[stage - 1]
        except TypeError:
            return values
        except IndexError as exc:
            raise ValueError(f"curriculum_stage={self.env.cfg.curriculum_stage} has no entry for {name}") from exc

    def disturbance_force_scale(self) -> float:
        return self.env.cfg.disturbance_force_scale * self.stage_value(
            self.cfg.disturbance_force_scale,
            "disturbance_force_scale",
        )

    def disturbance_moment_scale(self) -> float:
        return self.env.cfg.disturbance_moment_scale * self.stage_value(
            self.cfg.disturbance_moment_scale,
            "disturbance_moment_scale",
        )

    def disturbance_cts_force_scale(self) -> float:
        return self.env.cfg.disturbance_force_scale * self.stage_value(
            self.cfg.disturbance_cts_force_scale,
            "disturbance_cts_force_scale",
        )

    def reset_initial_state(self, env_ids: torch.Tensor, randomized: bool):
        env = self.env
        target = torch.tensor(self.cfg.target_pos, device=env.device, dtype=torch.float).repeat(len(env_ids), 1)
        virtual_xy_offset = torch.zeros(len(env_ids), 2, device=env.device)
        if randomized:
            virtual_xy_offset = virtual_xy_offset.uniform_(
                -self.stage_value(self.cfg.target_xy_offset_range, "target_xy_offset_range"),
                self.stage_value(self.cfg.target_xy_offset_range, "target_xy_offset_range"),
            )
            target_height_min, target_height_max = self.stage_value(
                self.cfg.target_height_range,
                "target_height_range",
            )
            target[:, 2] = torch.zeros(len(env_ids), device=env.device).uniform_(
                target_height_min,
                target_height_max,
            )

        env._desired_pos_w[env_ids] = env._terrain.env_origins[env_ids] + target
        env._virtual_xy_offset_w[env_ids] = 0.0
        env._virtual_xy_offset_w[env_ids, :2] = virtual_xy_offset
        env._target_pos_obs_w[env_ids] = env._desired_pos_w[env_ids] + env._virtual_xy_offset_w[env_ids]

        root_state = env._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] = env._terrain.env_origins[env_ids]
        root_state[:, 2] += self.stage_value(self.cfg.start_root_height, "start_root_height")

        if randomized:
            roll_pitch_range = self.stage_value(self.cfg.start_roll_pitch_range, "start_roll_pitch_range")
            yaw_range = self.stage_value(self.cfg.start_yaw_range, "start_yaw_range")
            xy_offset = torch.zeros(len(env_ids), 2, device=env.device).uniform_(
                -self.stage_value(self.cfg.start_xy_range, "start_xy_range"),
                self.stage_value(self.cfg.start_xy_range, "start_xy_range"),
            )
            roll = torch.zeros(len(env_ids), device=env.device).uniform_(
                -roll_pitch_range,
                roll_pitch_range,
            )
            pitch = torch.zeros(len(env_ids), device=env.device).uniform_(
                -roll_pitch_range,
                roll_pitch_range,
            )
            yaw = torch.zeros(len(env_ids), device=env.device).uniform_(-yaw_range, yaw_range)
        else:
            xy_offset = torch.zeros(len(env_ids), 2, device=env.device)
            roll = torch.zeros(len(env_ids), device=env.device)
            pitch = torch.zeros(len(env_ids), device=env.device)
            yaw = torch.zeros(len(env_ids), device=env.device)

        root_state[:, :2] += xy_offset
        root_state[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
        root_state[:, 7:13] = 0.0
        env._takeoff_spawn_pos_w[env_ids] = root_state[:, :3]
        env._initial_yaw[env_ids] = yaw
        env._initial_roll_pitch[env_ids] = torch.stack((roll, pitch), dim=1)
        env._previous_yaw[env_ids] = yaw
        env._unwrapped_yaw_error[env_ids] = 0.0
        env._previous_roll_pitch[env_ids] = env._initial_roll_pitch[env_ids]
        env._unwrapped_roll_pitch_error[env_ids] = 0.0

        joint_pos, joint_vel = env.vehicle.deterministic_joint_state(env_ids)
        env.vehicle.set_joint_group_state(
            joint_pos,
            joint_vel,
            env_ids,
            env.vehicle.spec.landing_tuck_joint_group,
            self.stage_value(self.cfg.start_morph_angle, "start_morph_angle"),
            0.0,
        )

        env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        env._robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        env._robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)

    def joint_action_direction(self, group_name: str) -> float:
        if group_name == self.env.vehicle.spec.landing_tuck_joint_group:
            return -1.0
        return 1.0

    def reset_policy_action_value(self, fill_actions_with_ones: bool) -> float:
        return 0.0

    def warm_start_action_filter(self) -> bool:
        return True

    def get_rewards(self) -> torch.Tensor:
        env = self.env
        died, _ = self.get_dones()

        desired_pos_w = torch.nan_to_num(env._desired_pos_w, nan=0.0, posinf=1e6, neginf=-1e6)
        root_pos_w = torch.nan_to_num(env._robot.data.root_link_pos_w, nan=0.0, posinf=1e6, neginf=-1e6)
        root_lin_vel_w = torch.nan_to_num(env._robot.data.root_com_lin_vel_w, nan=0.0, posinf=1e3, neginf=-1e3)
        root_ang_vel_b = torch.nan_to_num(env._robot.data.root_com_ang_vel_b, nan=0.0, posinf=1e3, neginf=-1e3)

        pos_error_vec = desired_pos_w - root_pos_w
        pos_error = torch.linalg.norm(pos_error_vec, dim=1)
        xy_error = torch.linalg.norm(pos_error_vec[:, :2], dim=1)
        xy_error_for_pen = torch.clamp(
            xy_error,
            max=self.stage_value(self.cfg.xy_dist_pen_error_cap, "xy_dist_pen_error_cap"),
        )

        valid_contact_time = env.scene["contact_sensor"].data.current_contact_time[:, env._valid_contact_ids]
        ground_contact = torch.any(valid_contact_time > 0.0, dim=1)
        no_ground_contact = ~ground_contact

        spawn_pos_w = torch.nan_to_num(env._takeoff_spawn_pos_w, nan=0.0, posinf=1e6, neginf=-1e6)
        spawn_z = spawn_pos_w[:, 2]
        root_z = root_pos_w[:, 2]
        target_z = desired_pos_w[:, 2]
        height_span = torch.clamp(target_z - spawn_z, min=1e-3)
        hover_peak_rew = self.stage_value(self.cfg.hover_height_rew_scale, "hover_height_rew_scale")
        hover_flat_bonus = self.stage_value(self.cfg.hover_height_flat_bonus, "hover_height_flat_bonus")
        climb_progress = torch.clamp((root_z - spawn_z) / height_span, 0.0, 1.0)
        below_target_rew = hover_flat_bonus + (hover_peak_rew - hover_flat_bonus) * climb_progress
        overshoot_progress = torch.clamp((root_z - target_z) / height_span, 0.0, 1.0)
        above_target_rew = hover_peak_rew * (1.0 - overshoot_progress)
        hover_height_rew = torch.where(root_z <= target_z, below_target_rew, above_target_rew)
        hover_height_rew = torch.where(
            no_ground_contact & ~died,
            hover_height_rew,
            torch.zeros_like(hover_height_rew),
        )
        lin_vel_norm = torch.linalg.norm(root_lin_vel_w, dim=1)
        ang_vel_norm = torch.linalg.norm(root_ang_vel_b, dim=1)
        target_lin_vel_w = torch.tensor(
            env.vehicle.spec.takeoff_target_lin_vel_w,
            device=env.device,
            dtype=root_lin_vel_w.dtype,
        ).repeat(env.num_envs, 1)
        slow_start_z = spawn_z + 0.8 * height_span
        slow_span = torch.clamp(target_z - slow_start_z, min=1e-3)
        slow_progress = torch.clamp((root_z - slow_start_z) / slow_span, 0.0, 1.0)
        target_lin_vel_w = target_lin_vel_w * (1.0 - slow_progress).unsqueeze(dim=1)
        # Only vertical velocity is penalized so lateral corrections are not discouraged near hover.
        target_lin_vel_error = torch.abs(root_lin_vel_w[:, 2] - target_lin_vel_w[:, 2])
        commanded_rotor_action = torch.nan_to_num(
            env.vehicle.rotor_action_values(filtered=False),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        low_rotor_threshold = self.stage_value(self.cfg.low_rotor_thrust_threshold, "low_rotor_thrust_threshold")
        low_rotor_thrust = torch.sum(
            torch.square(torch.clamp(low_rotor_threshold - commanded_rotor_action, min=0.0)),
            dim=1,
        )
        tilt = torch.nan_to_num(env.vehicle.landing_tuck_position(), nan=0.0, posinf=1e3, neginf=-1e3)
        action_mag = torch.sum(torch.square(commanded_rotor_action), dim=1)
        filtered_rotor_action = torch.nan_to_num(
            env.vehicle.rotor_action_values(filtered=True),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        realized_rotor_thrust = torch.nan_to_num(env.kT * filtered_rotor_action, nan=0.0, posinf=1e6, neginf=0.0)
        cosine_loss = 1.0 - torch.cos(tilt.clamp(0.0, torch.pi / 2))
        wasted_thrust = torch.sum(realized_rotor_thrust * cosine_loss.unsqueeze(dim=1), dim=1)
        wasted_thrust = torch.square(wasted_thrust)
        wasted_thrust = wasted_thrust * (root_z < target_z).float()
        action_rate = torch.nan_to_num(env.vehicle.rotor_action_rate(), nan=0.0, posinf=1e3, neginf=0.0)
        root_quat_w = torch.nan_to_num(env._robot.data.root_link_quat_w, nan=0.0, posinf=1.0, neginf=-1.0)
        rpy = self._rpy_from_quat(root_quat_w)
        roll_pitch = rpy[:, :2]
        roll_pitch_delta = self._wrap_to_pi(roll_pitch - env._previous_roll_pitch)
        env._unwrapped_roll_pitch_error += roll_pitch_delta
        env._previous_roll_pitch[:] = roll_pitch.detach()
        roll_pitch_error = torch.sum(torch.abs(env._unwrapped_roll_pitch_error), dim=1)
        yaw = rpy[:, 2]
        yaw_delta = self._wrap_to_pi(yaw - env._previous_yaw)
        env._unwrapped_yaw_error += yaw_delta
        env._previous_yaw[:] = yaw.detach()
        yaw_error = torch.abs(env._unwrapped_yaw_error)
        tilt_error_mapped = torch.exp(-torch.square(tilt) / 0.25)
        invalid_contact_time = env.scene["contact_sensor"].data.current_contact_time[:, env._invalid_contact_ids]
        bad_contact = torch.any(invalid_contact_time > 0.0, dim=1)

        lin_vel_tolerance = self.stage_value(self.cfg.hover_lin_vel_tolerance, "hover_lin_vel_tolerance")
        ang_vel_tolerance = self.stage_value(self.cfg.hover_ang_vel_tolerance, "hover_ang_vel_tolerance")
        low_lin_vel = lin_vel_norm < lin_vel_tolerance
        low_ang_vel = ang_vel_norm < ang_vel_tolerance
        hover_success = (
            (pos_error < self.stage_value(self.cfg.hover_pos_tolerance, "hover_pos_tolerance"))
            & low_lin_vel
            & low_ang_vel
            & ~died
        )
        env._ep_hover_success |= hover_success

        rewards = {
            "ground_contact_pen": ground_contact.float()
            * self.stage_value(self.cfg.ground_contact_pen_scale, "ground_contact_pen_scale")
            * env.step_dt,
            "low_rotor_thrust_pen": low_rotor_thrust
            * self.stage_value(self.cfg.low_rotor_thrust_pen_scale, "low_rotor_thrust_pen_scale")
            * env.step_dt,
            "yaw_angle_pen": yaw_error
            * self.stage_value(self.cfg.yaw_angle_pen_scale, "yaw_angle_pen_scale")
            * env.step_dt,
            "roll_pitch_angle_pen": roll_pitch_error
            * self.stage_value(self.cfg.roll_pitch_angle_pen_scale, "roll_pitch_angle_pen_scale")
            * env.step_dt,
            "hover_height_rew": hover_height_rew * env.step_dt,
            "xy_dist_pen": xy_error_for_pen
            * self.stage_value(self.cfg.xy_dist_pen_scale, "xy_dist_pen_scale")
            * env.step_dt,
            "target_lin_vel_pen": target_lin_vel_error
            * self.stage_value(self.cfg.target_lin_vel_pen_scale, "target_lin_vel_pen_scale")
            * env.step_dt,
            "action_mag_pen": action_mag
            * self.stage_value(self.cfg.action_mag_pen_scale, "action_mag_pen_scale")
            * env.step_dt,
            "wasted_thrust_pen": wasted_thrust
            * self.stage_value(self.cfg.wasted_thrust_pen_scale, "wasted_thrust_pen_scale")
            * env.step_dt,
            "action_rate_pen": action_rate
            * self.stage_value(self.cfg.action_rate_pen_scale, "action_rate_pen_scale")
            * env.step_dt,
            "tilt_rew": tilt_error_mapped * self.stage_value(self.cfg.tilt_rew_scale, "tilt_rew_scale") * env.step_dt,
            "invalid_contact_pen": bad_contact.float()
            * self.stage_value(self.cfg.invalid_contact_pen, "invalid_contact_pen")
            * env.step_dt,
        }
        rewards = {key: torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0) for key, value in rewards.items()}
        reward = self.reward_mixer.sum(rewards)
        for key, value in rewards.items():
            env._episode_sums[key] += value
        return reward

    def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        env = self.env
        time_out = env.episode_length_buf >= env.max_episode_length - 1
        finite_state = (
            torch.isfinite(env._robot.data.root_link_pos_w).all(dim=1)
            & torch.isfinite(env._robot.data.root_link_quat_w).all(dim=1)
            & torch.isfinite(env._robot.data.root_com_lin_vel_w).all(dim=1)
            & torch.isfinite(env._robot.data.root_com_ang_vel_b).all(dim=1)
        )
        return ~finite_state, time_out

    def log_episode(self, env_ids: torch.Tensor):
        env = self.env
        finite_state = (
            torch.isfinite(env._robot.data.root_link_pos_w[env_ids]).all(dim=1)
            & torch.isfinite(env._robot.data.root_link_quat_w[env_ids]).all(dim=1)
            & torch.isfinite(env._robot.data.root_com_lin_vel_w[env_ids]).all(dim=1)
            & torch.isfinite(env._robot.data.root_com_ang_vel_b[env_ids]).all(dim=1)
        )
        pos_error = torch.linalg.norm(
            env._desired_pos_w[env_ids] - env._robot.data.root_link_pos_w[env_ids],
            dim=1,
        ).mean()

        extras = {}
        for key in env._episode_sums.keys():
            episodic_sum_avg = self._finite_mean(env._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = torch.nan_to_num(
                episodic_sum_avg / env.max_episode_length_s,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            env._episode_sums[key][env_ids] = 0.0

        env.extras["log"] = {}
        env.extras["log"].update(extras)

        timeout = env.reset_time_outs[env_ids]
        success = env._ep_hover_success[env_ids]
        num_resets = max(len(env_ids), 1)
        env.extras["log"].update(
            {
                "Metrics/final_position_error": torch.nan_to_num(
                    pos_error,
                    nan=0.0,
                    posinf=1e6,
                    neginf=1e6,
                ).item(),
                "Takeoff/hover_success_frequency": torch.count_nonzero(success & timeout).item() / num_resets,
                "Takeoff/terminated_no_hover_frequency": torch.count_nonzero(
                    env.reset_terminated[env_ids] & ~success
                ).item()
                / num_resets,
                "Takeoff/non_finite_state_frequency": torch.count_nonzero(~finite_state).item() / num_resets,
            }
        )

    def reset_episode_state(self, env_ids: torch.Tensor):
        env = self.env
        env._ep_hover_success[env_ids] = False
        env._previous_yaw[env_ids] = env._initial_yaw[env_ids]
        env._unwrapped_yaw_error[env_ids] = 0.0
        env._previous_roll_pitch[env_ids] = env._initial_roll_pitch[env_ids]
        env._unwrapped_roll_pitch_error[env_ids] = 0.0

    @staticmethod
    def _finite_mean(values: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return torch.zeros((), device=values.device)
        return torch.nan_to_num(torch.mean(values), nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _rpy_from_quat(q: torch.Tensor) -> torch.Tensor:
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return torch.stack((roll, pitch, yaw), dim=1)

    @staticmethod
    def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))
